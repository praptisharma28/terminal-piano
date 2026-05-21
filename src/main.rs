use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::io::Write;
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind};
use crossterm::style::{Color, Print, ResetColor, SetBackgroundColor, SetForegroundColor};
use crossterm::terminal::{self, ClearType};
use crossterm::{cursor, execute, queue};

const ATTACK:     f32 = 0.01; // seconds to reach full volume
const DECAY:      f32 = 0.10; // seconds to drop to sustain level
const SUSTAIN:    f32 = 0.70; // volume level held while key is down (0.0 - 1.0)
const RELEASE:    f32 = 0.30; // seconds to fade out after key is released
const TREM_DEPTH: f32 = 0.75; // how deeply tremolo dips the volume (0.0 = none, 1.0 = silence)
const VIB_DEPTH:  f32 = 0.50; // vibrato pitch swing in semitones (±0.5 semitones)

const MIN_OCTAVE:    i32 = -3;
const MAX_OCTAVE:    i32 =  3;
const MIN_TRANSPOSE: i32 = -12;
const MAX_TRANSPOSE: i32 =  12;

#[derive(Clone, Copy, PartialEq)]
enum Stage {
    Attack,
    Decay,
    Sustain,
    Release,
}

#[derive(Clone, Copy, PartialEq)]
enum Waveform {
    Sine,
    Triangle,
    Square,
    Sawtooth,
}

impl Waveform {
    fn next(self) -> Self {
        match self {
            Waveform::Sine     => Waveform::Triangle,
            Waveform::Triangle => Waveform::Square,
            Waveform::Square   => Waveform::Sawtooth,
            Waveform::Sawtooth => Waveform::Sine,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Waveform::Sine     => "Sine     — pure, clean",
            Waveform::Triangle => "Triangle — soft, hollow",
            Waveform::Square   => "Square   — reedy, nasal",
            Waveform::Sawtooth => "Sawtooth — bright, buzzy",
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Scale {
    Major,
    NaturalMinor,
    PentatonicMajor,
    PentatonicMinor,
    Blues,
}

impl Scale {
    fn intervals(self) -> &'static [u8] {
        match self {
            Scale::Major           => &[0, 2, 4, 5, 7, 9, 11],
            Scale::NaturalMinor    => &[0, 2, 3, 5, 7, 8, 10],
            Scale::PentatonicMajor => &[0, 2, 4, 7, 9],
            Scale::PentatonicMinor => &[0, 3, 5, 7, 10],
            Scale::Blues           => &[0, 3, 5, 6, 7, 10],
        }
    }

    fn name(self) -> &'static str {
        match self {
            Scale::Major           => "Major",
            Scale::NaturalMinor    => "Natural Minor",
            Scale::PentatonicMajor => "Pentatonic Major",
            Scale::PentatonicMinor => "Pentatonic Minor",
            Scale::Blues           => "Blues",
        }
    }

    fn next(self) -> Self {
        match self {
            Scale::Major           => Scale::NaturalMinor,
            Scale::NaturalMinor    => Scale::PentatonicMajor,
            Scale::PentatonicMajor => Scale::PentatonicMinor,
            Scale::PentatonicMinor => Scale::Blues,
            Scale::Blues           => Scale::Major,
        }
    }
}

const ROOTS: [(&str, u8); 12] = [
    ("C", 0), ("C#", 1), ("D", 2), ("D#", 3), ("E", 4),  ("F", 5),
    ("F#", 6), ("G", 7), ("G#", 8), ("A", 9),  ("A#", 10), ("B", 11),
];

#[derive(Clone, Copy, PartialEq)]
enum ArpPattern { Up, Down, UpDown }

impl ArpPattern {
    fn next(self) -> Self {
        match self {
            ArpPattern::Up     => ArpPattern::Down,
            ArpPattern::Down   => ArpPattern::UpDown,
            ArpPattern::UpDown => ArpPattern::Up,
        }
    }

    fn name(self) -> &'static str {
        match self {
            ArpPattern::Up     => "Up",
            ArpPattern::Down   => "Down",
            ArpPattern::UpDown => "Up-Down",
        }
    }

    fn note_index(self, arp_idx: usize, n: usize) -> usize {
        match self {
            ArpPattern::Up   => arp_idx % n,
            ArpPattern::Down => (n - 1).saturating_sub(arp_idx % n),
            ArpPattern::UpDown => {
                if n <= 1 { return 0; }
                let cycle = 2 * (n - 1);
                let pos   = arp_idx % cycle;
                if pos < n { pos } else { cycle - pos }
            }
        }
    }
}

#[derive(Clone, Copy)]
enum RecordKind { Press, Release }

#[derive(Clone)]
struct RecordedEvent {
    timestamp: Duration, // time elapsed since recording started
    key:       char,
    kind:      RecordKind,
    velocity:  f32,
    waveform:  Waveform,
}

struct Note {
    phase:        f32,
    amplitude:    f32,
    stage:        Stage,
    velocity:     f32,
    waveform:     Waveform,
    key_released: bool,
}

impl Note {
    fn new(velocity: f32, waveform: Waveform) -> Self {
        Note { phase: 0.0, amplitude: 0.0, stage: Stage::Attack, velocity, waveform, key_released: false }
    }

    fn release(&mut self) {
        self.stage = Stage::Release;
    }

    fn is_finished(&self) -> bool {
        self.stage == Stage::Release && self.amplitude <= 0.0
    }

    fn tick(&mut self, freq: f32, sample_rate: f32) -> f32 {
        match self.stage {
            Stage::Attack => {
                self.amplitude += 1.0 / (ATTACK * sample_rate);
                if self.amplitude >= 1.0 {
                    self.amplitude = 1.0;
                    self.stage = Stage::Decay;
                }
            }
            Stage::Decay => {
                self.amplitude -= (1.0 - SUSTAIN) / (DECAY * sample_rate);
                if self.amplitude <= SUSTAIN {
                    self.amplitude = SUSTAIN;
                    self.stage = Stage::Sustain;
                }
            }
            Stage::Sustain => {}
            Stage::Release => {
                self.amplitude -= SUSTAIN / (RELEASE * sample_rate);
                if self.amplitude < 0.0 {
                    self.amplitude = 0.0;
                }
            }
        }

        let v  = self.velocity;
        let p  = self.phase;
        let pi = std::f32::consts::PI;

        let (wave, gain) = match self.waveform {
            Waveform::Sine => (
                (2.0 * pi * p).sin(),
                1.0,
            ),
            Waveform::Triangle => (
                  1.000 * (2.0 * pi * 1.0 * p).sin()
                - 0.111 * v * (2.0 * pi * 3.0 * p).sin()
                + 0.040 * v * (2.0 * pi * 5.0 * p).sin()
                - 0.020 * v * (2.0 * pi * 7.0 * p).sin(),
                0.90,
            ),
            Waveform::Square => (
                  1.000 * (2.0 * pi * 1.0 * p).sin()
                + 0.333 * v * (2.0 * pi * 3.0 * p).sin()
                + 0.200 * v * (2.0 * pi * 5.0 * p).sin()
                + 0.143 * v * (2.0 * pi * 7.0 * p).sin(),
                0.60,
            ),
            Waveform::Sawtooth => (
                  1.000       * (2.0 * pi * 1.0 * p).sin()
                + 0.500 * v   * (2.0 * pi * 2.0 * p).sin()
                + 0.333 * v   * (2.0 * pi * 3.0 * p).sin()
                + 0.250 * v*v * (2.0 * pi * 4.0 * p).sin()
                + 0.200 * v*v * (2.0 * pi * 5.0 * p).sin(),
                0.50,
            ),
        };

        let out = wave * gain * self.amplitude * self.velocity * 0.15;
        self.phase = (self.phase + freq / sample_rate) % 1.0;
        out
    }
}

type ActiveNotes = Arc<Mutex<HashMap<char, Note>>>;
type OctaveShift = Arc<Mutex<i32>>;
type Transpose   = Arc<Mutex<i32>>;
type ReverbMix   = Arc<Mutex<f32>>;
type IsPlaying   = Arc<Mutex<bool>>;
type IsLooping   = Arc<Mutex<bool>>;
type DelayOn     = Arc<Mutex<bool>>;
type DelayFb     = Arc<Mutex<f32>>;
type TremOn      = Arc<Mutex<bool>>;
type TremRate    = Arc<Mutex<f32>>;
type VibOn       = Arc<Mutex<bool>>;
type VibRate     = Arc<Mutex<f32>>;

// Schroeder reverb: four parallel comb filters.
struct Reverb {
    buffers:   [Vec<f32>; 4],
    positions: [usize; 4],
}

impl Reverb {
    fn new(sample_rate: f32) -> Self {
        let ms = |t: f32| (sample_rate * t * 0.001) as usize;
        Reverb {
            buffers: [
                vec![0.0; ms(29.7)], // delay lengths from Schroeder's original paper
                vec![0.0; ms(37.1)],
                vec![0.0; ms(41.1)],
                vec![0.0; ms(43.7)],
            ],
            positions: [0; 4],
        }
    }

    fn process(&mut self, input: f32) -> f32 {
        let gains = [0.805_f32, 0.827, 0.783, 0.764]; // feedback amount per filter
        let mut output = 0.0_f32;
        for i in 0..4 {
            let pos     = self.positions[i];
            let delayed = self.buffers[i][pos];
            self.buffers[i][pos] = input + gains[i] * delayed;
            self.positions[i]    = (pos + 1) % self.buffers[i].len();
            output += delayed;
        }
        output * 0.25
    }
}

// Tap delay: a single ring buffer that plays back the signal after a fixed time.
// feedback controls how much of the delayed output is mixed back in — higher = more echoes.
struct Delay {
    buffer:   Vec<f32>,
    position: usize,
}

impl Delay {
    fn new(sample_rate: f32, ms: f32) -> Self {
        Delay {
            buffer:   vec![0.0; (sample_rate * ms * 0.001) as usize],
            position: 0,
        }
    }

    fn process(&mut self, input: f32, feedback: f32) -> f32 {
        let delayed = self.buffer[self.position];
        self.buffer[self.position] = input + delayed * feedback;
        self.position = (self.position + 1) % self.buffer.len();
        delayed
    }
}

fn key_to_pitch_class(key: char) -> Option<u8> {
    match key {
        'a' | 'k' => Some(9),  // A
        'w' | 'o' => Some(10), // A#
        's' | 'l' => Some(11), // B
        'd' | ';' => Some(0),  // C
        'r'       => Some(1),  // C#
        'f'       => Some(2),  // D
        't'       => Some(3),  // D#
        'g'       => Some(4),  // E
        'h'       => Some(5),  // F
        'u'       => Some(6),  // F#
        'j'       => Some(7),  // G
        'i'       => Some(8),  // G#
        _         => None,
    }
}

fn note_name(pc: u8) -> &'static str {
    match pc {
        0  => "C",  1  => "C#", 2  => "D",  3  => "D#",
        4  => "E",  5  => "F",  6  => "F#", 7  => "G",
        8  => "G#", 9  => "A",  10 => "A#", 11 => "B",
        _  => "?",
    }
}

// transpose: semitone shift to apply to each key's pitch class before chord/scale matching
fn detect_chord(pressed: &HashSet<char>, transpose: i32) -> String {
    let pcs: Vec<u8> = pressed
        .iter()
        .filter_map(|&k| key_to_pitch_class(k)
            .map(|pc| (pc as i32 + transpose).rem_euclid(12) as u8))
        .collect::<std::collections::BTreeSet<u8>>()
        .into_iter()
        .collect();

    match pcs.len() {
        0 => return String::new(),
        1 => return note_name(pcs[0]).to_string(),
        _ => {}
    }

    let shapes: &[(&str, &[u8])] = &[
        ("maj",  &[0, 4, 7]),
        ("min",  &[0, 3, 7]),
        ("dim",  &[0, 3, 6]),
        ("aug",  &[0, 4, 8]),
        ("sus2", &[0, 2, 7]),
        ("sus4", &[0, 5, 7]),
        ("maj7", &[0, 4, 7, 11]),
        ("min7", &[0, 3, 7, 10]),
        ("7",    &[0, 4, 7, 10]),
        ("dim7", &[0, 3, 6, 9]),
        ("m7b5", &[0, 3, 6, 10]),
    ];

    for &root in &pcs {
        let mut intervals: Vec<u8> = pcs
            .iter()
            .map(|&pc| (pc + 12 - root) % 12)
            .collect();
        intervals.sort();
        intervals.dedup();

        for &(name, shape) in shapes {
            if intervals == shape {
                return format!("{} {}", note_name(root), name);
            }
        }
    }

    pcs.iter().map(|&pc| note_name(pc)).collect::<Vec<_>>().join(" ")
}

fn key_to_freq(key: char) -> Option<f32> {
    match key {
        'a' => Some(220.00), // A3
        'w' => Some(233.08), // A#3
        's' => Some(246.94), // B3
        'd' => Some(261.63), // C4 (middle C)
        'r' => Some(277.18), // C#4
        'f' => Some(293.66), // D4
        't' => Some(311.13), // D#4
        'g' => Some(329.63), // E4
        'h' => Some(349.23), // F4
        'u' => Some(369.99), // F#4
        'j' => Some(392.00), // G4
        'i' => Some(415.30), // G#4
        'k' => Some(440.00), // A4 (concert A)
        'o' => Some(466.16), // A#4
        'l' => Some(493.88), // B4
        ';' => Some(523.25), // C5
        _ => None,
    }
}

fn stop_playback(is_looping: &IsLooping, active_notes: &ActiveNotes) {
    *is_looping.lock().unwrap() = false;
    for note in active_notes.lock().unwrap().values_mut() {
        note.release();
    }
}

fn main() -> anyhow::Result<()> {
    let active_notes: ActiveNotes = Arc::new(Mutex::new(HashMap::new()));
    let octave_shift: OctaveShift = Arc::new(Mutex::new(0));
    let transpose:    Transpose   = Arc::new(Mutex::new(0));
    let reverb_mix:   ReverbMix   = Arc::new(Mutex::new(0.3));
    let is_playing:   IsPlaying   = Arc::new(Mutex::new(false));
    let is_looping:   IsLooping   = Arc::new(Mutex::new(false));
    let delay_on:     DelayOn     = Arc::new(Mutex::new(false));
    let delay_fb:     DelayFb     = Arc::new(Mutex::new(0.40));
    let tremolo_on:   TremOn      = Arc::new(Mutex::new(false));
    let tremolo_rate: TremRate    = Arc::new(Mutex::new(4.0)); // Hz
    let vibrato_on:   VibOn       = Arc::new(Mutex::new(false));
    let vibrato_rate: VibRate     = Arc::new(Mutex::new(5.0)); // Hz

    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .expect("no output device found");
    let config = device.default_output_config()?;
    let sample_rate = config.sample_rate().0 as f32;

    let notes_for_audio      = Arc::clone(&active_notes);
    let octave_for_audio     = Arc::clone(&octave_shift);
    let transp_for_audio     = Arc::clone(&transpose);
    let mix_for_audio        = Arc::clone(&reverb_mix);
    let delay_on_for_audio   = Arc::clone(&delay_on);
    let delay_fb_for_audio   = Arc::clone(&delay_fb);
    let trem_on_for_audio    = Arc::clone(&tremolo_on);
    let trem_rate_for_audio  = Arc::clone(&tremolo_rate);
    let vib_on_for_audio     = Arc::clone(&vibrato_on);
    let vib_rate_for_audio   = Arc::clone(&vibrato_rate);

    let mut reverb    = Reverb::new(sample_rate);
    let mut delay     = Delay::new(sample_rate, 300.0); // 300ms tap delay
    let mut lfo_phase = 0.0_f32; // tremolo LFO phase
    let mut vib_phase = 0.0_f32; // vibrato LFO phase

    let stream = device.build_output_stream(
        &config.into(),
        move |output: &mut [f32], _| {
            let mut notes  = notes_for_audio.lock().unwrap();
            let shift      = *octave_for_audio.lock().unwrap();
            let transp     = *transp_for_audio.lock().unwrap();
            let mix        = *mix_for_audio.lock().unwrap();
            let d_on       = *delay_on_for_audio.lock().unwrap();
            let d_fb       = *delay_fb_for_audio.lock().unwrap();
            let t_on       = *trem_on_for_audio.lock().unwrap();
            let t_rate     = *trem_rate_for_audio.lock().unwrap();
            let v_on       = *vib_on_for_audio.lock().unwrap();
            let v_rate     = *vib_rate_for_audio.lock().unwrap();
            // 2^shift doubles/halves frequency per octave; 2^(n/12) does the same per semitone
            let octave_mul = 2.0_f32.powi(shift);
            let transp_mul = 2.0_f32.powf(transp as f32 / 12.0);

            for sample in output.iter_mut() {
                let vib_lfo = (2.0 * std::f32::consts::PI * vib_phase).sin();
                let vib_mul = if v_on { 2.0_f32.powf(VIB_DEPTH * vib_lfo / 12.0) } else { 1.0 };
                vib_phase   = (vib_phase + v_rate / sample_rate) % 1.0;

                *sample = 0.0;
                for (key, note) in notes.iter_mut() {
                    if let Some(freq) = key_to_freq(*key) {
                        *sample += note.tick(freq * octave_mul * transp_mul * vib_mul, sample_rate);
                    }
                }
                notes.retain(|_, note| !note.is_finished());

                // signal chain: notes → delay → reverb → tremolo
                let dry      = *sample;
                let delayed  = delay.process(if d_on { dry } else { 0.0 }, d_fb);
                let pre_verb = if d_on { dry * 0.65 + delayed * 0.35 } else { dry };
                let wet      = reverb.process(pre_verb);
                let post_verb = pre_verb * (1.0 - mix) + wet * mix;

                let trem_lfo = (2.0 * std::f32::consts::PI * lfo_phase).sin();
                let trem_mul = if t_on { 1.0 - TREM_DEPTH * 0.5 * (1.0 - trem_lfo) } else { 1.0 };
                lfo_phase    = (lfo_phase + t_rate / sample_rate) % 1.0;

                *sample = post_verb * trem_mul;
            }
        },
        |err| eprintln!("audio error: {err}"),
        None,
    )?;

    stream.play()?;

    terminal::enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    execute!(stdout, terminal::Clear(ClearType::All), cursor::Hide)?;

    let mut velocity:     f32              = 5.0 / 9.0;
    let mut pedal_down:   bool             = false;
    let mut waveform:     Waveform         = Waveform::Sine;
    let mut recording:    bool             = false;
    let mut record_start: Option<Instant>  = None;
    let mut recorded:     Vec<RecordedEvent> = Vec::new();
    let mut scale:        Scale            = Scale::Major;
    let mut root_idx:     usize            = 0; // C
    let mut arp_on:       bool             = false;
    let mut arp_pattern:  ArpPattern       = ArpPattern::Up;
    let mut bpm:          f32              = 120.0;
    let mut held_notes:   Vec<char>        = Vec::new();
    let mut arp_idx:      usize            = 0;
    let mut arp_last:     Option<char>     = None;
    let mut last_tick:    Instant          = Instant::now();

    macro_rules! redraw {
        () => {{
            let pressed  = pressed_keys(&active_notes);
            let octave   = *octave_shift.lock().unwrap();
            let transp   = *transpose.lock().unwrap();
            let mix      = *reverb_mix.lock().unwrap();
            let playing  = *is_playing.lock().unwrap();
            let looping  = *is_looping.lock().unwrap();
            let d_on     = *delay_on.lock().unwrap();
            let d_fb     = *delay_fb.lock().unwrap();
            let t_on     = *tremolo_on.lock().unwrap();
            let t_rate   = *tremolo_rate.lock().unwrap();
            let v_on     = *vibrato_on.lock().unwrap();
            let v_rate   = *vibrato_rate.lock().unwrap();
            print_ui(&mut stdout, octave, transp, &pressed, velocity, pedal_down, waveform, mix,
                     recording, recorded.len(), playing, looping, scale, root_idx,
                     arp_on, arp_pattern, bpm,
                     d_on, d_fb, t_on, t_rate, v_on, v_rate)?;
        }};
    }

    redraw!();

    loop {
        let beat_dur = Duration::from_millis((60_000.0 / bpm) as u64);
        let wait = if arp_on {
            beat_dur.saturating_sub(last_tick.elapsed())
        } else {
            Duration::from_secs(60)
        };

        let got_event = event::poll(wait)?;

        if !got_event && arp_on {
            last_tick = Instant::now();
            let mut notes = active_notes.lock().unwrap();
            if let Some(prev) = arp_last {
                if let Some(n) = notes.get_mut(&prev) { n.release(); }
            }
            let mut sorted = held_notes.clone();
            sorted.sort_by_key(|&k| key_to_pitch_class(k).unwrap_or(0));
            if !sorted.is_empty() {
                let idx = arp_pattern.note_index(arp_idx, sorted.len());
                let key = sorted[idx];
                arp_idx += 1;
                arp_last = Some(key);
                notes.entry(key).or_insert_with(|| Note::new(velocity, waveform));
            }
            drop(notes);
            redraw!();
            continue;
        }

        if !got_event { continue; }

        if let Event::Key(KeyEvent { code, kind, .. }) = event::read()? {
            match code {
                KeyCode::Esc | KeyCode::Char('q') => break,

                // sustain pedal
                KeyCode::Char(' ') => {
                    if kind == KeyEventKind::Press && !pedal_down {
                        pedal_down = true;
                    } else if kind == KeyEventKind::Release && pedal_down {
                        pedal_down = false;
                        let mut notes = active_notes.lock().unwrap();
                        for note in notes.values_mut() {
                            if note.key_released { note.release(); }
                        }
                    }
                    redraw!();
                }

                // transpose
                KeyCode::PageUp if kind == KeyEventKind::Press => {
                    let mut t = transpose.lock().unwrap();
                    if *t < MAX_TRANSPOSE { *t += 1; }
                    drop(t);
                    redraw!();
                }
                KeyCode::PageDown if kind == KeyEventKind::Press => {
                    let mut t = transpose.lock().unwrap();
                    if *t > MIN_TRANSPOSE { *t -= 1; }
                    drop(t);
                    redraw!();
                }

                // toggle arpeggiator
                KeyCode::Tab if kind == KeyEventKind::Press => {
                    arp_on = !arp_on;
                    if !arp_on {
                        held_notes.clear();
                        arp_last = None;
                        arp_idx  = 0;
                        let mut notes = active_notes.lock().unwrap();
                        for note in notes.values_mut() { note.release(); }
                    }
                    last_tick = Instant::now();
                    redraw!();
                }

                // cycle arp pattern
                KeyCode::Char('`') if kind == KeyEventKind::Press => {
                    arp_pattern = arp_pattern.next();
                    arp_idx     = 0;
                    redraw!();
                }

                // BPM
                KeyCode::Up if kind == KeyEventKind::Press => {
                    bpm = (bpm + 10.0).min(300.0);
                    redraw!();
                }
                KeyCode::Down if kind == KeyEventKind::Press => {
                    bpm = (bpm - 10.0).max(40.0);
                    redraw!();
                }

                // toggle delay
                KeyCode::Char('e') if kind == KeyEventKind::Press => {
                    let mut on = delay_on.lock().unwrap();
                    *on = !*on;
                    drop(on);
                    redraw!();
                }

                // delay feedback
                KeyCode::Left if kind == KeyEventKind::Press => {
                    let mut fb = delay_fb.lock().unwrap();
                    *fb = (*fb - 0.1).max(0.1);
                    drop(fb);
                    redraw!();
                }
                KeyCode::Right if kind == KeyEventKind::Press => {
                    let mut fb = delay_fb.lock().unwrap();
                    *fb = (*fb + 0.1).min(0.7);
                    drop(fb);
                    redraw!();
                }

                // toggle tremolo
                KeyCode::Char('v') if kind == KeyEventKind::Press => {
                    let mut on = tremolo_on.lock().unwrap();
                    *on = !*on;
                    drop(on);
                    redraw!();
                }

                // tremolo rate
                KeyCode::Char('b') if kind == KeyEventKind::Press => {
                    let mut rate = tremolo_rate.lock().unwrap();
                    *rate = (*rate - 0.5).max(0.5);
                    drop(rate);
                    redraw!();
                }
                KeyCode::Char('n') if kind == KeyEventKind::Press => {
                    let mut rate = tremolo_rate.lock().unwrap();
                    *rate = (*rate + 0.5).min(10.0);
                    drop(rate);
                    redraw!();
                }

                // toggle vibrato
                KeyCode::Char('c') if kind == KeyEventKind::Press => {
                    let mut on = vibrato_on.lock().unwrap();
                    *on = !*on;
                    drop(on);
                    redraw!();
                }

                // vibrato rate
                KeyCode::Char('m') if kind == KeyEventKind::Press => {
                    let mut rate = vibrato_rate.lock().unwrap();
                    *rate = (*rate - 0.5).max(0.5);
                    drop(rate);
                    redraw!();
                }
                KeyCode::Char(',') if kind == KeyEventKind::Press => {
                    let mut rate = vibrato_rate.lock().unwrap();
                    *rate = (*rate + 0.5).min(10.0);
                    drop(rate);
                    redraw!();
                }

                // toggle recording
                KeyCode::Char('r') if kind == KeyEventKind::Press => {
                    if recording {
                        recording = false;
                    } else {
                        recorded.clear();
                        record_start = Some(Instant::now());
                        recording = true;
                    }
                    redraw!();
                }

                // play once — or stop if already playing
                KeyCode::Char('p') if kind == KeyEventKind::Press => {
                    let playing = *is_playing.lock().unwrap();
                    if playing {
                        stop_playback(&is_looping, &active_notes);
                    } else if !recorded.is_empty() {
                        *is_playing.lock().unwrap() = true;
                        let events      = recorded.clone();
                        let notes       = Arc::clone(&active_notes);
                        let playing_arc = Arc::clone(&is_playing);
                        let looping_arc = Arc::clone(&is_looping);
                        std::thread::spawn(move || {
                            let mut prev = Duration::ZERO;
                            for event in &events {
                                let gap = event.timestamp.saturating_sub(prev);
                                std::thread::sleep(gap);
                                prev = event.timestamp;
                                let mut notes = notes.lock().unwrap();
                                match event.kind {
                                    RecordKind::Press   => { notes.entry(event.key).or_insert_with(|| Note::new(event.velocity, event.waveform)); }
                                    RecordKind::Release => { if let Some(n) = notes.get_mut(&event.key) { n.release(); } }
                                }
                            }
                            *playing_arc.lock().unwrap() = false;
                            *looping_arc.lock().unwrap() = false;
                        });
                    }
                    redraw!();
                }

                // loop playback — or stop if already looping
                KeyCode::Char('y') if kind == KeyEventKind::Press => {
                    let playing = *is_playing.lock().unwrap();
                    if playing {
                        stop_playback(&is_looping, &active_notes);
                    } else if !recorded.is_empty() {
                        *is_playing.lock().unwrap() = true;
                        *is_looping.lock().unwrap() = true;
                        let events      = recorded.clone();
                        let notes       = Arc::clone(&active_notes);
                        let playing_arc = Arc::clone(&is_playing);
                        let looping_arc = Arc::clone(&is_looping);
                        std::thread::spawn(move || {
                            'playback: loop {
                                let mut prev = Duration::ZERO;
                                for event in &events {
                                    if !*looping_arc.lock().unwrap() { break 'playback; }
                                    let gap = event.timestamp.saturating_sub(prev);
                                    std::thread::sleep(gap);
                                    prev = event.timestamp;
                                    if !*looping_arc.lock().unwrap() { break 'playback; }
                                    let mut notes = notes.lock().unwrap();
                                    match event.kind {
                                        RecordKind::Press   => { notes.entry(event.key).or_insert_with(|| Note::new(event.velocity, event.waveform)); }
                                        RecordKind::Release => { if let Some(n) = notes.get_mut(&event.key) { n.release(); } }
                                    }
                                }
                                if !*looping_arc.lock().unwrap() { break; }
                            }
                            *playing_arc.lock().unwrap() = false;
                            *looping_arc.lock().unwrap() = false;
                        });
                    }
                    redraw!();
                }

                // cycle waveform
                KeyCode::Char('[') if kind == KeyEventKind::Press => {
                    waveform = waveform.next();
                    redraw!();
                }

                // reverb
                KeyCode::Char('-') if kind == KeyEventKind::Press => {
                    let mut mix = reverb_mix.lock().unwrap();
                    *mix = (*mix - 0.1).max(0.0);
                    drop(mix);
                    redraw!();
                }
                KeyCode::Char('=') if kind == KeyEventKind::Press => {
                    let mut mix = reverb_mix.lock().unwrap();
                    *mix = (*mix + 0.1).min(0.8);
                    drop(mix);
                    redraw!();
                }

                // octave
                KeyCode::Char('z') if kind == KeyEventKind::Press => {
                    let mut shift = octave_shift.lock().unwrap();
                    if *shift > MIN_OCTAVE { *shift -= 1; }
                    drop(shift);
                    redraw!();
                }
                KeyCode::Char('x') if kind == KeyEventKind::Press => {
                    let mut shift = octave_shift.lock().unwrap();
                    if *shift < MAX_OCTAVE { *shift += 1; }
                    drop(shift);
                    redraw!();
                }

                // scale type
                KeyCode::Char(']') if kind == KeyEventKind::Press => {
                    scale = scale.next();
                    redraw!();
                }

                // scale root
                KeyCode::Char('\\') if kind == KeyEventKind::Press => {
                    root_idx = (root_idx + 1) % ROOTS.len();
                    redraw!();
                }

                // velocity
                KeyCode::Char(c @ '1'..='9') if kind == KeyEventKind::Press => {
                    velocity = c.to_digit(10).unwrap() as f32 / 9.0;
                    redraw!();
                }

                // piano keys
                KeyCode::Char(c) if key_to_freq(c).is_some() => {
                    if arp_on {
                        if kind == KeyEventKind::Press && !held_notes.contains(&c) {
                            held_notes.push(c);
                            arp_idx = 0;
                        } else if kind == KeyEventKind::Release {
                            held_notes.retain(|&k| k != c);
                            if held_notes.is_empty() {
                                if let Some(prev) = arp_last {
                                    if let Some(n) = active_notes.lock().unwrap().get_mut(&prev) {
                                        n.release();
                                    }
                                }
                                arp_last = None;
                            }
                        }
                    } else {
                        let mut notes = active_notes.lock().unwrap();
                        if kind == KeyEventKind::Press {
                            notes.entry(c).or_insert_with(|| Note::new(velocity, waveform));
                            if recording {
                                if let Some(start) = record_start {
                                    recorded.push(RecordedEvent {
                                        timestamp: start.elapsed(), key: c,
                                        kind: RecordKind::Press, velocity, waveform,
                                    });
                                }
                            }
                        } else if kind == KeyEventKind::Release {
                            if let Some(note) = notes.get_mut(&c) {
                                if pedal_down { note.key_released = true; } else { note.release(); }
                            }
                            if recording {
                                if let Some(start) = record_start {
                                    recorded.push(RecordedEvent {
                                        timestamp: start.elapsed(), key: c,
                                        kind: RecordKind::Release, velocity, waveform,
                                    });
                                }
                            }
                        }
                    }
                    redraw!();
                }

                _ => {}
            }
        }
    }

    terminal::disable_raw_mode()?;
    execute!(stdout, cursor::Show, terminal::Clear(ClearType::All))?;
    println!("Goodbye!");
    Ok(())
}

fn collect_pressed(notes: &HashMap<char, Note>) -> HashSet<char> {
    notes
        .iter()
        .filter(|(_, n)| n.stage != Stage::Release)
        .map(|(k, _)| *k)
        .collect()
}

fn pressed_keys(active_notes: &ActiveNotes) -> HashSet<char> {
    collect_pressed(&active_notes.lock().unwrap())
}

fn write_key(
    stdout:      &mut impl Write,
    key:         char,
    pressed:     &HashSet<char>,
    scale_notes: &HashSet<u8>,
) -> anyhow::Result<()> {
    let label    = key.to_ascii_uppercase();
    let in_scale = key_to_pitch_class(key).map_or(false, |pc| scale_notes.contains(&pc));

    if pressed.contains(&key) {
        queue!(stdout, SetBackgroundColor(Color::Yellow), SetForegroundColor(Color::Black),
               Print(label), ResetColor)?;
    } else if in_scale {
        queue!(stdout, SetBackgroundColor(Color::DarkGreen), SetForegroundColor(Color::White),
               Print(label), ResetColor)?;
    } else {
        queue!(stdout, Print(label))?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn print_ui(
    stdout:       &mut impl Write,
    octave:       i32,
    transp:       i32,
    pressed:      &HashSet<char>,
    velocity:     f32,
    pedal_down:   bool,
    waveform:     Waveform,
    reverb:       f32,
    recording:    bool,
    event_count:  usize,
    is_playing:   bool,
    is_looping:   bool,
    scale:        Scale,
    root_idx:     usize,
    arp_on:       bool,
    arp_pattern:  ArpPattern,
    bpm:          f32,
    delay_on:     bool,
    delay_fb:     f32,
    tremolo_on:   bool,
    tremolo_rate: f32,
    vibrato_on:   bool,
    vibrato_rate: f32,
) -> anyhow::Result<()> {
    let (root_name, root_pc) = ROOTS[root_idx];
    // scale notes are shifted by transpose so highlighting stays in sync with what you hear
    let scale_notes: HashSet<u8> = scale.intervals()
        .iter()
        .map(|&i| (root_pc as i32 + i as i32 + transp).rem_euclid(12) as u8)
        .collect();
    let scale_label  = format!("Scale:    {} {}  (] = scale, \\ = root)", root_name, scale.name());
    let octave_label = match octave {
        0 => "Octave:  0  (default)".to_string(),
        n if n > 0 => format!("Octave: +{}  (Z/X to shift)", n),
        n => format!("Octave: {}  (Z/X to shift)", n),
    };
    let transp_label = match transp {
        0 => "Transpose:  0 semitones  (PgUp/PgDn)".to_string(),
        n if n > 0 => format!("Transpose: +{} semitones  (PgUp/PgDn)", n),
        n => format!("Transpose: {} semitones  (PgUp/PgDn)", n),
    };

    let level = (velocity * 9.0).round() as usize;
    let bar: String = (1..=9).map(|i| if i <= level { '█' } else { '░' }).collect();
    let vel_label   = format!("Velocity: {}  ({}/9 — keys 1-9)", bar, level);
    let pedal_label = if pedal_down { "Pedal: DOWN" } else { "Pedal: up  " };
    let wave_label  = format!("Waveform: {}  ([ to cycle)", waveform.name());

    let rvb_steps = (reverb * 10.0).round() as usize;
    let rvb_bar: String = (1..=8).map(|i| if i <= rvb_steps { '█' } else { '░' }).collect();
    let rvb_label = format!("Reverb:   {}  (- / = to adjust)", rvb_bar);

    let fb_steps = (delay_fb * 10.0).round() as usize;
    let fb_bar: String = (1..=7).map(|i| if i <= fb_steps { '█' } else { '░' }).collect();
    let delay_label = if delay_on {
        format!("Delay:    ON   fb: {}  (E=off, \u{25c4}\u{25ba}=feedback)", fb_bar)
    } else {
        format!("Delay:    off  fb: {}  (E to enable, \u{25c4}\u{25ba}=feedback)", fb_bar)
    };

    let trem_label = if tremolo_on {
        format!("Tremolo:  ON   {:.1} Hz  (V=off, B/N=rate)", tremolo_rate)
    } else {
        format!("Tremolo:  off  {:.1} Hz  (V to enable, B/N=rate)", tremolo_rate)
    };

    let vib_label = if vibrato_on {
        format!("Vibrato:  ON   {:.1} Hz  (C=off, M/,=rate)", vibrato_rate)
    } else {
        format!("Vibrato:  off  {:.1} Hz  (C to enable, M/,=rate)", vibrato_rate)
    };

    let arp_label = if arp_on {
        format!("Arp:      ON   {} BPM  {}  (Tab=off, \u{2191}\u{2193}=BPM, `=pattern)", bpm as u32, arp_pattern.name())
    } else {
        "Arp:      off  (Tab to enable)".to_string()
    };

    let tape_label = if is_looping {
        format!("Tape: LOOPING  ({} events)  — Y to stop", event_count)
    } else if is_playing {
        "Tape: playing once...  — P to stop".to_string()
    } else if recording {
        format!("Tape: RECORDING  ({} events)  — R to stop", event_count)
    } else {
        format!("Tape: {} events stored  — R=record  P=play once  Y=loop", event_count)
    };

    let chord = detect_chord(pressed, transp);

    queue!(stdout, cursor::MoveTo(0, 0), terminal::Clear(ClearType::All))?;

    let white_keys = ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';'];
    let black_keys = [Some('w'), None, Some('r'), None, Some('t'), Some('u'), Some('i'), None, Some('o'), None];

    writeln!(stdout, "\r")?;
    writeln!(stdout, "  Terminal Piano\r")?;
    writeln!(stdout, "\r")?;

    write!(stdout, "  White:  ")?;
    for key in white_keys {
        write_key(stdout, key, pressed, &scale_notes)?;
        write!(stdout, "  ")?;
    }
    writeln!(stdout, "\r")?;

    write!(stdout, "  Black:  ")?;
    for slot in black_keys {
        match slot {
            Some(key) => { write_key(stdout, key, pressed, &scale_notes)?; write!(stdout, "  ")?; }
            None      => { write!(stdout, "   ")?; }
        }
    }
    writeln!(stdout, "\r")?;

    writeln!(stdout, "\r")?;
    if chord.is_empty() {
        writeln!(stdout, "  Chord:   —\r")?;
    } else {
        write!(stdout, "  Chord:   ")?;
        queue!(stdout, SetForegroundColor(Color::Cyan), Print(format!("{:<16}", chord)), ResetColor)?;
        writeln!(stdout, "\r")?;
    }
    writeln!(stdout, "\r")?;
    writeln!(stdout, "  {}\r", octave_label)?;

    write!(stdout, "  ")?;
    if transp != 0 {
        queue!(stdout, SetForegroundColor(Color::Yellow), Print(&transp_label), ResetColor)?;
    } else {
        write!(stdout, "{}", transp_label)?;
    }
    writeln!(stdout, "\r")?;

    writeln!(stdout, "  {}\r", vel_label)?;
    writeln!(stdout, "  {}\r", pedal_label)?;
    writeln!(stdout, "  {}\r", wave_label)?;
    writeln!(stdout, "  {}\r", rvb_label)?;

    write!(stdout, "  ")?;
    if delay_on {
        queue!(stdout, SetForegroundColor(Color::Yellow), Print(&delay_label), ResetColor)?;
    } else {
        write!(stdout, "{}", delay_label)?;
    }
    writeln!(stdout, "\r")?;

    write!(stdout, "  ")?;
    if tremolo_on {
        queue!(stdout, SetForegroundColor(Color::Blue), Print(&trem_label), ResetColor)?;
    } else {
        write!(stdout, "{}", trem_label)?;
    }
    writeln!(stdout, "\r")?;

    write!(stdout, "  ")?;
    if vibrato_on {
        queue!(stdout, SetForegroundColor(Color::Magenta), Print(&vib_label), ResetColor)?;
    } else {
        write!(stdout, "{}", vib_label)?;
    }
    writeln!(stdout, "\r")?;

    writeln!(stdout, "  {}\r", scale_label)?;

    write!(stdout, "  ")?;
    if arp_on {
        queue!(stdout, SetForegroundColor(Color::Cyan), Print(&arp_label), ResetColor)?;
    } else {
        write!(stdout, "{}", arp_label)?;
    }
    writeln!(stdout, "\r")?;

    write!(stdout, "  ")?;
    if is_looping {
        queue!(stdout, SetForegroundColor(Color::Green), SetBackgroundColor(Color::DarkGreen),
               Print(&tape_label), ResetColor)?;
    } else if recording {
        queue!(stdout, SetForegroundColor(Color::Red), Print(&tape_label), ResetColor)?;
    } else if is_playing {
        queue!(stdout, SetForegroundColor(Color::Green), Print(&tape_label), ResetColor)?;
    } else {
        write!(stdout, "{}", tape_label)?;
    }
    writeln!(stdout, "\r")?;

    writeln!(stdout, "\r")?;
    writeln!(stdout, "  SPACE=pedal  Z/X=octave  PgUp/PgDn=transpose  1-9=vel  [=wave  -/==reverb\r")?;
    writeln!(stdout, "  E=delay  V=tremolo  C=vibrato  Tab=arp  R=record  P=play  Y=loop  Q=quit\r")?;

    stdout.flush()?;
    Ok(())
}
