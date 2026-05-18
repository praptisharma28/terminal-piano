use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::io::Write;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind};
use crossterm::style::{Color, Print, ResetColor, SetBackgroundColor, SetForegroundColor};
use crossterm::terminal::{self, ClearType};
use crossterm::{cursor, execute, queue};

const ATTACK:  f32 = 0.01; // seconds to reach full volume
const DECAY:   f32 = 0.10; // seconds to drop to sustain level
const SUSTAIN: f32 = 0.70; // volume level held while key is down (0.0 - 1.0)
const RELEASE: f32 = 0.30; // seconds to fade out after key is released

const MIN_OCTAVE: i32 = -3;
const MAX_OCTAVE: i32 =  3;

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

struct Note {
    phase:        f32,
    amplitude:    f32,
    stage:        Stage,
    velocity:     f32,     // 0.0 (silent) to 1.0 (full force), captured at key press
    waveform:     Waveform, // captured at key press
    key_released: bool,    // key was lifted while pedal was down — pedal is keeping this note alive
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

        // Each waveform is defined by its harmonic series.
        // Velocity scales upper harmonics — loud = bright, soft = warm.
        let (wave, gain) = match self.waveform {
            Waveform::Sine => (
                (2.0 * pi * p).sin(),
                1.0,
            ),
            Waveform::Triangle => (
                // odd harmonics, alternating sign, weights fall as 1/n²
                  1.000 * (2.0 * pi * 1.0 * p).sin()
                - 0.111 * v * (2.0 * pi * 3.0 * p).sin()
                + 0.040 * v * (2.0 * pi * 5.0 * p).sin()
                - 0.020 * v * (2.0 * pi * 7.0 * p).sin(),
                0.90,
            ),
            Waveform::Square => (
                // odd harmonics only, weights fall as 1/n
                  1.000 * (2.0 * pi * 1.0 * p).sin()
                + 0.333 * v * (2.0 * pi * 3.0 * p).sin()
                + 0.200 * v * (2.0 * pi * 5.0 * p).sin()
                + 0.143 * v * (2.0 * pi * 7.0 * p).sin(),
                0.60,
            ),
            Waveform::Sawtooth => (
                // all harmonics, weights fall as 1/n
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

type ActiveNotes  = Arc<Mutex<HashMap<char, Note>>>;
type OctaveShift  = Arc<Mutex<i32>>;
type ReverbMix    = Arc<Mutex<f32>>;

// Schroeder reverb: four parallel comb filters.
// Each delays the signal and feeds it back into itself.
// Different delay lengths create the illusion of multiple reflections.
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
        output * 0.25 // average across the four filters
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

fn detect_chord(pressed: &HashSet<char>) -> String {
    // Collect unique pitch classes regardless of octave, sorted
    let pcs: Vec<u8> = pressed
        .iter()
        .filter_map(|&k| key_to_pitch_class(k))
        .collect::<std::collections::BTreeSet<u8>>()
        .into_iter()
        .collect();

    match pcs.len() {
        0 => return String::new(),
        1 => return note_name(pcs[0]).to_string(),
        _ => {}
    }

    // (name, interval pattern from root in semitones)
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

    // Try every note as root — this handles all inversions
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

    // No named chord — list the notes
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

fn main() -> anyhow::Result<()> {
    let active_notes: ActiveNotes = Arc::new(Mutex::new(HashMap::new()));
    let octave_shift: OctaveShift = Arc::new(Mutex::new(0));
    let reverb_mix:   ReverbMix   = Arc::new(Mutex::new(0.3)); // default: light room reverb

    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .expect("no output device found");
    let config = device.default_output_config()?;
    let sample_rate = config.sample_rate().0 as f32;

    let notes_for_audio  = Arc::clone(&active_notes);
    let octave_for_audio = Arc::clone(&octave_shift);
    let mix_for_audio    = Arc::clone(&reverb_mix);

    let mut reverb = Reverb::new(sample_rate);

    let stream = device.build_output_stream(
        &config.into(),
        move |output: &mut [f32], _| {
            let mut notes = notes_for_audio.lock().unwrap();
            let shift = *octave_for_audio.lock().unwrap();
            let mix   = *mix_for_audio.lock().unwrap();
            // 2^shift: shift=1 doubles the frequency (one octave up), shift=-1 halves it
            let octave_multiplier = 2.0_f32.powi(shift);

            for sample in output.iter_mut() {
                *sample = 0.0;
                for (key, note) in notes.iter_mut() {
                    if let Some(freq) = key_to_freq(*key) {
                        *sample += note.tick(freq * octave_multiplier, sample_rate);
                    }
                }
                notes.retain(|_, note| !note.is_finished());

                let dry = *sample;
                let wet = reverb.process(dry);
                *sample = dry * (1.0 - mix) + wet * mix;
            }
        },
        |err| eprintln!("audio error: {err}"),
        None,
    )?;

    stream.play()?;

    terminal::enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    execute!(stdout, terminal::Clear(ClearType::All), cursor::Hide)?;

    let mut velocity:   f32      = 5.0 / 9.0;    // default: mezzo-forte (5 out of 9)
    let mut pedal_down: bool     = false;
    let mut waveform:   Waveform = Waveform::Sine;

    print_ui(&mut stdout, 0, &HashSet::new(), velocity, pedal_down, waveform, 0.3)?;

    loop {
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
                            if note.key_released {
                                note.release();
                            }
                        }
                    }
                    let pressed = pressed_keys(&active_notes);
                    let octave  = *octave_shift.lock().unwrap();
                    let mix     = *reverb_mix.lock().unwrap();
                    print_ui(&mut stdout, octave, &pressed, velocity, pedal_down, waveform, mix)?;
                }

                // cycle waveform
                KeyCode::Char('[') if kind == KeyEventKind::Press => {
                    waveform = waveform.next();
                    let pressed = pressed_keys(&active_notes);
                    let octave  = *octave_shift.lock().unwrap();
                    let mix     = *reverb_mix.lock().unwrap();
                    print_ui(&mut stdout, octave, &pressed, velocity, pedal_down, waveform, mix)?;
                }

                // reverb mix: - to decrease, = to increase
                KeyCode::Char('-') if kind == KeyEventKind::Press => {
                    let mut mix = reverb_mix.lock().unwrap();
                    *mix = (*mix - 0.1).max(0.0);
                    let m = *mix;
                    drop(mix);
                    let pressed = pressed_keys(&active_notes);
                    let octave  = *octave_shift.lock().unwrap();
                    print_ui(&mut stdout, octave, &pressed, velocity, pedal_down, waveform, m)?;
                }

                KeyCode::Char('=') if kind == KeyEventKind::Press => {
                    let mut mix = reverb_mix.lock().unwrap();
                    *mix = (*mix + 0.1).min(0.8);
                    let m = *mix;
                    drop(mix);
                    let pressed = pressed_keys(&active_notes);
                    let octave  = *octave_shift.lock().unwrap();
                    print_ui(&mut stdout, octave, &pressed, velocity, pedal_down, waveform, m)?;
                }

                KeyCode::Char('z') if kind == KeyEventKind::Press => {
                    let mut shift = octave_shift.lock().unwrap();
                    if *shift > MIN_OCTAVE {
                        *shift -= 1;
                        let s = *shift;
                        drop(shift);
                        let pressed = pressed_keys(&active_notes);
                        let mix     = *reverb_mix.lock().unwrap();
                        print_ui(&mut stdout, s, &pressed, velocity, pedal_down, waveform, mix)?;
                    }
                }

                KeyCode::Char('x') if kind == KeyEventKind::Press => {
                    let mut shift = octave_shift.lock().unwrap();
                    if *shift < MAX_OCTAVE {
                        *shift += 1;
                        let s = *shift;
                        drop(shift);
                        let pressed = pressed_keys(&active_notes);
                        let mix     = *reverb_mix.lock().unwrap();
                        print_ui(&mut stdout, s, &pressed, velocity, pedal_down, waveform, mix)?;
                    }
                }

                // number keys 1-9 set velocity
                KeyCode::Char(c @ '1'..='9') if kind == KeyEventKind::Press => {
                    let n = c.to_digit(10).unwrap() as f32;
                    velocity = n / 9.0;
                    let pressed = pressed_keys(&active_notes);
                    let octave  = *octave_shift.lock().unwrap();
                    let mix     = *reverb_mix.lock().unwrap();
                    print_ui(&mut stdout, octave, &pressed, velocity, pedal_down, waveform, mix)?;
                }

                KeyCode::Char(c) => {
                    let pressed = {
                        let mut notes = active_notes.lock().unwrap();
                        if kind == KeyEventKind::Press && key_to_freq(c).is_some() {
                            notes.entry(c).or_insert_with(|| Note::new(velocity, waveform));
                        } else if kind == KeyEventKind::Release {
                            if let Some(note) = notes.get_mut(&c) {
                                if pedal_down {
                                    note.key_released = true;
                                } else {
                                    note.release();
                                }
                            }
                        }
                        collect_pressed(&notes)
                    };
                    let octave = *octave_shift.lock().unwrap();
                    let mix    = *reverb_mix.lock().unwrap();
                    print_ui(&mut stdout, octave, &pressed, velocity, pedal_down, waveform, mix)?;
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

fn write_key(stdout: &mut impl Write, key: char, pressed: &HashSet<char>) -> anyhow::Result<()> {
    let label = key.to_ascii_uppercase();
    if pressed.contains(&key) {
        queue!(
            stdout,
            SetBackgroundColor(Color::Yellow),
            SetForegroundColor(Color::Black),
            Print(label),
            ResetColor
        )?;
    } else {
        queue!(stdout, Print(label))?;
    }
    Ok(())
}

fn print_ui(
    stdout:     &mut impl Write,
    octave:     i32,
    pressed:    &HashSet<char>,
    velocity:   f32,
    pedal_down: bool,
    waveform:   Waveform,
    reverb:     f32,
) -> anyhow::Result<()> {
    let octave_label = match octave {
        0 => "Octave:  0  (default)".to_string(),
        n if n > 0 => format!("Octave: +{}  (Z to go down)", n),
        n => format!("Octave: {}  (X to go up)", n),
    };

    let level = (velocity * 9.0).round() as usize;
    let bar: String = (1..=9).map(|i| if i <= level { '█' } else { '░' }).collect();
    let vel_label   = format!("Velocity: {}  ({}/9 — keys 1-9)", bar, level);
    let pedal_label = if pedal_down { "Pedal: DOWN" } else { "Pedal: up  " };
    let wave_label  = format!("Waveform: {}  ([ to cycle)", waveform.name());

    let rvb_steps = (reverb * 10.0).round() as usize;
    let rvb_bar: String = (1..=8).map(|i| if i <= rvb_steps { '█' } else { '░' }).collect();
    let rvb_label = format!("Reverb:   {}  (- / = to adjust)", rvb_bar);

    let chord = detect_chord(pressed);

    queue!(stdout, cursor::MoveTo(0, 0), terminal::Clear(ClearType::All))?;

    let white_keys = ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';'];
    let black_keys = [Some('w'), None, Some('r'), None, Some('t'), Some('u'), Some('i'), None, Some('o'), None];

    writeln!(stdout, "\r")?;
    writeln!(stdout, "  Terminal Piano\r")?;
    writeln!(stdout, "\r")?;

    write!(stdout, "  White:  ")?;
    for key in white_keys {
        write_key(stdout, key, pressed)?;
        write!(stdout, "  ")?;
    }
    writeln!(stdout, "\r")?;

    write!(stdout, "  Black:  ")?;
    for slot in black_keys {
        match slot {
            Some(key) => { write_key(stdout, key, pressed)?; write!(stdout, "  ")?; }
            None      => { write!(stdout, "   ")?; }
        }
    }
    writeln!(stdout, "\r")?;

    writeln!(stdout, "\r")?;
    if chord.is_empty() {
        writeln!(stdout, "  Chord:   —\r")?;
    } else {
        write!(stdout, "  Chord:   ")?;
        queue!(
            stdout,
            SetForegroundColor(Color::Cyan),
            Print(format!("{:<16}", chord)),
            ResetColor
        )?;
        writeln!(stdout, "\r")?;
    }
    writeln!(stdout, "\r")?;
    writeln!(stdout, "  {}\r", octave_label)?;
    writeln!(stdout, "  {}\r", vel_label)?;
    writeln!(stdout, "  {}\r", pedal_label)?;
    writeln!(stdout, "  {}\r", wave_label)?;
    writeln!(stdout, "  {}\r", rvb_label)?;
    writeln!(stdout, "\r")?;
    writeln!(stdout, "  SPACE = pedal  |  Z/X = octave  |  1-9 = velocity  |  [ = waveform  |  -/= = reverb  |  Q / ESC = quit\r")?;

    stdout.flush()?;
    Ok(())
}
