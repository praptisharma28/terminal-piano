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

struct Note {
    phase:     f32,
    amplitude: f32,
    stage:     Stage,
    velocity:  f32, // 0.0 (silent) to 1.0 (full force), captured at key press
}

impl Note {
    fn new(velocity: f32) -> Self {
        Note { phase: 0.0, amplitude: 0.0, stage: Stage::Attack, velocity }
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

        let v = self.velocity;

        // higher harmonics scale with velocity: loud = bright, soft = warm
        let harmonics: &[(f32, f32)] = &[
            (1.0, 1.00),       // fundamental — always present
            (2.0, 0.50 * v),   // octave up
            (3.0, 0.25 * v),   // fifth above that
            (4.0, 0.12 * v*v), // two octaves up — drops fast at low velocity
            (5.0, 0.06 * v*v), // major third above that
        ];

        let mut sample = 0.0_f32;
        for (multiple, weight) in harmonics {
            sample += weight * (2.0 * std::f32::consts::PI * self.phase * multiple).sin();
        }
        sample *= self.amplitude * self.velocity * 0.12;

        self.phase = (self.phase + freq / sample_rate) % 1.0;
        sample
    }
}

type ActiveNotes = Arc<Mutex<HashMap<char, Note>>>;
type OctaveShift = Arc<Mutex<i32>>;

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

    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .expect("no output device found");
    let config = device.default_output_config()?;
    let sample_rate = config.sample_rate().0 as f32;

    let notes_for_audio  = Arc::clone(&active_notes);
    let octave_for_audio = Arc::clone(&octave_shift);

    let stream = device.build_output_stream(
        &config.into(),
        move |output: &mut [f32], _| {
            let mut notes = notes_for_audio.lock().unwrap();
            let shift = *octave_for_audio.lock().unwrap();
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
            }
        },
        |err| eprintln!("audio error: {err}"),
        None,
    )?;

    stream.play()?;

    terminal::enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    execute!(stdout, terminal::Clear(ClearType::All), cursor::Hide)?;

    // default velocity: mezzo-forte (5 out of 9)
    let mut velocity: f32 = 5.0 / 9.0;
    print_ui(&mut stdout, 0, &HashSet::new(), velocity)?;

    loop {
        if let Event::Key(KeyEvent { code, kind, .. }) = event::read()? {
            match code {
                KeyCode::Esc | KeyCode::Char('q') => break,

                KeyCode::Char('z') if kind == KeyEventKind::Press => {
                    let mut shift = octave_shift.lock().unwrap();
                    if *shift > MIN_OCTAVE {
                        *shift -= 1;
                        let s = *shift;
                        drop(shift);
                        let pressed = pressed_keys(&active_notes);
                        print_ui(&mut stdout, s, &pressed, velocity)?;
                    }
                }

                KeyCode::Char('x') if kind == KeyEventKind::Press => {
                    let mut shift = octave_shift.lock().unwrap();
                    if *shift < MAX_OCTAVE {
                        *shift += 1;
                        let s = *shift;
                        drop(shift);
                        let pressed = pressed_keys(&active_notes);
                        print_ui(&mut stdout, s, &pressed, velocity)?;
                    }
                }

                // number keys 1-9 set velocity
                KeyCode::Char(c @ '1'..='9') if kind == KeyEventKind::Press => {
                    let n = c.to_digit(10).unwrap() as f32;
                    velocity = n / 9.0;
                    let pressed = pressed_keys(&active_notes);
                    let octave = *octave_shift.lock().unwrap();
                    print_ui(&mut stdout, octave, &pressed, velocity)?;
                }

                KeyCode::Char(c) => {
                    let pressed = {
                        let mut notes = active_notes.lock().unwrap();
                        if kind == KeyEventKind::Press && key_to_freq(c).is_some() {
                            notes.entry(c).or_insert_with(|| Note::new(velocity));
                        } else if kind == KeyEventKind::Release {
                            if let Some(note) = notes.get_mut(&c) {
                                note.release();
                            }
                        }
                        collect_pressed(&notes)
                    };
                    let octave = *octave_shift.lock().unwrap();
                    print_ui(&mut stdout, octave, &pressed, velocity)?;
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

fn print_ui(stdout: &mut impl Write, octave: i32, pressed: &HashSet<char>, velocity: f32) -> anyhow::Result<()> {
    let octave_label = match octave {
        0 => "Octave:  0  (default)".to_string(),
        n if n > 0 => format!("Octave: +{}  (Z to go down)", n),
        n => format!("Octave: {}  (X to go up)", n),
    };

    let level = (velocity * 9.0).round() as usize;
    let bar: String = (1..=9).map(|i| if i <= level { '█' } else { '░' }).collect();
    let vel_label = format!("Velocity: {}  ({}/9 — keys 1-9)", bar, level);

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
    writeln!(stdout, "  {}\r", octave_label)?;
    writeln!(stdout, "  {}\r", vel_label)?;
    writeln!(stdout, "\r")?;
    writeln!(stdout, "  Z = octave down  |  X = octave up  |  1-9 = velocity  |  Q / ESC = quit\r")?;

    stdout.flush()?;
    Ok(())
}
