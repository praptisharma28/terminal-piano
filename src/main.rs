use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind};
use crossterm::terminal::{self, ClearType};
use crossterm::{cursor, execute, queue};
use std::io::Write;

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
}

impl Note {
    fn new() -> Self {
        Note { phase: 0.0, amplitude: 0.0, stage: Stage::Attack }
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

        // harmonics: each partial is a multiple of the base frequency, decreasing in volume
        let harmonics: &[(f32, f32)] = &[
            (1.0, 1.00), // fundamental
            (2.0, 0.50), // octave up
            (3.0, 0.25), // fifth above that
            (4.0, 0.12), // two octaves up
            (5.0, 0.06), // major third above that
        ];

        let mut sample = 0.0_f32;
        for (multiple, weight) in harmonics {
            sample += weight * (2.0 * std::f32::consts::PI * self.phase * multiple).sin();
        }
        sample *= self.amplitude * 0.10;

        self.phase = (self.phase + freq / sample_rate) % 1.0;
        sample
    }
}

type ActiveNotes   = Arc<Mutex<HashMap<char, Note>>>;
type OctaveShift   = Arc<Mutex<i32>>;

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
    print_ui(&mut stdout, 0)?;

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
                        print_ui(&mut stdout, s)?;
                    }
                }

                KeyCode::Char('x') if kind == KeyEventKind::Press => {
                    let mut shift = octave_shift.lock().unwrap();
                    if *shift < MAX_OCTAVE {
                        *shift += 1;
                        let s = *shift;
                        drop(shift);
                        print_ui(&mut stdout, s)?;
                    }
                }

                KeyCode::Char(c) => {
                    let mut notes = active_notes.lock().unwrap();
                    if kind == KeyEventKind::Press && key_to_freq(c).is_some() {
                        notes.entry(c).or_insert_with(Note::new);
                    } else if kind == KeyEventKind::Release {
                        if let Some(note) = notes.get_mut(&c) {
                            note.release();
                        }
                    }
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

fn print_ui(stdout: &mut impl Write, octave: i32) -> anyhow::Result<()> {
    let label = match octave {
        0 => "Octave:  0  (default)".to_string(),
        n if n > 0 => format!("Octave: +{}  (Z to go down)", n),
        n => format!("Octave: {}  (X to go up)", n),
    };

    queue!(stdout, cursor::MoveTo(0, 0), terminal::Clear(ClearType::All))?;
    writeln!(stdout, "\r")?;
    writeln!(stdout, "  Terminal Piano\r")?;
    writeln!(stdout, "\r")?;
    writeln!(stdout, "  White keys:  A  S  D  F  G  H  J  K  L  ;\r")?;
    writeln!(stdout, "  Black keys:  W     R     T     U  I     O\r")?;
    writeln!(stdout, "\r")?;
    writeln!(stdout, "  {}\r", label)?;
    writeln!(stdout, "\r")?;
    writeln!(stdout, "  Z = octave down  |  X = octave up  |  Q / ESC = quit\r")?;
    stdout.flush()?;
    Ok(())
}
