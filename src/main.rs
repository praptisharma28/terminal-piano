use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind};
use crossterm::terminal::{self, ClearType};
use crossterm::{cursor, execute};

type ActiveNotes = Arc<Mutex<HashMap<char, f32>>>;

fn key_to_freq(key: char) -> Option<f32> {
    match key {
        'a' => Some(220.00),
        'w' => Some(233.08),
        's' => Some(246.94),
        'd' => Some(261.63),
        'r' => Some(277.18),
        'f' => Some(293.66),
        't' => Some(311.13),
        'g' => Some(329.63),
        'h' => Some(349.23),
        'u' => Some(369.99),
        'j' => Some(392.00),
        'i' => Some(415.30),
        'k' => Some(440.00),
        'o' => Some(466.16),
        'l' => Some(493.88),
        ';' => Some(523.25),
        _ => None,
    }
}

fn main() -> anyhow::Result<()> {
    let active_notes: ActiveNotes = Arc::new(Mutex::new(HashMap::new()));

    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .expect("no output device found");
    let config = device.default_output_config()?;
    let sample_rate = config.sample_rate().0 as f32;

    let notes_for_audio = Arc::clone(&active_notes);

    let stream = device.build_output_stream(
        &config.into(),
        move |output: &mut [f32], _| {
            let mut notes = notes_for_audio.lock().unwrap();
            for sample in output.iter_mut() {
                *sample = 0.0;
                for (key, phase) in notes.iter_mut() {
                    if let Some(freq) = key_to_freq(*key) {
                        *sample += 0.15 * (2.0 * std::f32::consts::PI * *phase).sin();
                        *phase = (*phase + freq / sample_rate) % 1.0;
                    }
                }
            }
        },
        |err| eprintln!("audio error: {err}"),
        None,
    )?;

    stream.play()?;

    terminal::enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    execute!(stdout, terminal::Clear(ClearType::All), cursor::Hide)?;
    print_keyboard();

    loop {
        if let Event::Key(KeyEvent { code, kind, .. }) = event::read()? {
            match code {
                KeyCode::Esc | KeyCode::Char('q') => break,
                KeyCode::Char(c) => {
                    let mut notes = active_notes.lock().unwrap();
                    if kind == KeyEventKind::Press && key_to_freq(c).is_some() {
                        notes.entry(c).or_insert(0.0);
                    } else if kind == KeyEventKind::Release {
                        notes.remove(&c);
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

fn print_keyboard() {
    println!("\r");
    println!("  Terminal Piano\r");
    println!("\r");
    println!("  White keys:  A  S  D  F  G  H  J  K  L  ;\r");
    println!("  Black keys:  W     R     T     U  I     O\r");
    println!("\r");
    println!("  Q or ESC to quit\r");
    println!("\r");
}
