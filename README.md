# terminal-piano

A fun hobby project - a piano you can play in your terminal, built in Rust. Used to be a pianist in school :)) this is my way of merging that with systems programming.


## How to play

```
cargo run
```

| White keys | A  S  D  F  G  H  J  K  L  ;  |
|------------|-------------------------------|
| Black keys | W     R     T     U  I     O  |

Hold multiple keys at once to play chords. Press `Q` or `ESC` to quit.

## Key map

```
A  W  S  R  D  F  T  G  H  U  J  I  K  O  L  ;
A3 A#3 B3 C4 C#4 D4 D#4 E4 F4 F#4 G4 G#4 A4 A#4 B4 C5
```

## Built with

- [`cpal`](https://github.com/RustAudio/cpal) — cross-platform audio output
- [`crossterm`](https://github.com/crossterm-rs/crossterm) — terminal input and control
