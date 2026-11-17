//! Standalone 2ndPressing Declicker (nih-plug standalone wrapper; run with
//! `--help` for audio backend and device options). Requires the
//! `standalone` feature:
//! `cargo build -p pressing-plugin --release --features standalone`.

use nih_plug::prelude::*;
use pressing_plugin::SecondPressing;

fn main() {
    nih_export_standalone::<SecondPressing>();
}
