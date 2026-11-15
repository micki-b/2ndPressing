//! `pressing`: 2ndPressing command-line tool.
//!
//! - `pressing declick IN OUT`: remove clicks from a file (WAV, AIFF, FLAC,
//!   ALAC/AAC .m4a) and write a WAV;
//! - `pressing bench`: single-thread real-time factor on synthetic material;
//! - `pressing parity DIR`: parity checks against Python reference fixtures.

mod audio;
mod bench;
mod declick;
mod gapless;
mod parity;
mod synth;

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};
use pressing_core::{OutputMode, Protection};

#[derive(Parser)]
#[command(
    name = "pressing",
    version,
    about = "2ndPressing: LP click and pop removal"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Remove clicks from an audio file and write a WAV (same rate and
    /// channel count; latency compensated).
    Declick {
        /// Input: WAV, AIFF, FLAC, or ALAC/AAC in .m4a.
        input: PathBuf,
        /// Output WAV (24-bit integer for integer sources of up to 24 bits,
        /// else 32-bit float).
        output: PathBuf,
        /// Trigger threshold in logits (default: the model's, 2.75); lower
        /// is more sensitive.
        #[arg(long, short = 't')]
        threshold: Option<f32>,
        /// What to write: repaired audio, the removed clicks, or the input.
        #[arg(long, value_enum, default_value_t = Mode::Declick)]
        mode: Mode,
        /// Percussion protection: leave claps/drums on a beat grid alone.
        /// standard ≈ 1 s look-ahead, extended ≈ 2 s (protects more of the
        /// start of a rhythmic passage), off = no protection.
        #[arg(long, value_enum, default_value_t = ProtectionArg::Standard)]
        protection: ProtectionArg,
        /// Also list every detected region.
        #[arg(long)]
        report: bool,
        /// Write 32-bit float WAV even for integer sources.
        #[arg(long)]
        float: bool,
        /// Only decode the first N seconds (testing).
        #[arg(long, hide = true)]
        max_seconds: Option<f64>,
    },
    /// Benchmark the engine, network and LSAR single-threaded.
    Bench {
        /// Seconds of synthetic audio per timed run.
        #[arg(long, default_value_t = 20.0)]
        seconds: f64,
        /// Block size for `Declicker::process`.
        #[arg(long, default_value_t = 512)]
        block: usize,
        /// Sample rate in Hz.
        #[arg(long, default_value_t = 48_000)]
        rate: u32,
        /// Percussion protection mode for the engine benchmark.
        #[arg(long, value_enum, default_value_t = ProtectionArg::Off)]
        protection: ProtectionArg,
    },
    /// Run fixture parity checks (features, scores, masks, repaired).
    Parity {
        /// Directory with manifest.json and the fixture .bin arrays.
        dir: PathBuf,
    },
}

/// `--mode` values.
#[derive(Clone, Copy, ValueEnum)]
enum Mode {
    /// Repaired signal.
    Declick,
    /// Input minus repaired: what was removed.
    ClicksOnly,
    /// Input unchanged.
    Bypass,
}

/// `--protection` values.
#[derive(Clone, Copy, ValueEnum)]
enum ProtectionArg {
    /// No percussion protection (lowest latency).
    Off,
    /// About 1 s of look-ahead.
    Standard,
    /// About 2 s of look-ahead.
    Extended,
}

impl From<ProtectionArg> for Protection {
    fn from(p: ProtectionArg) -> Self {
        match p {
            ProtectionArg::Off => Protection::Off,
            ProtectionArg::Standard => Protection::Standard,
            ProtectionArg::Extended => Protection::Extended,
        }
    }
}

impl From<Mode> for OutputMode {
    fn from(m: Mode) -> Self {
        match m {
            Mode::Declick => OutputMode::Declick,
            Mode::ClicksOnly => OutputMode::ClicksOnly,
            Mode::Bypass => OutputMode::Bypass,
        }
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.cmd {
        Cmd::Declick {
            input,
            output,
            threshold,
            mode,
            protection,
            report,
            float,
            max_seconds,
        } => declick::run(&declick::DeclickArgs {
            input,
            output,
            threshold,
            mode: mode.into(),
            protection: protection.into(),
            report,
            float,
            max_seconds,
        }),
        Cmd::Bench {
            seconds,
            block,
            rate,
            protection,
        } => bench::run(&bench::BenchArgs {
            seconds,
            block,
            rate,
            protection: protection.into(),
        }),
        Cmd::Parity { dir } => parity::run(&dir),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}
