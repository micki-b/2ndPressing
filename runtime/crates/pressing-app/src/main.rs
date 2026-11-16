//! 2ndPressing Declicker, standalone desktop app.
//!
//! `2ndpressing` opens the window. Diagnostics without a window:
//! - `2ndpressing --list`: input and output devices with their native
//!   channel counts and rates.
//! - `2ndpressing --probe "DEVICE" [SECONDS]`: open an input natively and
//!   report callbacks and peak level (tells "device busy / no permission"
//!   from "digital silence" from "working").
//! - `2ndpressing --check "INPUT" "OUTPUT" [SECONDS] [RATE]`: run the whole
//!   path (input -> declicker, protection Off -> output, channels 1-2) and
//!   print the counters.
//!
//! On Windows the app is a windowed program (no console window behind it);
//! the diagnostics attach to the console they were started from.

#![cfg_attr(windows, windows_subsystem = "windows")]

mod audio;
mod resample;
mod settings;
mod ui;

/// Windows: a windowed program has no console, so print diagnostics to the
/// console of the shell that started it (Command Prompt / PowerShell).
#[cfg(windows)]
fn attach_parent_console() {
    #[link(name = "kernel32")]
    extern "system" {
        fn AttachConsole(process_id: u32) -> i32;
    }
    const ATTACH_PARENT_PROCESS: u32 = u32::MAX;
    // SAFETY: plain Win32 call with a constant argument; failure (no parent
    // console) is harmless and ignored.
    unsafe {
        AttachConsole(ATTACH_PARENT_PROCESS);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    #[cfg(windows)]
    if !args.is_empty() {
        attach_parent_console();
    }
    match args.first().map(String::as_str) {
        Some("--list") => {
            let (ins, outs) = audio::scan();
            let (di, dout) = audio::default_names();
            let show = |title: &str, list: &[audio::DeviceInfo], def: &Option<String>| {
                println!("{title}");
                for d in list {
                    let star = if def.as_deref() == Some(d.name.as_str()) { "*" } else { " " };
                    let rates: Vec<String> =
                        d.rates.iter().map(|r| format!("{:.1}", *r as f64 / 1000.0)).collect();
                    println!(
                        " {star} {:<28} {:>2} ch   {} kHz (now {:.1})",
                        d.name,
                        d.channels,
                        rates.join(" / "),
                        d.default_rate as f64 / 1000.0
                    );
                }
            };
            show("Inputs (* = system default)", &ins, &di);
            show("Outputs", &outs, &dout);
        }
        Some("--probe") => {
            let Some(name) = args.get(1) else {
                eprintln!("usage: 2ndpressing --probe \"DEVICE\" [SECONDS]");
                std::process::exit(2);
            };
            let secs = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(3.0);
            match audio::probe(name, secs) {
                Ok(s) => println!("{s}"),
                Err(e) => {
                    eprintln!("{e}");
                    std::process::exit(1);
                }
            }
        }
        Some("--check") => {
            // --check "INPUT" "OUTPUT" [SECONDS] [RATE]: run the full path
            // (input -> declicker -> output) without a window.
            let (Some(input), Some(output)) = (args.get(1), args.get(2)) else {
                eprintln!("usage: 2ndpressing --check \"INPUT\" \"OUTPUT\" [SECONDS] [RATE]");
                std::process::exit(2);
            };
            let secs: f32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(4.0);
            let (ins, _) = audio::scan();
            let rate = args
                .get(4)
                .and_then(|s| s.parse().ok())
                .or_else(|| ins.iter().find(|d| &d.name == input).map(|d| d.default_rate))
                .unwrap_or(48_000);
            let cfg = audio::SessionConfig {
                input: input.clone(),
                in_route: audio::Route::FIRST_PAIR,
                output: output.clone(),
                out_route: audio::Route::FIRST_PAIR,
                rate,
            };
            let shared = std::sync::Arc::new(audio::Shared::new(
                pressing_plugin::SENSITIVITY_DEFAULT,
                pressing_plugin::PercussionProtection::Off,
                pressing_plugin::Monitor::Declick,
            ));
            let model = pressing_core::Model::embedded();
            match audio::start(&cfg, &model, &shared) {
                Ok(s) => {
                    std::thread::sleep(std::time::Duration::from_secs_f32(secs));
                    println!("{}", audio::report(&s, &shared));
                }
                Err(e) => {
                    eprintln!("could not start: {e}");
                    std::process::exit(1);
                }
            }
        }
        Some("--help" | "-h") => {
            println!(
                "2ndpressing [--list | --probe \"DEVICE\" [SECONDS] | \
                 --check \"INPUT\" \"OUTPUT\" [SECONDS] [RATE]]"
            );
        }
        _ => {
            if let Err(e) = ui::run() {
                eprintln!("2ndPressing: {e}");
                std::process::exit(1);
            }
        }
    }
}
