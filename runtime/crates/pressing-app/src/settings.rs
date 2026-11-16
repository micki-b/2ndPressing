//! Remembered settings (JSON in the platform's per-user config folder).

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::audio::Route;

/// Everything the app remembers between launches.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub input: Option<String>,
    pub in_route: Route,
    pub output: Option<String>,
    pub out_route: Route,
    /// Requested input rate (`None`: the device's current rate).
    pub rate: Option<u32>,
    pub sensitivity: f32,
    /// 0 Off, 1 Standard, 2 Extended.
    pub protection: u8,
    /// 0 Repaired, 1 Clicks only.
    pub monitor: u8,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            input: None,
            in_route: Route::FIRST_PAIR,
            output: None,
            out_route: Route::FIRST_PAIR,
            rate: None,
            sensitivity: pressing_plugin::SENSITIVITY_DEFAULT,
            protection: 2,
            monitor: 0,
        }
    }
}

/// `<config dir>/2ndPressing/app.json`.
pub fn path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME").map(PathBuf::from);
    let base = if cfg!(target_os = "macos") {
        home.map(|h| h.join("Library/Application Support"))
    } else if cfg!(windows) {
        std::env::var_os("APPDATA").map(PathBuf::from)
    } else {
        std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| home.map(|h| h.join(".config")))
    }?;
    Some(base.join("2ndPressing").join("app.json"))
}

pub fn load() -> Settings {
    path()
        .and_then(|p| std::fs::read(p).ok())
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default()
}

pub fn save(s: &Settings) {
    let Some(p) = path() else { return };
    if let Some(dir) = p.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if let Ok(json) = serde_json::to_vec_pretty(s) {
        let _ = std::fs::write(p, json);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_and_partial_files() {
        let s = Settings {
            input: Some("H6".into()),
            rate: Some(44_100),
            ..Settings::default()
        };
        let back: Settings = serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
        assert_eq!(back, s);
        // Older or hand-edited files with missing keys still load.
        let partial: Settings = serde_json::from_str(r#"{"output":"Bose USB Audio"}"#).unwrap();
        assert_eq!(partial.output.as_deref(), Some("Bose USB Audio"));
        assert_eq!(partial.protection, 2);
    }
}
