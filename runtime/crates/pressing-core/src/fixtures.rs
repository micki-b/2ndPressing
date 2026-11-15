//! Loader for the Python-generated parity fixtures (raw LE arrays + manifest).

use std::path::{Path, PathBuf};

pub struct Fixtures {
    pub dir: PathBuf,
    pub manifest: serde_json::Value,
    pub windows: usize,
    pub channels: usize,
    pub samples: usize,
}

impl Fixtures {
    /// `None` if the directory has no manifest (tests then skip).
    pub fn open(dir: &Path) -> Option<Fixtures> {
        let text = std::fs::read_to_string(dir.join("manifest.json")).ok()?;
        let manifest: serde_json::Value = serde_json::from_str(&text).ok()?;
        let g = |k: &str| manifest[k].as_u64().unwrap_or(0) as usize;
        Some(Fixtures {
            dir: dir.to_path_buf(),
            windows: g("windows"),
            channels: g("channels"),
            samples: g("samples"),
            manifest,
        })
    }

    pub fn f32s(&self, name: &str) -> Vec<f32> {
        let b = std::fs::read(self.dir.join(format!("{name}.bin"))).expect("fixture array");
        b.as_chunks::<4>()
            .0
            .iter()
            .map(|&c| f32::from_le_bytes(c))
            .collect()
    }

    pub fn u8s(&self, name: &str) -> Vec<u8> {
        std::fs::read(self.dir.join(format!("{name}.bin"))).expect("fixture array")
    }

    /// Index of (window, channel, sample) in a (W, C, L) array.
    pub fn idx(&self, w: usize, c: usize, n: usize) -> usize {
        (w * self.channels + c) * self.samples + n
    }

    /// Committed small fixtures and (optional) large ones.
    pub fn small() -> Option<Fixtures> {
        Fixtures::open(&Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures"))
    }

    pub fn large() -> Option<Fixtures> {
        Fixtures::open(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures-large"))
    }
}
