//! 2ndPressing core: streaming click detection (ClickNet) and LSAR repair.
//!
//! Spec: `docs/runtime-v1.md`. The engine reproduces the Python reference
//! (`tools/src/pressing_tools/nn/method.py`) up to float rounding, with a fixed
//! latency, and never allocates inside [`Declicker::process`].

pub mod engine;
pub mod features;
pub mod fixtures;
pub mod lsar;
pub mod model;
pub mod net;
pub mod post;

pub use engine::{Declicker, OutputMode, ScoreSource, Stats};
pub use model::{FeatureConfig, Model, ModelError, NetConfig, PostConfig};
