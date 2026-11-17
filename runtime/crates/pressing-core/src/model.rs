//! Model loading: `weights.json` (layout, configs) + `weights.bin` (f32 LE).

use serde::Deserialize;

/// Input features per channel: `[g1_self, g2_self, g1_other]`.
pub const N_FEATURES: usize = 3;

/// Network shape (weights.json -> net_config).
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct NetConfig {
    #[serde(rename = "C")]
    pub c: usize,
    pub blocks: usize,
    pub stem_k: usize,
    pub stride: usize,
    #[serde(rename = "D")]
    pub d: usize,
}

/// Feature constants (weights.json -> features).
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct FeatureConfig {
    #[serde(rename = "EPS")]
    pub eps: f64,
    #[serde(rename = "FLOOR")]
    pub floor: f64,
    #[serde(rename = "NW")]
    pub nw: usize,
    #[serde(rename = "ASINH_SCALE")]
    pub asinh_scale: f64,
}

/// Largest feature window accepted by [`FeatureConfig::validate`] (2^20 samples;
/// [`crate::features::FeatureState`] allocates `4 * NW` f64).
pub const MAX_NW: usize = 1 << 20;

impl FeatureConfig {
    /// Checks the constants the feature formulas need: `1 <= NW <= MAX_NW`, and
    /// `EPS`, `FLOOR`, `ASINH_SCALE` finite and `> 0` (otherwise `ln(|d| + EPS)`,
    /// `d / s` or `/ ASINH_SCALE` can produce -inf/NaN, or the window mean divides
    /// by zero).
    pub fn validate(&self) -> Result<(), ModelError> {
        let bad = |m: &str| Err(ModelError::Json(format!("features {self:?}: {m}")));
        if self.nw == 0 || self.nw > MAX_NW {
            return bad(&format!("NW must be in 1..={MAX_NW}"));
        }
        for (name, v) in [
            ("EPS", self.eps),
            ("FLOOR", self.floor),
            ("ASINH_SCALE", self.asinh_scale),
        ] {
            if !(v.is_finite() && v > 0.0) {
                return bad(&format!("{name} must be finite and > 0"));
            }
        }
        Ok(())
    }
}

/// Post-processing chosen on val (weights.json -> post).
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct PostConfig {
    pub threshold: f32,
    pub ext_drop: f32,
    pub merge: usize,
    pub pre: usize,
    pub post: usize,
    pub max_region: usize,
    pub min_contrast: f32,
    pub contrast_context: usize,
}

#[derive(Debug)]
pub enum ModelError {
    Json(String),
    Layout(String),
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelError::Json(s) => write!(f, "weights.json: {s}"),
            ModelError::Layout(s) => write!(f, "weights layout: {s}"),
        }
    }
}

impl std::error::Error for ModelError {}

/// A loaded ClickNet model. Tensors are stored by layer name in the order of
/// weights.json `layers` (e.g. `stem.weight` [C,3,8], `blocks.0.dil.weight`
/// [C,C,3], `blocks.0.mix.weight` [C,C,1], `head1.weight`, `head2.weight` ...).
#[derive(Clone, Debug)]
pub struct Model {
    pub net: NetConfig,
    pub features: FeatureConfig,
    pub post: PostConfig,
    /// (name, shape, data) for every tensor, in file order.
    pub tensors: Vec<(String, Vec<usize>, Vec<f32>)>,
}

pub const EMBEDDED_JSON: &str = include_str!("../../../models/clicknet-r2/weights.json");
pub const EMBEDDED_BIN: &[u8] = include_bytes!("../../../models/clicknet-r2/weights.bin");

impl Model {
    /// Parse weights.json + weights.bin; validates shapes against net_config.
    ///
    /// Checks, each with a descriptive [`ModelError`]:
    /// - `format` (if present) is `clicknet-v1-weights`, `dtype` (if present) is `f32le`,
    ///   `total_bytes` (if present) equals `bin.len()`, top-level `D` (if present) equals
    ///   `net_config.D`, `features.N_FEATURES` (if present) is 3;
    /// - `features` passes [`FeatureConfig::validate`]; `post.threshold`, `post.ext_drop`
    ///   and `post.min_contrast` are finite;
    /// - `net_config` is supported: `C > 0`, `1 <= blocks <= 16`, `stride == 4` (the streaming
    ///   interface emits 4 logits per frame), `stem_k >= stride`;
    /// - every layer has `nbytes == 4 * prod(shape)`, a 4-byte aligned offset and lies
    ///   inside `bin`; names are unique; all values are finite;
    /// - the layer set is exactly `stem.{weight,bias}` `[C,3,stem_k]`/`[C]`,
    ///   `blocks.{i}.dil.{weight,bias}` `[C,C,3]`/`[C]`, `blocks.{i}.mix.{weight,bias}`
    ///   `[C,C,1]`/`[C]` for `i < blocks`, `head1.{weight,bias}` `[C,C,1]`/`[C]` and
    ///   `head2.{weight,bias}` `[stride,C,1]`/`[stride]`.
    pub fn from_json_and_bin(json: &str, bin: &[u8]) -> Result<Model, ModelError> {
        let value: serde_json::Value =
            serde_json::from_str(json).map_err(|e| ModelError::Json(e.to_string()))?;
        if let Some(n) = value.get("features").and_then(|f| f.get("N_FEATURES")) {
            if n.as_u64() != Some(N_FEATURES as u64) {
                return Err(ModelError::Json(format!(
                    "features.N_FEATURES is {n}, the runtime supports {N_FEATURES}"
                )));
            }
        }
        let hdr: Header =
            serde_json::from_value(value).map_err(|e| ModelError::Json(e.to_string()))?;
        hdr.check_meta(bin.len())?;
        let net = hdr.net_config;
        check_net_config(&net)?;
        hdr.features.validate()?;
        check_post_config(&hdr.post)?;
        let expected = expected_shapes(&net);

        let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::with_capacity(hdr.layers.len());
        for layer in &hdr.layers {
            let name = layer.name.as_str();
            if tensors.iter().any(|t| t.0 == name) {
                return Err(layout(format!("duplicate tensor `{name}`")));
            }
            let want = expected
                .iter()
                .find(|(n, _)| n == name)
                .ok_or_else(|| layout(format!("unexpected tensor `{name}`")))?;
            if layer.shape != want.1 {
                return Err(layout(format!(
                    "tensor `{name}` has shape {:?}, expected {:?} for net_config {net:?}",
                    layer.shape, want.1
                )));
            }
            let count = layer
                .shape
                .iter()
                .try_fold(1usize, |a, &b| a.checked_mul(b));
            let nbytes = count.and_then(|n| n.checked_mul(4));
            if nbytes != Some(layer.nbytes) {
                return Err(layout(format!(
                    "tensor `{name}`: nbytes {} does not match shape {:?} (f32)",
                    layer.nbytes, layer.shape
                )));
            }
            if layer.offset % 4 != 0 {
                return Err(layout(format!(
                    "tensor `{name}`: offset {} is not 4-byte aligned",
                    layer.offset
                )));
            }
            let end = layer
                .offset
                .checked_add(layer.nbytes)
                .filter(|&e| e <= bin.len());
            let Some(end) = end else {
                return Err(layout(format!(
                    "tensor `{name}`: bytes {}..{}+{} exceed weights.bin length {}",
                    layer.offset,
                    layer.offset,
                    layer.nbytes,
                    bin.len()
                )));
            };
            let data: Vec<f32> = bin[layer.offset..end]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&c| f32::from_le_bytes(c))
                .collect();
            if let Some(i) = data.iter().position(|v| !v.is_finite()) {
                return Err(layout(format!(
                    "tensor `{name}`: non-finite value at index {i}"
                )));
            }
            tensors.push((layer.name.clone(), layer.shape.clone(), data));
        }
        if let Some((missing, _)) = expected
            .iter()
            .find(|(n, _)| !tensors.iter().any(|t| &t.0 == n))
        {
            return Err(layout(format!("missing tensor `{missing}`")));
        }

        Ok(Model {
            net,
            features: hdr.features,
            post: hdr.post,
            tensors,
        })
    }

    /// The model shipped with the runtime (R2: C=24, D=512, threshold 2.75).
    pub fn embedded() -> Model {
        Model::from_json_and_bin(EMBEDDED_JSON, EMBEDDED_BIN).expect("embedded model is valid")
    }

    /// Tensor data by name.
    pub fn tensor(&self, name: &str) -> Option<&[f32]> {
        self.tensors
            .iter()
            .find(|t| t.0 == name)
            .map(|t| t.2.as_slice())
    }
}

// ---- weights.json parsing and validation ----

fn layout(msg: String) -> ModelError {
    ModelError::Layout(msg)
}

/// One entry of weights.json `layers`.
#[derive(Deserialize)]
struct LayerEntry {
    name: String,
    shape: Vec<usize>,
    offset: usize,
    nbytes: usize,
}

/// The parts of weights.json the runtime reads (unknown keys are ignored).
#[derive(Deserialize)]
struct Header {
    format: Option<String>,
    dtype: Option<String>,
    total_bytes: Option<usize>,
    #[serde(rename = "D")]
    d: Option<usize>,
    net_config: NetConfig,
    features: FeatureConfig,
    post: PostConfig,
    layers: Vec<LayerEntry>,
}

impl Header {
    fn check_meta(&self, bin_len: usize) -> Result<(), ModelError> {
        if let Some(f) = &self.format {
            if f != "clicknet-v1-weights" {
                return Err(ModelError::Json(format!("unsupported format `{f}`")));
            }
        }
        if let Some(t) = &self.dtype {
            if t != "f32le" {
                return Err(ModelError::Json(format!(
                    "unsupported dtype `{t}` (want f32le)"
                )));
            }
        }
        if let Some(n) = self.total_bytes {
            if n != bin_len {
                return Err(layout(format!(
                    "total_bytes {n} does not match weights.bin length {bin_len}"
                )));
            }
        }
        if let Some(d) = self.d {
            if d != self.net_config.d {
                return Err(ModelError::Json(format!(
                    "top-level D {d} disagrees with net_config.D {}",
                    self.net_config.d
                )));
            }
        }
        Ok(())
    }
}

fn check_net_config(net: &NetConfig) -> Result<(), ModelError> {
    let bad = |m: &str| Err(ModelError::Json(format!("net_config {net:?}: {m}")));
    if net.c == 0 {
        return bad("C must be > 0");
    }
    if net.blocks == 0 || net.blocks > 16 {
        return bad("blocks must be in 1..=16");
    }
    if net.stride != 4 {
        return bad("stride must be 4 (the streaming interface emits 4 logits per frame)");
    }
    if net.stem_k < net.stride {
        return bad("stem_k must be >= stride");
    }
    Ok(())
}

fn check_post_config(post: &PostConfig) -> Result<(), ModelError> {
    for (name, v) in [
        ("threshold", post.threshold),
        ("ext_drop", post.ext_drop),
        ("min_contrast", post.min_contrast),
    ] {
        if !v.is_finite() {
            return Err(ModelError::Json(format!(
                "post.{name} must be finite, got {v}"
            )));
        }
    }
    Ok(())
}

/// Every tensor the network needs, with its shape, in canonical order.
fn expected_shapes(net: &NetConfig) -> Vec<(String, Vec<usize>)> {
    let c = net.c;
    let mut v = vec![
        ("stem.weight".to_string(), vec![c, N_FEATURES, net.stem_k]),
        ("stem.bias".to_string(), vec![c]),
    ];
    for i in 0..net.blocks {
        v.push((format!("blocks.{i}.dil.weight"), vec![c, c, 3]));
        v.push((format!("blocks.{i}.dil.bias"), vec![c]));
        v.push((format!("blocks.{i}.mix.weight"), vec![c, c, 1]));
        v.push((format!("blocks.{i}.mix.bias"), vec![c]));
    }
    v.push(("head1.weight".to_string(), vec![c, c, 1]));
    v.push(("head1.bias".to_string(), vec![c]));
    v.push(("head2.weight".to_string(), vec![net.stride, c, 1]));
    v.push(("head2.bias".to_string(), vec![net.stride]));
    v
}
