//! Audio file input (symphonia: WAV, AIFF, FLAC, ALAC/AAC in .m4a/.mp4) and
//! WAV output (hound: 24-bit integer or 32-bit float).

use std::fs::File;
use std::path::Path;

use anyhow::{bail, Context, Result};
use symphonia::core::audio::{AudioBufferRef, SampleBuffer};
use symphonia::core::codecs::{
    CodecParameters, DecoderOptions, CODEC_TYPE_AAC, CODEC_TYPE_ALAC, CODEC_TYPE_NULL,
};
use symphonia::core::errors::Error as SymError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::{Metadata, MetadataOptions};
use symphonia::core::probe::Hint;

use crate::gapless::{self, Trim};

/// Decoded audio, planar f32 in [-1, 1) (integer sources are scaled by
/// `2^-(bits-1)` by symphonia, which is exact for up to 24-bit samples).
pub struct Decoded {
    /// One `Vec` per channel, all the same length.
    pub channels: Vec<Vec<f32>>,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// `true` if the decoder produced integer samples (PCM, FLAC, ALAC).
    pub integer: bool,
    /// Bits per sample reported by the container/codec, if any.
    pub bits_per_sample: Option<u32>,
    /// Codec short name (e.g. `alac`, `flac`, `pcm_s24le`).
    pub codec: String,
    /// `true` if decoding stopped early because of `max_frames`.
    pub truncated: bool,
    /// Gapless trim that was applied (MP4 only; see [`crate::gapless`]).
    pub trim: Option<Trim>,
    /// Frames dropped at the start / end by `trim`.
    pub trimmed: (u64, u64),
    /// Packets that failed to decode; each was replaced by silence of the
    /// packet's duration so the timeline stays aligned.
    pub bad_packets: u64,
    /// Frames of silence inserted for `bad_packets` (before trimming).
    pub concealed_frames: u64,
}

impl Decoded {
    /// Frames (samples per channel).
    pub fn frames(&self) -> usize {
        self.channels.first().map_or(0, Vec::len)
    }
}

/// Warnings about individual bad packets printed before summarising.
const MAX_PACKET_WARNINGS: u64 = 10;

/// Decode the default audio track of `path`, keeping at most `max_seconds`
/// of audio (`None` = whole file).
///
/// MP4/M4A input is trimmed for encoder delay and padding (iTunSMPB or edit
/// list, see [`crate::gapless`]); AAC without such information prints a
/// warning because the output then starts with the encoder priming.
/// Undecodable packets are replaced by silence of the packet's duration
/// (warned and counted in [`Decoded::bad_packets`]).
pub fn decode(path: &Path, max_seconds: Option<f64>) -> Result<Decoded> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    let fmt_opts = FormatOptions {
        enable_gapless: true,
        ..Default::default()
    };
    let mut probed = symphonia::default::get_probe()
        .format(&hint, mss, &fmt_opts, &MetadataOptions::default())
        .with_context(|| format!("unsupported or unreadable format: {}", path.display()))?;
    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .context("no audio track")?;
    let track_id = track.id;
    let params = track.codec_params.clone();
    let codec = symphonia::default::get_codecs()
        .get_codec(params.codec)
        .map_or_else(
            || format!("{:?}", params.codec),
            |d| d.short_name.to_string(),
        );
    let mut decoder = symphonia::default::get_codecs()
        .make(&params, &DecoderOptions::default())
        .with_context(|| format!("no decoder for codec {codec}"))?;

    // ---- Gapless trim ----
    // symphonia trims itself when the reader fills in delay/padding; the MP4
    // reader does not, so read iTunSMPB or the edit list ourselves.
    let mut trim = None;
    if params.delay.is_none() && params.padding.is_none() {
        let mut smpb = find_itunsmpb(&mut format.metadata());
        if smpb.is_none() {
            if let Some(mut m) = probed.metadata.get() {
                smpb = find_itunsmpb(&mut m);
            }
        }
        trim = smpb.as_deref().and_then(gapless::parse_itunsmpb);
        if trim.is_none() {
            if let Some(sr) = params.sample_rate {
                match gapless::mp4_edit_trim(path, sr) {
                    Ok(t) => trim = t,
                    Err(e) => eprintln!("warning: edit list ignored: {e:#}"),
                }
            }
        }
        if trim.is_none() && params.codec == CODEC_TYPE_AAC {
            eprintln!(
                "warning: AAC without gapless info (no iTunSMPB tag or edit list): the output \
                 starts with the encoder priming (typically 1024-2112 samples) and is offset \
                 and longer than the source"
            );
        }
    }
    let mut window = gapless::Window::new(trim);

    let mut channels: Vec<Vec<f32>> = Vec::new();
    let mut sample_rate = params.sample_rate;
    let mut integer = None;
    let mut sbuf: Option<SampleBuffer<f32>> = None;
    let mut limit_set = false;
    let (mut bad_packets, mut concealed_frames) = (0u64, 0u64);

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(SymError::ResetRequired) => bail!("stream reset (chained streams) not supported"),
            Err(e) => return Err(e).context("reading packet"),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(d) => Some(d),
            Err(SymError::DecodeError(e)) => {
                // Corrupt packet: replace it by silence of the same duration
                // so everything after it stays at the right time.
                bad_packets += 1;
                concealed_frames += packet.dur();
                if bad_packets <= MAX_PACKET_WARNINGS {
                    let at = sample_rate.map_or(String::new(), |r| {
                        format!(" at {:.3} s", packet.ts() as f64 / f64::from(r))
                    });
                    eprintln!(
                        "warning: undecodable packet{at} ({} frames) replaced by silence: {e}",
                        packet.dur()
                    );
                }
                None
            }
            Err(e) => return Err(e).context("decoding"),
        };
        let (nch, frames) = match &decoded {
            Some(d) => (d.spec().channels.count(), d.frames()),
            None => (
                if channels.is_empty() {
                    params
                        .channels
                        .context("undecodable first packet and unknown channel count")?
                        .count()
                } else {
                    channels.len()
                },
                usize::try_from(packet.dur()).context("packet duration")?,
            ),
        };
        if channels.is_empty() {
            channels = vec![Vec::new(); nch];
        } else if nch != channels.len() {
            bail!(
                "channel count changed mid-stream ({} -> {nch})",
                channels.len()
            );
        }
        if let Some(d) = &decoded {
            sample_rate = Some(d.spec().rate);
            integer.get_or_insert(!matches!(
                d,
                AudioBufferRef::F32(_) | AudioBufferRef::F64(_)
            ));
        }
        if !limit_set {
            if let (Some(s), Some(r)) = (max_seconds, sample_rate) {
                window.set_limit((s * f64::from(r)).round() as u64);
                limit_set = true;
            }
        }
        if frames == 0 {
            continue;
        }
        let keep = window.take(frames);
        match decoded {
            Some(d) => {
                if sbuf.as_ref().is_some_and(|b| b.capacity() < frames * nch) {
                    sbuf = None;
                }
                let (cap, spec) = (d.capacity() as u64, *d.spec());
                let buf = sbuf.get_or_insert_with(|| SampleBuffer::<f32>::new(cap, spec));
                buf.copy_planar_ref(d);
                for (c, ch) in channels.iter_mut().enumerate() {
                    let base = c * frames;
                    ch.extend_from_slice(&buf.samples()[base + keep.start..base + keep.end]);
                }
            }
            None => {
                for ch in &mut channels {
                    ch.resize(ch.len() + keep.len(), 0.0);
                }
            }
        }
        if window.limit_reached() {
            break;
        }
    }
    if bad_packets > MAX_PACKET_WARNINGS {
        eprintln!(
            "warning: {bad_packets} undecodable packets in total ({concealed_frames} frames of silence)"
        );
    }

    if channels.is_empty() || channels[0].is_empty() {
        bail!("{}: no audio decoded", path.display());
    }
    let truncated = window.limit_reached();
    if let Some(t) = trim {
        let got = channels[0].len() as u64;
        if let Some(k) = t.keep.filter(|&k| !truncated && got < k) {
            eprintln!(
                "warning: {} declares {k} frames after the delay but the stream has {got}",
                t.source
            );
        }
    }
    Ok(Decoded {
        channels,
        sample_rate: sample_rate.context("unknown sample rate")?,
        integer: integer.unwrap_or(true),
        bits_per_sample: params.bits_per_sample.or_else(|| alac_bit_depth(&params)),
        codec,
        truncated,
        trim,
        trimmed: (window.dropped_start(), window.dropped_end()),
        bad_packets,
        concealed_frames,
    })
}

/// Value of an iTunes `iTunSMPB` tag in any metadata revision.
fn find_itunsmpb(m: &mut Metadata<'_>) -> Option<String> {
    loop {
        if let Some(rev) = m.current() {
            let hit = rev
                .tags()
                .iter()
                .find(|t| t.key.to_ascii_lowercase().ends_with("itunsmpb"));
            if let Some(t) = hit {
                return Some(t.value.to_string());
            }
        }
        m.pop()?;
    }
}

/// Bit depth from an ALAC magic cookie (`ALACSpecificConfig`: 24 or 48
/// bytes, bit depth at byte 5); symphonia does not copy it to the params.
fn alac_bit_depth(params: &CodecParameters) -> Option<u32> {
    if params.codec != CODEC_TYPE_ALAC {
        return None;
    }
    let cookie = params.extra_data.as_deref()?;
    matches!(cookie.len(), 24 | 48).then(|| u32::from(cookie[5]))
}

/// Output sample format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutFormat {
    /// 24-bit signed integer; `round(x * 2^23)`, clamped, no dither.
    Int24,
    /// 32-bit IEEE float.
    Float32,
}

/// Write planar channels as an interleaved WAV. Returns the number of samples
/// clamped (integer output only).
pub fn write_wav(path: &Path, channels: &[&[f32]], rate: u32, fmt: OutFormat) -> Result<u64> {
    let n = channels.first().map_or(0, |c| c.len());
    if channels.iter().any(|c| c.len() != n) {
        bail!("channel length mismatch");
    }
    let spec = hound::WavSpec {
        channels: u16::try_from(channels.len()).context("too many channels")?,
        sample_rate: rate,
        bits_per_sample: match fmt {
            OutFormat::Int24 => 24,
            OutFormat::Float32 => 32,
        },
        sample_format: match fmt {
            OutFormat::Int24 => hound::SampleFormat::Int,
            OutFormat::Float32 => hound::SampleFormat::Float,
        },
    };
    let mut w = hound::WavWriter::create(path, spec)
        .with_context(|| format!("creating {}", path.display()))?;
    let mut clamped = 0u64;
    for i in 0..n {
        for ch in channels {
            match fmt {
                OutFormat::Int24 => {
                    let (v, clip) = to_i24(ch[i]);
                    clamped += u64::from(clip);
                    w.write_sample(v)?;
                }
                OutFormat::Float32 => w.write_sample(ch[i])?,
            }
        }
    }
    w.finalize()
        .with_context(|| format!("finalising {}", path.display()))?;
    Ok(clamped)
}

/// `round(x * 2^23)` clamped to the 24-bit range; NaN maps to 0. Returns the
/// value and whether it was clamped. Exact inverse of symphonia's integer
/// scaling for samples of up to 24 bits.
pub fn to_i24(x: f32) -> (i32, bool) {
    const MAX: f64 = 8_388_607.0;
    const MIN: f64 = -8_388_608.0;
    if x.is_nan() {
        return (0, true);
    }
    let v = (f64::from(x) * 8_388_608.0).round();
    if v > MAX {
        (MAX as i32, true)
    } else if v < MIN {
        (MIN as i32, true)
    } else {
        (v as i32, false)
    }
}

#[cfg(test)]
mod tests {
    use super::to_i24;

    #[test]
    fn i24_round_trip_and_clamp() {
        for v in [-8_388_608, -1, 0, 1, 12_345, 8_388_607] {
            let x = v as f32 / 8_388_608.0;
            assert_eq!(to_i24(x), (v, false));
        }
        assert_eq!(to_i24(1.5), (8_388_607, true));
        assert_eq!(to_i24(-1.5), (-8_388_608, true));
        assert_eq!(to_i24(f32::NAN), (0, true));
    }
}
