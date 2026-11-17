//! Gapless trimming for MP4/M4A input.
//!
//! symphonia 0.5's MP4 reader does not apply encoder delay/padding, so AAC
//! decodes start with the encoder priming (typically 2112 frames from Apple's
//! encoder, 1024 from ffmpeg's) and end with padding. The trim is read from
//! (in order of preference):
//! 1. the iTunes `iTunSMPB` tag (`delay`, `padding`, original frame count);
//! 2. the audio track's edit list (`moov/trak/edts/elst`): one non-empty edit
//!    at rate 1, `media_time` = frames to skip, `segment_duration` = frames
//!    to keep.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::ops::Range;
use std::path::Path;

use anyhow::{Context, Result};

/// Where a trim came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrimSource {
    /// iTunes `iTunSMPB` tag.
    ITunSmpb,
    /// MP4 edit list (`elst`).
    EditList,
}

impl std::fmt::Display for TrimSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            TrimSource::ITunSmpb => "iTunSMPB",
            TrimSource::EditList => "edit list",
        })
    }
}

/// Frames to drop at the start and (optionally) the number of frames to keep
/// after them, at the decoder's output rate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Trim {
    /// Leading frames to drop (encoder delay / priming).
    pub skip: u64,
    /// Frames to keep after `skip`; `None` keeps everything to the end.
    pub keep: Option<u64>,
    /// Where the values came from.
    pub source: TrimSource,
}

impl Trim {
    /// `true` if applying the trim can change anything.
    pub fn is_noop(&self) -> bool {
        self.skip == 0 && self.keep.is_none()
    }
}

// ---- iTunSMPB ----

/// Parses an `iTunSMPB` value: space-separated hex fields
/// `00000000 <delay> <padding> <original frame count> ...`. A zero frame count
/// keeps everything after the delay (the padding field alone is not used:
/// for ALAC it is relative to the packet-rounded length, which symphonia
/// already trims).
pub fn parse_itunsmpb(value: &str) -> Option<Trim> {
    let f: Vec<u64> = value
        .split_whitespace()
        .map(|h| u64::from_str_radix(h, 16))
        .collect::<Result<_, _>>()
        .ok()?;
    if f.len() < 4 {
        return None;
    }
    Some(Trim {
        skip: f[1],
        keep: (f[3] > 0).then_some(f[3]),
        source: TrimSource::ITunSmpb,
    })
}

// ---- MP4 edit list ----

/// Edit list of the first sound track of an MP4 file plus the timescales
/// needed to interpret it.
#[derive(Debug, PartialEq, Eq)]
struct EditInfo {
    movie_timescale: u32,
    media_timescale: u32,
    /// `(segment_duration, media_time, media_rate_int)`.
    entries: Vec<(u64, i64, i16)>,
}

/// Reads the edit list of the first sound track (`hdlr` = `soun`) of the MP4
/// file at `path` and converts it to a [`Trim`] at `sample_rate`.
///
/// Returns `Ok(None)` if the file is not MP4, has no edit list, or the edit
/// list is a no-op. Returns `Err` with a human-readable reason if the edit list
/// exists but cannot be applied (several edits, rate != 1, ...); callers should
/// warn and continue untrimmed.
pub fn mp4_edit_trim(path: &Path, sample_rate: u32) -> Result<Option<Trim>> {
    let Some(info) = read_edit_info(path)? else {
        return Ok(None);
    };
    edit_trim(&info, sample_rate)
}

fn edit_trim(info: &EditInfo, sample_rate: u32) -> Result<Option<Trim>> {
    // Leading empty edits (media_time = -1) delay presentation; they are not
    // material in the stream, so they are ignored.
    let media: Vec<_> = info.entries.iter().filter(|e| e.1 >= 0).collect();
    let &&(dur, media_time, rate) = match media.as_slice() {
        [] => return Ok(None),
        [one] => one,
        _ => anyhow::bail!(
            "edit list has {} media edits (only one is supported)",
            media.len()
        ),
    };
    if rate != 1 {
        anyhow::bail!("edit list rate {rate} is not supported");
    }
    if info.media_timescale == 0 || info.movie_timescale == 0 {
        anyhow::bail!("edit list with a zero timescale");
    }
    let sr = u128::from(sample_rate);
    let conv = |v: u64, ts: u32| -> (u64, bool) {
        let num = u128::from(v) * sr;
        let ts = u128::from(ts);
        (
            u64::try_from((num + ts / 2) / ts).unwrap_or(u64::MAX),
            num % ts == 0,
        )
    };
    let (skip, _) = conv(media_time as u64, info.media_timescale);
    // segment_duration 0 means "to the end" (fragmented files). The end is
    // only trimmed if it converts to an exact frame count; a coarse movie
    // timescale (e.g. 1000) could otherwise cut real samples.
    let keep = match conv(dur, info.movie_timescale) {
        (k, true) if dur > 0 => Some(k),
        _ => None,
    };
    let t = Trim {
        skip,
        keep,
        source: TrimSource::EditList,
    };
    Ok((!t.is_noop()).then_some(t))
}

/// Largest `moov` we are willing to load (sample tables of very long files
/// are a few MB).
const MAX_MOOV: u64 = 256 << 20;

fn read_edit_info(path: &Path) -> Result<Option<EditInfo>> {
    let mut f = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let file_len = f.metadata()?.len();
    let mut pos = 0u64;
    let mut first = true;
    while pos + 8 <= file_len {
        f.seek(SeekFrom::Start(pos))?;
        let mut h = [0u8; 8];
        f.read_exact(&mut h)?;
        let mut size = u64::from(u32::from_be_bytes([h[0], h[1], h[2], h[3]]));
        let typ = [h[4], h[5], h[6], h[7]];
        let mut hdr = 8u64;
        if first && &typ != b"ftyp" {
            return Ok(None);
        }
        first = false;
        if size == 1 {
            let mut b = [0u8; 8];
            f.read_exact(&mut b)?;
            size = u64::from_be_bytes(b);
            hdr = 16;
        } else if size == 0 {
            size = file_len - pos;
        }
        if size < hdr {
            return Ok(None);
        }
        if &typ == b"moov" {
            let len = size - hdr;
            if len > MAX_MOOV {
                anyhow::bail!("moov atom too large ({len} bytes)");
            }
            let mut body = vec![0u8; len as usize];
            f.read_exact(&mut body).context("reading moov")?;
            return Ok(parse_moov(&body));
        }
        pos = pos.saturating_add(size);
    }
    Ok(None)
}

/// Iterates `(type, body)` over the atoms in `buf`; stops at the first
/// malformed header.
fn atoms(buf: &[u8]) -> impl Iterator<Item = ([u8; 4], &[u8])> {
    let mut pos = 0usize;
    std::iter::from_fn(move || {
        let h = buf.get(pos..pos + 8)?;
        let mut size = u32::from_be_bytes([h[0], h[1], h[2], h[3]]) as usize;
        let typ = [h[4], h[5], h[6], h[7]];
        let mut hdr = 8;
        if size == 1 {
            let b = buf.get(pos + 8..pos + 16)?;
            size = usize::try_from(u64::from_be_bytes(b.try_into().ok()?)).ok()?;
            hdr = 16;
        } else if size == 0 {
            size = buf.len() - pos;
        }
        if size < hdr {
            return None;
        }
        let body = buf.get(pos + hdr..pos.checked_add(size)?)?;
        pos += size;
        Some((typ, body))
    })
}

fn child<'a>(buf: &'a [u8], name: &[u8; 4]) -> Option<&'a [u8]> {
    atoms(buf).find(|(t, _)| t == name).map(|(_, b)| b)
}

fn be_u32(b: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_be_bytes(b.get(at..at + 4)?.try_into().ok()?))
}

fn be_u64(b: &[u8], at: usize) -> Option<u64> {
    Some(u64::from_be_bytes(b.get(at..at + 8)?.try_into().ok()?))
}

/// Timescale of an `mvhd`/`mdhd` body (version 0 or 1).
fn header_timescale(b: &[u8]) -> Option<u32> {
    match b.first()? {
        0 => be_u32(b, 12),
        1 => be_u32(b, 20),
        _ => None,
    }
}

fn parse_elst(b: &[u8]) -> Option<Vec<(u64, i64, i16)>> {
    let version = *b.first()?;
    let n = be_u32(b, 4)? as usize;
    let mut out = Vec::with_capacity(n.min(64));
    let mut o = 8;
    for _ in 0..n {
        let (dur, time) = if version == 1 {
            let d = be_u64(b, o)?;
            let t = be_u64(b, o + 8)? as i64;
            o += 16;
            (d, t)
        } else {
            let d = u64::from(be_u32(b, o)?);
            let t = i64::from(be_u32(b, o + 4)? as i32);
            o += 8;
            (d, t)
        };
        let rate = i16::from_be_bytes(b.get(o..o + 2)?.try_into().ok()?);
        o += 4;
        out.push((dur, time, rate));
    }
    Some(out)
}

fn parse_moov(moov: &[u8]) -> Option<EditInfo> {
    let movie_timescale = header_timescale(child(moov, b"mvhd")?)?;
    for (t, trak) in atoms(moov) {
        if &t != b"trak" {
            continue;
        }
        let Some(mdia) = child(trak, b"mdia") else {
            continue;
        };
        // hdlr: version/flags (4), pre_defined (4), handler_type (4).
        let is_sound = child(mdia, b"hdlr").and_then(|h| h.get(8..12)) == Some(b"soun");
        if !is_sound {
            continue;
        }
        let media_timescale = header_timescale(child(mdia, b"mdhd")?)?;
        let entries = child(trak, b"edts")
            .and_then(|e| child(e, b"elst"))
            .and_then(parse_elst)?;
        return Some(EditInfo {
            movie_timescale,
            media_timescale,
            entries,
        });
    }
    None
}

// ---- Frame window ----

/// Selects the frames `[skip, skip + keep.min(limit))` of a stream that is
/// delivered in chunks.
#[derive(Debug)]
pub struct Window {
    skip: u64,
    keep: u64,
    limit: u64,
    pos: u64,
}

impl Window {
    /// Window for `trim` (no limit yet).
    pub fn new(trim: Option<Trim>) -> Self {
        Window {
            skip: trim.map_or(0, |t| t.skip),
            keep: trim.and_then(|t| t.keep).unwrap_or(u64::MAX),
            limit: u64::MAX,
            pos: 0,
        }
    }

    /// Keep at most `limit` frames (e.g. `--max-seconds`).
    pub fn set_limit(&mut self, limit: u64) {
        self.limit = limit;
    }

    fn end(&self) -> u64 {
        self.skip.saturating_add(self.keep.min(self.limit))
    }

    /// Feeds a chunk of `frames` frames; returns the range of the chunk to
    /// keep (possibly empty).
    pub fn take(&mut self, frames: usize) -> Range<usize> {
        let start = self.pos;
        let stop = start + frames as u64;
        self.pos = stop;
        let lo = start.max(self.skip).min(stop);
        let hi = stop.min(self.end()).max(lo);
        (lo - start) as usize..(hi - start) as usize
    }

    /// `true` once the `limit` (not the trim) has been reached: decoding
    /// can stop.
    pub fn limit_reached(&self) -> bool {
        self.limit < self.keep && self.pos >= self.end()
    }

    /// Frames dropped at the start so far.
    pub fn dropped_start(&self) -> u64 {
        self.skip.min(self.pos)
    }

    /// Frames delivered past the end of the window.
    pub fn dropped_end(&self) -> u64 {
        self.pos.saturating_sub(self.end())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn itunsmpb_values() {
        let aac = " 00000000 00000840 000000C0 0000000000017700 00000000 00000000";
        assert_eq!(
            parse_itunsmpb(aac),
            Some(Trim {
                skip: 2112,
                keep: Some(96_000),
                source: TrimSource::ITunSmpb
            })
        );
        // Apple ALAC: no delay; padding relative to the packet-rounded length.
        let alac = " 00000000 00000000 00000900 0000000000017700 00000000";
        let t = parse_itunsmpb(alac).unwrap();
        assert_eq!((t.skip, t.keep), (0, Some(96_000)));
        assert_eq!(parse_itunsmpb("00000000 00000840"), None);
        assert_eq!(parse_itunsmpb("zz 1 2 3"), None);
        let t = parse_itunsmpb("0 840 0 0").unwrap();
        assert_eq!((t.skip, t.keep), (2112, None));
    }

    fn info(movie: u32, media: u32, entries: Vec<(u64, i64, i16)>) -> EditInfo {
        EditInfo {
            movie_timescale: movie,
            media_timescale: media,
            entries,
        }
    }

    #[test]
    fn edit_list_conversion() {
        // ffmpeg AAC: skip 1024, keep 96000.
        let t = edit_trim(&info(48_000, 48_000, vec![(96_000, 1024, 1)]), 48_000)
            .unwrap()
            .unwrap();
        assert_eq!((t.skip, t.keep), (1024, Some(96_000)));
        // Leading empty edit ignored.
        let t = edit_trim(
            &info(48_000, 48_000, vec![(500, -1, 1), (96_000, 2112, 1)]),
            48_000,
        )
        .unwrap()
        .unwrap();
        assert_eq!((t.skip, t.keep), (2112, Some(96_000)));
        // Coarse movie timescale: the end is not trimmed unless exact.
        let t = edit_trim(&info(1000, 44_100, vec![(2001, 2112, 1)]), 44_100)
            .unwrap()
            .unwrap();
        assert_eq!((t.skip, t.keep), (2112, None));
        let t = edit_trim(&info(1000, 48_000, vec![(2000, 0, 1)]), 48_000)
            .unwrap()
            .unwrap();
        assert_eq!((t.skip, t.keep), (0, Some(96_000)));
        // No-op and unsupported lists.
        assert_eq!(
            edit_trim(&info(1000, 48_000, vec![(0, 0, 1)]), 48_000).unwrap(),
            None
        );
        assert!(edit_trim(&info(1000, 48_000, vec![(5, 0, 1), (5, 9, 1)]), 48_000).is_err());
        assert!(edit_trim(&info(1000, 48_000, vec![(5, 0, 2)]), 48_000).is_err());
    }

    #[test]
    fn parses_moov_atoms() {
        fn atom(t: &[u8; 4], body: &[u8]) -> Vec<u8> {
            let mut v = ((body.len() + 8) as u32).to_be_bytes().to_vec();
            v.extend_from_slice(t);
            v.extend_from_slice(body);
            v
        }
        let mut hd = vec![0u8; 20];
        hd[12..16].copy_from_slice(&44_100u32.to_be_bytes());
        let mut hdlr = vec![0u8; 12];
        hdlr[8..12].copy_from_slice(b"soun");
        let mut elst = vec![0, 0, 0, 0, 0, 0, 0, 1];
        elst.extend_from_slice(&1000u32.to_be_bytes());
        elst.extend_from_slice(&(-1i32).to_be_bytes());
        elst.extend_from_slice(&[0, 1, 0, 0]);
        let mdia = [atom(b"mdhd", &hd), atom(b"hdlr", &hdlr)].concat();
        let trak = [atom(b"edts", &atom(b"elst", &elst)), atom(b"mdia", &mdia)].concat();
        let moov = [atom(b"mvhd", &hd), atom(b"trak", &trak)].concat();
        assert_eq!(
            parse_moov(&moov),
            Some(info(44_100, 44_100, vec![(1000, -1, 1)]))
        );
    }

    #[test]
    fn window_selects_frames() {
        let mut w = Window::new(Some(Trim {
            skip: 5,
            keep: Some(10),
            source: TrimSource::EditList,
        }));
        assert!(w.take(4).is_empty());
        assert_eq!(w.take(4), 1..4);
        assert_eq!(w.take(4), 0..4);
        assert_eq!(w.take(4), 0..3);
        assert!(w.take(4).is_empty());
        assert_eq!((w.dropped_start(), w.dropped_end()), (5, 5));
        assert!(!w.limit_reached());

        let mut w = Window::new(None);
        w.set_limit(6);
        assert_eq!(w.take(4), 0..4);
        assert!(!w.limit_reached());
        assert_eq!(w.take(4), 0..2);
        assert!(w.limit_reached());
    }
}
