"""Discovery and decoding for the VinylStudio + iTunes library.

Raw sides:   <RAW>/Artist~Album_[LP]~side1.m4a  (+ .m4a.crd)
De-clicked:  <ITUNES>/Artist/Album [LP]/NN Title.m4a

Names are matched after normalisation (lower-case, alphanumerics only), so
"10000_Maniacs" matches "10,000 Maniacs".
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import signal

TAG_RE = re.compile(r"\[(?:[^\]]*\s)?(LP|45)\]", re.IGNORECASE)


# MARK: - Name matching


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", s.lower())


@dataclass
class Side:
    path: Path
    artist: str
    album: str          # includes the [LP]/[45] tag
    side: str           # "side1", ...

    @property
    def crd(self) -> Path:
        return self.path.with_name(self.path.name + ".crd")

    @property
    def key(self) -> tuple[str, str]:
        return norm(self.artist), norm(self.album)


@dataclass
class Album:
    dir: Path
    sides: list[Side] = field(default_factory=list)

    @property
    def tracks(self) -> list[Path]:
        return sorted(p for p in self.dir.iterdir()
                      if p.suffix.lower() in (".m4a", ".flac", ".wav", ".aif", ".aiff")
                      and not p.name.startswith("."))


def find_sides(raw_root: Path) -> list[Side]:
    sides = []
    for p in sorted(raw_root.glob("*~*~*.m4a")) + sorted(raw_root.glob("*~*~*.flac")):
        artist, album, side = p.stem.split("~", 2)
        if TAG_RE.search(album):
            sides.append(Side(p, artist.replace("_", " "), album.replace("_", " "), side))
    return sides


def match_albums(raw_root: Path, itunes_root: Path) -> tuple[list[Album], list[Side]]:
    """Return (albums with their sides, sides with no iTunes match)."""
    index: dict[tuple[str, str], Path] = {}
    for d in itunes_root.glob("*/*"):
        if d.is_dir() and TAG_RE.search(d.name):
            index[(norm(d.parent.name), norm(d.name))] = d
    albums: dict[Path, Album] = {}
    unmatched = []
    for s in find_sides(raw_root):
        d = index.get(s.key)
        if d is None:
            unmatched.append(s)
            continue
        albums.setdefault(d, Album(d)).sides.append(s)
    return list(albums.values()), unmatched


# MARK: - Decoding (ffmpeg)


def probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=sample_rate,channels,codec_name,duration_ts,duration",
         "-of", "json", str(path)], capture_output=True, check=True, text=True).stdout
    st = json.loads(out)["streams"][0]
    return {"sr": int(st["sample_rate"]), "channels": int(st["channels"]),
            "codec": st["codec_name"], "duration": float(st.get("duration") or 0)}


def decode(path: Path, start: int | None = None, frames: int | None = None,
           sr: int | None = None, mono: bool = False) -> np.ndarray:
    """Decode to float32 (frames, channels). start/frames are in the file's
    native sample rate; sr resamples the result (ffmpeg swr)."""
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-map", "0:a:0"]
    filters = []
    if start is not None or frames is not None:
        s = start or 0
        trim = f"atrim=start_sample={s}"
        if frames is not None:
            trim += f":end_sample={s + frames}"
        filters += [trim, "asetpts=PTS-STARTPTS"]
    if sr is not None:
        filters.append(f"aresample={sr}")
    if filters:
        cmd += ["-af", ",".join(filters)]
    ch = 1 if mono else probe(path)["channels"]
    cmd += ["-ac", str(ch), "-f", "f32le", "-acodec", "pcm_f32le", "-"]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {path}: {res.stderr.decode()[-500:]}")
    return np.frombuffer(res.stdout, "<f4").reshape(-1, ch).copy()


# MARK: - Locating a track inside a side


@dataclass
class Placement:
    track: Path
    side: Path
    offset: int          # track sample 0 == side sample `offset` (side rate)
    frames: int          # track length in side samples
    score: float         # normalised correlation at coarse stage (0..1)
    same_rate: bool


def _ncc_peak(hay: np.ndarray, needle: np.ndarray) -> tuple[int, float]:
    c = signal.correlate(hay, needle, mode="valid", method="fft")
    # normalise by local energy of hay for a comparable score
    cs = np.concatenate([[0.0], np.cumsum(hay.astype(np.float64) ** 2)])
    win = np.maximum(cs[len(needle):] - cs[:-len(needle)], 0.0)
    e = np.sqrt(win) * np.linalg.norm(needle)
    ncc = np.abs(c) / (e + 1e-9)
    i = int(np.argmax(ncc))
    return i, float(ncc[i])


def locate(track: Path, side: Path, side_coarse: np.ndarray | None = None,
           coarse_sr: int = 2000, probe_s: float = 20.0) -> Placement:
    """Find where `track` sits inside `side`.

    Coarse: correlate a mid-track excerpt at `coarse_sr` against the whole
    side. Fine: correlate the whole track at full rate near the estimate.
    """
    ti, si = probe(track), probe(side)
    if side_coarse is None:
        side_coarse = decode(side, sr=coarse_sr, mono=True)[:, 0]
    t_coarse = decode(track, sr=coarse_sr, mono=True)[:, 0]
    # Excerpt from the middle avoids fades and lead-in silence.
    n = min(int(probe_s * coarse_sr), len(t_coarse))
    a = max(0, len(t_coarse) // 2 - n // 2)
    i, score = _ncc_peak(side_coarse, t_coarse[a:a + n])
    est = (i - a) * si["sr"] / coarse_sr          # side samples where track starts

    # Fine stage at full side rate (track resampled if needed). Correlate the
    # *whole* track: short excerpts of periodic music can lock onto a peak one
    # pitch period away (seen as ~165-sample errors).
    sr = si["sr"]
    track_full = decode(track, sr=sr if ti["sr"] != sr else None, mono=True)[:, 0]
    margin = 4 * int(math.ceil(sr / coarse_sr)) + 256
    hay_start = max(0, int(est) - margin)
    hay = decode(side, start=hay_start, frames=len(track_full) + 2 * margin, mono=True)[:, 0]
    j, _ = _ncc_peak(hay, track_full)
    offset = hay_start + j
    return Placement(track, side, int(offset), len(track_full), score, ti["sr"] == sr)
