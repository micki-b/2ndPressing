"""Reader for VinylStudio click-repair sidecar files (``*.m4a.crd``).

The format was reverse-engineered from the library on 2026-11-14; VinylStudio
does not document it. Everything below is verified to parse files
byte-exactly, but the *meaning* of some fields is still a hypothesis.

Layout (little-endian)::

    char[4]  magic      "CRDF"
    u32      channels   (2 for stereo)
    repeated per channel:
        repeated records:
            u16  kind
            u16  flags          (usually 0; meaning unknown)
            u32  start          first sample, in the raw side file
            u32  end            sample offset end (exclusive for payloads)
            kind 1 or 2:  u16 param, then (end - start) x i16 samples
            kind 11/12:   u16 value (7..127; likely detection strength)
        u16  0              end-of-channel marker

Verified: kind 1/2 payload samples are the *replacement* waveform (smooth
interpolation) at 16-bit scale, while the raw side audio at the same
positions still contains the click. Kind 11/12 records carry no samples;
whether VinylStudio applied them in exported tracks is still to be confirmed
against the de-clicked iTunes tracks.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAGIC = b"CRDF"
PAYLOAD_KINDS = (1, 2)
VALUE_KINDS = (11, 12)


# MARK: - Data types


@dataclass
class CrdRecord:
    channel: int
    kind: int
    flags: int
    start: int
    end: int
    param: int                          # kind 1/2: header u16; kind 11/12: value
    replacement: np.ndarray | None      # int16 samples for kind 1/2, else None

    @property
    def length(self) -> int:
        return self.end - self.start


class CrdFormatError(ValueError):
    pass


# MARK: - Parsing


def parse_crd(data: bytes) -> tuple[int, list[CrdRecord]]:
    if data[:4] != MAGIC:
        raise CrdFormatError("not a CRDF file")
    (channels,) = struct.unpack_from("<I", data, 4)
    off, ch, recs = 8, 0, []
    while ch < channels:
        if off + 2 > len(data):
            raise CrdFormatError(f"truncated in channel {ch}")
        if struct.unpack_from("<H", data, off)[0] == 0:
            off += 2
            ch += 1
            continue
        if off + 14 > len(data):
            raise CrdFormatError(f"truncated record at {off}")
        kind, flags, start, end = struct.unpack_from("<HHII", data, off)
        (param,) = struct.unpack_from("<H", data, off + 12)
        if end < start:
            raise CrdFormatError(f"end < start at {off}")
        if kind in PAYLOAD_KINDS:
            n = end - start
            body = data[off + 14: off + 14 + 2 * n]
            if len(body) != 2 * n:
                raise CrdFormatError(f"truncated payload at {off}")
            recs.append(CrdRecord(ch, kind, flags, start, end, param,
                                  np.frombuffer(body, "<i2").copy()))
            off += 14 + 2 * n
        elif kind in VALUE_KINDS:
            recs.append(CrdRecord(ch, kind, flags, start, end, param, None))
            off += 14
        else:
            raise CrdFormatError(f"unknown record kind {kind} at {off}")
    if off != len(data):
        raise CrdFormatError(f"{len(data) - off} trailing bytes")
    return channels, recs


def read_crd(path: Path) -> tuple[int, list[CrdRecord]]:
    return parse_crd(Path(path).read_bytes())
