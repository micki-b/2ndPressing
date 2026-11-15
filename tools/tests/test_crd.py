import struct

import numpy as np
import pytest

from pressing_tools.crd import CrdFormatError, parse_crd


def _payload_rec(kind, start, samples, param=0, flags=0):
    end = start + len(samples)
    return (struct.pack("<HHIIH", kind, flags, start, end, param)
            + np.asarray(samples, "<i2").tobytes())


def _value_rec(kind, start, end, value):
    return struct.pack("<HHIIH", kind, 0, start, end, value)


def test_parses_two_channels_with_all_kinds():
    data = (b"CRDF" + struct.pack("<I", 2)
            + _payload_rec(2, 100, [-5, -4, -3]) + _value_rec(12, 200, 230, 127)
            + b"\x00\x00"
            + _payload_rec(1, 50, [7, 8], param=3, flags=7) + _value_rec(11, 300, 310, 9)
            + b"\x00\x00")
    channels, recs = parse_crd(data)
    assert channels == 2
    assert [(r.channel, r.kind, r.start, r.end) for r in recs] == [
        (0, 2, 100, 103), (0, 12, 200, 230), (1, 1, 50, 52), (1, 11, 300, 310)]
    assert recs[0].replacement.tolist() == [-5, -4, -3]
    assert recs[1].param == 127 and recs[1].replacement is None
    assert recs[2].flags == 7 and recs[2].param == 3


def test_rejects_trailing_garbage_and_unknown_kind():
    base = b"CRDF" + struct.pack("<I", 1)
    with pytest.raises(CrdFormatError):
        parse_crd(base + b"\x00\x00" + b"xx")
    with pytest.raises(CrdFormatError):
        parse_crd(base + _value_rec(5, 0, 1, 1) + b"\x00\x00")
