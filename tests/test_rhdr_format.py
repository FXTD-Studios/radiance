"""radiance.core.rhdr: the shared .rhdr encoder, on its own.

The layout is fixed by the one reader, _parseRHDR in js/radiance_viewer.js:
a 12-byte "<4sHHHH" header (magic, width, height, channels, flags) and a zlib
stream of H x W x C samples, fp16 for flags 0 and fp32 for flags 1. What the
four existing writers produce is pinned separately in test_rhdr_writers.py.

numpy only: these run on the no-torch CI lane too.
"""
import os
import struct
import zlib

import numpy as np
import pytest

from radiance.core import rhdr

ZLIB_STORED = b"\x78\x01"   # zlib header written by levels 0 and 1
ZLIB_DEFAULT = b"\x78\x9c"  # zlib header written by level 6 (2-5 write 78 5e)


def _ramp(h=3, w=2, c=4):
    return (np.arange(h * w * c, dtype=np.float32).reshape(h, w, c) - 5.0) * 0.75


# ── header ───────────────────────────────────────────────────────────────────

def test_header_is_magic_width_height_channels_flags_little_endian():
    data = rhdr.encode(np.zeros((3, 2, 4), np.float32))
    assert data[:12] == b"RHDR" + struct.pack("<HHHH", 2, 3, 4, 0)
    assert rhdr.HEADER.size == 12


def test_fp32_sets_flags_1():
    data = rhdr.encode(np.zeros((3, 2, 4), np.float32), fp32=True)
    assert struct.unpack("<H", data[10:12])[0] == rhdr.FLAG_FP32 == 1


def test_a_2d_array_is_one_channel():
    data = rhdr.encode(np.zeros((5, 7), np.float32))
    assert struct.unpack("<4sHHHH", data[:12]) == (b"RHDR", 7, 5, 1, 0)


@pytest.mark.parametrize("shape", [(4,), (1, 2, 3, 4)])
def test_other_ranks_are_refused(shape):
    with pytest.raises(ValueError, match="RHDR needs"):
        rhdr.encode(np.zeros(shape, np.float32))


@pytest.mark.parametrize("shape", [(1, 65536, 1), (65536, 1, 1), (1, 1, 65536)])
def test_a_dimension_past_uint16_is_a_value_error_not_a_struct_error(shape):
    # The Viewer catches (IOError, OSError, ValueError) around its writes;
    # struct.error is none of those.
    with pytest.raises(ValueError, match="uint16"):
        rhdr.encode(np.zeros(shape, np.float32))


def test_the_largest_header_value_is_accepted():
    flags, px = rhdr.decode(rhdr.encode(np.zeros((1, 65535), np.float32)))
    assert px.shape == (1, 65535, 1)


# ── payload ──────────────────────────────────────────────────────────────────

def test_payload_is_the_fp16_samples_in_c_order():
    pixels = _ramp()
    data = rhdr.encode(pixels)
    assert zlib.decompress(data[12:]) == pixels.astype(np.float16).tobytes()


def test_payload_is_the_fp32_samples_in_c_order():
    pixels = _ramp()
    data = rhdr.encode(pixels, fp32=True)
    assert zlib.decompress(data[12:]) == pixels.tobytes()


def test_a_non_contiguous_view_is_written_in_c_order():
    base = _ramp(h=2, w=3, c=4)
    view = base.transpose(1, 0, 2)              # (3, 2, 4), not C-contiguous
    assert not view.flags["C_CONTIGUOUS"]
    flags, px = rhdr.decode(rhdr.encode(view, fp32=True))
    np.testing.assert_array_equal(px, view)


def test_float64_input_is_cast_like_astype():
    pixels = _ramp().astype(np.float64) / 3.0
    assert zlib.decompress(rhdr.encode(pixels, fp32=True)[12:]) == pixels.astype(np.float32).tobytes()
    assert zlib.decompress(rhdr.encode(pixels)[12:]) == pixels.astype(np.float16).tobytes()


def test_the_input_array_is_not_modified():
    pixels = np.array([[[1e5, -1e5, 0.5]]], np.float32)
    before = pixels.copy()
    rhdr.encode(pixels)
    np.testing.assert_array_equal(pixels, before)


# ── fp16 range ───────────────────────────────────────────────────────────────

def test_fp16_clamps_to_its_finite_range_by_default():
    pixels = np.array([[[1e5, -1e5, 65504.0, 3.0]]], np.float32)
    flags, px = rhdr.decode(rhdr.encode(pixels))
    assert np.isfinite(px).all(), "1e5 must clamp to the fp16 range, not become inf"
    np.testing.assert_array_equal(px[0, 0], np.array([65504.0, -65504.0, 65504.0, 3.0], np.float16))


def test_fp16_keeps_a_true_inf_and_nan():
    # The clamp is for finite overflow. A sample that is already +-inf (or
    # NaN) is a fault in the source the Viewer flags; clamping inf to 65504
    # made it look like a bright pixel.
    pixels = np.array([[[np.inf, -np.inf, np.nan, 1e5]]], np.float32)
    flags, px = rhdr.decode(rhdr.encode(pixels))
    assert np.isposinf(px[0, 0, 0]) and np.isneginf(px[0, 0, 1])
    assert np.isnan(px[0, 0, 2])
    assert px[0, 0, 3] == np.float16(65504.0)


def test_fp16_clamping_leaves_in_range_values_bit_identical():
    pixels = _ramp()
    assert zlib.decompress(rhdr.encode(pixels)[12:]) == pixels.astype(np.float16).tobytes()


def test_there_is_no_way_to_write_an_unclamped_fp16_file():
    # Every writer used to choose; the VAE export and the depth sidecar chose
    # not to clamp and wrote inf. The option is gone.
    with pytest.raises(TypeError):
        rhdr.encode(_ramp(), clamp_f16=False)


def test_fp32_is_never_clamped():
    pixels = np.array([[[1e5, -1e5, 1e30]]], np.float32)
    flags, px = rhdr.decode(rhdr.encode(pixels, fp32=True))
    np.testing.assert_array_equal(px, pixels)


# ── zlib level ───────────────────────────────────────────────────────────────

def test_level_0_stores_and_is_the_default():
    pixels = _ramp()
    data = rhdr.encode(pixels)
    assert data[12:14] == ZLIB_STORED
    # Stored, not merely tagged as level 0: 2 header bytes, 5 block-header
    # bytes and a 4-byte checksum around the raw samples.
    assert len(data) - 12 == pixels.astype(np.float16).nbytes + 11
    assert rhdr.encode(pixels, level=6)[12:14] == ZLIB_DEFAULT


@pytest.mark.parametrize("level", [0, 1, 6, 9])
def test_every_level_decodes_to_the_same_samples(level):
    pixels = _ramp()
    np.testing.assert_array_equal(rhdr.decode(rhdr.encode(pixels, level=level))[1],
                                  rhdr.decode(rhdr.encode(pixels))[1])


# ── write ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kwargs", [{}, {"fp32": True}, {"level": 6}])
def test_write_puts_exactly_the_encoded_bytes_on_disk(tmp_path, kwargs):
    pixels = _ramp()
    path = tmp_path / "frame.rhdr"
    size = rhdr.write(str(path), pixels, **kwargs)
    on_disk = path.read_bytes()
    assert on_disk == rhdr.encode(pixels, **kwargs)
    assert size == len(on_disk)


def test_write_creates_no_file_when_encoding_fails(tmp_path):
    path = tmp_path / "too_wide.rhdr"
    with pytest.raises(ValueError):
        rhdr.write(str(path), np.zeros((1, 65536), np.float32))
    assert not path.exists()


def test_write_into_a_missing_folder_is_an_oserror(tmp_path):
    with pytest.raises(OSError):
        rhdr.write(os.path.join(str(tmp_path), "missing", "f.rhdr"), _ramp())


# ── decode ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("fp32, dtype", [(False, np.float16), (True, np.float32)])
def test_decode_round_trips(fp32, dtype):
    pixels = _ramp()
    flags, px = rhdr.decode(rhdr.encode(pixels, fp32=fp32))
    assert flags == int(fp32)
    assert px.dtype == dtype and px.shape == pixels.shape
    np.testing.assert_array_equal(px, pixels.astype(dtype))


def test_decode_reads_the_fp32_bit_like_the_js_parser():
    # _parseRHDR tests (flags & 1); higher bits are ignored.
    data = bytearray(rhdr.encode(_ramp(), fp32=True))
    data[10:12] = struct.pack("<H", 0b11)
    flags, px = rhdr.decode(bytes(data))
    assert px.dtype == np.float32


@pytest.mark.parametrize("corrupt, message", [
    (lambda d: b"XHDR" + d[4:], "not an RHDR"),
    (lambda d: d[:11], "shorter than"),
    (lambda d: d[:10] + struct.pack("<H", 1) + d[12:], "header says"),   # fp16 data, fp32 flag
    (lambda d: d[:12] + b"not zlib", "not a zlib stream"),
])
def test_decode_rejects_what_the_parser_rejects(corrupt, message):
    with pytest.raises(ValueError, match=message):
        rhdr.decode(corrupt(rhdr.encode(_ramp())))
