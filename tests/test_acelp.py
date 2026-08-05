from __future__ import annotations

import hashlib

import numpy as np
import pytest

from audio_interleaver.acelp import (
    ACELP_FRAME_BITS,
    ACELP_FRAME_SAMPLES,
    AcelpEngine,
    AcelpSymbols,
    TetraAcelpCodec,
    snap_acelp_chunk_ms,
)
from audio_interleaver.audio import InterleavePattern, LoadedAudio, RegionInsert


def audio(values, sample_rate=8000, channels=1) -> LoadedAudio:
    samples = np.asarray(values, dtype=np.float32)
    if samples.ndim == 0:
        samples = np.full((sample_rate, channels), samples, dtype=np.float32)
    elif samples.ndim == 1:
        samples = samples[:, np.newaxis]
    return LoadedAudio(samples, sample_rate)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (-10, 30),
        (30, 30),
        (44, 30),
        (45, 60),
        (371, 360),
        (1999, 1980),
    ],
)
def test_acelp_chunk_duration_snaps_to_legal_frames(value, expected):
    assert snap_acelp_chunk_ms(value) == expected


def test_reference_codec_keeps_state_across_frames_and_round_trips_shape():
    codec = TetraAcelpCodec()
    symbols = codec.encode(np.zeros(2 * ACELP_FRAME_SAMPLES, dtype=np.int16))

    assert symbols.packed.shape == (2, 18)
    assert symbols.unpacked_bits().shape == (2, ACELP_FRAME_BITS)
    assert not np.array_equal(symbols.packed[0], symbols.packed[1])
    assert hashlib.sha256(symbols.packed.tobytes()).hexdigest() == (
        "409a755778a5a0cdb21b322c5d2b40504338b9db241f9b27296ee4418d28f54b"
    )
    assert codec.decode(symbols).shape == (2 * ACELP_FRAME_SAMPLES,)


def test_spe_format_has_zero_bfi_and_one_little_endian_word_per_bit():
    packed = np.zeros((1, 18), dtype=np.uint8)
    packed[0, 0] = 0b10100000
    symbols = AcelpSymbols(packed)

    words = np.frombuffer(symbols.to_spe_bytes(), dtype="<i2")

    assert len(words) == 138
    assert words[:5].tolist() == [0, 1, 0, 1, 0]
    assert set(words) <= {0, 1}


class RecordingCodec:
    def __init__(self):
        self.encoded_lengths = []

    def encode(self, pcm):
        frame_count = len(pcm) // ACELP_FRAME_SAMPLES
        self.encoded_lengths.append(len(pcm))
        value = len(self.encoded_lengths)
        return AcelpSymbols(np.full((frame_count, 18), value, dtype=np.uint8))

    def decode(self, symbols):
        return np.zeros(symbols.frame_count * ACELP_FRAME_SAMPLES, dtype=np.int16)


def test_source_a_anchors_timeline_even_when_b_is_longer():
    engine = AcelpEngine(
        audio(np.zeros(700)),
        audio(np.zeros(7000)),
        slot_ms=60,
        codec=RecordingCodec(),
    )

    assert engine.slot_count == 2
    assert engine.total_frames == 960
    assert engine.duration == pytest.approx(0.12)


def test_one_stream_encodes_only_active_b_chunks_in_one_call():
    codec = RecordingCodec()
    engine = AcelpEngine(
        audio(np.zeros(1440)),
        audio(np.ones(10000) * 0.1),
        slot_ms=60,
        codec=codec,
    )

    symbols = engine.render_symbols(
        InterleavePattern(fill=1.0), "one_stream"
    )

    # A has three 60 ms slots; only the middle B slot is encoded.
    assert codec.encoded_lengths == [1440, 480]
    assert np.all(symbols.packed[:2] == 1)
    assert np.all(symbols.packed[2:4] == 2)
    assert np.all(symbols.packed[4:] == 1)


def test_restart_mode_resets_for_every_active_b_chunk():
    codec = RecordingCodec()
    engine = AcelpEngine(
        audio(np.zeros(1920)),
        audio(np.ones(10000) * 0.1),
        slot_ms=60,
        codec=codec,
    )

    engine.render_symbols(
        RegionInsert(b_source_slot=0, output_slot=1, length_slots=2),
        "restart_each_chunk",
    )

    assert codec.encoded_lengths == [1920, 480, 480]


def test_stereo_is_averaged_and_resampled_to_codec_rate():
    stereo = np.column_stack((np.full(1600, 0.5), np.full(1600, -0.5)))
    engine = AcelpEngine(audio(stereo, sample_rate=16000), audio(0.0), slot_ms=30)

    assert engine.sample_rate == 8000
    assert engine.channels == 1
    np.testing.assert_allclose(engine.source_a.samples, 0.0, atol=1e-7)
