from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from audio_interleaver.audio import (
    AudioEngine,
    AudioError,
    LoadedAudio,
    load_wav,
    select_source,
    write_wav,
)


def audio(values, sample_rate=1000, channels=1) -> LoadedAudio:
    samples = np.asarray(values, dtype=np.float32)
    if samples.ndim == 0:
        samples = np.full((sample_rate, channels), samples, dtype=np.float32)
    elif samples.ndim == 1:
        samples = samples[:, np.newaxis]
    return LoadedAudio(samples, sample_rate)


@pytest.mark.parametrize(
    ("position", "expected"),
    [
        (0.0, "AAAAAAAA"),
        (0.25, "AAABAAAB"),
        (0.5, "ABABABAB"),
        (0.75, "ABBBABBB"),
        (1.0, "BBBBBBBB"),
    ],
)
def test_source_selection_is_deterministic_and_even(position, expected):
    assert "".join(select_source(index, position) for index in range(8)) == expected


def test_crossfader_values_are_clamped():
    assert select_source(3, -2) == "A"
    assert select_source(3, 4) == "B"


def test_center_uses_shared_timeline_slots():
    source_a = audio(np.full(1000, 0.1))
    source_b = audio(np.full(1000, 0.8))
    engine = AudioEngine(source_a, source_b, smoothing_ms=0)

    rendered = engine.render(0.5)[:, 0]

    np.testing.assert_allclose(rendered[:360], 0.1)
    np.testing.assert_allclose(rendered[360:720], 0.8)
    np.testing.assert_allclose(rendered[720:], 0.1)


def test_endpoints_select_only_one_source():
    source_a = audio(np.full(900, -0.25))
    source_b = audio(np.full(900, 0.75))
    engine = AudioEngine(source_a, source_b)

    np.testing.assert_allclose(engine.render(0.0), -0.25)
    np.testing.assert_allclose(engine.render(1.0), 0.75)


def test_shorter_source_loops_until_longer_source_ends():
    source_a = audio(np.arange(500, dtype=np.float32) / 1000)
    source_b = audio(np.zeros(1200, dtype=np.float32))
    engine = AudioEngine(source_a, source_b, smoothing_ms=0)

    rendered = engine.render(0.0)[:, 0]

    assert len(rendered) == 1200
    np.testing.assert_allclose(rendered[:500], source_a.samples[:, 0])
    np.testing.assert_allclose(rendered[500:1000], source_a.samples[:, 0])
    np.testing.assert_allclose(rendered[1000:], source_a.samples[:200, 0])


def test_final_partial_slot_is_preserved():
    engine = AudioEngine(
        audio(np.zeros(850, dtype=np.float32)),
        audio(np.ones(800, dtype=np.float32)),
        smoothing_ms=0,
    )

    assert engine.slot_count == 3
    assert len(engine.render_slot(2, 0.0)[0]) == 130
    assert len(engine.render(0.5)) == 850


def test_sample_rates_and_channels_are_normalized():
    mono_1khz = audio(np.linspace(-0.5, 0.5, 1000), sample_rate=1000)
    stereo_2khz = audio(
        np.column_stack(
            (np.linspace(-0.25, 0.25, 2000), np.linspace(0.25, -0.25, 2000))
        ),
        sample_rate=2000,
    )

    engine = AudioEngine(mono_1khz, stereo_2khz)

    assert engine.sample_rate == 2000
    assert engine.channels == 2
    assert abs(engine.source_a.frames - 2000) <= 1
    np.testing.assert_allclose(
        engine.source_a.samples[:, 0], engine.source_a.samples[:, 1]
    )


def test_source_switch_has_equal_power_smoothing():
    engine = AudioEngine(audio(0.0), audio(1.0), smoothing_ms=5)

    first, previous = engine.render_slot(0, 0.5)
    second, _ = engine.render_slot(1, 0.5, previous)

    assert np.all(first == 0.0)
    transition = second[:5, 0]
    assert transition[0] == pytest.approx(0.0)
    assert transition[-1] == pytest.approx(1.0)
    assert np.all(np.diff(transition) >= 0)
    np.testing.assert_allclose(second[5:], 1.0)


def test_configurable_chunk_and_crossfade_durations():
    engine = AudioEngine(audio(0.0), audio(1.0), slot_ms=250, smoothing_ms=20)

    assert engine.slot_frames == 250
    assert engine.smoothing_frames == 20
    assert engine.slot_count == 4


def test_wav_round_trip(tmp_path):
    path = tmp_path / "round-trip.wav"
    samples = np.column_stack(
        (np.linspace(-0.8, 0.8, 100), np.linspace(0.8, -0.8, 100))
    ).astype(np.float32)

    write_wav(path, samples, 48000)
    loaded = load_wav(path)

    assert loaded.sample_rate == 48000
    assert loaded.channels == 2
    np.testing.assert_allclose(loaded.samples, samples, atol=1 / 32768)


def test_non_wav_and_empty_audio_are_rejected(tmp_path):
    flac_path = tmp_path / "not-wav.flac"
    sf.write(flac_path, np.zeros(10, dtype=np.float32), 1000, format="FLAC")
    with pytest.raises(AudioError, match="not a WAV"):
        load_wav(flac_path)

    with pytest.raises(AudioError, match="no audio frames"):
        LoadedAudio(np.empty((0, 1), dtype=np.float32), 1000)
