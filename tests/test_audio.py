from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from audio_interleaver.audio import (
    AudioEngine,
    AudioError,
    InterleavePattern,
    LoadedAudio,
    RegionInsert,
    load_wav,
    occurrence_capacity,
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


def sources(pattern: InterleavePattern, slot_count: int = 8) -> str:
    return "".join(
        select_source(index, slot_count, pattern) for index in range(slot_count)
    )


@pytest.mark.parametrize(
    ("fill", "expected"),
    [
        (0.0, "AAAAAAAA"),
        (0.25, "ABAAAAAA"),
        (0.5, "ABABAAAA"),
        (0.75, "ABABABAA"),
        (1.0, "ABABABAB"),
    ],
)
def test_occurrences_are_revealed_from_left_to_right(fill, expected):
    assert sources(InterleavePattern(fill=fill)) == expected


def test_fill_values_are_clamped():
    assert sources(InterleavePattern(fill=-2)) == "AAAAAAAA"
    assert sources(InterleavePattern(fill=4)) == "ABABABAB"


def test_occurrence_capacity_matches_meaningful_fill_steps():
    assert occurrence_capacity(12, InterleavePattern(b_chunks_per_occurrence=1)) == 6
    assert occurrence_capacity(12, InterleavePattern(b_chunks_per_occurrence=2)) == 4


def test_increasing_fill_never_moves_or_removes_an_active_b_slot():
    previous_b_slots: set[int] = set()
    for value in range(101):
        current = sources(
            InterleavePattern(fill=value / 100, b_chunks_per_occurrence=2), 30
        )
        current_b_slots = {
            index for index, source in enumerate(current) if source == "B"
        }
        assert previous_b_slots <= current_b_slots
        previous_b_slots = current_b_slots


def test_moving_first_b_right_keeps_filled_occurrences_until_they_stop_fitting():
    pattern = InterleavePattern(fill=0.5, first_alternate_slot=7)
    assert sources(pattern, 12) == "AAAAAAABABAB"


@pytest.mark.parametrize(
    ("burst", "expected"),
    [(1, "ABABABABA"), (2, "ABBABBABB"), (3, "ABBBABBBA")],
)
def test_burst_size_controls_chunks_per_occurrence(burst, expected):
    pattern = InterleavePattern(fill=1.0, b_chunks_per_occurrence=burst)
    assert sources(pattern, 9) == expected


@pytest.mark.parametrize(
    ("burst", "expected"),
    [(1, "BABABABAB"), (2, "BBABBABBA"), (3, "BBBABBBAB")],
)
def test_starting_b_occurrence_uses_the_configured_burst_size(burst, expected):
    pattern = InterleavePattern(
        fill=1.0,
        starts_with="B",
        b_chunks_per_occurrence=burst,
    )

    assert sources(pattern, 9) == expected


def test_starting_b_burst_is_anchored_even_with_zero_optional_fill():
    pattern = InterleavePattern(
        fill=0.0,
        starts_with="B",
        b_chunks_per_occurrence=3,
    )

    assert sources(pattern, 8) == "BBBAAAAA"


def test_start_source_and_first_alternate_position_control_the_prefix():
    start_a = InterleavePattern(
        fill=1.0, first_alternate_slot=3, b_chunks_per_occurrence=2
    )
    start_b = InterleavePattern(
        fill=1.0,
        starts_with="B",
        first_alternate_slot=3,
        b_chunks_per_occurrence=2,
    )

    assert sources(start_a, 9) == "AAABBABBA"
    assert sources(start_b, 9) == "BBBABBABB"
    anchored_b = InterleavePattern(
        fill=0.0, starts_with="B", first_alternate_slot=3
    )
    assert sources(anchored_b, 8) == "BBBAAAAA"


def test_pattern_keeps_a_timeline_fixed_while_b_advances_continuously():
    source_a = audio(np.repeat(np.arange(6, dtype=np.float32), 360))
    source_b = audio(np.repeat(np.arange(10, 16, dtype=np.float32), 360))
    engine = AudioEngine(source_a, source_b, smoothing_ms=0)

    pattern = InterleavePattern(fill=1.0)
    rendered = engine.render(pattern)[:, 0].reshape(6, 360)

    np.testing.assert_allclose(rendered[:, 0], [0, 10, 2, 11, 4, 12])

    assert [
        engine.source_chunk_index_for_slot(index, pattern)
        for index in range(engine.slot_count)
    ] == [0, 0, 2, 1, 4, 2]


def test_source_region_returns_exact_whole_chunks_with_final_padding():
    engine = AudioEngine(
        audio(np.zeros(600, dtype=np.float32)),
        audio(np.arange(450, dtype=np.float32)),
        slot_ms=100,
        smoothing_ms=0,
    )

    region = engine.source_region("B", source_start_ms=300, chunk_count=2)[:, 0]

    np.testing.assert_allclose(region[:150], np.arange(300, 450, dtype=np.float32))
    np.testing.assert_allclose(region[150:], 0.0)


def test_first_inserted_b_slot_starts_with_b_chunk_zero():
    source_a = audio(np.zeros(2160, dtype=np.float32))
    source_b = audio(np.arange(2160, dtype=np.float32) / 10000)
    engine = AudioEngine(source_a, source_b, smoothing_ms=0)
    pattern = InterleavePattern(fill=1.0, first_alternate_slot=3)

    rendered = engine.render(pattern)[:, 0]

    np.testing.assert_allclose(rendered[1080:1440], source_b.samples[:360, 0])


def test_source_shorter_than_a_chunk_restarts_and_silence_pads_each_occurrence():
    short_source = audio(np.linspace(0.1, 0.5, 100, dtype=np.float32))
    engine = AudioEngine(
        short_source,
        audio(np.zeros(720, dtype=np.float32)),
        smoothing_ms=0,
    )

    rendered = engine.render(InterleavePattern(fill=0.0))[:, 0]

    for slot_start in (0, 360):
        np.testing.assert_allclose(
            rendered[slot_start : slot_start + 100], short_source.samples[:, 0]
        )
        np.testing.assert_allclose(rendered[slot_start + 100 : slot_start + 360], 0.0)


def test_zero_fill_selects_only_source_a():
    source_a = audio(np.full(900, -0.25))
    source_b = audio(np.full(900, 0.75))
    engine = AudioEngine(source_a, source_b)

    rendered = engine.render(InterleavePattern(fill=0.0))[:, 0]
    np.testing.assert_allclose(rendered[:900], -0.25)
    np.testing.assert_allclose(rendered[900:], 0.0)


def test_shorter_source_loops_as_complete_silence_padded_chunks():
    source_a = audio(np.arange(500, dtype=np.float32) / 1000)
    source_b = audio(np.zeros(1200, dtype=np.float32))
    engine = AudioEngine(source_a, source_b)

    rendered = engine.render(InterleavePattern(fill=0.0))[:, 0]

    assert len(rendered) == 1440
    np.testing.assert_allclose(rendered[:360], source_a.samples[:360, 0])
    np.testing.assert_allclose(rendered[360:500], source_a.samples[360:, 0])
    np.testing.assert_allclose(rendered[500:720], 0.0)
    np.testing.assert_allclose(rendered[720:1080], source_a.samples[:360, 0])
    np.testing.assert_allclose(rendered[1080:1200], source_a.samples[360:480, 0])
    np.testing.assert_allclose(rendered[1200:], 0.0)


def test_region_insert_uses_independent_b_source_and_output_positions():
    chunk_values_a = np.repeat(np.arange(6, dtype=np.float32), 100)
    chunk_values_b = np.repeat(np.arange(10, 16, dtype=np.float32), 100)
    engine = AudioEngine(
        audio(chunk_values_a),
        audio(chunk_values_b),
        slot_ms=100,
        smoothing_ms=0,
    )
    settings = RegionInsert(b_source_ms=200, output_slot=1, length_slots=3)

    rendered = engine.render(settings)[:, 0].reshape(6, 100)

    np.testing.assert_allclose(rendered[:, 0], [0, 12, 13, 14, 4, 5])


def test_region_insert_is_clipped_only_by_the_output_timeline():
    settings = RegionInsert(b_source_ms=0, output_slot=4, length_slots=3)
    assert "".join(select_source(index, 6, settings) for index in range(6)) == "AAAABB"


def test_region_insert_can_replace_chunks_after_b_ends_with_silence():
    source_a = audio(np.full(500, 0.25, dtype=np.float32))
    source_b = audio(np.full(150, 0.75, dtype=np.float32))
    engine = AudioEngine(
        source_a,
        source_b,
        slot_ms=100,
        smoothing_ms=0,
    )
    settings = RegionInsert(
        b_source_ms=100,
        output_slot=1,
        length_slots=4,
        silence_after_b_end=True,
    )

    rendered = engine.render(settings)[:, 0].reshape(5, 100)

    np.testing.assert_allclose(rendered[0], 0.25)
    np.testing.assert_allclose(rendered[1, :50], 0.75)
    np.testing.assert_allclose(rendered[1, 50:], 0.0)
    np.testing.assert_allclose(rendered[2:], 0.0)


def test_source_region_can_pad_all_chunks_after_source_end():
    engine = AudioEngine(
        audio(np.zeros(500, dtype=np.float32)),
        audio(np.ones(150, dtype=np.float32)),
        slot_ms=100,
        smoothing_ms=0,
    )

    region = engine.source_region(
        "B", source_start_ms=100, chunk_count=3, silence_after_end=True
    )[:, 0]

    np.testing.assert_allclose(region[:50], 1.0)
    np.testing.assert_allclose(region[50:], 0.0)


def test_region_insert_can_start_between_chunk_boundaries():
    source_a = audio(np.zeros(400, dtype=np.float32))
    source_b = audio(np.arange(400, dtype=np.float32) / 1000)
    engine = AudioEngine(source_a, source_b, slot_ms=100, smoothing_ms=0)

    rendered = engine.render(
        RegionInsert(b_source_ms=35, output_slot=1, length_slots=2)
    )[:, 0]

    np.testing.assert_allclose(rendered[100:300], source_b.samples[35:235, 0])


def test_final_partial_slot_is_preserved_and_padded_with_silence():
    source_a = audio(np.full(850, 0.25, dtype=np.float32))
    engine = AudioEngine(
        source_a,
        audio(np.ones(800, dtype=np.float32)),
        smoothing_ms=0,
    )

    assert engine.slot_count == 3
    assert engine.content_frames == 850
    assert engine.total_frames == 1080
    pattern = InterleavePattern(fill=0.0)
    final_slot = engine.render_slot(2, pattern)[0]
    assert len(final_slot) == 360
    np.testing.assert_allclose(final_slot[:130], 0.25)
    np.testing.assert_allclose(final_slot[130:], 0.0)
    assert len(engine.render(pattern)) == 1080


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


def test_source_switch_has_equal_power_smoothing_without_changing_chunk_order():
    source_a = audio(np.arange(1000, dtype=np.float32) / 2000)
    source_b = audio(np.ones(1000, dtype=np.float32))
    engine = AudioEngine(source_a, source_b, smoothing_ms=5)

    pattern = InterleavePattern(fill=1.0)
    first, previous = engine.render_slot(0, pattern)
    second, _ = engine.render_slot(1, pattern, 0, previous, first)

    np.testing.assert_allclose(first[:, 0], source_a.samples[:360, 0])
    transition = second[:5, 0]
    assert transition[0] == pytest.approx(source_a.samples[355, 0])
    assert transition[-1] == pytest.approx(1.0, abs=1e-7)
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
