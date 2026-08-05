from __future__ import annotations

import threading

import numpy as np

from audio_interleaver.audio import (
    AudioEngine,
    InterleavePattern,
    LoadedAudio,
    RegionInsert,
)
from audio_interleaver.playback import (
    AudioPreviewController,
    PlaybackController,
    RenderedPlaybackController,
)


class FakeOutputStream:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.writes = []
        self.aborted = False
        self.stopped = False
        self.call_threads = []
        self.instances.append(self)

    def start(self):
        self.call_threads.append(("start", threading.get_ident()))

    def write(self, samples):
        self.call_threads.append(("write", threading.get_ident()))
        self.writes.append(samples.copy())

    def abort(self):
        self.call_threads.append(("abort", threading.get_ident()))
        self.aborted = True

    def stop(self):
        self.call_threads.append(("stop", threading.get_ident()))
        self.stopped = True

    def close(self):
        self.call_threads.append(("close", threading.get_ident()))


def test_pattern_is_sampled_before_each_playback_slot(monkeypatch):
    monkeypatch.setattr("audio_interleaver.playback.sd.OutputStream", FakeOutputStream)
    source_a = LoadedAudio(np.full((1080, 1), 0.1, dtype=np.float32), 1000)
    source_b = LoadedAudio(np.full((1080, 1), 0.9, dtype=np.float32), 1000)
    engine = AudioEngine(source_a, source_b, smoothing_ms=0)
    values = iter(
        (
            InterleavePattern(fill=0.0),
            InterleavePattern(fill=1.0),
            InterleavePattern(fill=0.0),
        )
    )
    finished = threading.Event()
    natural = []
    errors = []
    controller = PlaybackController(
        on_position=lambda _position: None,
        on_finished=lambda did_finish: (natural.append(did_finish), finished.set()),
        on_error=errors.append,
    )

    assert controller.start(engine, lambda: next(values))
    assert finished.wait(2)

    stream = FakeOutputStream.instances[-1]
    rendered = np.concatenate(stream.writes)
    np.testing.assert_allclose(rendered[:360], 0.1)
    np.testing.assert_allclose(rendered[360:720], 0.9)
    np.testing.assert_allclose(rendered[720:], 0.1)
    assert max(map(len, stream.writes)) == 20
    assert natural == [True]
    assert errors == []


def test_loop_replays_the_complete_result(monkeypatch):
    FakeOutputStream.instances.clear()
    monkeypatch.setattr("audio_interleaver.playback.sd.OutputStream", FakeOutputStream)
    source_a = LoadedAudio(np.full((720, 1), 0.1, dtype=np.float32), 1000)
    source_b = LoadedAudio(np.full((720, 1), 0.9, dtype=np.float32), 1000)
    engine = AudioEngine(source_a, source_b, smoothing_ms=0)
    loop_values = iter((True, False))
    positions = []
    finished = threading.Event()
    natural = []
    controller = PlaybackController(
        on_position=positions.append,
        on_finished=lambda did_finish: (natural.append(did_finish), finished.set()),
        on_error=lambda _message: None,
    )

    assert controller.start(
        engine, lambda: InterleavePattern(fill=1.0), lambda: next(loop_values)
    )
    assert finished.wait(2)

    stream = FakeOutputStream.instances[-1]
    rendered = np.concatenate(stream.writes)
    assert len(rendered) == 1440
    np.testing.assert_allclose(rendered[:720], rendered[720:])
    assert positions == [0.36, 0.72, 0.0, 0.36, 0.72]
    assert natural == [True]


def test_region_insert_playback_uses_selected_b_source_window(monkeypatch):
    FakeOutputStream.instances.clear()
    monkeypatch.setattr("audio_interleaver.playback.sd.OutputStream", FakeOutputStream)
    source_a = LoadedAudio(
        np.repeat(np.array((0.1, 0.2, 0.3), dtype=np.float32), 100)[:, None],
        1000,
    )
    source_b = LoadedAudio(
        np.repeat(np.array((0.6, 0.7, 0.8), dtype=np.float32), 100)[:, None],
        1000,
    )
    engine = AudioEngine(source_a, source_b, slot_ms=100, smoothing_ms=0)
    finished = threading.Event()
    controller = PlaybackController(
        on_position=lambda _position: None,
        on_finished=lambda _natural: finished.set(),
        on_error=lambda _message: None,
    )

    assert controller.start(
        engine,
        lambda: RegionInsert(b_source_ms=150, output_slot=1, length_slots=1),
    )
    assert finished.wait(2)

    rendered = np.concatenate(FakeOutputStream.instances[-1].writes)
    np.testing.assert_allclose(rendered[:100], 0.1)
    np.testing.assert_allclose(rendered[100:150], 0.7)
    np.testing.assert_allclose(rendered[150:200], 0.8)
    np.testing.assert_allclose(rendered[200:], 0.3)


def test_audio_preview_controller_plays_the_supplied_buffer(monkeypatch):
    FakeOutputStream.instances.clear()
    monkeypatch.setattr("audio_interleaver.playback.sd.OutputStream", FakeOutputStream)
    source = LoadedAudio(np.full((250, 1), 0.4, dtype=np.float32), 2000)
    finished = threading.Event()
    results = []
    controller = AudioPreviewController(
        on_finished=lambda preview_id, natural: (
            results.append((preview_id, natural)),
            finished.set(),
        ),
        on_error=lambda _message: None,
    )

    assert controller.start(source, "source-A")
    assert finished.wait(2)

    stream = FakeOutputStream.instances[-1]
    assert stream.kwargs["samplerate"] == 2000
    np.testing.assert_allclose(np.concatenate(stream.writes), source.samples)
    assert len(stream.writes[-1]) < len(stream.writes[0])
    assert stream.stopped
    assert results == [("source-A", True)]


class BlockingPreviewStream(FakeOutputStream):
    first_write = threading.Event()
    release_write = threading.Event()

    def write(self, samples):
        super().write(samples)
        self.first_write.set()
        self.release_write.wait(0.1)

    def abort(self):
        super().abort()
        self.release_write.set()


def test_audio_preview_stop_interrupts_playback_between_small_blocks(monkeypatch):
    BlockingPreviewStream.instances.clear()
    BlockingPreviewStream.first_write.clear()
    BlockingPreviewStream.release_write.clear()
    monkeypatch.setattr(
        "audio_interleaver.playback.sd.OutputStream", BlockingPreviewStream
    )
    source = LoadedAudio(np.full((2000, 1), 0.4, dtype=np.float32), 1000)
    finished = threading.Event()
    results = []
    controller = AudioPreviewController(
        on_finished=lambda preview_id, natural: (
            results.append((preview_id, natural)),
            finished.set(),
        ),
        on_error=lambda _message: None,
    )

    assert controller.start(source, "source-A")
    assert BlockingPreviewStream.first_write.wait(2)
    controller.stop(wait=True)
    assert finished.wait(2)

    stream = BlockingPreviewStream.instances[-1]
    assert not stream.aborted
    assert not stream.stopped
    assert sum(len(block) for block in stream.writes) < source.frames
    assert results == [("source-A", False)]


def test_stream_lifecycle_stays_on_playback_worker(monkeypatch):
    FakeOutputStream.instances.clear()
    monkeypatch.setattr("audio_interleaver.playback.sd.OutputStream", FakeOutputStream)
    source = LoadedAudio(np.full((100, 1), 0.4, dtype=np.float32), 1000)
    finished = threading.Event()
    controller = AudioPreviewController(
        on_finished=lambda _preview_id, _natural: finished.set(),
        on_error=lambda _message: None,
    )

    assert controller.start(source, "source-A")
    assert finished.wait(2)

    stream = FakeOutputStream.instances[-1]
    method_threads = {thread_id for _method, thread_id in stream.call_threads}
    assert method_threads == {stream.call_threads[0][1]}
    assert threading.get_ident() not in method_threads
    assert [method for method, _thread in stream.call_threads][-2:] == [
        "stop",
        "close",
    ]


def test_preview_can_be_restarted_repeatedly_without_overlapping_streams(monkeypatch):
    FakeOutputStream.instances.clear()
    monkeypatch.setattr("audio_interleaver.playback.sd.OutputStream", FakeOutputStream)
    source = LoadedAudio(np.full((100, 1), 0.4, dtype=np.float32), 1000)
    finished_count = 0
    finished = threading.Event()

    def on_finished(_preview_id, _natural):
        nonlocal finished_count
        finished_count += 1
        finished.set()

    controller = AudioPreviewController(on_finished, lambda _message: None)
    for index in range(50):
        finished.clear()
        assert controller.start(source, f"region-B-{index}")
        assert finished.wait(2)
        assert not controller.is_playing

    assert finished_count == 50
    assert len(FakeOutputStream.instances) == 50


def test_rendered_playback_reports_position_for_fixed_acelp_result(monkeypatch):
    FakeOutputStream.instances.clear()
    monkeypatch.setattr("audio_interleaver.playback.sd.OutputStream", FakeOutputStream)
    source = LoadedAudio(np.full((400, 1), 0.2, dtype=np.float32), 8000)
    positions = []
    finished = threading.Event()
    results = []
    controller = RenderedPlaybackController(
        on_position=positions.append,
        on_finished=lambda natural: (results.append(natural), finished.set()),
        on_error=lambda _message: None,
    )

    assert controller.start(source)
    assert finished.wait(2)

    stream = FakeOutputStream.instances[-1]
    assert stream.kwargs["samplerate"] == 8000
    assert stream.kwargs["channels"] == 1
    np.testing.assert_allclose(np.concatenate(stream.writes), source.samples)
    assert positions[-1] == source.duration
    assert results == [True]
