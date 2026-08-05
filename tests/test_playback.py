from __future__ import annotations

import threading

import numpy as np

from audio_interleaver.audio import AudioEngine, InterleavePattern, LoadedAudio
from audio_interleaver.playback import PlaybackController


class FakeOutputStream:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.writes = []
        self.aborted = False
        self.instances.append(self)

    def start(self):
        pass

    def write(self, samples):
        self.writes.append(samples.copy())

    def abort(self):
        self.aborted = True

    def close(self):
        pass


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

    writes = FakeOutputStream.instances[-1].writes
    assert len(writes) == 3
    np.testing.assert_allclose(writes[0], 0.1)
    np.testing.assert_allclose(writes[1], 0.9)
    np.testing.assert_allclose(writes[2], 0.1)
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

    writes = FakeOutputStream.instances[-1].writes
    assert len(writes) == 4
    np.testing.assert_allclose(writes[0], writes[2])
    np.testing.assert_allclose(writes[1], writes[3])
    assert positions == [0.36, 0.72, 0.0, 0.36, 0.72]
    assert natural == [True]
