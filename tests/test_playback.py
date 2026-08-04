from __future__ import annotations

import threading

import numpy as np

from audio_interleaver.audio import AudioEngine, LoadedAudio
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


def test_crossfader_is_sampled_before_each_playback_slot(monkeypatch):
    monkeypatch.setattr("audio_interleaver.playback.sd.OutputStream", FakeOutputStream)
    source_a = LoadedAudio(np.full((1080, 1), 0.1, dtype=np.float32), 1000)
    source_b = LoadedAudio(np.full((1080, 1), 0.9, dtype=np.float32), 1000)
    engine = AudioEngine(source_a, source_b, smoothing_ms=0)
    values = iter((0.0, 1.0, 0.0))
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

