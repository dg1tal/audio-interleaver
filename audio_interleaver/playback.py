"""Live slot-by-slot playback for the interleaver."""

from __future__ import annotations

from collections.abc import Callable
import threading

import sounddevice as sd

from .audio import AudioEngine, SourceId

CrossfaderProvider = Callable[[], float]
LoopProvider = Callable[[], bool]
PositionCallback = Callable[[float], None]
FinishedCallback = Callable[[bool], None]
ErrorCallback = Callable[[str], None]


class PlaybackController:
    """Stream rendered slots on a worker thread.

    The crossfader provider is sampled immediately before each slot is rendered,
    which makes UI changes take effect at the next chunk boundary.
    """

    def __init__(
        self,
        on_position: PositionCallback,
        on_finished: FinishedCallback,
        on_error: ErrorCallback,
    ) -> None:
        self._on_position = on_position
        self._on_finished = on_finished
        self._on_error = on_error
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._stream: sd.OutputStream | None = None
        self._lock = threading.Lock()

    @property
    def is_playing(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def start(
        self,
        engine: AudioEngine,
        crossfader: CrossfaderProvider,
        loop: LoopProvider | None = None,
    ) -> bool:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                args=(engine, crossfader, loop),
                name="audio-playback",
                daemon=True,
            )
            self._thread.start()
        return True

    def stop(self, wait: bool = False) -> None:
        self._stop_event.set()
        with self._lock:
            stream = self._stream
            thread = self._thread
        if stream is not None:
            try:
                stream.abort()
            except sd.PortAudioError:
                pass
        if wait and thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)

    def _run(
        self,
        engine: AudioEngine,
        crossfader: CrossfaderProvider,
        loop: LoopProvider | None,
    ) -> None:
        natural_finish = False
        try:
            stream = sd.OutputStream(
                samplerate=engine.sample_rate,
                channels=engine.channels,
                dtype="float32",
            )
            with self._lock:
                self._stream = stream
            stream.start()
            while not self._stop_event.is_set():
                previous_source: SourceId | None = None
                for slot_index in range(engine.slot_count):
                    if self._stop_event.is_set():
                        break
                    slot, previous_source = engine.render_slot(
                        slot_index, crossfader(), previous_source
                    )
                    stream.write(slot)
                    if self._stop_event.is_set():
                        break
                    played_frames = min(
                        (slot_index + 1) * engine.slot_frames, engine.total_frames
                    )
                    self._on_position(played_frames / engine.sample_rate)
                if self._stop_event.is_set():
                    break
                if loop is not None and loop():
                    self._on_position(0.0)
                    continue
                natural_finish = True
                break
        except Exception as exc:  # PortAudio exposes several environment-specific errors.
            if not self._stop_event.is_set():
                self._on_error(f"Playback failed: {exc}")
        finally:
            with self._lock:
                stream = self._stream
                self._stream = None
            if stream is not None:
                try:
                    stream.close()
                except sd.PortAudioError:
                    pass
            with self._lock:
                self._thread = None
            self._on_finished(natural_finish)
