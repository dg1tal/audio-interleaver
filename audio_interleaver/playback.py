"""Live slot-by-slot playback for the interleaver."""

from __future__ import annotations

from collections.abc import Callable
import threading

import sounddevice as sd

from .audio import AudioEngine, InterleaveSettings, LoadedAudio, SourceId

SettingsProvider = Callable[[], InterleaveSettings]
LoopProvider = Callable[[], bool]
PositionCallback = Callable[[float], None]
FinishedCallback = Callable[[bool], None]
ErrorCallback = Callable[[str], None]
PreviewFinishedCallback = Callable[[str, bool], None]


class PlaybackController:
    """Stream rendered slots on a worker thread.

    Interleave settings are sampled immediately before each slot is rendered,
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
        settings: SettingsProvider,
        loop: LoopProvider | None = None,
    ) -> bool:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                args=(engine, settings, loop),
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
        settings: SettingsProvider,
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
                previous_chunk = None
                for slot_index in range(engine.slot_count):
                    if self._stop_event.is_set():
                        break
                    current_settings = settings()
                    source_id = engine.source_for_slot(slot_index, current_settings)
                    source_chunk_index = engine.source_chunk_index_for_slot(
                        slot_index, current_settings, source_id
                    )
                    slot, previous_source = engine.render_slot(
                        slot_index,
                        current_settings,
                        source_chunk_index,
                        previous_source,
                        previous_chunk,
                    )
                    previous_chunk = slot
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


class AudioPreviewController:
    """Play one loaded audio buffer at a time on a worker thread."""

    def __init__(
        self,
        on_finished: PreviewFinishedCallback,
        on_error: ErrorCallback,
    ) -> None:
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

    def start(self, audio: LoadedAudio, preview_id: str) -> bool:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                args=(audio, preview_id),
                name="audio-preview",
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

    def _run(self, audio: LoadedAudio, preview_id: str) -> None:
        natural_finish = False
        try:
            stream = sd.OutputStream(
                samplerate=audio.sample_rate,
                channels=audio.channels,
                dtype="float32",
            )
            with self._lock:
                self._stream = stream
            stream.start()
            if not self._stop_event.is_set():
                stream.write(audio.samples)
            natural_finish = not self._stop_event.is_set()
        except Exception as exc:  # PortAudio errors vary by host and device.
            if not self._stop_event.is_set():
                self._on_error(f"Preview playback failed: {exc}")
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
            self._on_finished(preview_id, natural_finish)
