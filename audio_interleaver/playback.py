"""Live slot-by-slot playback for the interleaver."""

from __future__ import annotations

from collections.abc import Callable
import itertools
import logging
import threading

import sounddevice as sd

from .audio import AudioEngine, InterleaveSettings, LoadedAudio, SourceId

SettingsProvider = Callable[[], InterleaveSettings]
LoopProvider = Callable[[], bool]
PositionCallback = Callable[[float], None]
FinishedCallback = Callable[[bool], None]
ErrorCallback = Callable[[str], None]
PreviewFinishedCallback = Callable[[str, bool], None]
PREVIEW_BLOCK_SECONDS = 0.02
_SESSION_IDS = itertools.count(1)
_LOG = logging.getLogger(__name__)


def _write_blocks(
    stream: sd.OutputStream,
    samples,
    sample_rate: int,
    stop_event: threading.Event,
) -> bool:
    """Write short blocks, returning false when playback was interrupted."""

    block_frames = max(1, round(sample_rate * PREVIEW_BLOCK_SECONDS))
    for start in range(0, len(samples), block_frames):
        if stop_event.is_set():
            return False
        stream.write(samples[start : start + block_frames])
    return not stop_event.is_set()


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
        self._session_id = 0

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
            self._session_id = next(_SESSION_IDS)
            self._thread = threading.Thread(
                target=self._run,
                args=(self._session_id, engine, settings, loop),
                name="audio-playback",
                daemon=True,
            )
            self._thread.start()
        return True

    def stop(self, wait: bool = False) -> None:
        self._stop_event.set()
        with self._lock:
            thread = self._thread
            session_id = self._session_id
        _LOG.info("playback stop requested session=%s wait=%s", session_id, wait)
        if wait and thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
            if thread.is_alive():
                _LOG.warning("playback stop timed out session=%s", session_id)

    def _run(
        self,
        session_id: int,
        engine: AudioEngine,
        settings: SettingsProvider,
        loop: LoopProvider | None,
    ) -> None:
        natural_finish = False
        try:
            _LOG.info(
                "playback opening session=%s rate=%s channels=%s slots=%s slot_ms=%s",
                session_id,
                engine.sample_rate,
                engine.channels,
                engine.slot_count,
                engine.slot_ms,
            )
            stream = sd.OutputStream(
                samplerate=engine.sample_rate,
                channels=engine.channels,
                dtype="float32",
            )
            with self._lock:
                self._stream = stream
            stream.start()
            _LOG.info("playback started session=%s", session_id)
            loop_count = 0
            while not self._stop_event.is_set():
                previous_source: SourceId | None = None
                previous_chunk = None
                for slot_index in range(engine.slot_count):
                    if self._stop_event.is_set():
                        break
                    current_settings = settings()
                    source_id = engine.source_for_slot(slot_index, current_settings)
                    slot, previous_source = engine.render_slot(
                        slot_index,
                        current_settings,
                        None,
                        previous_source,
                        previous_chunk,
                    )
                    previous_chunk = slot
                    if not _write_blocks(
                        stream, slot, engine.sample_rate, self._stop_event
                    ):
                        break
                    played_frames = min(
                        (slot_index + 1) * engine.slot_frames, engine.total_frames
                    )
                    self._on_position(played_frames / engine.sample_rate)
                if self._stop_event.is_set():
                    break
                if loop is not None and loop():
                    loop_count += 1
                    _LOG.info(
                        "playback looping session=%s iteration=%s",
                        session_id,
                        loop_count,
                    )
                    self._on_position(0.0)
                    continue
                natural_finish = True
                break
        except Exception as exc:  # PortAudio exposes several environment-specific errors.
            _LOG.exception("playback failed session=%s", session_id)
            if not self._stop_event.is_set():
                self._on_error(f"Playback failed: {exc}")
        finally:
            with self._lock:
                stream = self._stream
                self._stream = None
            if stream is not None:
                if natural_finish:
                    try:
                        stream.stop()
                    except Exception:
                        _LOG.exception("playback stop failed session=%s", session_id)
                try:
                    stream.close()
                except Exception:
                    _LOG.exception("playback close failed session=%s", session_id)
            _LOG.info(
                "playback closed session=%s natural=%s",
                session_id,
                natural_finish,
            )
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
        self._session_id = 0

    @property
    def is_playing(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def start(self, audio: LoadedAudio, preview_id: str) -> bool:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._stop_event.clear()
            self._session_id = next(_SESSION_IDS)
            self._thread = threading.Thread(
                target=self._run,
                args=(self._session_id, audio, preview_id),
                name="audio-preview",
                daemon=True,
            )
            self._thread.start()
        return True

    def stop(self, wait: bool = False) -> None:
        self._stop_event.set()
        with self._lock:
            thread = self._thread
            session_id = self._session_id
        _LOG.info("preview stop requested session=%s wait=%s", session_id, wait)
        if wait and thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
            if thread.is_alive():
                _LOG.warning("preview stop timed out session=%s", session_id)

    def _run(self, session_id: int, audio: LoadedAudio, preview_id: str) -> None:
        natural_finish = False
        try:
            _LOG.info(
                "preview opening session=%s target=%s rate=%s channels=%s frames=%s",
                session_id,
                preview_id,
                audio.sample_rate,
                audio.channels,
                audio.frames,
            )
            stream = sd.OutputStream(
                samplerate=audio.sample_rate,
                channels=audio.channels,
                dtype="float32",
            )
            with self._lock:
                self._stream = stream
            stream.start()
            _LOG.info("preview started session=%s target=%s", session_id, preview_id)
            natural_finish = _write_blocks(
                stream, audio.samples, audio.sample_rate, self._stop_event
            )
        except Exception as exc:  # PortAudio errors vary by host and device.
            _LOG.exception("preview failed session=%s target=%s", session_id, preview_id)
            if not self._stop_event.is_set():
                self._on_error(f"Preview playback failed: {exc}")
        finally:
            with self._lock:
                stream = self._stream
                self._stream = None
            if stream is not None:
                if natural_finish:
                    try:
                        # Blocking writes may return while the host still owns
                        # buffered tail samples. stop() drains them; close()
                        # alone is allowed to discard them on some backends.
                        stream.stop()
                    except Exception:
                        _LOG.exception(
                            "preview stop failed session=%s target=%s",
                            session_id,
                            preview_id,
                        )
                try:
                    stream.close()
                except Exception:
                    _LOG.exception(
                        "preview close failed session=%s target=%s",
                        session_id,
                        preview_id,
                    )
            _LOG.info(
                "preview closed session=%s target=%s natural=%s",
                session_id,
                preview_id,
                natural_finish,
            )
            with self._lock:
                self._thread = None
            self._on_finished(preview_id, natural_finish)


class RenderedPlaybackController:
    """Play a fixed rendered buffer with position reporting and optional looping."""

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
        self._session_id = 0

    @property
    def is_playing(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def start(self, audio: LoadedAudio, loop: LoopProvider | None = None) -> bool:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._stop_event.clear()
            self._session_id = next(_SESSION_IDS)
            self._thread = threading.Thread(
                target=self._run,
                args=(self._session_id, audio, loop),
                name="rendered-audio-playback",
                daemon=True,
            )
            self._thread.start()
        return True

    def stop(self, wait: bool = False) -> None:
        self._stop_event.set()
        with self._lock:
            thread = self._thread
            session_id = self._session_id
        _LOG.info("rendered stop requested session=%s wait=%s", session_id, wait)
        if wait and thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
            if thread.is_alive():
                _LOG.warning("rendered stop timed out session=%s", session_id)

    def _run(
        self, session_id: int, audio: LoadedAudio, loop: LoopProvider | None
    ) -> None:
        natural_finish = False
        try:
            _LOG.info(
                "rendered opening session=%s rate=%s channels=%s frames=%s",
                session_id,
                audio.sample_rate,
                audio.channels,
                audio.frames,
            )
            stream = sd.OutputStream(
                samplerate=audio.sample_rate,
                channels=audio.channels,
                dtype="float32",
            )
            with self._lock:
                self._stream = stream
            stream.start()
            _LOG.info("rendered started session=%s", session_id)
            block_frames = max(1, round(audio.sample_rate * PREVIEW_BLOCK_SECONDS))
            loop_count = 0
            while not self._stop_event.is_set():
                for start in range(0, audio.frames, block_frames):
                    if self._stop_event.is_set():
                        break
                    end = min(audio.frames, start + block_frames)
                    stream.write(audio.samples[start:end])
                    self._on_position(end / audio.sample_rate)
                if self._stop_event.is_set():
                    break
                if loop is not None and loop():
                    loop_count += 1
                    _LOG.info(
                        "rendered looping session=%s iteration=%s",
                        session_id,
                        loop_count,
                    )
                    self._on_position(0.0)
                    continue
                natural_finish = True
                break
        except Exception as exc:
            _LOG.exception("rendered playback failed session=%s", session_id)
            if not self._stop_event.is_set():
                self._on_error(f"Playback failed: {exc}")
        finally:
            with self._lock:
                stream = self._stream
                self._stream = None
            if stream is not None:
                if natural_finish:
                    try:
                        stream.stop()
                    except Exception:
                        _LOG.exception("rendered stop failed session=%s", session_id)
                try:
                    stream.close()
                except Exception:
                    _LOG.exception("rendered close failed session=%s", session_id)
            _LOG.info(
                "rendered closed session=%s natural=%s",
                session_id,
                natural_finish,
            )
            with self._lock:
                self._thread = None
            self._on_finished(natural_finish)
