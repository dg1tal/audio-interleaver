"""ETSI TETRA ACELP symbol-domain encoding, interleaving, and decoding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Callable, Literal

import numpy as np
import soxr

try:
    from . import _tetra_acelp as _native_codec
except ImportError:  # Allows the raw-wave UI to explain an unbuilt extension.
    _native_codec = None

from .audio import (
    AudioError,
    InterleaveSettings,
    LoadedAudio,
    RegionInsert,
    RenderingCancelled,
    SourceId,
    select_source,
)

ACELP_SAMPLE_RATE = 8000
ACELP_FRAME_MS = 30
ACELP_FRAME_SAMPLES = 240
ACELP_FRAME_BITS = 137
ACELP_PACKED_BYTES = 18
ACELP_MIN_CHUNK_MS = 30
ACELP_MAX_CHUNK_MS = 1980

ProcessingStage = Literal["raw", "acelp"]
BEncoderMode = Literal["one_stream", "restart_each_chunk"]
ProgressCallback = Callable[[float], None]


def snap_acelp_chunk_ms(value: int | float) -> int:
    """Snap milliseconds to a legal 30 ms ACELP duration, ties upward."""

    clamped = min(ACELP_MAX_CHUNK_MS, max(ACELP_MIN_CHUNK_MS, float(value)))
    frames = int(np.floor(clamped / ACELP_FRAME_MS + 0.5))
    return frames * ACELP_FRAME_MS


def _codec_audio(audio: LoadedAudio) -> LoadedAudio:
    mono = np.mean(audio.samples, axis=1, dtype=np.float32)
    if audio.sample_rate != ACELP_SAMPLE_RATE:
        mono = soxr.resample(
            mono, audio.sample_rate, ACELP_SAMPLE_RATE, quality="HQ"
        ).astype(np.float32, copy=False)
    return LoadedAudio(
        np.ascontiguousarray(mono[:, np.newaxis], dtype=np.float32),
        ACELP_SAMPLE_RATE,
        audio.path,
    )


def _float_to_pcm16(samples: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    scaled = np.where(clipped < 0, clipped * 32768.0, clipped * 32767.0)
    return np.ascontiguousarray(np.rint(scaled), dtype="<i2")


@dataclass(frozen=True, slots=True)
class AcelpSymbols:
    """Packed 137-bit ACELP speech frames, one 18-byte row per 30 ms."""

    packed: np.ndarray

    def __post_init__(self) -> None:
        packed = np.asarray(self.packed, dtype=np.uint8)
        if packed.ndim != 2 or packed.shape[1] != ACELP_PACKED_BYTES:
            raise ValueError("ACELP symbols must have shape (frames, 18)")
        if len(packed) == 0:
            raise ValueError("ACELP symbols cannot be empty")
        object.__setattr__(self, "packed", np.ascontiguousarray(packed))

    @property
    def frame_count(self) -> int:
        return int(len(self.packed))

    def unpacked_bits(self) -> np.ndarray:
        return np.unpackbits(self.packed, axis=1)[:, :ACELP_FRAME_BITS]

    def to_spe_bytes(self) -> bytes:
        words = np.zeros((self.frame_count, ACELP_FRAME_BITS + 1), dtype="<i2")
        words[:, 1:] = self.unpacked_bits()
        return words.tobytes()

    def write_spe(self, path: str | Path) -> None:
        output_path = Path(path)
        try:
            output_path.write_bytes(self.to_spe_bytes())
        except OSError as exc:
            raise AudioError(f"Could not write {output_path.name}: {exc}") from exc


class TetraAcelpCodec:
    """Small array-oriented wrapper around the vendored fixed-point codec."""

    def __init__(self) -> None:
        if _native_codec is None:
            raise AudioError(
                "The native TETRA ACELP extension is not built. "
                "Install the project with 'python -m pip install -e .'."
            )

    def encode(self, pcm: np.ndarray) -> AcelpSymbols:
        samples = np.asarray(pcm, dtype="<i2").reshape(-1)
        if len(samples) == 0 or len(samples) % ACELP_FRAME_SAMPLES:
            raise ValueError("PCM must contain complete 240-sample ACELP frames")
        packed = _native_codec.encode_stream(
            np.ascontiguousarray(samples).tobytes()
        )
        return AcelpSymbols(
            np.frombuffer(packed, dtype=np.uint8).reshape(-1, ACELP_PACKED_BYTES)
        )

    def decode(self, symbols: AcelpSymbols) -> np.ndarray:
        pcm = _native_codec.decode_stream(symbols.packed.tobytes())
        return np.frombuffer(pcm, dtype="<i2").copy()


@dataclass(slots=True)
class AcelpEngine:
    """Source-A anchored interleaver operating on complete ACELP frames."""

    source_a: LoadedAudio
    source_b: LoadedAudio
    slot_ms: int = 360
    codec: TetraAcelpCodec | None = None

    def __post_init__(self) -> None:
        if self.slot_ms != snap_acelp_chunk_ms(self.slot_ms):
            raise ValueError("ACELP chunk duration must be a multiple of 30 ms")
        self.source_a = _codec_audio(self.source_a)
        self.source_b = _codec_audio(self.source_b)
        if self.codec is None:
            self.codec = TetraAcelpCodec()

    @property
    def sample_rate(self) -> int:
        return ACELP_SAMPLE_RATE

    @property
    def channels(self) -> int:
        return 1

    @property
    def slot_frames(self) -> int:
        return self.slot_ms * ACELP_SAMPLE_RATE // 1000

    @property
    def codec_frames_per_slot(self) -> int:
        return self.slot_ms // ACELP_FRAME_MS

    @property
    def slot_count(self) -> int:
        return max(1, int(np.ceil(self.source_a.frames / self.slot_frames)))

    @property
    def total_frames(self) -> int:
        return self.slot_count * self.slot_frames

    @property
    def duration(self) -> float:
        return self.total_frames / ACELP_SAMPLE_RATE

    def source_for_slot(
        self, slot_index: int, settings: InterleaveSettings
    ) -> SourceId:
        return select_source(slot_index, self.slot_count, settings)

    def source_chunk_count(self, source_id: SourceId) -> int:
        source = self.source_a if source_id == "A" else self.source_b
        return max(1, int(np.ceil(source.frames / self.slot_frames)))

    def source_chunk_index_for_slot(
        self,
        slot_index: int,
        settings: InterleaveSettings,
        source_id: SourceId | None = None,
    ) -> int:
        selected = source_id or self.source_for_slot(slot_index, settings)
        if isinstance(settings, RegionInsert):
            if selected == "B":
                return settings.b_source_slot + slot_index - settings.output_slot
            return slot_index
        if selected == "A":
            return slot_index
        return sum(
            self.source_for_slot(index, settings) == "B"
            for index in range(slot_index)
        )

    def _chunk_float(self, source_id: SourceId, chunk_index: int) -> np.ndarray:
        source = self.source_a if source_id == "A" else self.source_b
        actual_index = chunk_index % self.source_chunk_count(source_id)
        start = actual_index * self.slot_frames
        result = np.zeros((self.slot_frames, 1), dtype=np.float32)
        copied = min(self.slot_frames, max(0, source.frames - start))
        if copied:
            result[:copied] = source.samples[start : start + copied]
        return result

    def preview_chunk(
        self, source_id: SourceId, chunk_index: int, output_slot_index: int
    ) -> np.ndarray:
        if output_slot_index < 0 or output_slot_index >= self.slot_count:
            raise IndexError("output_slot_index is outside the output timeline")
        return self._chunk_float(source_id, chunk_index)

    def source_region(
        self,
        source_id: SourceId,
        first_chunk: int,
        chunk_count: int,
        silence_after_end: bool = False,
    ) -> np.ndarray:
        if first_chunk < 0 or chunk_count < 1:
            raise ValueError("invalid source region")
        if (
            not silence_after_end
            and first_chunk + chunk_count > self.source_chunk_count(source_id)
        ):
            raise ValueError("source region extends beyond the available chunks")
        return np.concatenate(
            [
                np.zeros((self.slot_frames, 1), dtype=np.float32)
                if first_chunk + i >= self.source_chunk_count(source_id)
                else self._chunk_float(source_id, first_chunk + i)
                for i in range(chunk_count)
            ]
        )

    def _a_pcm(self) -> np.ndarray:
        padded = np.zeros(self.total_frames, dtype=np.float32)
        padded[: self.source_a.frames] = self.source_a.samples[:, 0]
        return _float_to_pcm16(padded)

    def _b_chunk_pcm(self, chunk_index: int) -> np.ndarray:
        return _float_to_pcm16(self._chunk_float("B", chunk_index)[:, 0])

    def render_symbols(
        self,
        settings: InterleaveSettings,
        b_encoder_mode: BEncoderMode = "one_stream",
        progress: ProgressCallback | None = None,
        cancel_event: Event | None = None,
    ) -> AcelpSymbols:
        if b_encoder_mode not in ("one_stream", "restart_each_chunk"):
            raise ValueError("unknown B encoder mode")
        assert self.codec is not None
        if cancel_event is not None and cancel_event.is_set():
            raise RenderingCancelled("Rendering was cancelled.")

        a_symbols = self.codec.encode(self._a_pcm())
        if cancel_event is not None and cancel_event.is_set():
            raise RenderingCancelled("Rendering was cancelled.")
        mixed = a_symbols.packed.copy()
        active = [
            slot for slot in range(self.slot_count)
            if self.source_for_slot(slot, settings) == "B"
        ]
        if progress is not None:
            progress(0.35 if active else 1.0)
        if not active:
            return AcelpSymbols(mixed)

        b_chunks = []
        for slot in active:
            chunk_index = self.source_chunk_index_for_slot(slot, settings, "B")
            silent_b_tail = (
                isinstance(settings, RegionInsert)
                and settings.silence_after_b_end
                and chunk_index >= self.source_chunk_count("B")
            )
            b_chunks.append(
                np.zeros(self.slot_frames, dtype=np.int16)
                if silent_b_tail
                else self._b_chunk_pcm(chunk_index)
            )
        if cancel_event is not None and cancel_event.is_set():
            raise RenderingCancelled("Rendering was cancelled.")
        if b_encoder_mode == "one_stream":
            encoded = self.codec.encode(np.concatenate(b_chunks)).packed
            encoded_chunks = np.split(encoded, len(active))
        else:
            encoded_chunks = []
            for index, chunk in enumerate(b_chunks):
                if cancel_event is not None and cancel_event.is_set():
                    raise RenderingCancelled("Rendering was cancelled.")
                encoded_chunks.append(self.codec.encode(chunk).packed)
                if progress is not None:
                    progress(0.35 + 0.55 * (index + 1) / len(b_chunks))

        for index, (slot, encoded_chunk) in enumerate(zip(active, encoded_chunks)):
            first = slot * self.codec_frames_per_slot
            last = first + self.codec_frames_per_slot
            mixed[first:last] = encoded_chunk
            if progress is not None and b_encoder_mode == "one_stream":
                progress(0.35 + 0.55 * (index + 1) / len(active))
        if progress is not None:
            progress(1.0)
        return AcelpSymbols(mixed)

    def render(
        self,
        settings: InterleaveSettings,
        b_encoder_mode: BEncoderMode = "one_stream",
        progress: ProgressCallback | None = None,
        cancel_event: Event | None = None,
    ) -> np.ndarray:
        assert self.codec is not None
        symbols = self.render_symbols(
            settings, b_encoder_mode, progress, cancel_event
        )
        if cancel_event is not None and cancel_event.is_set():
            raise RenderingCancelled("Rendering was cancelled.")
        pcm = self.codec.decode(symbols)
        return (pcm.astype(np.float32) / 32768.0)[:, np.newaxis]


__all__ = [
    "ACELP_FRAME_BITS",
    "ACELP_FRAME_MS",
    "ACELP_FRAME_SAMPLES",
    "ACELP_MAX_CHUNK_MS",
    "ACELP_MIN_CHUNK_MS",
    "ACELP_SAMPLE_RATE",
    "AcelpEngine",
    "AcelpSymbols",
    "BEncoderMode",
    "ProcessingStage",
    "TetraAcelpCodec",
    "snap_acelp_chunk_ms",
]
