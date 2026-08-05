"""Audio loading, normalization, chunk selection, and offline rendering."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from threading import Event
from typing import Callable, Literal

import numpy as np
import soundfile as sf
import soxr

SourceId = Literal["A", "B"]
ProgressCallback = Callable[[float], None]


class AudioError(RuntimeError):
    """Raised when an input cannot be used by the interleaver."""


class RenderingCancelled(RuntimeError):
    """Raised when an offline render is cancelled."""


@dataclass(frozen=True, slots=True)
class LoadedAudio:
    """Decoded floating-point audio with frames on axis 0."""

    samples: np.ndarray
    sample_rate: int
    path: Path | None = None

    def __post_init__(self) -> None:
        samples = np.asarray(self.samples, dtype=np.float32)
        if samples.ndim == 1:
            samples = samples[:, np.newaxis]
        if samples.ndim != 2:
            raise AudioError("Audio samples must have frames and channels.")
        if len(samples) == 0:
            raise AudioError("The WAV file contains no audio frames.")
        if samples.shape[1] not in (1, 2):
            raise AudioError("Only mono and stereo WAV files are supported.")
        if self.sample_rate <= 0:
            raise AudioError("The WAV file has an invalid sample rate.")
        if not np.isfinite(samples).all():
            raise AudioError("The WAV file contains invalid sample values.")
        object.__setattr__(self, "samples", np.ascontiguousarray(samples))

    @property
    def channels(self) -> int:
        return int(self.samples.shape[1])

    @property
    def frames(self) -> int:
        return int(self.samples.shape[0])

    @property
    def duration(self) -> float:
        return self.frames / self.sample_rate


def load_wav(path: str | Path) -> LoadedAudio:
    """Load a mono or stereo WAV as float32 samples."""

    wav_path = Path(path)
    try:
        info = sf.info(wav_path)
        if not info.format.upper().startswith("WAV"):
            raise AudioError(f"{wav_path.name} is not a WAV file.")
        samples, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
    except AudioError:
        raise
    except (OSError, RuntimeError, sf.SoundFileError) as exc:
        raise AudioError(f"Could not read {wav_path.name}: {exc}") from exc
    return LoadedAudio(samples=samples, sample_rate=int(sample_rate), path=wav_path)


def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int) -> None:
    """Write a 16-bit PCM WAV, clipping only values outside the valid range."""

    output_path = Path(path)
    try:
        safe_samples = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
        sf.write(output_path, safe_samples, sample_rate, format="WAV", subtype="PCM_16")
    except (OSError, RuntimeError, sf.SoundFileError) as exc:
        raise AudioError(f"Could not write {output_path.name}: {exc}") from exc


@dataclass(frozen=True, slots=True)
class InterleavePattern:
    """Settings for a repeating, left-to-right B-chunk insertion pattern."""

    fill: float = 0.5
    starts_with: SourceId = "A"
    first_alternate_slot: int = 1
    b_chunks_per_occurrence: int = 1

    def __post_init__(self) -> None:
        if self.starts_with not in ("A", "B"):
            raise ValueError("starts_with must be A or B")
        if self.first_alternate_slot < 1:
            raise ValueError(
                "first_alternate_slot must leave at least one leading slot"
            )
        if self.b_chunks_per_occurrence < 1:
            raise ValueError("b_chunks_per_occurrence must be positive")


def select_source(
    slot_index: int,
    slot_count: int,
    pattern: InterleavePattern,
) -> SourceId:
    """Select a source from the active left-to-right occurrence pattern."""

    if slot_count <= 0:
        raise ValueError("slot_count must be positive")
    if slot_index < 0 or slot_index >= slot_count:
        raise IndexError("slot_index is outside the output timeline")

    first_alternate = min(pattern.first_alternate_slot, slot_count)
    if slot_index < first_alternate:
        return pattern.starts_with

    burst = pattern.b_chunks_per_occurrence
    cycle_length = burst + 1
    if pattern.starts_with == "A":
        first_b = first_alternate
    else:
        if slot_index == first_alternate:
            return "A"
        first_b = first_alternate + 1

    if first_b >= slot_count or slot_index < first_b:
        return "A"

    occurrence_count = math.ceil((slot_count - first_b) / cycle_length)
    fill = float(np.clip(pattern.fill, 0.0, 1.0))
    active_occurrences = min(
        occurrence_count,
        math.floor(fill * occurrence_count + 0.5),
    )
    relative_slot = slot_index - first_b
    occurrence_index, position_in_cycle = divmod(relative_slot, cycle_length)
    if occurrence_index < active_occurrences and position_in_cycle < burst:
        return "B"
    return "A"


def _match_channels(samples: np.ndarray, channels: int) -> np.ndarray:
    if samples.shape[1] == channels:
        return samples
    if samples.shape[1] == 1 and channels == 2:
        return np.repeat(samples, 2, axis=1)
    raise AudioError("The channel layouts cannot be normalized.")


def _resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return samples
    converted = soxr.resample(samples, source_rate, target_rate, quality="HQ")
    return np.ascontiguousarray(converted, dtype=np.float32)


@dataclass(slots=True)
class AudioEngine:
    """Normalized pair of sources whose chunks are placed into output slots."""

    source_a: LoadedAudio
    source_b: LoadedAudio
    slot_ms: float = 360.0
    smoothing_ms: float = 5.0

    def __post_init__(self) -> None:
        if self.slot_ms <= 0:
            raise ValueError("slot_ms must be positive")
        if self.smoothing_ms < 0 or self.smoothing_ms >= self.slot_ms:
            raise ValueError("smoothing_ms must be non-negative and shorter than a slot")

        target_rate = max(self.source_a.sample_rate, self.source_b.sample_rate)
        target_channels = max(self.source_a.channels, self.source_b.channels)
        self.source_a = self._normalize(self.source_a, target_rate, target_channels)
        self.source_b = self._normalize(self.source_b, target_rate, target_channels)

    @staticmethod
    def _normalize(audio: LoadedAudio, sample_rate: int, channels: int) -> LoadedAudio:
        samples = _match_channels(audio.samples, channels)
        samples = _resample(samples, audio.sample_rate, sample_rate)
        return LoadedAudio(samples=samples, sample_rate=sample_rate, path=audio.path)

    @property
    def sample_rate(self) -> int:
        return self.source_a.sample_rate

    @property
    def channels(self) -> int:
        return self.source_a.channels

    @property
    def total_frames(self) -> int:
        return max(self.source_a.frames, self.source_b.frames)

    @property
    def duration(self) -> float:
        return self.total_frames / self.sample_rate

    @property
    def slot_frames(self) -> int:
        return max(1, round(self.sample_rate * self.slot_ms / 1000.0))

    @property
    def smoothing_frames(self) -> int:
        return max(0, round(self.sample_rate * self.smoothing_ms / 1000.0))

    @property
    def slot_count(self) -> int:
        return math.ceil(self.total_frames / self.slot_frames)

    def source_for_slot(
        self, slot_index: int, pattern: InterleavePattern
    ) -> SourceId:
        return select_source(slot_index, self.slot_count, pattern)

    def _source(self, source_id: SourceId) -> LoadedAudio:
        return self.source_a if source_id == "A" else self.source_b

    def _chunk(self, source_id: SourceId, chunk_index: int, length: int) -> np.ndarray:
        """Read the next independent chunk from a source, looping if necessary."""

        source = self._source(source_id).samples
        result = np.empty((length, self.channels), dtype=np.float32)
        written = 0
        position = (chunk_index * self.slot_frames) % len(source)
        while written < length:
            available = min(length - written, len(source) - position)
            result[written : written + available] = source[position : position + available]
            written += available
            position = 0
        return result

    def render_slot(
        self,
        slot_index: int,
        pattern: InterleavePattern,
        source_chunk_index: int | None = None,
        previous_source: SourceId | None = None,
        previous_chunk: np.ndarray | None = None,
    ) -> tuple[np.ndarray, SourceId]:
        """Render the selected source's next chunk with optional switch smoothing."""

        if slot_index < 0 or slot_index >= self.slot_count:
            raise IndexError("slot_index is outside the output timeline")
        start = slot_index * self.slot_frames
        length = min(self.slot_frames, self.total_frames - start)
        source_id = self.source_for_slot(slot_index, pattern)
        if source_chunk_index is None:
            source_chunk_index = sum(
                self.source_for_slot(index, pattern) == source_id
                for index in range(slot_index)
            )
        if source_chunk_index < 0:
            raise ValueError("source_chunk_index must be non-negative")
        result = self._chunk(source_id, source_chunk_index, length)

        fade_length = min(
            self.smoothing_frames,
            length,
            len(previous_chunk) if previous_chunk is not None else 0,
        )
        if (
            previous_source is not None
            and previous_source != source_id
            and previous_chunk is not None
            and fade_length >= 2
        ):
            outgoing = previous_chunk[-fade_length:]
            angles = np.linspace(0.0, math.pi / 2.0, fade_length, dtype=np.float32)
            outgoing_gain = np.cos(angles)[:, np.newaxis]
            incoming_gain = np.sin(angles)[:, np.newaxis]
            result[:fade_length] = (
                outgoing * outgoing_gain + result[:fade_length] * incoming_gain
            )
            np.clip(result[:fade_length], -1.0, 1.0, out=result[:fade_length])
        return result, source_id

    def render(
        self,
        pattern: InterleavePattern,
        progress: ProgressCallback | None = None,
        cancel_event: Event | None = None,
    ) -> np.ndarray:
        """Render the complete output using a fixed pattern snapshot."""

        output = np.empty((self.total_frames, self.channels), dtype=np.float32)
        source_chunk_indices: dict[SourceId, int] = {"A": 0, "B": 0}
        previous_source: SourceId | None = None
        previous_chunk: np.ndarray | None = None
        for slot_index in range(self.slot_count):
            if cancel_event is not None and cancel_event.is_set():
                raise RenderingCancelled("Rendering was cancelled.")
            source_id = self.source_for_slot(slot_index, pattern)
            slot, previous_source = self.render_slot(
                slot_index,
                pattern,
                source_chunk_indices[source_id],
                previous_source,
                previous_chunk,
            )
            source_chunk_indices[source_id] += 1
            previous_chunk = slot
            start = slot_index * self.slot_frames
            output[start : start + len(slot)] = slot
            if progress is not None:
                progress((start + len(slot)) / self.total_frames)
        return output
