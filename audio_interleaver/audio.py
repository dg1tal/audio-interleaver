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


@dataclass(frozen=True, slots=True)
class RegionInsert:
    """A contiguous window from source B placed into the A output timeline."""

    b_source_slot: int = 0
    output_slot: int = 0
    length_slots: int = 1
    silence_after_b_end: bool = False

    def __post_init__(self) -> None:
        if self.b_source_slot < 0:
            raise ValueError("b_source_slot must be non-negative")
        if self.output_slot < 0:
            raise ValueError("output_slot must be non-negative")
        if self.length_slots < 1:
            raise ValueError("length_slots must be positive")


InterleaveSettings = InterleavePattern | RegionInsert


def occurrence_capacity(slot_count: int, pattern: InterleavePattern) -> int:
    """Return the number of meaningful optional B occurrences."""

    if slot_count <= 0:
        return 0
    cycle_length = pattern.b_chunks_per_occurrence + 1
    reference_first_b = 1 if pattern.starts_with == "A" else 2
    return max(
        0,
        math.ceil((slot_count - reference_first_b) / cycle_length),
    )


def select_source(
    slot_index: int,
    slot_count: int,
    settings: InterleaveSettings,
) -> SourceId:
    """Select a source for either supported interleave mode."""

    if slot_count <= 0:
        raise ValueError("slot_count must be positive")
    if slot_index < 0 or slot_index >= slot_count:
        raise IndexError("slot_index is outside the output timeline")

    if isinstance(settings, RegionInsert):
        region_end = min(slot_count, settings.output_slot + settings.length_slots)
        if settings.output_slot <= slot_index < region_end:
            return "B"
        return "A"

    pattern = settings
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

    occurrence_count = occurrence_capacity(slot_count, pattern)
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
    smoothing_ms: float = 0.0

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
    def content_frames(self) -> int:
        return max(self.source_a.frames, self.source_b.frames)

    @property
    def total_frames(self) -> int:
        return self.slot_count * self.slot_frames

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
        return math.ceil(self.content_frames / self.slot_frames)

    def source_for_slot(
        self, slot_index: int, settings: InterleaveSettings
    ) -> SourceId:
        return select_source(slot_index, self.slot_count, settings)

    def source_chunk_count(self, source_id: SourceId) -> int:
        return math.ceil(self._source(source_id).frames / self.slot_frames)

    def source_chunk_index_for_slot(
        self,
        slot_index: int,
        settings: InterleaveSettings,
        source_id: SourceId | None = None,
    ) -> int:
        selected_source = source_id or self.source_for_slot(slot_index, settings)
        if isinstance(settings, RegionInsert):
            if selected_source == "B":
                return settings.b_source_slot + (slot_index - settings.output_slot)
            return slot_index
        if selected_source == "A":
            # A is the fixed output timeline. Inserting B replaces an A chunk;
            # it must not pause A's source position.
            return slot_index
        return sum(
            self.source_for_slot(index, settings) == selected_source
            for index in range(slot_index)
        )

    def _source(self, source_id: SourceId) -> LoadedAudio:
        return self.source_a if source_id == "A" else self.source_b

    def _chunk(self, source_id: SourceId, chunk_index: int, length: int) -> np.ndarray:
        """Read an independent source chunk, padding a sub-chunk file."""

        source = self._source(source_id).samples
        source_chunk_count = self.source_chunk_count(source_id)
        source_chunk_index = chunk_index % source_chunk_count
        source_start = source_chunk_index * self.slot_frames
        result = np.zeros((length, self.channels), dtype=np.float32)
        copied = min(length, len(source) - source_start)
        result[:copied] = source[source_start : source_start + copied]
        return result

    def preview_chunk(
        self,
        source_id: SourceId,
        chunk_index: int,
        output_slot_index: int,
    ) -> np.ndarray:
        """Return a source chunk as it appears in an output preview slot."""

        if output_slot_index < 0 or output_slot_index >= self.slot_count:
            raise IndexError("output_slot_index is outside the output timeline")
        if chunk_index < 0:
            raise ValueError("chunk_index must be non-negative")
        result = self._chunk(source_id, chunk_index, self.slot_frames)
        output_start = output_slot_index * self.slot_frames
        content_end = min(
            self.slot_frames,
            max(0, self.content_frames - output_start),
        )
        if content_end < self.slot_frames:
            result[content_end:] = 0.0
        return result

    def source_region(
        self,
        source_id: SourceId,
        first_chunk: int,
        chunk_count: int,
        silence_after_end: bool = False,
    ) -> np.ndarray:
        """Return a whole-chunk source selection, padding its final chunk."""

        if first_chunk < 0:
            raise ValueError("first_chunk must be non-negative")
        if chunk_count < 1:
            raise ValueError("chunk_count must be positive")
        if (
            not silence_after_end
            and first_chunk + chunk_count > self.source_chunk_count(source_id)
        ):
            raise ValueError("source region extends beyond the available chunks")
        return np.concatenate(
            [
                np.zeros((self.slot_frames, self.channels), dtype=np.float32)
                if first_chunk + offset >= self.source_chunk_count(source_id)
                else self._chunk(
                    source_id, first_chunk + offset, self.slot_frames
                )
                for offset in range(chunk_count)
            ]
        )

    def render_slot(
        self,
        slot_index: int,
        settings: InterleaveSettings,
        source_chunk_index: int | None = None,
        previous_source: SourceId | None = None,
        previous_chunk: np.ndarray | None = None,
    ) -> tuple[np.ndarray, SourceId]:
        """Render the selected source's next chunk with optional switch smoothing."""

        if slot_index < 0 or slot_index >= self.slot_count:
            raise IndexError("slot_index is outside the output timeline")
        start = slot_index * self.slot_frames
        length = min(self.slot_frames, self.total_frames - start)
        source_id = self.source_for_slot(slot_index, settings)
        if source_chunk_index is None:
            source_chunk_index = self.source_chunk_index_for_slot(
                slot_index, settings, source_id
            )
        if source_chunk_index < 0:
            raise ValueError("source_chunk_index must be non-negative")
        silent_b_tail = (
            source_id == "B"
            and isinstance(settings, RegionInsert)
            and settings.silence_after_b_end
            and source_chunk_index >= self.source_chunk_count("B")
        )
        result = (
            np.zeros((self.slot_frames, self.channels), dtype=np.float32)
            if silent_b_tail
            else self.preview_chunk(source_id, source_chunk_index, slot_index)
        )

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
        settings: InterleaveSettings,
        progress: ProgressCallback | None = None,
        cancel_event: Event | None = None,
    ) -> np.ndarray:
        """Render the complete output using a fixed settings snapshot."""

        output = np.empty((self.total_frames, self.channels), dtype=np.float32)
        previous_source: SourceId | None = None
        previous_chunk: np.ndarray | None = None
        for slot_index in range(self.slot_count):
            if cancel_event is not None and cancel_event.is_set():
                raise RenderingCancelled("Rendering was cancelled.")
            source_id = self.source_for_slot(slot_index, settings)
            source_chunk_index = self.source_chunk_index_for_slot(
                slot_index, settings, source_id
            )
            slot, previous_source = self.render_slot(
                slot_index,
                settings,
                source_chunk_index,
                previous_source,
                previous_chunk,
            )
            previous_chunk = slot
            start = slot_index * self.slot_frames
            output[start : start + len(slot)] = slot
            if progress is not None:
                progress((start + len(slot)) / self.total_frames)
        return output
