"""Pattern-based WAV interleaver."""

from .audio import (
    AudioEngine,
    AudioError,
    InterleavePattern,
    InterleaveSettings,
    LoadedAudio,
    RegionInsert,
    load_wav,
    occurrence_capacity,
    write_wav,
)

__all__ = [
    "AudioEngine",
    "AudioError",
    "InterleavePattern",
    "InterleaveSettings",
    "LoadedAudio",
    "RegionInsert",
    "load_wav",
    "occurrence_capacity",
    "write_wav",
]
__version__ = "0.1.0"
