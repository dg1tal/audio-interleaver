"""Pattern-based WAV interleaver."""

from .audio import (
    AudioEngine,
    AudioError,
    InterleavePattern,
    LoadedAudio,
    load_wav,
    write_wav,
)

__all__ = [
    "AudioEngine",
    "AudioError",
    "InterleavePattern",
    "LoadedAudio",
    "load_wav",
    "write_wav",
]
__version__ = "0.1.0"
