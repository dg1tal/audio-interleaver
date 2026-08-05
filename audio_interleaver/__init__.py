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
from .acelp import (
    AcelpEngine,
    AcelpSymbols,
    TetraAcelpCodec,
    snap_acelp_chunk_ms,
)

__all__ = [
    "AudioEngine",
    "AudioError",
    "AcelpEngine",
    "AcelpSymbols",
    "InterleavePattern",
    "InterleaveSettings",
    "LoadedAudio",
    "RegionInsert",
    "TetraAcelpCodec",
    "load_wav",
    "occurrence_capacity",
    "snap_acelp_chunk_ms",
    "write_wav",
]
__version__ = "0.1.0"
