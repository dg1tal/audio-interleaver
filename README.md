# Audio Interleaver

Audio Interleaver is a desktop application that divides two WAV files into
independent, configurable chunks and places those chunks into a repeating
output pattern. B occurrences are revealed from left to right, so adding more
never moves an existing B chunk toward the beginning of the timeline.

The pattern controls choose whether the output starts with A or B, where the
first alternate-source chunk occurs, and how many consecutive B chunks each
occurrence contains. One B chunk per occurrence produces `A B A B A B`; two
produce `A B B A B B`; three produce `A B B B A B B B`. The occurrence-fill
slider progressively enables those complete B groups from left to right and
has exactly one meaningful step per available occurrence.

The mode selector also provides **Region Insert**. In this mode, source A forms
the baseline output timeline and a contiguous window from source B replaces a
contiguous range of A chunks. Independent snapping controls select the first B
source chunk, the region length in whole chunks, and the output chunk where the
replacement begins. The B source window always remains intact; only placement
beyond the end of the output timeline clips the inserted region.

The preview places source A's waveform above the interleave lanes and source
B's waveform below them. Waveform portions used by the current pattern retain
their source color; inactive portions remain visible in gray. Chunk boundaries,
waveforms, and the playback marker share the same horizontal alignment. A
source waveform begins at that source's first selected slot, so B sample zero
aligns with the first inserted B chunk. The application header displays the
current Git commit hash for preview and bug report identification. Packaged
builds can provide it through the `AUDIO_INTERLEAVER_COMMIT` environment
variable.

Each source advances to its own next chunk whenever it is selected; the files
are not aligned on a shared timeline. When a boundary switches sources, an
equal-power transition crossfades the previous chunk's tail into the next
chunk's beginning without changing chunk order or output duration. The
crossfade is adjustable from 0 to 50 ms and defaults to 0 ms. When the sources
have different lengths, the shorter source loops and the timeline covers the
longer source. A partial final output chunk is retained and padded with silence.
Chunk duration is adjustable from 50 to 2000 ms. The chunk-duration and
crossfade controls are arranged side by side to keep the interface compact;
both support exact numerical entry as well as slider adjustment. If an entire
WAV is shorter than one chunk, every occurrence restarts at source sample zero,
plays the file once, and silence-pads the rest of the chunk.

## Requirements

- Python 3.11 or newer
- A working audio output device
- WAV input files with mono or stereo audio

The application supports WAV input and 16-bit PCM WAV output. Sources with
different sample rates are resampled to the higher rate; mono is promoted to
stereo when paired with a stereo source.

### Linux system packages

Linux users may need the system packages for Qt, PortAudio, and libsndfile. On
Debian/Ubuntu, install them before the Python dependencies:

```bash
sudo apt update
sudo apt install python3-venv libportaudio2 libsndfile1 libegl1
```

The macOS and Windows Python wheels normally include everything else needed.

## Set up the repository-local environment

The `.venv` directory is intentionally ignored by Git.

### macOS and Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

For an application-only installation, use `requirements.txt` instead of
`requirements-dev.txt`. The exact dependency versions are pinned in both the
requirements files and `pyproject.toml`.

## Run

With the virtual environment activated:

```bash
python -m audio_interleaver
```

You can also install the project command into the virtual environment:

```bash
python -m pip install -e .
audio-interleaver
```

Load source A and source B, choose **Pattern Interleave** or **Region Insert**,
configure that mode's controls and the chunk/crossfade durations, then press
**Play**. Interleave changes made during playback take effect when the next
chunk starts. Changing either duration stops playback and rebuilds the
timeline. Use the **Loop** checkbox to restart the complete result whenever
playback reaches the end. Turn it off during playback to stop looping after the
current pass. Use **Export WAV…** to render a complete file using a snapshot of
the settings.

## Test

```bash
python -m pytest
```

The test suite covers deterministic occurrence patterns, B burst sizes, start
and insertion controls, sequential source chunks, looping, partial final slots,
format normalization, transition smoothing, live updates, WAV I/O, and a
headless UI smoke test.

The included GitHub Actions workflow runs this suite on macOS, Windows, and
Linux with the oldest and newest supported Python versions.
