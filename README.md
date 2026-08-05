# Audio Interleaver

Audio Interleaver is a desktop application that divides two WAV files into
independent, configurable chunks and places those chunks into output slots. The
selection crossfader controls how many chunks come from each source:

- **A only:** every slot uses source A.
- **Center:** slots alternate B, A, B, A, inserting source B from the start.
- **B only:** every slot uses source B.
- **Between those points:** A and B slots are distributed evenly in the chosen
  proportion.

This is a chunk-selection crossfader, not a conventional volume mixer. Each
source advances to its own next chunk whenever it is selected; the files are
not aligned on a shared timeline. When a boundary switches sources, an
equal-power transition crossfades the previous chunk's tail into the next
chunk's beginning without changing chunk order or output duration. The
crossfade is adjustable from 0 to 50 ms (5 ms by default). When the sources
have different lengths, the shorter source loops and the result ends with the
longer source. Chunk duration is adjustable from 50 to 2000 ms.

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

Load source A and source B, choose the chunk and crossfade durations,
move the selection crossfader, and press **Play**. Selection changes made during
playback take effect when the next chunk starts. Changing either duration stops
playback and rebuilds the timeline. Use the **Loop** checkbox to restart the
complete result whenever playback reaches the end. Turn it off during playback
to stop looping after the current pass. Use **Export WAV…** to render a complete
file using a snapshot of the current settings.

## Test

```bash
python -m pytest
```

The test suite covers deterministic selection, sequential source chunks,
looping, partial final slots, format normalization, transition smoothing, live slot
updates, WAV I/O, and a headless UI smoke test.

The included GitHub Actions workflow runs this suite on macOS, Windows, and
Linux with the oldest and newest supported Python versions.
