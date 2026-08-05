# Audio Interleaver

Audio Interleaver is a desktop application that divides two WAV files into
independent, configurable chunks and places those chunks into a repeating
output pattern. B occurrences are revealed from left to right, so adding more
never moves an existing B chunk toward the beginning of the timeline.

The **Processing Stage** switch selects either the original **Raw wave** path
or an **ACELP symbols** path based on the fixed-point ETSI TETRA full-rate
speech codec. Raw mode replaces PCM samples. ACELP mode downmixes each source
to mono, resamples it to 8 kHz, encodes 30 ms speech frames, replaces complete
groups of 137 codec bits, and continuously decodes the mixed symbol stream.
It does not apply TETRA channel coding or radio-channel interleaving.

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
replacement begins. By default, the B source window always remains intact and
only placement beyond the end of the output timeline clips the inserted region.
Enable **Silence after B ends** to extend the region through the remaining
output timeline: B plays once from the selected source chunk, then zero-valued
chunks replace A instead of B looping.

The preview places source A's waveform above the interleave lanes and source
B's waveform below them. Source A remains visible in gray beneath slots
replaced by B. Source B appears only in active B slots, avoiding ghost waveform
chunks in intervening A slots. Chunk boundaries, waveforms, and the playback
marker share the same horizontal alignment. B sample zero aligns with the
first inserted B chunk. The application header displays the current Git commit
hash for preview and bug report identification. Packaged builds can provide it
through the `AUDIO_INTERLEAVER_COMMIT` environment variable.

Source A is the fixed output reference: a B insertion replaces the A material
at that position rather than pausing A. Source B advances continuously through
only its selected slots, so `A B A B A B` uses chunks `A0, B0, A2, B1, A4,
B2`. When a boundary switches sources, an equal-power transition crossfades
the previous chunk's tail into the next chunk's beginning without changing
chunk order or output duration. The crossfade is adjustable from 0 to 50 ms
and defaults to 0 ms. When the sources have different lengths, the shorter
source loops and the timeline covers the longer source. A partial final output
chunk is retained and padded with silence. Chunk duration is adjustable from
50 to 2000 ms. The chunk-duration and crossfade controls are arranged side by
side to keep the interface compact; both support exact numerical entry as well
as slider adjustment. If an entire WAV is shorter than one chunk, every
occurrence restarts at source sample zero, plays the file once, and
silence-pads the rest of the chunk.

## Requirements

- Python 3.11 or newer
- A C compiler supported by Python/setuptools when installing from source
- A working audio output device
- WAV input files with mono or stereo audio

The application supports WAV input and 16-bit PCM WAV output. Sources with
different sample rates are resampled to the higher rate; mono is promoted to
stereo when paired with a stereo source.

In ACELP mode, source A is the complete fixed output timeline. Its final chunk
is padded with silence when necessary. Only B chunks active in the current
pattern or inserted region are encoded. **One stream** carries encoder state
continuously across active B chunks; **Restart encoder every B chunk** starts
each selected B chunk from fresh state. The decoder remains continuous, so
state discontinuities introduced by symbol replacement remain audible.
For an extended Region Insert, silent tail chunks are encoded from zero PCM
using the selected B encoder-state mode before their symbols replace source A.

ACELP chunk durations are legal multiples of the codec's 30 ms speech frame.
The slider provides values from 30 to 1,980 ms; typed values snap to the
nearest legal duration, with exact ties rounded up. Raw and ACELP modes
remember durations independently. PCM crossfading is disabled in ACELP mode.

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
python -m pip install -e ".[dev]"
```

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

For an application-only installation, use `python -m pip install -e .`. This
also builds the native TETRA ACELP extension. Dependency versions remain pinned
in the requirements files and `pyproject.toml`.

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
current pass. Each source card's **Play preview** button auditions its complete
loaded file and changes to **Stop preview** during playback. In Region Insert
mode, **Preview B region** auditions exactly the selected whole-chunk source B
window, including final-chunk silence padding. Use **Export WAV…** to render a
complete file using a snapshot of the settings.

In ACELP mode, pressing **Play** first prepares a fixed symbol-stage result.
Settings and source loading remain locked until playback stops, as noted by
the on-screen banner. Source-card previews continue to play the original WAV
files. **Export WAV…** writes the decoded 8 kHz mono result, while **Export
ACELP symbols…** writes the mixed stream as an ETSI simulation-format `.spe`
file. Each 30 ms frame contains 138 little-endian 16-bit words: a zero
bad-frame indicator followed by 137 words containing zero or one.

The vendored codec wrapper and provenance are documented in
`vendor/tetra_codec/README.md`. The codec algorithm and reference source are
copyrighted by ETSI and should be redistributed only under the applicable
ETSI terms.

## Test

```bash
python -m pytest
```

The test suite covers deterministic occurrence patterns, B burst sizes, start
and insertion controls, sequential source chunks, looping, partial final slots,
format normalization, transition smoothing, live updates, WAV I/O, and a
headless UI smoke test. ACELP coverage includes native encoder state, frame
packing, `.spe` serialization, source-A anchoring, active-only B encoding,
per-chunk resets, downmixing, resampling, duration snapping, and UI lockout.

The included GitHub Actions workflow runs this suite on macOS, Windows, and
Linux with the oldest and newest supported Python versions.
