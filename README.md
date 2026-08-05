# Audio Interleaver

Audio Interleaver is a desktop tool for replacing chunks of source A with
chunks from source B. It accepts mono or stereo WAV files and exports 16-bit
PCM WAV audio.

Two processing stages are available:

- **Raw wave** replaces PCM samples.
- **ACELP symbols** encodes both sources with the fixed-point ETSI TETRA
  full-rate speech codec, replaces 137-bit vocoder frames, then decodes the
  result. It does not simulate TETRA channel coding or radio interleaving.

## Interleave modes

**Pattern Interleave** creates a repeating A/B pattern. Controls select the
starting source, first alternate-source chunk, B burst size, and number of
active B occurrences. Source A stays fixed on the output timeline; selected B
chunks advance in order.

**Region Insert** replaces a contiguous part of A with a selected region of B.
Controls set the B source position, region length, and output position. Enable
**Add silence after B ends** to extend the region past the end of B. The
remaining selected chunks contain silence instead of wrapped B audio.

The timeline shows both source waveforms, selected chunks, and playback
position. **Preview B region** plays the current Region Insert selection.

## Audio behavior

Raw chunk duration ranges from 50 to 2,000 ms. An optional 0–50 ms equal-power
crossfade is applied when the selected source changes. Short sources loop by
whole chunks; partial chunks are padded with silence.

ACELP audio is mono at 8 kHz and uses 30 ms frames. Chunk durations therefore
range from 30 to 1,980 ms in 30 ms steps. Typed values snap to the nearest legal
duration. Crossfading is disabled.

Source A is encoded as one complete stream. Only active B chunks are encoded:

- **One stream** preserves B encoder state across active chunks.
- **Restart every chunk** resets the B encoder for each chunk.

For an extended Region Insert, zero PCM is encoded for the silent B tail using
the selected encoder-state mode. Decoding always uses one continuous stream.

**Export WAV…** writes the rendered result. In ACELP mode, **Export ACELP
symbols…** also writes an ETSI simulation-format `.spe` file. Each 30 ms frame
contains 138 little-endian 16-bit words: one bad-frame indicator followed by
137 bit values.

## Requirements

- Python 3.11 or newer
- A supported C compiler
- An audio output device
- WAV input files

Linux users may need Qt, PortAudio, and libsndfile packages. On Debian/Ubuntu:

```bash
sudo apt update
sudo apt install python3-venv libportaudio2 libsndfile1 libegl1
```

## Install

Create the repository-local virtual environment on macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

On Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Use `python -m pip install -e .` to omit test dependencies. Installation also
builds the native TETRA ACELP extension.

## Run

```bash
python -m audio_interleaver
```

Load both sources, choose a processing stage and interleave mode, configure the
chunk settings, then play or export the result. Source preview buttons play the
original WAV files. ACELP settings are locked while a result is being prepared
or played.

The window displays the current Git commit. Packaged builds can set it with the
`AUDIO_INTERLEAVER_COMMIT` environment variable.

## Test

```bash
python -m pytest
```

GitHub Actions runs the suite on macOS, Windows, and Linux. Codec source and
provenance are documented in `vendor/tetra_codec/README.md`; redistribution is
subject to the applicable ETSI terms.
