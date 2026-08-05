# TETRA ACELP reference codec

The files in `source/` and `include/` are vendored from the public
[`outerplane/tetra-codec`](https://github.com/outerplane/tetra-codec) wrapper at
commit `21d884064478d63306ec654378e666ae41503d00`. That project adapts the
fixed-point speech codec published with ETSI EN 300 395-2 V1.3.1 (2005-01)
into a re-entrant C library.

The codec algorithm and reference implementation are copyrighted by ETSI.
Their inclusion here follows the distribution terms under which ETSI publishes
the electronic attachment to EN 300 395-2. Keep this provenance notice with
redistributions of the source.

`python_module.c` is the Audio Interleaver CPython adapter. It accepts complete
streams of 240-sample PCM frames and returns 18-byte packed representations of
the 137 codec bits. Each adapter call starts with fresh encoder or decoder
state; state remains continuous between all frames in that call.
