# TETRA ACELP reference codec

The files in `source/` and `include/` are vendored from the public
[`outerplane/tetra-codec`](https://github.com/outerplane/tetra-codec) wrapper at
commit `21d884064478d63306ec654378e666ae41503d00`. That project adapts the
fixed-point speech codec published with ETSI EN 300 395-2 V1.3.1 (2005-01)
into a re-entrant C library.

The codec algorithm and reference implementation are copyrighted by ETSI and
their respective copyright holders. The files under `source/` and `include/`
are not covered by the Audio Interleaver MIT License. Their use and
redistribution are subject to the applicable ETSI terms, including clause 9.2
of the ETSI IPR Policy. No patent licence is granted by the Audio Interleaver
project or implied by the ETSI software copyright licences. See the repository
root `THIRD_PARTY_NOTICES.md` for the full provenance and scope notice, and keep
both notices with redistributions of the source.

`python_module.c` is the original Audio Interleaver CPython adapter and is
licensed under the MIT License in the repository root `LICENSE`. It accepts
complete streams of 240-sample PCM frames and returns 18-byte packed
representations of the 137 codec bits. Each adapter call starts with fresh
encoder or decoder state; state remains continuous between all frames in that
call.
