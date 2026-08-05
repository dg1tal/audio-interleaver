# Third-party notices

## ETSI TETRA speech codec

Audio Interleaver contains source derived from the TETRA speech codec described
by ETSI EN 300 395-2 V1.3.1 (2005-01), "Speech codec for full-rate traffic
channel; Part 2: TETRA codec."

The vendored files came from the public
[`outerplane/tetra-codec`](https://github.com/outerplane/tetra-codec) wrapper at
commit `21d884064478d63306ec654378e666ae41503d00`, which in turn adapts the ETSI
reference implementation.

The following files are third-party material and are **not** licensed under the
Audio Interleaver MIT License:

- `vendor/tetra_codec/include/tetra-codec.h`
- `vendor/tetra_codec/source/tetra-codec-impl.c`
- `vendor/tetra_codec/source/tetra-codec-impl.h`
- `vendor/tetra_codec/source/tetra-codec.c`

At the pinned upstream commit, the first three files are byte-identical to the
upstream copies. `source/tetra-codec.c` differs only by removal of two blank
lines. `vendor/tetra_codec/python_module.c` is the original Audio Interleaver
CPython adapter and is covered by the MIT License in `LICENSE`.

The codec algorithm and reference implementation are copyrighted by ETSI and
their respective copyright holders. Use and redistribution of the contributed
software and derivative works are subject to the applicable ETSI terms,
including clause 9.2 of the ETSI IPR Policy in the
[ETSI Directives](https://portal.etsi.org/Resources/ETSI-Directives).
Those terms include purpose limitations and an "AS IS" warranty disclaimer.
No patent licence is granted by the Audio Interleaver project, and ETSI clause
9.2.4 states that no patent licence is granted by implication, estoppel, or
otherwise under its software copyright licences.

Recipients are responsible for determining whether their use, modification,
or redistribution is permitted and whether any additional patent or other
licences are required.
