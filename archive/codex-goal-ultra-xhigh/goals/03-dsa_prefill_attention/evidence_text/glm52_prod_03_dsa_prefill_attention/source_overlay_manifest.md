# Source-overlay manifest

Date: 2026-07-22

## Measured candidate

Both `metadata_before.json` and `metadata_after.json` in the exact raw-pool
bundle record:

- candidate tree SHA256:
  `8c0eeeb8fb0a61df6a1b6b0d7a6ec7bb6e3ca2717a20c8d9cb1b3d95b83704`;
- `candidate.py` SHA256:
  `f7dd87ca9859530613c0be880291acfbedab05649e3c2151d47a207b12569837`.

After the bundle ended at 15:07 UTC, only the candidate README was expanded at
15:16 UTC with source/build provenance. The Python candidate and both overlay
headers predate the bundle and are unchanged. The documented current tree hash
is therefore
`b18a347300b4ea724bc788e6eea520571d1c8242eff8c9d50c02ef795895f43d`;
the changed tree hash is documentation-only, while the measured executable
candidate hash remains exact.

Current executable-source hashes:

- `candidate.py`: `f7dd87ca9859530613c0be880291acfbedab05649e3c2151d47a207b12569837`;
- relocated/modified `fmhaKernels.cuh`:
  `11c1b51d1725ab321cdb89badcb8155c78ba2b84577034933e2757e7ba234b62`;
- relocated `fmhaRunner.cuh`:
  `6921d6ecd653615035e753826b74d472c02828e56590934e4297775cf6f3c71f`.

## Upstream base

- repository: `https://github.com/flashinfer-ai/flashinfer`;
- annotated tag `v0.6.12` object:
  `dd86b5bd7bac37cc9a0cc537c29c15a61a4aca99`;
- peeled source commit:
  `d768c14e7cf5dd5df45a8a1de78ae815879f108a`;
- stock `fmhaKernels.cuh`:
  `728a29df0f717990ed8b7987600c84e2f2c836200c882b733bdf79e29416d0b0`;
- stock `fmhaRunner.cuh`:
  `5198b71149a96ebec048483029c7c15a92446cac0e7b0292554afa79daff6141`.

The stock header hashes match both the installed FlashInfer 0.6.12 package and
the peeled upstream commit. Overlay differences are mechanical include-path
relocation plus the one exact-shape Q32/Q16 selector predicate. The JIT library
paths are persisted separately in `trace-q32.json` and `trace-q16.json`; no
installed file was overwritten.

The goal-scoped overlay and all executable benchmark/profile drivers are
committed in the isolated Kernel-Harness worktree at
`d00181769a6041dc3803de056a1f70cafdb9d483`. SGLang imports none of them; its
separate rollback worktree remains clean at `5a444f66cf5764d2d76003a3a4c4631af152a253`.
