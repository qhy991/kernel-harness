# Stock versus `enable_pdl=False`

## Decision

Reject the PDL-off trial. It does not clear the 3% optimization gate:

- Nsight Systems mean over 25 launches changes by -0.0186% (nominally faster).
- Nsight Systems median changes by +0.0154% (nominally slower).
- The single-launch Nsight Compute duration changes by +0.2938% (slower).
- The normalized 4888-row device SASS sequence is identical.

The signs disagree and every magnitude is noise-scale. There is no profile
evidence for a production change, so stock launch policy remains the correct
fallback/default.

## Side-by-side measurements

| Dimension | Stock | `enable_pdl=False` | Trial minus stock |
| --- | ---: | ---: | ---: |
| Nsight Systems mean | 832326.5 ns | 832171.7 ns | -154.8 ns (-0.0186%) |
| Nsight Systems median | 831647.0 ns | 831775.0 ns | +128.0 ns (+0.0154%) |
| Nsight Compute duration | 838.752 us | 841.216 us | +2.464 us (+0.2938%) |
| SM throughput | 69.3217% | 69.3439% | +0.0222 points |
| Tensor-pipe elapsed activity | 62.6120% | 62.6328% | +0.0208 points |
| Max memory-hierarchy throughput | 56.2842% | 56.1378% | -0.1464 points |
| DRAM read utilization | 3.1630% | 3.1637% | +0.0007 points |
| L2 hit rate | 94.4579% | 94.4745% | +0.0166 points |
| Achieved occupancy | 24.8159% | 24.8341% | +0.0182 points |
| Long-scoreboard cycles/issue | 5.8863 | 5.8907 | +0.0045 |
| Long-scoreboard sample share | 60.6395% | 60.5645% | -0.0750 points |
| Registers/thread | 128 | 128 | 0 |
| Shared memory/block | 220672 B | 220672 B | 0 |

The target kernel name, grid (4096), block (512), resource limits, instruction
sequence, and dominant stall signature are unchanged. PDL-off changes neither the
selected vendor kernel nor its device code.

## SASS comparison method

[`analysis/compare_profiles.py`](analysis/compare_profiles.py) is an offline,
standard-library-only comparison. It reads the retained JSON/CSV/text exports and
does not initialize CUDA. For SASS it:

1. extracts every address-bearing instruction row from both full source-counter
   exports;
2. replaces process-specific absolute relocation addresses;
3. removes one numeric sampling field that Nsight attaches to an overflowing
   `UTCQMMA ... !UPT` display column; and
4. compares and hashes the resulting instruction sequences.

Both sequences contain 4888 rows and hash to
`219febc4b68acdbf9df034b8a77021e3f37b02115afb5bc3ce2cd226efbc4b81`.
This normalization does not remove opcodes, operands, predicates, or
instruction-relative immediates.

## Reproduce the offline analysis

From the Kernel-Harness repository root:

```bash
python3 profile/dsa-prefill-trtllm-m4096-pdl-compare-20260722/analysis/compare_profiles.py
```

The script rewrites:

- [`analysis/comparison.json`](analysis/comparison.json), the complete
  machine-readable comparison; and
- [`analysis/comparison.txt`](analysis/comparison.txt), a compact reviewer view.

The raw inputs remain in the adjacent
[`stock`](../dsa-prefill-trtllm-m4096-stock-20260722/REPORT.md) and
[`pdl-off`](../dsa-prefill-trtllm-m4096-pdl-off-20260722/REPORT.md) profile
packages.

## Caveats

The Systems values aggregate 25 launches; the Compute values are one replayed
launch per variant and are diagnostic only. Nsight Compute also reports that
Work ID/Cluster Launch Control can affect metrics derived from work counts. The
kernel is vendor AOT, so source-line mapping is unavailable even though actual
SASS and per-PC samples are retained. None of these caveats changes the decision:
the observed differences are orders of magnitude below the required 3% threshold.
