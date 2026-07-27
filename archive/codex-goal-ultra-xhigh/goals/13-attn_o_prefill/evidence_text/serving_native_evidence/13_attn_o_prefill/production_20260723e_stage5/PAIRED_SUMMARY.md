# Paired performance summary

All rows use the exact production packed FP8 shape `M=4096, N=6144, K=16384`
on B200. Each numbered series contains 30 alternating reference/candidate
samples on one physical GPU. A candidate needs a paired-median speedup of at
least `1.03`, correctness, and a non-regressing containing region.

| Measurement | Series | Reference p50 (ms) | Candidate p50 (ms) | Paired p50 speedup | Paired p10–p90 | Correct | Gate |
|---|---:|---:|---:|---:|---:|---|---|
| Identity calibration | 1 | 0.304624 | 0.304416 | 0.999373 | 0.986373–1.013562 | yes | neutral |
| Identity calibration | 2 | 0.305232 | 0.305104 | 0.998949 | 0.982838–1.014064 | yes | neutral |
| Identity calibration | 3 | 0.308160 | 0.308064 | 0.995850 | 0.971366–1.017775 | yes | neutral |
| Existing compiled-NK dispatcher, leaf | 1 | 0.316448 | 0.314144 | 1.001274 | 0.962130–1.034696 | yes | neutral |
| Existing compiled-NK dispatcher, leaf | 2 | 0.308800 | 0.308544 | 0.994634 | 0.959610–1.022227 | yes | neutral |
| Existing compiled-NK dispatcher, leaf | 3 | 0.320608 | 0.316784 | 1.006626 | 0.957312–1.054500 | yes | neutral |
| Existing dispatcher, `Fp8LinearMethod.apply` | 1 | 0.393872 | 0.382240 | 1.009760 | 0.934704–1.073937 | yes | neutral |
| Existing dispatcher, `Fp8LinearMethod.apply` | 2 | 0.386704 | 0.383360 | 1.001332 | 0.967254–1.062979 | yes | neutral |
| Existing dispatcher, `Fp8LinearMethod.apply` | 3 | 0.387008 | 0.389600 | 0.997545 | 0.942590–1.065350 | yes | neutral |
| Five-stage serving-runner probe | 1 | 0.303040 | 0.298304 | 1.014827 | 0.988125–1.039484 | yes | neutral |
| Five-stage serving-runner probe | 2 | 0.303488 | 0.298176 | 1.026786 | 0.993133–1.055331 | yes | neutral |
| Five-stage serving-runner probe | 3 | 0.308160 | 0.300688 | 1.026509 | 0.988405–1.067246 | yes | neutral |
| Five-stage fair direct leaf | 1 | 0.292464 | 0.299744 | 0.973239 | 0.959119–0.992861 | yes | regress |
| Five-stage fair direct leaf | 2 | 0.294480 | 0.302752 | 0.978281 | 0.954058–0.999698 | yes | regress |
| Five-stage fair direct leaf | 3 | 0.314112 | 0.321824 | 0.972641 | 0.942042–0.998281 | yes | regress |
| Five-stage `Fp8LinearMethod.apply` | 1 | 0.347984 | 0.360368 | 0.961955 | 0.914997–1.017332 | yes | regress |
| Five-stage `Fp8LinearMethod.apply` | 2 | 0.365456 | 0.372784 | 0.972547 | 0.916888–1.029123 | yes | regress |
| Five-stage `Fp8LinearMethod.apply` | 3 | 0.372928 | 0.382608 | 0.976800 | 0.922604–1.028011 | yes | regress |

The serving-runner probe compares a candidate that calls the isolated overlay
leaf directly with a reference that enters through the production SGLang
wrapper. It is retained as workload evidence, but it does not isolate the
source change and none of its series reaches the 3% gate. The fair direct-leaf
and full-region series are the adjudicating comparisons: they alternate the
stock six-stage and experimental five-stage kernels under the same call
contract, and both regress in every series.

The full-precision records, including every sample, are in
`paired_summary.json`'s `raw_sources`.
