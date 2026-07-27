# Flexible-GPU paired campaign summary

- Campaign: `flex-20260723T160729Z`
- Scheduler allocation: physical GPU 1 (`GPU-5b9be10b-5bfc-b658-9b31-f7ae8516dc54`) exposed as logical GPU 0
- Wrapper evidence: `wrapper.log` (SHA-256 `f19c7093be421879245e103b668686ed5e69b743d43e46871921dc73a7e10f4d`)
- Validated raw paired artifacts: 24/24
- All producer correctness checks passed: `true`

The paired p50 is the median of the raw per-pair `reference_ms / candidate_ms` ratios; the 3% gate is `>= 1.03`.

## Device snapshots

| Stage | Raw file | P-state | Graphics MHz | SM MHz | Memory MHz | Power W | Temp C |
|---|---|---|---:|---:|---:|---:|---:|
| start | `analysis/device_start.json` | P0 | 645 | 645 | 3996 | 146.32 | 33 |
| after_paired | `analysis/device_after_paired.json` | P0 | 780 | 780 | 3996 | 188.46 | 35 |
| after_nsys | `analysis/device_after_nsys.json` | P0 | 757 | 757 | 3996 | 148.55 | 34 |
| end | `analysis/device_end.json` | P0 | 682 | 682 | 3996 | 181.79 | 35 |

## Stock baseline context

| Bucket | Session | Reference p50 (ms) | Raw file |
|---|---:|---:|---|
| M16 | 1 | 0.04297599941492081 | `analysis/baseline_stock_m16_r1.json` |
| M16 | 2 | 0.04358400031924248 | `analysis/baseline_stock_m16_r2.json` |
| M16 | 3 | 0.043696001172065735 | `analysis/baseline_stock_m16_r3.json` |
| M32 | 1 | 0.05375999957323074 | `analysis/baseline_stock_m32_r1.json` |
| M32 | 2 | 0.04729599878191948 | `analysis/baseline_stock_m32_r2.json` |
| M32 | 3 | 0.048239998519420624 | `analysis/baseline_stock_m32_r3.json` |

## Paired measurements

| Mode | Bucket | Variant | Session | Reference p50 (ms) | Candidate p50 (ms) | Paired p50 speedup | Gate | Correctness |
|---|---|---|---:|---:|---:|---:|---|---|
| eager | M16 | control | 1 | 0.04495999962091446 | 0.04411200061440468 | 1.0074143831071285 | FAIL | PASS |
| eager | M16 | control | 2 | 0.045823998749256134 | 0.0453919991850853 | 1.0145773036887573 | FAIL | PASS |
| eager | M16 | control | 3 | 0.05167999863624573 | 0.05124799907207489 | 1.0131166737531179 | FAIL | PASS |
| eager | M16 | candidate | 1 | 0.04879999905824661 | 0.04742399975657463 | 1.0247605324688704 | FAIL | PASS |
| eager | M16 | candidate | 2 | 0.04766400158405304 | 0.0475040003657341 | 1.0141020166184174 | FAIL | PASS |
| eager | M16 | candidate | 3 | 0.044815998524427414 | 0.044256001710891724 | 1.0166338247476818 | FAIL | PASS |
| eager | M32 | control | 1 | 0.049775999039411545 | 0.04926399886608124 | 1.0075336967568256 | FAIL | PASS |
| eager | M32 | control | 2 | 0.050416000187397 | 0.04963199980556965 | 1.0176904516579697 | FAIL | PASS |
| eager | M32 | control | 3 | 0.05020799860358238 | 0.049456000328063965 | 0.995030797227415 | FAIL | PASS |
| eager | M32 | candidate | 1 | 0.05076799914240837 | 0.050416000187397 | 1.004522407464212 | FAIL | PASS |
| eager | M32 | candidate | 2 | 0.051103999838232994 | 0.04995200037956238 | 1.0219715699264755 | FAIL | PASS |
| eager | M32 | candidate | 3 | 0.04955200105905533 | 0.04934399947524071 | 0.9986881213880754 | FAIL | PASS |
| cuda_graph | M16 | control | 1 | 0.030592000111937523 | 0.031247999519109726 | 0.9913507725534355 | FAIL | PASS |
| cuda_graph | M16 | control | 2 | 0.03175999969244003 | 0.03198400139808655 | 0.9912824946674033 | FAIL | PASS |
| cuda_graph | M16 | control | 3 | 0.03158400021493435 | 0.031599998474121094 | 0.9938223852210077 | FAIL | PASS |
| cuda_graph | M16 | candidate | 1 | 0.030640000477433205 | 0.03094400092959404 | 0.9893568266789268 | FAIL | PASS |
| cuda_graph | M16 | candidate | 2 | 0.030672000721096992 | 0.030880000442266464 | 0.9900068480784652 | FAIL | PASS |
| cuda_graph | M16 | candidate | 3 | 0.031072000041604042 | 0.03139200061559677 | 0.9891241342909625 | FAIL | PASS |
| cuda_graph | M32 | control | 1 | 0.03681600093841553 | 0.037087999284267426 | 0.9899225084373786 | FAIL | PASS |
| cuda_graph | M32 | control | 2 | 0.03625600039958954 | 0.03667199984192848 | 0.9894458221606957 | FAIL | PASS |
| cuda_graph | M32 | control | 3 | 0.036720000207424164 | 0.036896001547575 | 0.9939953932657186 | FAIL | PASS |
| cuda_graph | M32 | candidate | 1 | 0.03683200106024742 | 0.036959998309612274 | 0.9921051020609164 | FAIL | PASS |
| cuda_graph | M32 | candidate | 2 | 0.036928001791238785 | 0.037647999823093414 | 0.9856620163222459 | FAIL | PASS |
| cuda_graph | M32 | candidate | 3 | 0.03620800003409386 | 0.036448001861572266 | 0.9912701811979512 | FAIL | PASS |

## Three-session groups

| Mode | Bucket | Variant | Session speedups | Median session speedup | 3% passes | Repeated gate |
|---|---|---|---|---:|---:|---|
| eager | M16 | control | 1.0074143831071285, 1.0145773036887573, 1.0131166737531179 | 1.0131166737531179 | 0/3 | FAIL |
| eager | M16 | candidate | 1.0247605324688704, 1.0141020166184174, 1.0166338247476818 | 1.0166338247476818 | 0/3 | FAIL |
| eager | M32 | control | 1.0075336967568256, 1.0176904516579697, 0.995030797227415 | 1.0075336967568256 | 0/3 | FAIL |
| eager | M32 | candidate | 1.004522407464212, 1.0219715699264755, 0.9986881213880754 | 1.004522407464212 | 0/3 | FAIL |
| cuda_graph | M16 | control | 0.9913507725534355, 0.9912824946674033, 0.9938223852210077 | 0.9913507725534355 | 0/3 | FAIL |
| cuda_graph | M16 | candidate | 0.9893568266789268, 0.9900068480784652, 0.9891241342909625 | 0.9893568266789268 | 0/3 | FAIL |
| cuda_graph | M32 | control | 0.9899225084373786, 0.9894458221606957, 0.9939953932657186 | 0.9899225084373786 | 0/3 | FAIL |
| cuda_graph | M32 | candidate | 0.9921051020609164, 0.9856620163222459, 0.9912701811979512 | 0.9912701811979512 | 0/3 | FAIL |

## Validation outcome

- Candidate groups with repeated 3% sessions: 0/4.
- Candidate sessions passing the 3% gate: 0/12.
- Baseline r1-to-r3 change: M16 0.01675357797252186; M32 -0.10267859184579964.
