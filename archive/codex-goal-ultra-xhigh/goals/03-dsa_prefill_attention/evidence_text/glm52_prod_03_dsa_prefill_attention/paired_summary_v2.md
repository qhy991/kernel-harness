# Audited paired summary

Decision-bearing comparisons use paired ratios within one scheduler allocation;
absolute times are not compared across physical GPUs.

| Scope | Runs × pairs | Reference p50s (ms) | Candidate p50s (ms) | Median run paired speedup | Gate |
|---|---:|---|---|---:|---|
| exact raw-pool stock control | 3 × 30 | 0.874624 / 0.888160 / 0.977648 | 0.882448 / 0.887040 / 0.968560 | 1.000624x | neutral |
| exact raw-pool Q32 Swaps | 3 × 30 | 0.871088 / 0.873424 / 0.864928 | 1.608112 / 1.612208 / 1.600384 | 0.541023x | reject |
| exact raw-pool Q16 Swaps | 3 × 30 | 0.866640 / 0.967984 / 0.865968 | 2.924160 / 3.022368 / 2.923248 | 0.296514x | reject |
| checkpoint-free backend stock control | 3 × 20 | 0.990112 / 0.979568 / 0.993392 | 0.994096 / 0.979088 / 0.994288 | 0.998658x | neutral |
| DP4 rank-max stock control | 3 × 20 | 0.911072 / 0.940944 / 0.941648 | 0.905200 / 0.943040 / 0.952880 | 1.003564x | diagnostic neutral |

Two raw-pool series show allocation-local absolute-time excursions, but both
arms move together and every Q32/Q16 paired result remains decisively negative.
The fixed promotion threshold is 1.03x; no source tactic approaches it.

The older `paired_summary.*` files summarize the inherited PDL experiment. Per
[`AUDIT_CORRECTIONS.md`](AUDIT_CORRECTIONS.md), its 0.996817x value is direct
external-candidate leaf evidence, not SGLang-integrated timing.
