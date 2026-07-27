# archive/

Campaign result archives that are **not** part of the live harness contract.

| Directory / file | Description |
|---|---|
| [`TEST_RESULTS_0724.md`](TEST_RESULTS_0724.md) | **2026-07-24 汇总**：serving bake-off 数值 + Codex/e2e 结案（无确认上线收益）。 |
| [`0720-Best-GLM-52/`](0720-Best-GLM-52/) | 2026-07-20 KDA-Pilot best agent candidates for GLM-5.2 tasks (wins, near-misses, and evidence-backed no-gos). |
| [`codex-goal-ultra-xhigh/`](codex-goal-ultra-xhigh/) | 2026-07-22..23 Codex GLM-5.2 production goals (`gpt-5.6-sol` / reasoning `ultra`): custom candidates, REPORT/FINAL text, and evidence text from `~/glm52-goal-runs`. |
| [`serving_bakeoff_0720_vs_codex/`](serving_bakeoff_0720_vs_codex/) | Serving-native paired A/B bake-off of 0720 ports vs Codex candidates (production packed UE8M0 ABI). |

These trees are for provenance and replay. Default task seeds under `testbench/tasks/` remain the frozen starting points unless you explicitly pass `--candidate` at a path here.
