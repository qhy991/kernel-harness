# archive/

Campaign result archives that are **not** part of the live harness contract.

| Directory | Description |
|---|---|
| [`0720-Best-GLM-52/`](0720-Best-GLM-52/) | 2026-07-20 KDA-Pilot best agent candidates for GLM-5.2 tasks (wins, near-misses, and evidence-backed no-gos). |
| [`0723-amd-glm52/`](0723-amd-glm52/) | 2026-07-23 MI300X GLM-5.2 tuned operators (8: o_proj×3, index_k×3, dsa_attn×2). Geomean 2.22× over the aiter production baseline; each passes the opbench correctness gate. |

These trees are for provenance and replay. Default task seeds under `testbench/tasks/` remain the frozen starting points unless you explicitly pass `--candidate` at a path here.
