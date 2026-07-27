# 05-indexer_k_weights_prefill

- Campaign: Codex GLM-5.2 production goals (`gpt-5.6-sol`, `model_reasoning_effort=ultra`)
- Source worktree: `/home/qinhaiyan/glm52-goal-runs/05-indexer_k_weights_prefill`
- Harness branch: `goal/glm52-prod-05-indexer_k_weights_prefill` @ `1e38cef`
- Disposition (from EXPERIMENT_RESULTS / archive triage): `no-replacement (inner-gate)`
- Note: TGV regress; torch.mm/single-stream miss 1.03×
- Analysis status: analyzed
- Archived custom candidates: indexer_single_stream.py, indexer_wk_cutedsl_tgv.py, indexer_wk_flashinfer.py, indexer_wk_torch_mm.py
- Primary reports in this archive: reports/indexer-wk-weights-prefill-m4096-20260722/REPORT.md
- Evidence text files archived: 163
- Full profile binaries (nsys/ncu/sqlite) intentionally **not** copied; remain under source `profile/` / `evidence/`.
