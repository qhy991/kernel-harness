# 02-dsa_decode_attention

- Campaign: Codex GLM-5.2 production goals (`gpt-5.6-sol`, `model_reasoning_effort=ultra`)
- Source worktree: `/home/qinhaiyan/glm52-goal-runs/02-dsa_decode_attention`
- Harness branch: `goal/glm52-prod-02-dsa_decode_attention` @ `d91dc16`
- Disposition (from EXPERIMENT_RESULTS / archive triage): `no-replacement`
- Note: scheduler regress; bank-layout correctness fail
- Analysis status: analyzed
- Archived custom candidates: dsa_direct_trtllm.py, dsa_flashmla_kv_bank_layout.py, dsa_flashmla_kv_sched_override.py, dsa_pdl_disabled.py, dsa_preallocated_out.py, dsa_split9/, dsa_tensor_bmm1_scale.py
- Primary reports in this archive: reports/dsa_flashmla_kv_production_20260722/REPORT.md, reports/dsa_flashmla_kv_scheduler_campaign_20260722/REPORT.md
- Evidence text files archived: 0
- Full profile binaries (nsys/ncu/sqlite) intentionally **not** copied; remain under source `profile/` / `evidence/`.
