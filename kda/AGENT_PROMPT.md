You are running a Kernel Design Agents (KDA) Phase-1 planning session in this Kernel-Harness git worktree.

Non-negotiables:
1. Use English for all files you write and all user-facing status.
2. Do NOT implement the kernel yet. This session ends after Humanize gen-plan finishes.
3. Do NOT ask the user questions unless a Humanize tool hard-requires it; prefer autonomous decisions and --direct mode.
4. Read `kda/prompts/phase1.md` fully and treat it as the task contract.
5. Inspect `testbench/tasks/glm52/index_k_proj_decode/` (`task.json`, `definition.json`, `reference.py`, `solution.py`, `workload.jsonl`).
6. You may read KernelWiki / ncu-report-skill and Kernel-Harness testbench docs as needed.
7. Write a high-quality implementation draft to `kda/docs/draft.md`.
8. Immediately after the draft exists, invoke:
   `/humanize:gen-plan --input kda/docs/draft.md --output kda/docs/plan.md --direct`
9. When gen-plan completes, verify `kda/docs/plan.md` exists, print a short English summary, then print exactly:
   `KDA-PHASE1-GENPLAN-DONE glm52/index_k_proj_decode`

Do not start `/humanize:start-rlcr-loop` in this session.
