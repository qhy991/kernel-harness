# Source and build provenance

## Repositories

- Kernel-Harness base:
  `bcd005409e65786af82c86f621507ebef12b2766`
- Kernel-Harness final source before report consolidation:
  `1d49116a5a31ef7e70e569d8ecf6fab6466f6428`
- SGLang base and final:
  `f93f8867b4bc124c9809c9110ec7361ed11b6b4a`
- SGLang net source diff: empty

The Kernel-Harness commits add only production-shaped serving workloads,
candidate adapters, correctness checks, campaign drivers, and evidence
renderers. They do not edit the frozen GLM-5.2 oracle, task metadata, timing,
anti-cheat code, or `legacy/`.

## Decision-bearing source hashes

| File | SHA-256 |
|---|---|
| SGLang `dsa_indexer.py` | `2399c19c2ecce0c16e33fdc75eccb49383ea1fdba84da1458e191dd39173b730` |
| SGLang `server_args.py` | `1ca51d89b2dcbe14157b3c2fd04596015299bd5f283c8fc57dccaa7594407943` |
| SGLang `glm4_moe.py` | `2c19a1ba456edd5635df6d39c970fd51935d907599f3527d05a8bfa89e9db69f` |
| final `serving_native/indexer_region.py` | `1f838a78730dabf75cea20c912aa09c92afd84532cb624b23551065d677f5861` |
| measured region candidate | `0db1bd447b24b8f1ac004ef557963a5e7499321be88392132a5fdd594675c238` |
| measured score candidate | `b93d7c327d287212f461e3799b0ff9cec26e7ba9b74b18d62f5a4643acddd013` |

Each campaign has its own source manifest because the containing workloads and
TP4 diagnostic were added after the score campaign:

- score: `runs/20260723T113910Z/source_manifest.sha256`, measured at
  Kernel-Harness `6c86a8de3138dfdf883a5c47924f4fc1d0862abb`;
- complete indexer/DSA:
  `region_runs/20260723T120153Z/source_manifest.sha256`, measured at
  `63f1bcf5a937b9e2aeeeff09d39934947c370882`;
- TP4 diagnostic:
  `tp4_runs/20260723T121417Z/source_manifest.sha256`, measured at
  `fbf2410c06135a8a29d7de2a5506b7806299a088`.

## Runtime stack

The decision-bearing campaigns used the repo-local `.venv` and the isolated
SGLang checkout:

- NVIDIA B200, SM100, 148 SMs;
- PyTorch `2.11.0+cu130`, CUDA runtime `13.0`;
- SGLang package metadata `0.5.15`, source checkout at the SHA above;
- FlashInfer Python/cubin `0.6.12`;
- sglang-kernel `0.4.4`;
- Triton `3.6.0`.

The campaign `check_env.log` records the selected physical GPU and module
checkout. Each `pip_freeze.log`, GPU identity file, repository status file,
source manifest, and raw result is covered by its run artifact manifest.

No installed package was overwritten. CuTe-DSL compilation and tactic setup
occur through the shipped SGLang backend outside timed replay. The candidates
are external Python dispatch adapters; the final SGLang worktree remains
unchanged.
