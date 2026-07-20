# DeepGEMM-GLM52 — best registered variants

Source tree: `/home/qinhaiyan/DeepGEMM-GLM52` (`glm52-experiments`)

| Variant ID | Commit | Result |
|---|---|---|
| `glm52-qb-sm100-sms-clamp` | `0b39e97…` | q_b ~29.3% HBM（>25%） |
| **`glm52-qb-sm100-fused-ue8m0-pack`** | **`41c6235…`** | **q_b ~36.3–36.5% HBM（>35%）— 推荐** |

完整表见同目录 `VARIANT_REGISTRY.md`。

加载方式：`KDA-Pilot-Exp/llm/scripts/deepgemm_glm52/loader.py` → `deep_gemm_experimental`；
勿覆盖 Kernel-Harness `.venv` 里的 stock `deep_gemm`。
