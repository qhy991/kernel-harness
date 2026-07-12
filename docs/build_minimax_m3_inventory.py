#!/usr/bin/env python3
"""Build a MiniMax-M3 B200 operator-backend inventory CSV, matching the schema of
sglang_b200_operator_backend_inventory.xlsx (12 cols) + test-result + perf columns.

xlsx schema (header on row 3 of the xlsx):
  模型 | 检查点 | 算子名称 | 算子所处阶段 | 执行阶段 | 输入规模 | 输出规模 |
  数据精度 | B200实际后端 | Python调度路径 | 底层Kernel/库 | CI验证来源

Extra columns appended for this report:
  本机测试结果 | 实测延迟 | 实测带宽/吞吐 | 缺口/备注

Performance data sources (run on B200, harness venv, amd_add_m3 worktree):
  - bench_minimax_qknorm_rope.py        (op 29/35)
  - bench_minimax_store_kv_index.py     (op 30)
  - bench_minimax_decode_topk.py        (op 32)
"""
import csv

MODEL = "MiniMax-M3"
CKPT_BF16 = "MiniMaxAI/MiniMax-M3"
CKPT_MXFP8 = "MiniMaxAI/MiniMax-M3-MXFP8"

# M3 text-config (HF config.json):
#   num_q_heads=64, num_kv_heads=4 (GQA 16:1), head_dim=128, rotary_dim=64,
#   sparse_num_index_heads=4, sparse_index_dim=128, sparse_block_size=128,
#   sparse_init_block=0, sparse_local_block=1, sparse_score_type="max",
#   num_local_experts=128, num_experts_per_tok=4, scoring_func=sigmoid,
#   routed_scaling_factor=2.0, swiglu_alpha=1.702, swiglu_limit=7.0,
#   max_position_embeddings=1048576, torch_dtype=bfloat16

# Each row: the 12 xlsx columns + 4 extra. perf latency/throughput are strings
# (ranges from the bench scripts; "n/a" when not benchmarked).
ROWS = [
    # op 28
    ["op28", CKPT_BF16, "Main Q/K/V 投影 (per-head GQA)", "每层主注意力", "Prefill+Decode",
     "q [M,64,128]; kv [M,4,128]", "q/k/v [M,64/4,128]", "BF16 (MXFP8检查点为MXFP8)",
     "标准 Linear → cuBLAS BF16 GEMM (MXFP8检查点走 deep_gemm/flashinfer MoE backend)",
     "python/sglang/srt/layers/linear.py:143 (LinearBase)", "cuBLAS GEMM", "—",
     "NO-OP-KERNEL (M3主QKV走标准Linear,无M3专属kernel)", "n/a", "n/a",
     "M3非MLA,per-head GQA(num_q_heads%num_kv_heads==0)"],

    # op 29 / 35
    ["op29/35", CKPT_BF16, "Indexer Q投影 + 融合 Gemma-RMSNorm + 局部RoPE", "每层稀疏indexer前", "Prefill+Decode",
     "q [T,64,128]+k[T,4,128]+idx_q[T,4,128]+idx_k[T,1,128]", "原位norm+rope,同形状",
     "BF16 (cos|sin cache需FP32)",
     "JIT CUDA fused_gemma_qknorm_rope (单launch, [q|k|idx_q|idx_k]分组)",
     "python/sglang/jit_kernel/minimax_qknorm_rope.py:51,152; csrc/minimax/fused_gemma_qknorm_rope.cuh",
     "JIT CUDA (load_jit, 不在sgl_kernel wheel)", "test/registered/jit/minimax/test_minimax_qknorm_rope.py; test/registered/jit/minimax/test_minimax_m3_cuda_config.py",
     "PASS (79 + 新增M3(64,4,4)grouped 1case)",
     "fused 1.64-192.7us (T=1..8192)", "峰值1372.8 GB/s @T=1024; combined_one_launch比two_launches快2.4x",
     "rotary_dim=64,partial_rotary=0.5; 现有grid未覆盖(64,4,4),已补"],

    # op 30
    ["op30", CKPT_BF16, "Indexer K/V cache 融合写入 (store_kv_index)", "稀疏层KV cache写入", "Prefill+Decode",
     "k/v [T,4*128]; idx_k [T,128]; idx_v [T,128]", "k_cache/v_cache [N,4*128]; idx_k/v_cache [N,128]",
     "BF16 (uniform dtype)",
     "JIT CUDA store_kv_index (单launch写主K/V+idx K[+idx V])",
     "python/sglang/jit_kernel/minimax_store_kv_index.py:31; csrc/minimax/fused_store_kv_index.cuh",
     "JIT CUDA", "test/registered/jit/minimax/test_minimax_store_kv_index.py; test_minimax_m3_cuda_config.py",
     "PASS (96 + 新增M3 shape 1case)",
     "fused 1.41-4.89us (T=16..16384)", "峰值4815.5 GB/s @T=16384; 比separate快~12x",
     "SGLANG_OPT_USE_MINIMAX_FUSED_KV_INDEX_STORE默认True; 需CUDA+16B对齐"],

    # op 31
    ["op31", CKPT_BF16, "Prefill Indexer Score + topk (单launch)", "稀疏层prefill score", "Prefill",
     "idx_q [total_q,4,128] × idx_k_cache [slots,1,128]", "score [4,total_q,num_blocks]; topk_idx [4,total_q,topk]",
     "BF16计算→FP32 score",
     "Triton autotune _flash_attn_fwd_with_block_score_kernel (score max/lse + 内联topk)",
     "python/sglang/srt/layers/attention/minimax_sparse_ops/prefill/flash_with_topk_idx.py:62,417,491",
     "Triton (JIT, on-device)", "test/registered/jit/minimax/test_minimax_m3_cuda_config.py::test_m3_prefill_score_topk (NEW)",
     "PASS (新写, 1case; B=2,L=256,topk=8)",
     "n/a (无prefill bench脚本)", "n/a",
     "★缺口1: 测试用退化shape(长度1-per-seq)经decode等价做oracle; 真实多token prefill未压; autotune慢"],

    # op 32
    ["op32", CKPT_BF16, "Decode Indexer Score + topk", "稀疏层decode score", "Decode",
     "score [4,batch,max_seqblock] → topk_idx [4,batch,topk]",
     "topk_idx [4,batch,topk] int32",
     "FP32 score",
     "JIT CUDA minimax_decode_topk (radix单stage); 回落Triton _topk_index_partial+merge (2stage)",
     "python/sglang/jit_kernel/minimax_decode_topk.py:43; csrc/minimax/minimax_decode_topk.cuh:270",
     "JIT CUDA radix / Triton 2stage", "test/registered/jit/minimax/test_minimax_decode_topk.py; test_minimax_decode_topk_page_table.py; test_minimax_m3_cuda_config.py",
     "PASS (118 radix + 30 page_table + 新增topk∈{4,8,16,32}×ctx∈{512,2048} 8case)",
     "JIT radix 2.25-7.50us (ctx=4096..524288, b=1..256)",
     "JIT比Triton 2stage快2-30x (ctx=524288,b=256: 7.5us vs 239us)",
     "SGLANG_OPT_USE_MINIMAX_DECODE_TOPK_RADIX默认True+shape门控(topk≤32); 新增topk=4补现有grid{16,32,64}缺口"],

    # op 33
    ["op33", CKPT_BF16, "Prefill 主稀疏注意力 (GQA share, topk block)", "稀疏层prefill主attn", "Prefill",
     "q [total_q,64,128] × k_cache [slots,4,128] under topk_idx",
     "o [total_q,64,128]",
     "BF16",
     "Triton _gqa_share_sparse_fwd_kernel; B200生产路径=msa_sparse_prefill_main(fmha_sm100,SM100)",
     "python/sglang/srt/layers/attention/minimax_sparse_ops/prefill/topk_sparse.py:258; msa.py:92,142",
     "Triton / fmha_sm100 (外部)", "test/registered/jit/minimax/test_minimax_m3_cuda_config.py::test_m3_prefill_main_sparse (NEW, Triton路径)",
     "PASS (Triton路径, 1case; B=2,L=256,topk=8)",
     "n/a (无prefill bench)", "n/a",
     "★缺口2: B200生产fast path=msa_sparse_prefill_main需fmha_sm100 package,本机未装(msa_available()=False); 测试只覆盖Triton回落"],

    # op 34
    ["op34", CKPT_BF16, "Decode 主稀疏注意力 (GQA share, topk block)", "稀疏层decode主attn", "Decode",
     "q [batch,64,128] × k_cache [slots,4,128] under topk_idx",
     "o [batch,64,128]",
     "BF16",
     "Triton flash_decode_with_gqa_share_sparse; B200生产=msa_sparse_decode_main(fmha_sm100,eager非cuda-graph)",
     "python/sglang/srt/layers/attention/minimax_sparse_ops/decode/topk_sparse.py:299; msa.py:316,362",
     "Triton / fmha_sm100 (外部)", "minimax_sparse_ops/tests/test_sparse_gqa.py",
     "PASS (20 case, Triton路径, 77s)",
     "n/a", "n/a",
     "★缺口2同op33: msa decode非cuda-graph safe(msa.py:194-209); fmha_sm100未装故未测生产路径"],

    # op 36 (ROCm skip)
    ["op36", CKPT_BF16, "Gemma-RMSNorm (独立, Triton)", "norm层", "Prefill+Decode",
     "[*,6144] fp32", "[*,6144]",
     "BF16→FP32",
     "ROCm gfx94x/gfx95x Triton _gemma_rmsnorm_kernel (CUDA走sgl_kernel.gemma_rmsnorm C++)",
     "python/sglang/jit_kernel/minimax_m3/rmsnorm.py:19,96",
     "Triton (ROCm) / sgl_kernel C++ (CUDA)", "python/sglang/jit_kernel/tests/test_minimax_m3_rmsnorm.py",
     "SKIP (B200上1 skipped, ROCm gfx95专属)",
     "n/a", "n/a",
     "layernorm.py:88-91用sgl_kernel.gemma_rmsnorm(C++)非此Triton; 两个不同实现"],

    # op 37
    ["op37", CKPT_MXFP8, "MoE Router + TopK (sigmoid)", "MoE层router", "Prefill+Decode",
     "[M,6144]×[6144,128_experts]", "logits [M,128]; topk_idx [M,4]; weights",
     "BF16计算",
     "标准sglang FusedMoE router; cuBLAS/DeepGEMM _jit_dsv3_router_gemm; topk=sigmoid+routed_scaling_factor折入weights",
     "python/sglang/srt/layers/moe/topk.py; python/sglang/jit_kernel/dsv3_router_gemm.py:65",
     "cuBLAS/DeepGEMM + sglang topk", "sgl-kernel/tests/test_moe_topk_sigmoid.py; test/registered/kernels/test_fused_topk_deepseek.py",
     "PASS (774 case)", "n/a", "n/a",
     "M3用sigmoid routing,routed_scaling_factor=2.0已折入topk_weights; 复用标准MoE无M3专属kernel"],

    # op 38 (B200 M3-MXFP8 NOT this path!)
    ["op38", CKPT_MXFP8, "MoE MXFP8 GateUp/Act/Down", "MoE层expert GEMM", "Prefill+Decode",
     "E×[M_e,6144]×[6144,2*3072] MXFP8 e4m3+E8M0",
     "E×[M_e,3072]→[M_e,6144]",
     "MXFP8 (e4m3+E8M0 scales)",
     "★B200上不走此路径! Triton runner显式raise(不支持NV MXFP8); B200走 --moe-runner-backend deep_gemm(mega_moe)或flashinfer_trtllm/cutlass",
     "python/sglang/srt/layers/moe/moe_runner/triton.py:114-117 (gate); triton_utils/mxfp8_moe_amd_gfx95.py:358 (ROCm实现)",
     "ROCm: Triton dot_scaled groupGEMM; B200: deep_gemm mega_moe(未映射)",
     "python/sglang/jit_kernel/tests/test_minimax_m3_mxfp8.py (ROCm)",
     "SKIP (B200上1 skipped; 此kernel仅ROCm gfx950)",
     "n/a", "n/a",
     "★缺口3(最重要): B200 M3-MXFP8实际MoE走deep_gemm mega_moe(SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE),不在op28-43映射,未测; op38仅ROCm"],

    # op 39
    ["op39", CKPT_MXFP8, "SwiGLU-OAI split", "MoE层激活", "Prefill+Decode",
     "[*,2*3072]→[*,3072]", "[*,3072]",
     "FP32 math",
     "Triton _swiglu_oai_kernel (仅ROCm gfx95x)",
     "python/sglang/jit_kernel/minimax_m3/swiglu.py:18,55",
     "Triton (ROCm)", "python/sglang/jit_kernel/tests/test_minimax_m3_mxfp8.py",
     "SKIP (B200, ROCm专属)", "n/a", "n/a",
     "OAI式SwiGLU: alpha*x*sigmoid(alpha*x)*(up+beta)+clamp; swiglu_alpha=1.702,limit=7.0"],

    # op 40
    ["op40", CKPT_MXFP8, "SwiGLU-OAI + MXFP8 quant (融合)", "MoE层激活+量化", "Prefill+Decode",
     "[*,2*3072]→[*,3072] fp8_e4m3 + [*,96] E8M0", "同左",
     "MXFP8量化",
     "Triton _swiglu_oai_mxfp8_quant_kernel (单launch, 仅ROCm)",
     "python/sglang/jit_kernel/minimax_m3/swiglu.py:96,159",
     "Triton (ROCm)", "python/sglang/jit_kernel/tests/test_minimax_m3_mxfp8.py",
     "SKIP (B200, ROCm专属)", "n/a", "n/a",
     "SGLANG_MINIMAX_M3_FUSED_SWIGLU_MXFP8=true opt-in; 实验性A/B"],

    # op 41
    ["op41", CKPT_MXFP8, "MoE Combine (融合 topk routes)", "MoE层combine", "Prefill+Decode",
     "[M,topk=4,6144]→[M,6144]", "[M,6144]",
     "BF16",
     "Triton _combine_topk_routes_kernel (仅ROCm) 或 .sum(dim=1) 回落",
     "python/sglang/srt/layers/moe/moe_runner/triton_utils/mxfp8_moe_amd_gfx95.py:191,220",
     "Triton (ROCm) / torch.sum", "python/sglang/jit_kernel/tests/test_minimax_m3_mxfp8.py",
     "SKIP (B200, ROCm专属)", "n/a", "n/a",
     "SGLANG_MINIMAX_M3_FUSED_MOE_COMBINE默认False; 实验性"],

    # op 42
    ["op42", CKPT_BF16, "KV Pool (MiniMaxSparseKVPool, 3子池)", "KV cache基础设施", "—",
     "3子池: main K/V[slots,4,128]+idx_kv[slots,1,128]+idx_k[slots,1,128]",
     "paged KV存储",
     "BF16",
     "组合MHATokenToKVPool(main)+index_kv_pool(K+V)+index_k_pool(K-only)",
     "python/sglang/srt/mem_cache/memory_pool.py:3538,3593,3606,3621,3743,3788",
     "sglang MHATokenToKVPool (CUDA paged)", "test/registered/unit/mem_cache/test_minimax_sparse_pool_host_unit.py; test_minimax_sparse_pool_pd_unit.py",
     "PASS (host 7+1skip+3subtests; pd 2)", "n/a", "n/a",
     "is_minimax_sparse(config)自动检测(arch=MiniMaxM3SparseFor*+sparse_attention_config)"],

    # op 43
    ["op43", CKPT_BF16, "Hybrid Cache 策略 (HiRadixCache dispatch)", "L1/L2 cache tier", "—",
     "MiniMaxSparseKVPool经_MiniMaxSparseStrategy调度", "L1(host)/L2(nvme+rdma) tier",
     "BF16",
     "build_minimax_sparse_hicache_stack; attach到HiRadixCache",
     "python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py:969,1219; hiradix_cache.py:145",
     "sglang HiRadixCache + hicache", "test/registered/unit/mem_cache/test_unified_radix_hicache_dispatch.py",
     "NOT-RUN (端到端集成测试,超出算子级范围)", "n/a", "n/a",
     "无PP支持; L2仅支持K-only sparse层"],
]

# xlsx schema (12) + extra (4) = 16 cols. First col is op id (for readability;
# the xlsx uses 模型 as first col, so we keep that and put op id inside 算子名称).
HEADER = [
    "模型", "检查点", "算子名称", "算子所处阶段", "执行阶段",
    "输入规模", "输出规模", "数据精度", "B200实际后端", "Python调度路径",
    "底层Kernel/库", "CI验证来源",
    "本机测试结果(B200)", "实测延迟", "实测带宽/吞吐", "缺口/备注",
]

DST = "/home/qinhaiyan/kernel-harness/docs/minimax_m3_operator_backend_inventory.csv"
with open(DST, "w", newline="") as f:
    w = csv.writer(f)
    # title rows to match xlsx style
    w.writerow(["MiniMax-M3 · B200 实际算子后端 + 测试/性能清单"])
    w.writerow(["配置来源: HF MiniMaxAI/MiniMax-M3 text_config; 测试环境: B200 + harness venv + amd_add_m3 worktree@e28a0c81; 性能来自 test/registered/jit/benchmark/minimax/"])
    w.writerow(HEADER)
    for row in ROWS:
        # row[0] is op id tag -> fold into 算子名称 prefix
        op_tag = row[0]
        out = [MODEL, row[1], f"[{op_tag}] {row[2]}"] + row[3:]
        w.writerow(out)

print(f"wrote {len(ROWS)} operator rows -> {DST}")
# summary
from collections import Counter
verdicts = Counter()
for r in ROWS:
    v = r[12].split("(")[0].strip().split(" ")[0] if r[12] else ""
    # first word of 测试结果 col
    v = r[12].split()[0] if r[12] else "?"
    verdicts[v] += 1
print("verdict counts:", dict(verdicts))
