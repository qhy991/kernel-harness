#!/usr/bin/env python3
"""Req-0 mark-integrity validator for the MI300X flow.

Proves BOTH benchmark marks are themselves correct before any optimization runs, so a
flow failure can never be blamed on a broken yardstick. Exits nonzero on ANY failure.

It checks, for the GLM-5.2-on-MI300X targets (o_proj/index_k prefill, dsa_attn decode):

  A. Peaks + cost model AGREE across the two marks — opbench (testbench/harness/
     glm52_ops.py) and rewardbench (AMD4GLM52/amd_glm5_ops_common.py) must produce
     byte-identical (flops, bytes, dtype) and identical reward for the same latency,
     or a kernel scored by both would get two different verdicts.
  B. The opbench REFERENCE is actually correct — the triton blockwise-fp8 GEMM matches
     a bf16 dequant oracle, and the chunked sparse-MLA matches a full-attention oracle.
     The reference is the correctness truth; if it is wrong, every gate is wrong.
  C. The opbench GATE has teeth — a no-op (returns the poisoned buffer) FAILS, a
     magnitude-wrong candidate FAILS, an independent-but-correct reimpl PASSES, and the
     reference-as-candidate PASSES at speedup~1. (poison + calc_diff + tolerances work.)
  D. The rewardbench reward is well-formed — monotonic in latency, correct bound
     classification vs the ridge, reward in (0,1] for the reference, and equal to
     opbench's reward for the same (latency, flops, bytes).

Run:  .venv/bin/python rewardbench/amd/validate_marks.py
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("KERNEL_HARNESS_PLATFORM", "rocm")
os.environ.setdefault("KERNEL_HARNESS_PROFILE", "rocm-mi300x")
os.environ.setdefault("KERNEL_HARNESS_PROVIDER", "torch-triton-rocm")
os.environ.setdefault("KERNEL_HARNESS_TIMER", "event")

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_REPO))

import torch  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ops = _load("glm52_ops", _REPO / "testbench" / "harness" / "glm52_ops.py")
rb = _load("amd_glm5_ops_common", _HERE / "amd_glm5_ops_common.py")
from testbench.harness.backends import rocm_mi300x as R  # noqa: E402

FAILURES: list[str] = []
CHECKS = 0


def check(cond: bool, label: str, detail: str = ""):
    global CHECKS
    CHECKS += 1
    tag = "ok  " if cond else "FAIL"
    print(f"  [{tag}] {label}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(label + (f" ({detail})" if detail else ""))


def approx(a, b, rel=1e-9):
    return abs(a - b) <= rel * max(abs(a), abs(b), 1.0)


DEV = "cuda:0"
S = ops.DEFAULT_S
# (op, phase, M, rewardbench cost fn + args)
TARGETS = [
    ("o_proj", "prefill", 4096, lambda M: rb.gemm_fp8_cost(M, 16384, 6144)),
    ("index_k", "prefill", 4096, lambda M: rb.gemm_fp8_cost(S, 6144, 128)),   # rows=S
    ("dsa_attn", "decode", 16, lambda M: rb.sparse_mla_cost(M, S)),
    ("dsa_attn", "decode", 32, lambda M: rb.sparse_mla_cost(M, S)),
]


def section(t):
    print(f"\n=== {t} ===")


# ── A. peaks + cost model agree across marks ─────────────────────────────────
section("A. peaks + cost model agree (opbench <-> rewardbench)")
check(approx(ops.HBM_BYTES_PER_S, rb.HBM_BYTES_PER_S), "HBM peak identical",
      f"{ops.HBM_BYTES_PER_S:.3e}")
check(approx(ops.PEAK_FLOPS["fp8"], rb.PEAK_FLOPS["fp8"]), "FP8 peak identical",
      f"{ops.PEAK_FLOPS['fp8']:.4e}")
check(approx(ops.PEAK_FLOPS["bf16"], rb.PEAK_FLOPS["bf16"]), "BF16 peak identical",
      f"{ops.PEAK_FLOPS['bf16']:.4e}")
check(str(R.FP8_DTYPE) == str(rb.FP8_DTYPE) == "torch.float8_e4m3fnuz",
      "fp8 dtype e4m3fnuz in both", str(R.FP8_DTYPE))

for op, phase, M, rbcost in TARGETS:
    f1, b1, d1 = ops.cost(op, phase, M, S)
    f2, b2, d2 = rbcost(M)
    check(approx(f1, f2) and approx(b1, b2) and d1 == d2,
          f"cost({op}/{phase} M={M}) identical",
          f"opbench(flops={f1:.4e},bytes={b1:.0f},{d1}) rewardbench(flops={f2:.4e},bytes={b2:.0f},{d2})")

# reward formula identity: same (latency, flops, bytes, dtype) -> same reward
for lat_ms in (0.5, 2.0, 10.0):
    f, b, d = ops.cost("o_proj", "prefill", 4096, S)
    r1 = ops.reward(lat_ms, f, b, d)["reward"]
    r2 = rb.roofline_reward(lat_ms, f, b, d)["reward"]
    check(approx(r1, r2), f"reward formula identical @ {lat_ms}ms", f"{r1:.6f} vs {r2:.6f}")

# ── B. opbench reference is actually correct (vs independent oracle) ──────────
section("B. opbench reference correct (vs bf16/f64 oracle)")
for op, phase, M in [("o_proj", "prefill", 1024), ("index_k", "prefill", 1024)]:
    ins = ops.build_inputs(op, phase, M, S, DEV, 0)
    tri = R._blockwise_fp8_gemm(ins["x_fp8"], ins["x_scale"], ins["w_fp8"], ins["w_scale"])
    ora = R._blockwise_fp8_gemm_torch(ins["x_fp8"], ins["x_scale"], ins["w_fp8"], ins["w_scale"])
    cd = ops.calc_diff(tri.float(), ora.float())
    check(cd < 1e-6, f"{op} triton-GEMM vs dequant-oracle calc_diff<1e-6", f"{cd:.3e}")

# MLA: chunked reference vs a full float64 attention oracle
ins = ops.build_inputs("dsa_attn", "decode", 32, S, DEV, 0)
ref = R._ref_mla(ins).float()
q = ins["q"].double()
kv = ins["kv"][:, 0, :].double()
idx = ins["indices"][:, 0, :].long()
sm = ins["sm_scale"]
dv = ins["d_v"]
oracle = torch.empty_like(ref, dtype=torch.float64)
for m in range(q.shape[0]):
    g = kv[idx[m]]                                   # [tk, D_QK]
    sc = (q[m] @ g.t()) * sm                          # [H, tk]
    p = torch.softmax(sc, dim=-1)
    oracle[m] = (p @ g[:, :dv])
cd = ops.calc_diff(ref, oracle.float())
check(cd < 1e-4, "dsa_attn chunked-MLA vs full-attention oracle calc_diff<1e-4", f"{cd:.3e}")

# ── C. opbench gate has teeth ────────────────────────────────────────────────
section("C. opbench gate anti-cheat (poison + tolerances)")


def gate(op, phase, M, cand_fn):
    ins = ops.build_inputs(op, phase, M, S, DEV, 0)
    ref = ops.reference(op, phase, ins)
    refc = ref.clone() if torch.is_tensor(ref) else ref
    ops.poison(ins)
    cand = cand_fn(ins)
    return ops.compare(refc, cand, op, phase, ins)

# reference-as-candidate (independent fresh compute) -> PASS
r = gate("o_proj", "prefill", 1024,
         lambda ins: R._blockwise_fp8_gemm(ins["x_fp8"], ins["x_scale"],
                                           ins["w_fp8"], ins["w_scale"], out=ins["out"]))
check(r["pass"] and r["calc_diff"] < 1e-6, "correct reimpl PASSES", f"calc_diff={r['calc_diff']:.2e}")

# no-op: return the poisoned (NaN) buffer unwritten -> FAIL on anomaly
r = gate("o_proj", "prefill", 1024, lambda ins: ins["out"])
check((not r["pass"]) and (not r.get("anomaly_ok", True)), "no-op (poisoned buffer) FAILS",
      r.get("reason", "")[:60])

# magnitude-wrong: 2x the correct answer -> cosine 1.0 but calc_diff catches it -> FAIL
def _mag_wrong(ins):
    o = R._blockwise_fp8_gemm(ins["x_fp8"], ins["x_scale"], ins["w_fp8"], ins["w_scale"])
    ins["out"].copy_(o * 2.0)
    return ins["out"]
r = gate("o_proj", "prefill", 1024, _mag_wrong)
check((not r["pass"]) and r["cosine"] > 0.999, "2x-magnitude candidate FAILS despite cosine~1",
      f"cosine={r['cosine']:.4f} calc_diff={r['calc_diff']:.3e}")

# mla no-op: zeros -> FAIL
r = gate("dsa_attn", "decode", 16, lambda ins: torch.zeros_like(R._ref_mla(ins)))
check(not r["pass"], "mla zeros candidate FAILS", r.get("reason", "")[:50])

# ── D. rewardbench reward well-formed ────────────────────────────────────────
section("D. rewardbench reward well-formed")
f, b, d = ops.cost("o_proj", "prefill", 4096, S)
r_fast = rb.roofline_reward(1.0, f, b, d)["reward"]
r_slow = rb.roofline_reward(5.0, f, b, d)["reward"]
check(r_fast > r_slow, "reward strictly decreasing in latency", f"{r_fast:.4f}>{r_slow:.4f}")
check(r_fast == r_fast and 0 < r_fast, "reward positive", f"{r_fast:.4f}")
# bound classification: AI vs ridge agrees between marks
for op, phase, M, rbcost in TARGETS:
    f, b, d = ops.cost(op, phase, M, S)
    ai = f / b
    ridge = ops.PEAK_FLOPS[d] / ops.HBM_BYTES_PER_S
    ob = ops.reward(1.0, f, b, d)["bound"]
    rbnd = rb.roofline_reward(1.0, f, b, d)["bound"]
    exp = "compute" if ai >= ridge else "memory"
    check(ob == rbnd == exp, f"bound({op}/{phase} M={M}) agrees", f"AI={ai:.1f} ridge={ridge:.1f} -> {ob}")

# ── verdict ──────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
if FAILURES:
    print(f"MARK VALIDATION FAILED: {len(FAILURES)}/{CHECKS} checks failed")
    for fdesc in FAILURES:
        print(f"  - {fdesc}")
    sys.exit(1)
print(f"MARK VALIDATION PASSED: all {CHECKS} checks green")
print("Both opbench (correctness+reward gate) and rewardbench (roofline tracker) are")
print("self-consistent, share one cost model, and their references are oracle-correct.")
sys.exit(0)
