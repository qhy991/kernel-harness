#!/usr/bin/env python3
"""Extract the decisive round-2 metrics from the NCU reports."""
import json, sys
sys.path.insert(0, "/opt/nvidia/nsight-compute/2026.1.1/extras/python")
import ncu_report

KEYS = [
    "gpu__time_duration.sum",
    "dram__bytes_read.sum", "dram__bytes_write.sum",
    "dram__bytes_read.sum.pct_of_peak_sustained_elapsed",
    "dram__bytes_read.sum.per_second",
    "dram__bytes_write.sum.per_second",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__t_sector_hit_rate.pct",
    "l1tex__t_sector_hit_rate.pct",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "launch__grid_size", "launch__block_size", "launch__cluster_dim_x",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block_dynamic", "launch__shared_mem_per_block_static",
    "launch__waves_per_multiprocessor",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "sm__cycles_elapsed.avg", "sm__cycles_active.avg",
]
STALLS = [
 "long_scoreboard","short_scoreboard","wait","barrier","membar","sleeping",
 "math_pipe_throttle","mio_throttle","lg_throttle","tex_throttle",
 "not_selected","branch_resolving","dispatch_stall","drain",
 "no_instruction","misc","selected",
]

def get(action, name):
    m = action[name] if name in action.metric_names() else None
    if m is None:
        return None
    try:
        return m.value()
    except Exception:
        return None

out = {}
for arm in ("candidate", "stock"):
    ctx = ncu_report.load_report(f"{sys.argv[1]}/reports/full_{arm}.ncu-rep")
    rng = ctx.range_by_idx(0)
    act = rng.action_by_idx(0)
    rec = {"kernel": act.name(), "demangled": act.name(ncu_report.IAction.NameBase_DEMANGLED)}
    for k in KEYS:
        rec[k] = get(act, k)
    st = {}
    for s in STALLS:
        st[s] = get(act, f"smsp__average_warps_issue_stalled_{s}_per_issue_active.ratio")
    rec["stalls_per_issue_active"] = st
    out[arm] = rec

# NCU rule-engine speedup estimates for the survivor
ctx = ncu_report.load_report(f"{sys.argv[1]}/reports/full_candidate.ncu-rep")
act = ctx.range_by_idx(0).action_by_idx(0)
rules = []
for r in act.rule_results():
    entry = {}
    for attr in ("rule_name", "focus_metric_value", "speedup_value", "speedup_type"):
        try:
            entry[attr] = getattr(r, attr)()
        except Exception:
            pass
    rules.append(entry)
out["candidate_rules"] = rules
print(json.dumps(out, indent=2, default=str))
