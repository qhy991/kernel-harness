#!/usr/bin/env python3
"""Attribute the survivor's warp stalls to SASS PCs and instruction text."""
import json, re, sys, collections
sys.path.insert(0, "/opt/nvidia/nsight-compute/2026.1.1/extras/python")
import ncu_report

REASONS = ["long_scoreboard","short_scoreboard","wait","barrier","membar",
           "mio_throttle","lg_throttle","math_pipe_throttle","not_selected",
           "dispatch_stall","drain","no_instructions","branch_resolving","selected"]

run = sys.argv[1]
sass_path = sys.argv[2]
ctx = ncu_report.load_report(f"{run}/reports/source_candidate.ncu-rep")
act = ctx.range_by_idx(0).action_by_idx(0)

# addr -> instruction text from the retained cubin disassembly
sass = {}
for line in open(sass_path, errors="ignore"):
    m = re.match(r"\s+/\*([0-9a-f]+)\*/\s+(.*?);", line)
    if m:
        sass[int(m.group(1), 16)] = m.group(2).strip()

totals = collections.Counter()
per_pc = collections.defaultdict(collections.Counter)
for reason in REASONS:
    name = f"smsp__pcsamp_warps_issue_stalled_{reason}"
    if name not in act.metric_names():
        continue
    m = act[name]
    for i in range(m.num_instances()):
        v = m.value(i)
        if not v:
            continue
        pc = m.correlation_ids().value(i)
        totals[reason] += v
        per_pc[pc][reason] += v

grand = sum(totals.values())
out = {"grand_total_samples": grand,
       "by_reason": {k: {"samples": v, "pct": 100.0*v/grand} for k, v in totals.most_common()},
       "top_pcs": []}
ranked = sorted(per_pc.items(), key=lambda kv: -sum(kv[1].values()))
for pc, counter in ranked[:25]:
    tot = sum(counter.values())
    try:
        info = act.source_info(pc)
        loc = f"{info.file_name()}:{info.line()}" if info else None
    except Exception:
        loc = None
    # cubin offsets in the retained SASS dump are relative to the function start
    out["top_pcs"].append({
        "pc": pc, "pct_of_all_stalls": 100.0*tot/grand, "samples": tot,
        "source": loc, "sass": sass.get(pc),
        "reasons": {k: v for k, v in counter.most_common()},
    })
print(json.dumps(out, indent=2))
