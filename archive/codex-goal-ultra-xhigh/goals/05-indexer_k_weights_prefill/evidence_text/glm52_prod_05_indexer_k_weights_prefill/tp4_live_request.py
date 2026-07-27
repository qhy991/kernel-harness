#!/usr/bin/env python3
"""HTTP helper for the locked GLM-5.2 TP4 live indexer diagnostic.

This helper deliberately uses only the Python standard library.  It records the
exact request and every HTTP response so a failed distributed launch is still
reviewable without relying on terminal scrollback.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import re
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


PROMPT_TOKENS = 4096
REQUEST_COUNT = 4
TOKEN_MODULUS = 153_000
MODULE_SCOPE_RE = re.compile(r"['\"]Module['\"]\s*:\s*['\"]([^'\"]+)['\"]")


def _write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _http_request(
    base_url: str,
    endpoint: str,
    *,
    method: str,
    payload: Any | None = None,
    timeout_s: float,
) -> dict[str, Any]:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{endpoint}",
        data=body,
        headers=headers,
        method=method,
    )
    started = time.perf_counter()
    result: dict[str, Any] = {
        "endpoint": endpoint,
        "method": method,
        "request_payload": payload,
        "url": request.full_url,
    }

    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw_body = response.read()
            result["status_code"] = response.status
            result["response_headers"] = dict(response.headers.items())
    except urllib.error.HTTPError as error:
        raw_body = error.read()
        result["status_code"] = error.code
        result["response_headers"] = dict(error.headers.items())
        result["exception"] = repr(error)
    except Exception as error:  # Preserve transport failures as evidence.
        result["status_code"] = None
        result["response_headers"] = {}
        result["response_body"] = ""
        result["exception"] = repr(error)
        result["elapsed_s"] = time.perf_counter() - started
        return result

    decoded_body = raw_body.decode("utf-8", errors="replace")
    result["response_body"] = decoded_body
    if decoded_body:
        try:
            result["response_json"] = json.loads(decoded_body)
        except json.JSONDecodeError:
            pass
    result["elapsed_s"] = time.perf_counter() - started
    return result


def _request_succeeded(result: dict[str, Any]) -> bool:
    status_code = result.get("status_code")
    if not isinstance(status_code, int) or not 200 <= status_code < 300:
        return False
    response_json = result.get("response_json")
    if isinstance(response_json, dict) and response_json.get("success") is False:
        return False
    return True


def _process_alive(pid: int) -> bool:
    # The shell is the nsys leader's parent, so a failed launch can remain as a
    # zombie until cleanup calls wait(). kill(pid, 0) treats that state as alive.
    try:
        stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        state = stat_text.rsplit(")", 1)[1].strip().split()[0]
        if state == "Z":
            return False
    except (FileNotFoundError, IndexError, PermissionError, OSError):
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _find_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _check_port_free(host: str, port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
    except OSError:
        return False
    return True


def _wait_health(args: argparse.Namespace) -> int:
    deadline = time.monotonic() + args.timeout_s
    attempt_count = 0
    last_result: dict[str, Any] | None = None

    while True:
        attempt_count += 1
        last_result = _http_request(
            args.base_url,
            "/health_generate",
            method="GET",
            timeout_s=args.request_timeout_s,
        )
        if _request_succeeded(last_result):
            _write_json(
                args.output,
                {
                    "attempt_count": attempt_count,
                    "healthy": True,
                    "last_response": last_result,
                },
            )
            return 0

        if not _process_alive(args.process_pid):
            _write_json(
                args.output,
                {
                    "attempt_count": attempt_count,
                    "healthy": False,
                    "last_response": last_result,
                    "reason": "tracked nsys process exited before health_generate",
                },
            )
            return 1

        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0:
            _write_json(
                args.output,
                {
                    "attempt_count": attempt_count,
                    "healthy": False,
                    "last_response": last_result,
                    "reason": "health_generate timeout",
                },
            )
            return 1
        time.sleep(min(args.poll_interval_s, remaining_s))


def _deterministic_request_payload(request_index: int) -> dict[str, Any]:
    input_ids = [
        1000 + (request_index * 104729 + index * 7919) % TOKEN_MODULUS
        for index in range(PROMPT_TOKENS)
    ]
    assert len(input_ids) == PROMPT_TOKENS
    return {
        "input_ids": input_ids,
        "rid": f"tp4-dp4-ep4-indexer-prefill-m4096-{request_index}",
        "routed_dp_rank": request_index,
        "sampling_params": {
            "ignore_eos": True,
            "max_new_tokens": 1,
            "temperature": 0,
        },
        "stream": False,
    }


def _server_info_mismatches(
    server_info: Any,
    *,
    expected_model: str,
    expected_revision: str,
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    expected_scalars = {
        # The launch supplies global 16384; SGLang resolves the per-DP-rank
        # server value to 4096 under DP4.
        "attention_backend": "dsa",
        "chunked_prefill_size": 4096,
        "context_length": 8192,
        "disable_flashinfer_autotune": True,
        "disable_radix_cache": True,
        "dp_size": 4,
        "dsa_prefill_backend": "trtllm",
        "dsa_topk_backend": "sgl-kernel",
        "enable_deepseek_v4_fp4_indexer": False,
        "enable_dp_attention": True,
        "enable_layerwise_nvtx_marker": True,
        "ep_size": 4,
        "load_format": "dummy",
        "kv_cache_dtype": "fp8_e4m3",
        "max_running_requests": 4,
        "max_prefill_tokens": 4096,
        "max_total_tokens": 8192,
        "mem_fraction_static": 0.8,
        "moe_a2a_backend": "deepep",
        "page_size": 64,
        "prefill_max_requests": 1,
        "quantization": "modelopt_fp4",
        "revision": expected_revision,
        "skip_tokenizer_init": True,
        "tp_size": 4,
        "trust_remote_code": True,
    }
    if not isinstance(server_info, dict):
        mismatches.append(
            {"field": "response_json", "expected": "object", "observed": None}
        )
    else:
        for field, expected in expected_scalars.items():
            observed = server_info.get(field)
            if observed != expected:
                mismatches.append(
                    {"field": field, "expected": expected, "observed": observed}
                )

        model_path = server_info.get("model_path")
        if model_path != expected_model:
            mismatches.append(
                {
                    "field": "model_path",
                    "expected": expected_model,
                    "observed": model_path,
                }
            )

        cuda_graph_config = server_info.get("cuda_graph_config")
        prefill_backend = None
        decode_backend = None
        if isinstance(cuda_graph_config, dict):
            prefill = cuda_graph_config.get("prefill")
            decode = cuda_graph_config.get("decode")
            if isinstance(prefill, dict):
                prefill_backend = prefill.get("backend")
            if isinstance(decode, dict):
                decode_backend = decode.get("backend")
        for phase, observed in (
            ("prefill", prefill_backend),
            ("decode", decode_backend),
        ):
            if observed != "disabled":
                mismatches.append(
                    {
                        "field": f"cuda_graph_config.{phase}.backend",
                        "expected": "disabled",
                        "observed": observed,
                    }
                )
    return mismatches


def _capture_server_info(args: argparse.Namespace) -> int:
    result = _http_request(
        args.base_url,
        "/server_info",
        method="GET",
        timeout_s=args.timeout_s,
    )
    mismatches = _server_info_mismatches(
        result.get("response_json"),
        expected_model=args.expected_model,
        expected_revision=args.expected_revision,
    )

    resolved_config_ok = _request_succeeded(result) and not mismatches
    result["validation"] = {
        "mismatches": mismatches,
        "resolved_config_ok": resolved_config_ok,
    }
    _write_json(args.output, result)
    return 0 if resolved_config_ok else 1


def _extract_prompt_tokens(generate_result: dict[str, Any]) -> int | None:
    response_json = generate_result.get("response_json")
    if not isinstance(response_json, dict):
        return None
    meta_info = response_json.get("meta_info")
    if not isinstance(meta_info, dict):
        return None
    prompt_tokens = meta_info.get("prompt_tokens")
    return prompt_tokens if isinstance(prompt_tokens, int) else None


def _run_profiled_request(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    request_payloads = [
        _deterministic_request_payload(request_index)
        for request_index in range(REQUEST_COUNT)
    ]
    start_payload = {"activities": ["CUDA_PROFILER"]}
    _write_json(output_dir / "generate_payloads.json", request_payloads)
    _write_json(output_dir / "start_profile_payload.json", start_payload)

    start_result = _http_request(
        args.base_url,
        "/start_profile",
        method="POST",
        payload=start_payload,
        timeout_s=args.control_timeout_s,
    )
    _write_json(output_dir / "start_profile_response.json", start_result)
    start_ok = _request_succeeded(start_result)

    generate_results: list[dict[str, Any]] = []
    generate_ok = False
    prompt_tokens: list[int | None] = []
    stop_result: dict[str, Any] | None = None
    stop_ok = False

    if start_ok:
        try:
            # Four simultaneous 4096-token requests give the DP4 scheduler one
            # local M4096 prefill on each rank. The trace gate below still
            # requires the Q/K M4096 grid signature on all four devices.
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=REQUEST_COUNT
            ) as executor:
                futures = [
                    executor.submit(
                        _http_request,
                        args.base_url,
                        "/generate",
                        method="POST",
                        payload=payload,
                        timeout_s=args.generate_timeout_s,
                    )
                    for payload in request_payloads
                ]
                generate_results = [future.result() for future in futures]
            _write_json(output_dir / "generate_responses.json", generate_results)
            prompt_tokens = [
                _extract_prompt_tokens(result) for result in generate_results
            ]
            generate_ok = all(
                _request_succeeded(result) and observed == PROMPT_TOKENS
                for result, observed in zip(generate_results, prompt_tokens)
            )
        finally:
            # A failed generation must still close cudaProfilerApi capture.
            stop_result = _http_request(
                args.base_url,
                "/stop_profile",
                method="POST",
                timeout_s=args.control_timeout_s,
            )
            _write_json(output_dir / "stop_profile_response.json", stop_result)
            stop_ok = _request_succeeded(stop_result)

    summary = {
        "expected_prompt_tokens_per_request": PROMPT_TOKENS,
        "expected_request_count": REQUEST_COUNT,
        "generate_ok": generate_ok,
        "observed_prompt_tokens_per_request": prompt_tokens,
        "observed_request_count": len(generate_results),
        "start_profile_ok": start_ok,
        "stop_profile_ok": stop_ok,
    }
    if not start_ok:
        summary["reason"] = "start_profile did not succeed; generate was not sent"
    elif len(prompt_tokens) != REQUEST_COUNT or any(
        observed != PROMPT_TOKENS for observed in prompt_tokens
    ):
        summary["reason"] = (
            "generate responses did not confirm four accepted 4096-token prompts"
        )
    elif not stop_ok:
        summary["reason"] = "stop_profile did not succeed"
    _write_json(output_dir / "request_status.json", summary)
    return 0 if start_ok and generate_ok and stop_ok else 1


def _stop_profile(args: argparse.Namespace) -> int:
    result = _http_request(
        args.base_url,
        "/stop_profile",
        method="POST",
        timeout_s=args.timeout_s,
    )
    _write_json(args.output, result)
    return 0 if _request_succeeded(result) else 1


def _trace_int(row: dict[str, str], field: str) -> int | None:
    try:
        return int(row.get(field, ""))
    except (TypeError, ValueError):
        return None


def _nvtx_scope(row: dict[str, str]) -> str:
    name = row.get("Name", "")
    return name.rsplit("/", 1)[0] if "/" in name else ""


def _nvtx_module(row: dict[str, str]) -> str:
    match = MODULE_SCOPE_RE.search(_nvtx_scope(row))
    return match.group(1) if match is not None else ""


def _indexer_parent(row: dict[str, str], child: str | None = None) -> str:
    module = _nvtx_module(row)
    if child is not None:
        suffix = f".{child}"
        if not module.endswith(suffix):
            return ""
        module = module[: -len(suffix)]
    return module if module.endswith(".self_attn.indexer") else ""


def _analyze_trace_rows(
    rows: list[dict[str, str]], expected_devices: int
) -> dict[str, Any]:
    q_aliases = (
        "fused_q_indexer_rope_hadamard_quant",
        "main_q_indexer_rope_first_quant",
    )
    k_aliases = (
        "fused_k_indexer_norm_rope_store",
        "dpsk_v32_k_indexer_norm_rope_store_p64",
    )

    q_rows = [
        row
        for row in rows
        if any(alias in row.get("Name", "") for alias in q_aliases)
    ]
    k_rows = [
        row
        for row in rows
        if any(alias in row.get("Name", "") for alias in k_aliases)
    ]
    gemm_rows = [
        row for row in rows if "nvjet_sm100_tst" in row.get("Name", "")
    ]

    def scope_contains(row: dict[str, str], *tokens: str) -> bool:
        scope = _nvtx_scope(row).lower()
        return bool(scope) and all(token in scope for token in tokens)

    # Q launches 8 blocks/token and K launches one block per four tokens. The
    # GEMM grids are not globally unique in a complete model, so attribution
    # additionally requires the layerwise Indexer NVTX scope and direct temporal
    # adjacency to its specialized Q/K kernel on the same stream.
    q_m4096_rows = [
        row
        for row in q_rows
        if row.get("GrdX") == "32768"
        and scope_contains(row, "self_attn.indexer")
    ]
    k_m4096_rows = [
        row
        for row in k_rows
        if row.get("GrdX") == "1024"
        and scope_contains(row, "self_attn.indexer")
    ]
    wq_m4096_rows = [
        row
        for row in gemm_rows
        if row.get("GrdX") == "512"
        and scope_contains(row, "self_attn.indexer.wq_b")
    ]
    wk_m4096_rows = [
        row
        for row in gemm_rows
        if row.get("GrdX") == "128"
        and scope_contains(row, "self_attn.indexer.wk_weights_proj")
    ]

    def context_key(row: dict[str, str]) -> tuple[str, str]:
        return row.get("Device", ""), row.get("Ctx", "")

    def stream_key(row: dict[str, str]) -> tuple[str, str, str]:
        return (*context_key(row), row.get("Strm", ""))

    events_by_stream: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        if _trace_int(row, "Start (ns)") is not None:
            events_by_stream.setdefault(stream_key(row), []).append(row)
    for stream_rows in events_by_stream.values():
        stream_rows.sort(key=lambda row: _trace_int(row, "Start (ns)") or -1)

    q_ids = {id(row) for row in q_m4096_rows}
    k_ids = {id(row) for row in k_m4096_rows}
    wq_ids = {id(row) for row in wq_m4096_rows}
    wk_ids = {id(row) for row in wk_m4096_rows}

    def compact_pair(
        predecessor: dict[str, str], target: dict[str, str]
    ) -> dict[str, Any]:
        predecessor_start = _trace_int(predecessor, "Start (ns)")
        predecessor_duration = _trace_int(predecessor, "Duration (ns)")
        target_start = _trace_int(target, "Start (ns)")
        gap_ns = None
        if (
            predecessor_start is not None
            and predecessor_duration is not None
            and target_start is not None
        ):
            gap_ns = target_start - predecessor_start - predecessor_duration
        return {
            "context": list(context_key(target)),
            "stream": target.get("Strm", ""),
            "predecessor_corr_id": predecessor.get("CorrId", ""),
            "predecessor_grid_x": predecessor.get("GrdX", ""),
            "predecessor_name": predecessor.get("Name", ""),
            "predecessor_module": _nvtx_module(predecessor),
            "predecessor_to_target_gap_ns": gap_ns,
            "target_corr_id": target.get("CorrId", ""),
            "target_grid_x": target.get("GrdX", ""),
            "target_name": target.get("Name", ""),
            "target_module": _nvtx_module(target),
            "target_nvtx_scope": _nvtx_scope(target),
            "target_start_ns": target_start,
        }

    q_pairs: list[dict[str, Any]] = []
    k_pairs: list[dict[str, Any]] = []

    def same_indexer_parent(
        predecessor: dict[str, str], target: dict[str, str], child: str
    ) -> bool:
        predecessor_parent = _indexer_parent(predecessor, child)
        return bool(predecessor_parent) and predecessor_parent == _indexer_parent(
            target
        )

    for stream_rows in events_by_stream.values():
        for index, row in enumerate(stream_rows):
            if index == 0:
                continue
            predecessor = stream_rows[index - 1]
            if (
                id(row) in q_ids
                and id(predecessor) in wk_ids
                and same_indexer_parent(predecessor, row, "wk_weights_proj")
            ):
                q_pairs.append(compact_pair(predecessor, row))
            if (
                id(row) in k_ids
                and id(predecessor) in wq_ids
                and same_indexer_parent(predecessor, row, "wq_b")
            ):
                k_pairs.append(compact_pair(predecessor, row))

    q_pairs_by_context: dict[tuple[str, str], list[dict[str, Any]]] = {}
    k_pairs_by_context: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for pair in q_pairs:
        q_pairs_by_context.setdefault(tuple(pair["context"]), []).append(pair)
    for pair in k_pairs:
        k_pairs_by_context.setdefault(tuple(pair["context"]), []).append(pair)

    # The paired stage-2 kernels should launch close together. Two milliseconds
    # is deliberately loose relative to the reconstructed capture while still
    # preventing an adjacent model layer from satisfying the same invocation.
    max_stage2_start_delta_ns = 2_000_000
    schedule_matches: list[dict[str, Any]] = []
    for context in sorted(set(q_pairs_by_context) & set(k_pairs_by_context)):
        for q_pair in q_pairs_by_context[context]:
            for k_pair in k_pairs_by_context[context]:
                if q_pair["stream"] == k_pair["stream"]:
                    continue
                if q_pair["target_module"] != k_pair["target_module"]:
                    continue
                q_start = q_pair["target_start_ns"]
                k_start = k_pair["target_start_ns"]
                if q_start is None or k_start is None:
                    continue
                delta_ns = abs(q_start - k_start)
                if delta_ns <= max_stage2_start_delta_ns:
                    schedule_matches.append(
                        {
                            "context": list(context),
                            "k_pair": k_pair,
                            "q_pair": q_pair,
                            "stage2_start_delta_ns": delta_ns,
                        }
                    )
                    break
            else:
                continue
            break

    exact_schedule_contexts = sorted(
        {tuple(match["context"]) for match in schedule_matches}
    )
    exact_schedule_devices = sorted(
        {context[0] for context in exact_schedule_contexts if context[0]}
    )
    indexer_kernel_name_counts: dict[str, int] = {}
    for row in rows:
        if ".self_attn.indexer" not in _nvtx_module(row):
            continue
        kernel_name = row.get("Name", "").rsplit("/", 1)[-1]
        indexer_kernel_name_counts[kernel_name] = (
            indexer_kernel_name_counts.get(kernel_name, 0) + 1
        )
    checks = {
        "expected_distinct_devices_match_full_dual_stream_schedule": len(
            exact_schedule_devices
        )
        >= expected_devices,
        "expected_contexts_match_full_dual_stream_schedule": len(
            exact_schedule_contexts
        )
        >= expected_devices,
        "k_m4096_indexer_scoped_grid_present": len(k_m4096_rows)
        >= expected_devices,
        "q_m4096_indexer_scoped_grid_present": len(q_m4096_rows)
        >= expected_devices,
        "wk_m4096_indexer_scoped_grid_present": len(wk_m4096_rows)
        >= expected_devices,
        "wq_m4096_indexer_scoped_grid_present": len(wq_m4096_rows)
        >= expected_devices,
    }
    analysis = {
        "checks": checks,
        "exact_schedule_contexts": [
            list(context) for context in exact_schedule_contexts
        ],
        "exact_schedule_devices": exact_schedule_devices,
        "expected_devices": expected_devices,
        "indexer_kernel_name_counts": dict(
            sorted(indexer_kernel_name_counts.items())
        ),
        "k_kernel_count": len(k_rows),
        "k_m4096_grid_count": len(k_m4096_rows),
        "k_preceded_by_wq_pairs": k_pairs,
        "max_stage2_start_delta_ns": max_stage2_start_delta_ns,
        "q_kernel_count": len(q_rows),
        "q_m4096_grid_count": len(q_m4096_rows),
        "q_preceded_by_wk_pairs": q_pairs,
        "schedule_matches": schedule_matches,
        "wk_m4096_grid_count": len(wk_m4096_rows),
        "wq_m4096_grid_count": len(wq_m4096_rows),
        "trace_reachability_ok": all(checks.values()),
    }
    return analysis


def _analyze_trace(args: argparse.Namespace) -> int:
    with Path(args.csv).open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    analysis = _analyze_trace_rows(rows, args.expected_devices)
    _write_json(args.output, analysis)
    return 0 if analysis["trace_reachability_ok"] else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    find_port = subparsers.add_parser("find-free-port")
    find_port.add_argument("--host", default="127.0.0.1")

    check_port = subparsers.add_parser("check-port")
    check_port.add_argument("--host", default="127.0.0.1")
    check_port.add_argument("--port", type=int, required=True)

    wait_health = subparsers.add_parser("wait-health")
    wait_health.add_argument("--base-url", required=True)
    wait_health.add_argument("--output", required=True)
    wait_health.add_argument("--process-pid", type=int, required=True)
    wait_health.add_argument("--timeout-s", type=float, default=1800.0)
    wait_health.add_argument("--poll-interval-s", type=float, default=2.0)
    wait_health.add_argument("--request-timeout-s", type=float, default=5.0)

    server_info = subparsers.add_parser("capture-server-info")
    server_info.add_argument("--base-url", required=True)
    server_info.add_argument("--expected-model", required=True)
    server_info.add_argument("--expected-revision", required=True)
    server_info.add_argument("--output", required=True)
    server_info.add_argument("--timeout-s", type=float, default=60.0)

    profile = subparsers.add_parser("run-profiled-request")
    profile.add_argument("--base-url", required=True)
    profile.add_argument("--output-dir", required=True)
    profile.add_argument("--control-timeout-s", type=float, default=120.0)
    profile.add_argument("--generate-timeout-s", type=float, default=1800.0)

    stop = subparsers.add_parser("stop-profile")
    stop.add_argument("--base-url", required=True)
    stop.add_argument("--output", required=True)
    stop.add_argument("--timeout-s", type=float, default=15.0)

    analyze = subparsers.add_parser("analyze-trace")
    analyze.add_argument("--csv", required=True)
    analyze.add_argument("--output", required=True)
    analyze.add_argument("--expected-devices", type=int, default=4)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.command == "find-free-port":
        print(_find_free_port(args.host))
        return 0
    if args.command == "check-port":
        return 0 if _check_port_free(args.host, args.port) else 1
    if args.command == "wait-health":
        return _wait_health(args)
    if args.command == "capture-server-info":
        return _capture_server_info(args)
    if args.command == "run-profiled-request":
        return _run_profiled_request(args)
    if args.command == "stop-profile":
        return _stop_profile(args)
    if args.command == "analyze-trace":
        return _analyze_trace(args)
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
