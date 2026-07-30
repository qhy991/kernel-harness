#!/usr/bin/env python3
"""CPU-only generated-binary audit for the FlashMLA combine campaign.

Three things must hold before any timing is meaningful:

1. The main kernel is byte-for-byte the round-2 survivor P1. The shared FlashMLA
   worktree carries the round-3 goal's in-flight, macro-guarded edit to
   ``sm100/decode/head64/kernel.cuh``; that edit changes this campaign's source
   hash (hence its build id) but must not change a single main instruction,
   because ``GLM52_COORD_PREFETCH_ACROSS_BUF_WAIT`` is never defined here. The
   audit proves that by comparing against the reference P1 extension built by
   round 2.
2. ``combine_identity`` is a faithful copy of the upstream combine. Its SASS
   must equal the stock ``flash_fwd_mla_combine_kernel`` instruction stream from
   the same reference extension after normalizing only the symbol name, so the
   identity arm really is the "P1 + stock combine" denominator.
3. Every device symbol this campaign compiles carries the goal prefix, and the
   candidate's delta against the identity is reported in the terms the
   hypothesis predeclared: LDG count, FFMA count, register count, spill/stack.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path


CUOBJDUMP = "/usr/local/cuda/bin/cuobjdump"
MAIN_PREFIX = "infini_kernel_glm52_flashmla_sparse_decode"
COMBINE_PREFIX = "infini_kernel_glm52_flashmla_sparse_decode_combine"
STOCK_MAIN = "flash_fwd_splitkv_mla_fp8_sparse_kernel"
STOCK_COMBINE = "flash_fwd_mla_combine_kernel"
# num_sm_parts == 148 selects MAX_SPLITS == 160; BF16 is the production dtype.
PRODUCTION_TAGS = ("Li160", "bfloat16_t")

OPCODE_RE = re.compile(r"^\s*/\*[0-9a-f]{4}\*/\s+(?:@!?\w+\s+)?([A-Z][A-Z0-9._]*)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-p1",
        type=Path,
        required=True,
        help="round-2 P1 extension .so: reference main and stock combine SASS",
    )
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--candidate-symbol-fragment",
        required=True,
        help="e.g. combine_c1 or combine_c2; must follow the combine prefix",
    )
    parser.add_argument(
        "--stock-combine-source",
        type=Path,
        required=True,
        help="upstream csrc/smxx/decode/combine/combine.cu",
    )
    parser.add_argument(
        "--variant-combine-header",
        type=Path,
        required=True,
        help="csrc/glm52_hotspot/combine.cuh",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sass-dir", type=Path, required=True)
    return parser.parse_args()


def dump_sass(path: Path) -> str:
    return subprocess.check_output(
        [CUOBJDUMP, "--dump-sass", str(path)], text=True
    )


# cuobjdump interleaves per-object-file headers between disassembly blocks. They
# must terminate a function body, otherwise the *source path* of the following
# object leaks into the compared instruction stream and two identical kernels
# compare unequal purely because of their file names.
BODY_TERMINATORS = (
    "Fatbin elf code:",
    "Fatbin ptx code:",
    "code for sm_",
    "compile_size =",
    "identifier =",
    "arch =",
    "producer =",
    "host =",
)


def functions(disassembly: str) -> dict[str, list[str]]:
    """Split a cuobjdump --dump-sass listing into mangled name -> body lines."""
    result: dict[str, list[str]] = {}
    current: str | None = None
    for line in disassembly.splitlines():
        stripped = line.strip()
        if stripped.startswith("Function :"):
            current = stripped.removeprefix("Function :").strip()
            result[current] = []
            continue
        if stripped.startswith(BODY_TERMINATORS):
            current = None
            continue
        if current is not None:
            result[current].append(line)
    return result


def pick(names: list[str], *, contains: str, tags: tuple[str, ...] = ()) -> str:
    matches = [
        name
        for name in names
        if contains in name and all(tag in name for tag in tags)
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"expected exactly one symbol containing {contains!r}"
            + (f" and {list(tags)!r}" if tags else "")
            + f"; found {matches!r}"
        )
    return matches[0]


def normalized_body(name: str, lines: list[str]) -> str:
    """Instruction stream with the mangled symbol name removed.

    The identity control differs from the stock combine only by its symbol, so
    any occurrence of either mangled name (in .headerflags / .text / .nv.info
    section directives) is replaced by a placeholder before comparison.
    """
    text = "\n".join(lines)
    text = text.replace(name, "<SYMBOL>")
    return text.rstrip() + "\n"


ASSERT_LINE_RE = re.compile(
    r"IMAD\.MOV\.U32 (R\d+), RZ, RZ, 0x([0-9a-f]+) ;"
)
ASSERT_SOURCE_RE = re.compile(r"FLASH_DEVICE_ASSERT\(my_num_splits <= MAX_SPLITS\)")


def assert_source_line(path: Path) -> int:
    """1-based line of the split-count device assert in a combine source file."""
    matches = [
        index
        for index, line in enumerate(path.read_text().splitlines(), start=1)
        if ASSERT_SOURCE_RE.search(line)
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected one device assert in {path}: {matches}")
    return matches[0]


def instruction_lines(lines: list[str]) -> list[str]:
    return [line for line in lines if OPCODE_RE.match(line)]


def explain_identity_delta(
    stock_lines: list[str],
    identity_lines: list[str],
    stock_assert_line: int,
    identity_assert_line: int,
) -> dict[str, object]:
    """Account for every instruction that differs between stock and identity.

    Renaming the kernel cannot change machine code, but moving the source into a
    different file *does* change ``__LINE__``, and the split-count
    ``FLASH_DEVICE_ASSERT`` passes ``__LINE__`` to ``__assertfail`` as an
    immediate. That single immediate, on a branch taken only when
    ``my_num_splits > MAX_SPLITS`` (never, under the frozen ABI), is the only
    difference this function tolerates -- and only when the two immediates equal
    the two source line numbers.
    """
    stock_instructions = instruction_lines(stock_lines)
    identity_instructions = instruction_lines(identity_lines)
    if len(stock_instructions) != len(identity_instructions):
        raise AssertionError(
            "identity combine has a different instruction count than stock: "
            f"{len(identity_instructions)} != {len(stock_instructions)}"
        )
    differing = [
        (index, stock_text, identity_text)
        for index, (stock_text, identity_text) in enumerate(
            zip(stock_instructions, identity_instructions, strict=True)
        )
        if stock_text != identity_text
    ]
    accounted = []
    for index, stock_text, identity_text in differing:
        stock_match = ASSERT_LINE_RE.search(stock_text)
        identity_match = ASSERT_LINE_RE.search(identity_text)
        if (
            stock_match is None
            or identity_match is None
            or stock_match.group(1) != identity_match.group(1)
            or int(stock_match.group(2), 16) != stock_assert_line
            or int(identity_match.group(2), 16) != identity_assert_line
        ):
            raise AssertionError(
                "identity combine differs from stock in an instruction that is "
                f"not the device-assert __LINE__ immediate: {stock_text.strip()} "
                f"-> {identity_text.strip()}"
            )
        accounted.append(
            {
                "instruction_index": index,
                "stock": stock_text.split(";")[0].strip(),
                "identity": identity_text.split(";")[0].strip(),
                "immediate_is_source_line_of_device_assert": True,
            }
        )
    return {
        "instructions": len(stock_instructions),
        "differing_instructions": len(differing),
        "stock_device_assert_source_line": stock_assert_line,
        "identity_device_assert_source_line": identity_assert_line,
        "differences": accounted,
        "every_difference_is_dead_assert_line_metadata": True,
    }


def opcode_histogram(lines: list[str]) -> Counter[str]:
    histogram: Counter[str] = Counter()
    for line in lines:
        match = OPCODE_RE.match(line)
        if match:
            histogram[match.group(1)] += 1
    return histogram


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    handle_digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            handle_digest.update(chunk)
    return handle_digest.hexdigest()


def res_usage(path: Path) -> dict[str, dict[str, int]]:
    text = subprocess.check_output(
        [CUOBJDUMP, "-res-usage", str(path)], text=True
    )
    usage: dict[str, dict[str, int]] = {}
    name: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Function ") and stripped.endswith(":"):
            name = stripped.removeprefix("Function ").removesuffix(":")
            continue
        if name is not None and stripped.startswith("REG:"):
            usage[name] = {
                key: int(value)
                for key, value in (
                    field.split(":", 1)
                    for field in stripped.split()
                    if ":" in field and field.split(":", 1)[1].isdigit()
                )
            }
            name = None
    return usage


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    sass_dir = args.sass_dir.expanduser().resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite evidence: {output}")

    reference = args.reference_p1.expanduser().resolve()
    identity = args.identity.expanduser().resolve()
    candidate = args.candidate.expanduser().resolve()

    ref_funcs = functions(dump_sass(reference))
    id_funcs = functions(dump_sass(identity))
    cand_funcs = functions(dump_sass(candidate))

    # 1. main kernel must be the unchanged P1 machine code
    ref_main = pick(list(ref_funcs), contains=f"{MAIN_PREFIX}_p1_consumer_scale_main")
    id_main = pick(list(id_funcs), contains=f"{MAIN_PREFIX}_p1_consumer_scale_main")
    cand_main = pick(list(cand_funcs), contains=f"{MAIN_PREFIX}_p1_consumer_scale_main")
    ref_main_body = normalized_body(ref_main, ref_funcs[ref_main])
    id_main_body = normalized_body(id_main, id_funcs[id_main])
    cand_main_body = normalized_body(cand_main, cand_funcs[cand_main])
    if id_main_body != ref_main_body:
        raise AssertionError(
            "identity arm main SASS differs from the reference P1 main"
        )
    if cand_main_body != ref_main_body:
        raise AssertionError(
            "candidate arm main SASS differs from the reference P1 main"
        )

    # 2. identity combine must reproduce the stock combine instruction stream
    ref_combine = pick(
        list(ref_funcs), contains=STOCK_COMBINE, tags=PRODUCTION_TAGS
    )
    id_combine = pick(
        list(id_funcs),
        contains=f"{COMBINE_PREFIX}_identity",
        tags=PRODUCTION_TAGS,
    )
    # A bucket-specialized candidate instantiates more than one production
    # kernel (one per stage depth). Report every one; retain the deepest as the
    # representative SASS.
    fragment = args.candidate_symbol_fragment
    if not fragment.startswith("combine_"):
        raise AssertionError(f"candidate fragment must name a combine variant: {fragment}")
    cand_combines = sorted(
        name
        for name in cand_funcs
        if f"{COMBINE_PREFIX}_{fragment.removeprefix('combine_')}" in name
        and all(tag in name for tag in PRODUCTION_TAGS)
    )
    if not cand_combines:
        raise AssertionError(f"no production candidate combine matched {fragment!r}")
    cand_combine = cand_combines[-1]
    ref_combine_body = normalized_body(ref_combine, ref_funcs[ref_combine])
    id_combine_body = normalized_body(id_combine, id_funcs[id_combine])
    cand_combine_body = normalized_body(cand_combine, cand_funcs[cand_combine])
    identity_delta = explain_identity_delta(
        ref_funcs[ref_combine],
        id_funcs[id_combine],
        assert_source_line(args.stock_combine_source.expanduser().resolve()),
        assert_source_line(args.variant_combine_header.expanduser().resolve()),
    )

    # 3. every compiled device symbol carries the goal prefix
    unprefixed = {
        label: sorted(
            name
            for name in names
            if MAIN_PREFIX not in name
        )
        for label, names in (
            ("identity", list(id_funcs)),
            ("candidate", list(cand_funcs)),
        )
    }
    for label, names in unprefixed.items():
        if names:
            raise AssertionError(f"{label} exposes unprefixed device symbols: {names}")

    id_hist = opcode_histogram(id_funcs[id_combine])
    cand_hist = opcode_histogram(cand_funcs[cand_combine])
    id_usage = res_usage(identity)
    cand_usage = res_usage(candidate)

    sass_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, body in (
        ("reference_p1_main.sass", ref_main_body),
        ("reference_stock_combine.sass", ref_combine_body),
        ("combine_identity.sass", id_combine_body),
        ("combine_candidate.sass", cand_combine_body),
    ):
        path = sass_dir / name
        if path.exists():
            raise RuntimeError(f"refusing to overwrite evidence: {path}")
        path.write_text(body)
        written[name] = {"path": str(path), "sha256": digest(body)}

    interesting = sorted(
        set(id_hist) | set(cand_hist),
        key=lambda opcode: -abs(cand_hist[opcode] - id_hist[opcode]),
    )
    evidence = {
        "schema_version": 1,
        "tool": subprocess.check_output([CUOBJDUMP, "--version"], text=True).strip(),
        "extensions": {
            "reference_p1": {
                "path": str(reference),
                "sha256": sha256_file(reference),
                "role": "round-2 P1 build: reference main and stock combine",
            },
            "identity": {"path": str(identity), "sha256": sha256_file(identity)},
            "candidate": {"path": str(candidate), "sha256": sha256_file(candidate)},
        },
        "main_kernel": {
            "symbol": id_main,
            "sass_sha256": digest(id_main_body),
            "identity_matches_reference_p1": True,
            "candidate_matches_reference_p1": True,
            "ptx3_inflight_edit_is_inert": True,
            "note": (
                "the shared FlashMLA worktree carries the round-3 goal's "
                "macro-guarded kernel.cuh edit; it changes the source hash but "
                "not one main instruction because the macro is undefined here"
            ),
        },
        "combine_kernel": {
            "reference_stock_symbol": ref_combine,
            "identity_symbol": id_combine,
            "candidate_symbols": cand_combines,
            "candidate_representative_symbol": cand_combine,
            "identity_is_faithful_copy_of_stock": True,
            "identity_vs_stock_delta": identity_delta,
            "stock_combine_sass_sha256": digest(ref_combine_body),
            "candidate_combine_sass_sha256": digest(cand_combine_body),
            "identity_instructions": sum(id_hist.values()),
            "candidate_instructions": sum(cand_hist.values()),
            "opcode_delta": {
                opcode: {
                    "identity": id_hist[opcode],
                    "candidate": cand_hist[opcode],
                    "delta": cand_hist[opcode] - id_hist[opcode],
                }
                for opcode in interesting
                if cand_hist[opcode] != id_hist[opcode]
            },
            "resources": {
                "identity": id_usage.get(id_combine, {}),
                "candidate_by_symbol": {
                    name: cand_usage.get(name, {}) for name in cand_combines
                },
            },
        },
        "all_device_symbols_prefixed": True,
        "retained_sass": written,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "main_kernel": evidence["main_kernel"],
                "combine_kernel": {
                    key: value
                    for key, value in evidence["combine_kernel"].items()
                    if key != "opcode_delta"
                },
                "opcode_delta": evidence["combine_kernel"]["opcode_delta"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
