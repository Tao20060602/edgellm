#!/usr/bin/env python3
"""Run a pinned Wikitext-2 perplexity gate for quantized KV-cache types."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import platform
import re
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import bench
import fetch_dataset


ROOT = Path(__file__).resolve().parents[1]
QUALITY_LOCK = ROOT / "locks" / "quality.json"
MODEL_LOCK = ROOT / "locks" / "model.json"
UPSTREAM_LOCK = ROOT / "locks" / "upstream.json"
SCHEMA_VERSION = 1
FLOAT = r"([+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|nan|inf(?:inity)?))"
PPL_RE = re.compile(r"Final estimate:\s*PPL\s*=\s*" + FLOAT + r"\s*\+/-\s*" + FLOAT, re.IGNORECASE)
CHUNK_RE = re.compile(
    r"\b(?:calculating|computing)\s+(?:perplexity\s+)?over\s+(\d+)\s+chunks\s*,\s*"
    r"n_ctx\s*=\s*(\d+)\s*,\s*batch_size\s*=\s*(\d+)",
    re.IGNORECASE,
)
CUDA_RE = re.compile(r"\bfound\s+(\d+)\s+CUDA\s+devices\b", re.IGNORECASE)
OFFLOAD_RE = re.compile(r"\boffloaded\s+(\d+)\s*/\s*(\d+)\s+layers\s+to\s+GPU\b", re.IGNORECASE)
KV_TYPES_RE = re.compile(r"\bK\s*\(\s*([a-zA-Z0-9_]+)\s*\).*?\bV\s*\(\s*([a-zA-Z0-9_]+)\s*\)", re.IGNORECASE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, help="path to the pinned-build llama-perplexity executable")
    parser.add_argument("--model", required=True, help="path to the locked Qwen3-0.6B Q8_0 GGUF")
    parser.add_argument("--upstream", required=True, help="clean llama.cpp checkout at the locked source revision")
    parser.add_argument("--dataset", required=True, help="verified local wiki.test.raw member")
    parser.add_argument("--out", required=True, help="new evidence directory; existing paths are never overwritten")
    parser.add_argument("--ctx", type=positive_int, default=2048, help="context size (the predeclared gate uses 2048)")
    parser.add_argument("--chunks", type=positive_int, default=8, help="maximum dataset chunks (the gate uses 8)")
    parser.add_argument("--gpu-layers", type=nonnegative_int, default=99)
    parser.add_argument("--threads", type=positive_int, default=4)
    parser.add_argument("--timeout", type=positive_int, default=300, help="per-case timeout in seconds")
    return parser


def positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected an integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return result


def nonnegative_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected an integer") from exc
    if result < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return result


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def command_for(binary: Path, model: Path, dataset: Path, args: argparse.Namespace, kv_type: str, lock: dict[str, Any]) -> list[str]:
    fixed = lock["evaluation"]
    return [
        str(binary),
        "-m", str(model),
        "-f", str(dataset),
        "-c", str(args.ctx),
        "--chunks", str(args.chunks),
        "-ctk", kv_type,
        "-ctv", kv_type,
        "-fa", fixed["flash_attention"],
        "-ngl", str(args.gpu_layers),
        "-t", str(args.threads),
        "-b", str(fixed["batch_tokens"]),
        "-ub", str(fixed["ubatch_tokens"]),
        "-v",
    ]


def create_metadata(args: argparse.Namespace, paths: dict[str, Path]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "starting",
        "created_at": bench.utc_now(),
        "requested": {
            "binary": str(paths["binary"]),
            "model": str(paths["model"]),
            "upstream": str(paths["upstream"]),
            "dataset": str(paths["dataset"]),
            "output": str(paths["out"]),
            "context_tokens": args.ctx,
            "chunks": args.chunks,
            "gpu_layers": args.gpu_layers,
            "threads": args.threads,
            "timeout_seconds_per_case": args.timeout,
        },
        "environment": {
            "hostname": socket.gethostname(),
            "python_version": sys.version,
            "platform": platform.platform(),
            "system": platform.system(),
            "machine": platform.machine(),
            "os_name": os.name,
        },
        "provenance": {},
        "limitations": [
            "The gate evaluates only the first configured context chunks from Wikitext-2; it is not a full-set or long-context quality result.",
            "The same-run f16/f16 comparison is a point-estimate gate; reported PPL uncertainty is retained but not folded into the threshold.",
            "No logits-dump file is requested or written.",
        ],
    }


def preflight(args: argparse.Namespace, paths: dict[str, Path], metadata: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    lock = load_json(QUALITY_LOCK)
    model_lock = load_json(MODEL_LOCK)
    upstream_lock = load_json(UPSTREAM_LOCK)
    if args.ctx != lock["evaluation"]["context_tokens"] or args.chunks != lock["evaluation"]["chunks"]:
        raise ValueError(
            f"predeclared gate is locked to ctx={lock['evaluation']['context_tokens']} and "
            f"chunks={lock['evaluation']['chunks']}; requested ctx={args.ctx}, chunks={args.chunks}"
        )
    for label in ("binary", "model", "dataset"):
        if not paths[label].is_file():
            raise FileNotFoundError(f"{label} file does not exist: {paths[label]}")
    if not paths["upstream"].is_dir():
        raise FileNotFoundError(f"upstream directory does not exist: {paths['upstream']}")

    source_sha = bench.run_git(paths["upstream"], "rev-parse", "HEAD")
    dirty = bool(bench.run_git(paths["upstream"], "status", "--porcelain"))
    if source_sha != upstream_lock["commit"]:
        raise ValueError(f"upstream SHA mismatch: expected {upstream_lock['commit']}, got {source_sha}")
    if dirty:
        raise ValueError("upstream checkout is dirty; refusing to attribute results to the pinned revision")

    model_sha = bench.sha256_file(paths["model"])
    if model_sha != model_lock["sha256"]:
        raise ValueError(f"model SHA256 mismatch: expected {model_lock['sha256']}, got {model_sha}")
    dataset_sha = bench.sha256_file(paths["dataset"])
    dataset_lock = lock["dataset"]
    if paths["dataset"].stat().st_size != dataset_lock["member_bytes"] or dataset_sha != dataset_lock["member_sha256"]:
        raise ValueError(
            f"dataset member mismatch: expected {dataset_lock['member_bytes']} bytes/{dataset_lock['member_sha256']}, "
            f"got {paths['dataset'].stat().st_size}/{dataset_sha}"
        )

    metadata["provenance"] = {
        "harness": {
            "quality_script": {"path": str(Path(__file__).resolve()), "sha256": bench.sha256_file(Path(__file__).resolve())},
            "fetch_script": {"path": str(Path(fetch_dataset.__file__).resolve()), "sha256": bench.sha256_file(Path(fetch_dataset.__file__).resolve())},
            "quality_lock_sha256": bench.sha256_file(QUALITY_LOCK),
            "model_lock_sha256": bench.sha256_file(MODEL_LOCK),
            "upstream_lock_sha256": bench.sha256_file(UPSTREAM_LOCK),
        },
        "binary": {"path": str(paths["binary"]), "sha256": bench.sha256_file(paths["binary"])},
        "model": {"path": str(paths["model"]), "sha256": model_sha, "lock": model_lock},
        "dataset": {"path": str(paths["dataset"]), "sha256": dataset_sha, "bytes": paths["dataset"].stat().st_size, "lock": dataset_lock},
        "upstream": {"path": str(paths["upstream"]), "git_sha": source_sha, "dirty": dirty},
        "build": bench.cmake_cache_info(paths["binary"]),
        "build_artifacts": {"shared_libraries": bench.shared_library_info(paths["binary"])},
    }
    return lock, model_lock


def parse_runtime(stdout: bytes, stderr: bytes, args: argparse.Namespace, kv_type: str) -> tuple[dict[str, Any] | None, str | None]:
    out_text = stdout.decode("utf-8", errors="replace")
    err_text = stderr.decode("utf-8", errors="replace")
    combined = out_text + "\n" + err_text

    ppl_matches = []
    for stream_name, stream_text in (("stdout", out_text), ("stderr", err_text)):
        ppl_matches.extend((stream_name, match) for match in PPL_RE.finditer(stream_text))
    if len(ppl_matches) != 1:
        return None, f"ppl_parse_error: expected exactly one Final estimate, found {len(ppl_matches)}"
    ppl_source, ppl_match = ppl_matches[0]
    ppl = float(ppl_match.group(1))
    uncertainty = float(ppl_match.group(2))
    if not math.isfinite(ppl) or ppl <= 0 or not math.isfinite(uncertainty) or uncertainty < 0:
        return None, f"non_finite_or_invalid_ppl: ppl={ppl!r}, uncertainty={uncertainty!r}"

    chunk_matches = list(CHUNK_RE.finditer(combined))
    if len(chunk_matches) != 1:
        return None, f"chunk_count_parse_error: expected one chunks/context/batch log, found {len(chunk_matches)}"
    actual_chunks, actual_ctx, actual_batch = (int(value) for value in chunk_matches[0].groups())
    if actual_chunks != args.chunks:
        return None, f"field_mismatch: chunks expected {args.chunks}, got {actual_chunks}"
    if actual_ctx != args.ctx:
        return None, f"field_mismatch: context expected {args.ctx}, got {actual_ctx}"
    if actual_batch != 512:
        return None, f"field_mismatch: batch expected 512, got {actual_batch}"

    kv_pairs = [(match.group(1).lower(), match.group(2).lower(), match.group(0)) for match in KV_TYPES_RE.finditer(combined)]
    if not kv_pairs:
        return None, "kv_type_parse_error: no runtime K/V cache type log found"
    if any(key_type != kv_type or value_type != kv_type for key_type, value_type, _ in kv_pairs):
        found = [(key_type, value_type) for key_type, value_type, _ in kv_pairs]
        return None, f"field_mismatch: requested K/V type {kv_type}/{kv_type}, runtime reported {found}"

    cuda_device_matches = [int(match.group(1)) for match in CUDA_RE.finditer(combined)]
    offload_matches = [(int(match.group(1)), int(match.group(2))) for match in OFFLOAD_RE.finditer(combined)]
    if len(offload_matches) != 1:
        return None, f"offload_parse_error: expected one model layer offload summary, found {len(offload_matches)}"
    offloaded, total_layers = offload_matches[0]
    if args.gpu_layers > 0:
        if not cuda_device_matches or max(cuda_device_matches) < 1:
            return None, "cuda_runtime_missing: no CUDA device discovery log found"
        if offloaded != min(args.gpu_layers, total_layers):
            return None, f"field_mismatch: requested {args.gpu_layers} GPU layers, runtime offloaded {offloaded}/{total_layers}"
    elif offloaded != 0:
        return None, f"field_mismatch: requested CPU-only, runtime offloaded {offloaded}/{total_layers} layers"

    # The pinned upstream log reports the actual chunk count, but not the full
    # tokenized input length or scored-target count. In perplexity.cpp, each
    # completed chunk increments count by n_ctx - n_ctx/2 - 1.
    scored_tokens = actual_chunks * (args.ctx - args.ctx // 2 - 1)
    return {
        "ppl": ppl,
        "uncertainty_absolute": uncertainty,
        "ppl_source_stream": ppl_source,
        "input_token_count": None,
        "input_token_count_status": "not_reported_by_pinned_upstream",
        "chunk_count": actual_chunks,
        "context_tokens": actual_ctx,
        "batch_tokens": actual_batch,
        "scored_next_token_targets": scored_tokens,
        "scored_token_count_basis": "derived from pinned perplexity.cpp count += n_ctx - first - 1 for each completed chunk, where first = n_ctx/2; chunk_count is parsed from this run's log",
        "runtime_kv_types": [{"key": key_type, "value": value_type, "source_line": line[:500]} for key_type, value_type, line in kv_pairs],
        "cuda_device_count_logs": cuda_device_matches,
        "gpu_layer_offload": {"offloaded": offloaded, "total": total_layers, "requested": args.gpu_layers},
    }, None


def save_case(out: Path, case_index: int, kv_type: str, args: argparse.Namespace, paths: dict[str, Path], lock: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    case_id = f"case-{case_index:02d}-{kv_type}"
    case_dir = out / "cases"
    stdout_path = case_dir / f"{case_id}.stdout.bin"
    stderr_path = case_dir / f"{case_id}.stderr.bin"
    case_path = case_dir / f"{case_id}.json"
    argv = command_for(paths["binary"], paths["model"], paths["dataset"], args, kv_type, lock)
    detail: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "kv_type": kv_type,
        "argv": argv,
        "cwd": str(paths["upstream"]),
        "status": "running",
        "started_at": bench.utc_now(),
        "stdout_path": str(stdout_path.relative_to(out)),
        "stderr_path": str(stderr_path.relative_to(out)),
    }
    bench.write_json(case_path, detail)
    stdout = b""
    stderr = b""
    error = None
    try:
        completed = subprocess.run(
            argv,
            cwd=str(paths["upstream"]),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=args.timeout,
            check=False,
        )
        stdout, stderr = completed.stdout, completed.stderr
        detail["return_code"] = completed.returncode
        if completed.returncode != 0:
            error = f"nonzero_exit: {completed.returncode}"
    except subprocess.TimeoutExpired as exc:
        stdout = bench.to_bytes(exc.stdout)
        stderr = bench.to_bytes(exc.stderr)
        detail["status"] = "timeout"
        error = f"timeout_after_seconds: {args.timeout}"
    except OSError as exc:
        detail["status"] = "launch_error"
        error = f"launch_error: {exc}"

    stdout_path.write_bytes(stdout)
    stderr_path.write_bytes(stderr)
    detail["finished_at"] = bench.utc_now()
    detail["stdout_bytes"] = len(stdout)
    detail["stderr_bytes"] = len(stderr)
    detail["stdout_sha256"] = bench.sha256_file(stdout_path)
    detail["stderr_sha256"] = bench.sha256_file(stderr_path)

    if error is None:
        if not stdout.strip() and not stderr.strip():
            error = "empty_output"
        else:
            parsed, parse_error = parse_runtime(stdout, stderr, args, kv_type)
            if parse_error:
                error = parse_error
            elif parsed is not None:
                detail["parsed"] = parsed
                detail["status"] = "success"
    if error is not None:
        if detail.get("status") not in ("timeout", "launch_error"):
            detail["status"] = "invalid_execution"
        detail["failure_reason"] = error
    bench.write_json(case_path, detail)
    return detail, error


def make_summary(cases: list[dict[str, Any]], lock: dict[str, Any]) -> dict[str, Any]:
    by_type = {case["kv_type"]: case["parsed"] for case in cases}
    baseline = by_type["f16"]["ppl"]
    gate: dict[str, Any] = {}
    thresholds = lock["gate"]["max_relative_ppl_increase_percent"]
    for kv_type, maximum in thresholds.items():
        mean_ppl = by_type[kv_type]["ppl"]
        relative = 100.0 * (mean_ppl / baseline - 1.0)
        gate[kv_type] = {
            "relative_ppl_increase_percent": relative,
            "maximum_percent": maximum,
            "passed": relative <= maximum,
            "uncertainty_absolute": by_type[kv_type]["uncertainty_absolute"],
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "success" if all(row["passed"] for row in gate.values()) else "evaluation_failed",
        "completed_at": bench.utc_now(),
        "baseline": {"kv_type": "f16/f16", "ppl": baseline, "uncertainty_absolute": by_type["f16"]["uncertainty_absolute"]},
        "gate": gate,
        "measurement_scope": {
            "context_tokens": cases[0]["parsed"]["context_tokens"],
            "chunk_count": cases[0]["parsed"]["chunk_count"],
            "input_token_count": cases[0]["parsed"]["input_token_count"],
            "input_token_count_status": cases[0]["parsed"]["input_token_count_status"],
            "scored_next_token_targets": cases[0]["parsed"]["scored_next_token_targets"],
            "scored_token_count_basis": cases[0]["parsed"]["scored_token_count_basis"],
            "dataset_coverage": "only first configured context chunks; not full test-set or 32K quality",
        },
        "cases": [
            {
                "case_id": case["case_id"],
                "kv_type": case["kv_type"],
                "ppl": case["parsed"]["ppl"],
                "uncertainty_absolute": case["parsed"]["uncertainty_absolute"],
                "input_token_count": case["parsed"]["input_token_count"],
                "input_token_count_status": case["parsed"]["input_token_count_status"],
                "chunk_count": case["parsed"]["chunk_count"],
                "scored_next_token_targets": case["parsed"]["scored_next_token_targets"],
                "scored_token_count_basis": case["parsed"]["scored_token_count_basis"],
                "runtime_kv_types": case["parsed"]["runtime_kv_types"],
                "gpu_layer_offload": case["parsed"]["gpu_layer_offload"],
                "stdout_sha256": case["stdout_sha256"],
                "stderr_sha256": case["stderr_sha256"],
                "case_file": f"cases/{case['case_id']}.json",
            }
            for case in cases
        ],
    }


def fail_invalid(out: Path, metadata: dict[str, Any], reason: str, stage: str, case_id: str | None = None) -> int:
    metadata["status"] = "invalid_execution"
    metadata["finished_at"] = bench.utc_now()
    metadata["failure"] = {"stage": stage, "reason": reason, "case_id": case_id}
    bench.write_json(out / "metadata.json", metadata)
    bench.write_json(out / "failure.json", {
        "schema_version": SCHEMA_VERSION,
        "status": "invalid_execution",
        "failed_at": metadata["finished_at"],
        "stage": stage,
        "reason": reason,
        "case_id": case_id,
    })
    print(f"quality run invalid during {stage}: {reason}", file=sys.stderr)
    return 1


def run(args: argparse.Namespace) -> int:
    paths = {name: bench.resolved_path(getattr(args, name)) for name in ("binary", "model", "upstream", "dataset", "out")}
    try:
        paths["out"].mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"output path already exists; refusing to overwrite: {paths['out']}", file=sys.stderr)
        return 2
    metadata = create_metadata(args, paths)
    bench.write_json(paths["out"] / "metadata.json", metadata)
    try:
        lock, _model_lock = preflight(args, paths, metadata)
        if args.gpu_layers != lock["evaluation"]["gpu_layers"]:
            metadata["quality_scope_note"] = "GPU layer count differs from the predeclared CUDA reference value; runtime offload is still verified."
        if args.threads != lock["evaluation"]["threads"]:
            metadata["quality_scope_note"] = "Thread count differs from the predeclared reference value; it does not change the PPL acceptance formula."
        plan = {
            "schema_version": SCHEMA_VERSION,
            "lock_sha256": bench.sha256_file(QUALITY_LOCK),
            "case_order": lock["evaluation"]["kv_types"],
            "cases": [
                {"case_id": f"case-{index:02d}-{kv_type}", "kv_type": kv_type,
                 "argv": command_for(paths["binary"], paths["model"], paths["dataset"], args, kv_type, lock),
                 "cwd": str(paths["upstream"])}
                for index, kv_type in enumerate(lock["evaluation"]["kv_types"], start=1)
            ],
        }
        bench.write_json(paths["out"] / "plan.json", plan)
    except Exception as exc:
        return fail_invalid(paths["out"], metadata, f"preflight_error: {exc}", "preflight")

    metadata["status"] = "running"
    metadata["started_at"] = bench.utc_now()
    bench.write_json(paths["out"] / "metadata.json", metadata)
    (paths["out"] / "cases").mkdir()
    completed: list[dict[str, Any]] = []
    for index, kv_type in enumerate(lock["evaluation"]["kv_types"], start=1):
        detail, error = save_case(paths["out"], index, kv_type, args, paths, lock)
        completed.append(detail)
        metadata["completed_cases"] = [case["case_id"] for case in completed]
        bench.write_json(paths["out"] / "metadata.json", metadata)
        if error:
            return fail_invalid(paths["out"], metadata, error, "case", detail["case_id"])
    summary = make_summary(completed, lock)
    bench.write_json(paths["out"] / "summary.json", summary)
    metadata["finished_at"] = summary["completed_at"]
    metadata["status"] = summary["status"]
    metadata["completed_case_count"] = len(completed)
    bench.write_json(paths["out"] / "metadata.json", metadata)
    if summary["status"] == "evaluation_failed":
        failures = {kv_type: values for kv_type, values in summary["gate"].items() if not values["passed"]}
        bench.write_json(paths["out"] / "failure.json", {
            "schema_version": SCHEMA_VERSION,
            "status": "evaluation_failed",
            "failed_at": summary["completed_at"],
            "stage": "quality_threshold",
            "reason": "one or more predeclared relative PPL thresholds were exceeded",
            "threshold_failures": failures,
        })
        print(f"quality gate failed; complete summary retained: {paths['out']}", file=sys.stderr)
        return 1
    print(f"quality gate passed: {paths['out']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
