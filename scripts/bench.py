#!/usr/bin/env python3
"""Run isolated llama-bench decode cases and preserve their evidence."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import random
import re
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
KV_MIB_RE = re.compile(
    r"\b(?:kv(?:[_ -]?(?:cache|buffer))?|key.?value|cache)\b"
    r".{0,200}?(\d+(?:\.\d+)?)\s*MiB\b",
    re.IGNORECASE,
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_csv_ints(value: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("values must be non-negative integers")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("duplicate values are not allowed")
    return values


def parse_csv_types(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",")]
    if not values or any(not item for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated KV types")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("duplicate KV types are not allowed")
    return values


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def gpu_layers_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected an integer") from exc
    if parsed < -1:
        raise argparse.ArgumentTypeError("value must be -1 or greater")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, help="path to llama-bench")
    parser.add_argument("--model", required=True, help="path to a local GGUF model")
    parser.add_argument("--upstream", required=True, help="llama.cpp git checkout used for provenance and cwd")
    parser.add_argument("--out", required=True, help="new output directory; existing paths are never overwritten")
    parser.add_argument("--depths", type=parse_csv_ints, default=parse_csv_ints("0,512"))
    parser.add_argument("--kv-types", type=parse_csv_types, default=parse_csv_types("f16,q8_0,q4_0"))
    parser.add_argument("--gpu-layers", type=gpu_layers_int, default=0)
    parser.add_argument("--threads", type=positive_int, default=4)
    parser.add_argument("--repetitions", type=positive_int, default=3)
    parser.add_argument("--gen", type=positive_int, default=32)
    parser.add_argument("--timeout", type=positive_int, default=180)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--dry-run", action="store_true", help="write a planning-only manifest without requiring binary or model files")
    return parser


def resolved_path(value: str) -> Path:
    return Path(value).expanduser().resolve(strict=False)


def run_git(upstream: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(upstream), *args],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=15,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {detail or result.returncode}")
    return result.stdout.decode("utf-8", errors="replace").strip()


def cmake_cache_info(binary: Path) -> dict[str, Any] | None:
    cache = binary.parent.parent / "CMakeCache.txt"
    if not cache.is_file():
        return None
    selected: dict[str, str] = {}
    for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith("#") or ":" not in line or "=" not in line:
            continue
        key_part, value = line.split("=", 1)
        key = key_part.split(":", 1)[0]
        if key == "CMAKE_BUILD_TYPE" or key.startswith(("GGML_", "LLAMA_")):
            selected[key] = value
    return {"path": str(cache), "sha256": sha256_file(cache), "selected_settings": selected}


def shared_library_info(binary: Path) -> list[dict[str, str]]:
    patterns = ("lib*.so", "lib*.so.*", "lib*.dylib", "*.dll")
    libraries = {path for pattern in patterns for path in binary.parent.glob(pattern) if path.is_file()}
    return [
        {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for path in sorted(libraries, key=lambda item: item.name.lower())
    ]


def base_metadata(args: argparse.Namespace, binary: Path, model: Path, upstream: Path) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "starting",
        "planning_only": args.dry_run,
        "created_at": utc_now(),
        "requested": {
            "binary": str(binary),
            "model": str(model),
            "upstream": str(upstream),
            "depths": args.depths,
            "kv_types": args.kv_types,
            "gpu_layers": args.gpu_layers,
            "threads": args.threads,
            "repetitions": args.repetitions,
            "gen": args.gen,
            "timeout_seconds": args.timeout,
            "seed": args.seed,
        },
        "environment": {
            "hostname": socket.gethostname(),
            "python_version": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "os_name": os.name,
        },
        "provenance": {
            "harness_script": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "binary": {"path": str(binary), "sha256": None},
            "model": {"path": str(model), "sha256": None},
            "upstream": {"path": str(upstream), "git_sha": None, "dirty": None},
            "build": None,
            "build_artifacts": {"shared_libraries": []},
        },
        "limitations": [
            "llama-bench generation throughput only; it does not measure TTFT or sampling/tokenization time",
            "no generation quality evaluation is performed",
            "peak RSS is not collected",
        ],
    }


def fill_provenance(metadata: dict[str, Any], args: argparse.Namespace, binary: Path, model: Path, upstream: Path) -> None:
    sha = run_git(upstream, "rev-parse", "HEAD")
    dirty = bool(run_git(upstream, "status", "--porcelain"))
    metadata["provenance"]["upstream"].update({"git_sha": sha, "dirty": dirty})
    if args.dry_run:
        metadata["provenance"]["binary"]["hash_status"] = "skipped_planning_only"
        metadata["provenance"]["model"]["hash_status"] = "skipped_planning_only"
        return
    for label, path in (("binary", binary), ("model", model)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} file does not exist: {path}")
        metadata["provenance"][label]["sha256"] = sha256_file(path)
    metadata["provenance"]["build"] = cmake_cache_info(binary)
    metadata["provenance"]["build_artifacts"]["shared_libraries"] = shared_library_info(binary)


def command_for(binary: Path, model: Path, args: argparse.Namespace, depth: int, kv_type: str) -> list[str]:
    return [
        str(binary),
        "-m", str(model),
        "-p", "0",
        "-n", str(args.gen),
        "-d", str(depth),
        "-ctk", kv_type,
        "-ctv", kv_type,
        "-fa", "on",
        "-ngl", str(args.gpu_layers),
        "-t", str(args.threads),
        "-r", str(args.repetitions),
        "-v",
        "-o", "json",
    ]


def make_plan(args: argparse.Namespace, binary: Path, model: Path, upstream: Path) -> dict[str, Any]:
    cases = [{"depth": depth, "kv_type": kv_type} for depth in args.depths for kv_type in args.kv_types]
    random.Random(args.seed).shuffle(cases)
    planned = []
    for index, case in enumerate(cases, start=1):
        case_id = f"case-{index:04d}"
        planned.append({
            "case_id": case_id,
            **case,
            "argv": command_for(binary, model, args, case["depth"], case["kv_type"]),
            "cwd": str(upstream),
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "planning_only": args.dry_run,
        "seed": args.seed,
        "randomized_order": planned,
    }


def parse_kv_runtime_sizes(stderr: bytes) -> list[dict[str, Any]] | None:
    text = stderr.decode("utf-8", errors="replace")
    found = []
    for line in text.splitlines():
        match = KV_MIB_RE.search(line)
        if match:
            found.append({
                "mib": float(match.group(1)),
                "source": "stderr",
                "source_line": line[:500],
            })
    return found or None


def int_field_matches(row: dict[str, Any], key: str, expected: int) -> bool:
    value = row.get(key)
    return isinstance(value, int) and not isinstance(value, bool) and value == expected


def validate_result(stdout: bytes, case: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any] | None, str | None]:
    if not stdout.strip():
        return None, "empty_stdout"
    try:
        payload = json.loads(stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"invalid_json: {exc}"
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
        return None, "expected_json_array_with_exactly_one_result_object"
    row = payload[0]
    expected_ints = {
        "n_prompt": 0,
        "n_gen": args.gen,
        "n_depth": case["depth"],
        "n_gpu_layers": args.gpu_layers,
        "n_threads": args.threads,
        "flash_attn": 1,
    }
    for key, expected in expected_ints.items():
        if not int_field_matches(row, key, expected):
            return None, f"field_mismatch: {key} expected integer {expected}, got {row.get(key)!r}"
    for key in ("type_k", "type_v"):
        if row.get(key) != case["kv_type"]:
            return None, f"field_mismatch: {key} expected {case['kv_type']!r}, got {row.get(key)!r}"
    backend = row.get("backends")
    if not isinstance(backend, str) or not backend:
        return None, f"field_mismatch: backends expected non-empty string, got {backend!r}"
    avg_ts = row.get("avg_ts")
    if isinstance(avg_ts, bool) or not isinstance(avg_ts, (int, float)) or not math.isfinite(avg_ts) or avg_ts <= 0:
        return None, f"field_mismatch: avg_ts expected finite positive number, got {avg_ts!r}"
    return row, None


def to_bytes(value: bytes | str | None) -> bytes:
    if value is None:
        return b""
    return value if isinstance(value, bytes) else value.encode("utf-8", errors="replace")


def fail_run(out: Path, metadata: dict[str, Any], stage: str, reason: str, case_id: str | None = None) -> int:
    metadata["status"] = "failed"
    metadata["finished_at"] = utc_now()
    metadata["failure"] = {"stage": stage, "reason": reason, "case_id": case_id}
    write_json(out / "metadata.json", metadata)
    write_json(out / "failure.json", {
        "schema_version": SCHEMA_VERSION,
        "failed_at": metadata["finished_at"],
        "stage": stage,
        "reason": reason,
        "case_id": case_id,
    })
    print(f"benchmark failed during {stage}: {reason}", file=sys.stderr)
    return 1


def run_case(out: Path, planned: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], str | None]:
    cases_dir = out / "cases"
    case_id = planned["case_id"]
    stdout_path = cases_dir / f"{case_id}.stdout.bin"
    stderr_path = cases_dir / f"{case_id}.stderr.bin"
    case_file = cases_dir / f"{case_id}.json"
    detail: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "depth": planned["depth"],
        "kv_type": planned["kv_type"],
        "argv": planned["argv"],
        "cwd": planned["cwd"],
        "status": "running",
        "started_at": utc_now(),
        "stdout_path": str(stdout_path.relative_to(out)),
        "stderr_path": str(stderr_path.relative_to(out)),
        "return_code": None,
        "kv_runtime_mib": None,
        "peak_rss_kib": None,
    }
    write_json(case_file, detail)
    stdout = b""
    stderr = b""
    error: str | None = None
    try:
        completed = subprocess.run(
            planned["argv"],
            cwd=planned["cwd"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=args.timeout,
            check=False,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        detail["return_code"] = completed.returncode
        if completed.returncode != 0:
            error = f"nonzero_exit: {completed.returncode}"
    except subprocess.TimeoutExpired as exc:
        stdout = to_bytes(exc.stdout)
        stderr = to_bytes(exc.stderr)
        error = f"timeout_after_seconds: {args.timeout}"
        detail["status"] = "timeout"
    except OSError as exc:
        error = f"launch_error: {exc}"
        detail["status"] = "launch_error"
    stdout_path.write_bytes(stdout)
    stderr_path.write_bytes(stderr)
    detail["finished_at"] = utc_now()
    detail["kv_runtime_mib"] = parse_kv_runtime_sizes(stderr)
    detail["stdout_bytes"] = len(stdout)
    detail["stderr_bytes"] = len(stderr)
    if error is None:
        row, validation_error = validate_result(stdout, planned, args)
        if validation_error:
            error = validation_error
        else:
            detail["status"] = "success"
            detail["result"] = row
            detail["selected_settings"] = {
                key: row.get(key)
                for key in ("backends", "type_k", "type_v", "n_gpu_layers", "n_threads", "flash_attn", "n_prompt", "n_gen", "n_depth")
            }
            detail["avg_ts"] = row["avg_ts"]
            detail["stddev_ts"] = row.get("stddev_ts")
    if error is not None:
        detail["status"] = detail.get("status") if detail.get("status") in ("timeout", "launch_error") else "failed"
        detail["failure_reason"] = error
    write_json(case_file, detail)
    return detail, error


def run(args: argparse.Namespace) -> int:
    binary = resolved_path(args.binary)
    model = resolved_path(args.model)
    upstream = resolved_path(args.upstream)
    out = resolved_path(args.out)
    try:
        out.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"output path already exists; refusing to overwrite: {out}", file=sys.stderr)
        return 2

    metadata = base_metadata(args, binary, model, upstream)
    write_json(out / "metadata.json", metadata)
    try:
        fill_provenance(metadata, args, binary, model, upstream)
        plan = make_plan(args, binary, model, upstream)
        write_json(out / "plan.json", plan)
    except Exception as exc:
        return fail_run(out, metadata, "preflight", str(exc))

    if args.dry_run:
        metadata["status"] = "planning_only"
        metadata["finished_at"] = utc_now()
        write_json(out / "metadata.json", metadata)
        print(f"planning-only manifest written: {out}")
        return 0

    metadata["status"] = "running"
    metadata["started_at"] = utc_now()
    write_json(out / "metadata.json", metadata)
    (out / "cases").mkdir()
    completed_cases = []
    for planned in plan["randomized_order"]:
        detail, error = run_case(out, planned, args)
        completed_cases.append(detail)
        if error:
            return fail_run(out, metadata, "case", error, planned["case_id"])

    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "success",
        "planning_only": False,
        "completed_at": utc_now(),
        "seed": args.seed,
        "measurement_count": len(completed_cases),
        "measurements": [
            {
                "case_id": case["case_id"],
                "depth": case["depth"],
                "kv_type": case["kv_type"],
                "avg_ts": case["avg_ts"],
                "stddev_ts": case["stddev_ts"],
                "selected_settings": case["selected_settings"],
                "kv_runtime_mib": case["kv_runtime_mib"],
                "case_file": f"cases/{case['case_id']}.json",
            }
            for case in completed_cases
        ],
    }
    write_json(out / "summary.json", summary)
    metadata["status"] = "success"
    metadata["finished_at"] = summary["completed_at"]
    metadata["completed_case_count"] = len(completed_cases)
    write_json(out / "metadata.json", metadata)
    print(f"benchmark completed: {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
