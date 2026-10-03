#!/usr/bin/env python3
"""Run isolated llama-bench decode or prefill cases and preserve their evidence."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import math
import os
import platform
import random
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
KV_MIB_RE = re.compile(
    r"\b(?:kv(?:[_ -]?(?:cache|buffer))?|key.?value|cache)\b"
    r".{0,200}?(\d+(?:\.\d+)?)\s*MiB\b",
    re.IGNORECASE,
)
CUDA_KV_MIB_RE = re.compile(r"\bCUDA0\b.*?\bKV\s+(?:buffer|cache)\s+size\s*=\s*(\d+(?:\.\d+)?)\s*MiB\b", re.IGNORECASE)
OFFLOAD_RE = re.compile(r"\boffloaded\s+(\d+)\s*/\s*(\d+)\s+layers?\b", re.IGNORECASE)
CUDA_DEVICE_RE = re.compile(r"\bDevice\s+(\d+)\s*:\s*(.+?),\s*compute capability\s+(\d+\.\d+)", re.IGNORECASE)


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
    parser.add_argument("--mode", choices=("decode", "prefill"), default="decode")
    parser.add_argument("--batch", type=positive_int, default=512)
    parser.add_argument("--ubatch", type=positive_int, default=512)
    parser.add_argument("--require-backend", choices=("cuda", "cpu", "any"), default="any")
    parser.add_argument("--monitor-memory", action="store_true", help="sample child process memory and capture GPU snapshots where available")
    parser.add_argument("--sample-interval-ms", type=positive_int, default=100)
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
            "mode": getattr(args, "mode", "decode"),
            "depth_semantics": "prompt_tokens" if getattr(args, "mode", "decode") == "prefill" else "context_depth",
            "batch": getattr(args, "batch", 512),
            "ubatch": getattr(args, "ubatch", 512),
            "require_backend": getattr(args, "require_backend", "any"),
            "monitor_memory": getattr(args, "monitor_memory", False),
            "sample_interval_ms": getattr(args, "sample_interval_ms", 100),
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
            "llama-bench throughput only; it does not measure TTFT or sampling/tokenization time",
            "no generation quality evaluation is performed",
            "optional process memory is child-only sampling, not a full-process-tree peak and not KV-specific",
            "GPU device snapshots describe whole-device state, not process GPU memory",
        ],
        "measurement_overhead": {
            "memory_monitor_requested": getattr(args, "monitor_memory", False),
            "sample_interval_ms": getattr(args, "sample_interval_ms", 100),
            "matched_arms": "Every case in this run uses the same monitoring settings; sampling and nvidia-smi queries can affect timing.",
        },
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
    mode = getattr(args, "mode", "decode")
    if mode == "prefill":
        n_prompt, n_gen, n_depth = depth, 0, 0
    else:
        n_prompt, n_gen, n_depth = 0, args.gen, depth
    return [
        str(binary),
        "-m", str(model),
        "-p", str(n_prompt),
        "-n", str(n_gen),
        "-d", str(n_depth),
        "-b", str(getattr(args, "batch", 512)),
        "-ub", str(getattr(args, "ubatch", 512)),
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
        "mode": getattr(args, "mode", "decode"),
        "depth_semantics": "prompt_tokens" if getattr(args, "mode", "decode") == "prefill" else "context_depth",
        "batch": getattr(args, "batch", 512),
        "ubatch": getattr(args, "ubatch", 512),
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


def parse_proc_memory_status(status_text: str) -> dict[str, int | None]:
    values: dict[str, int | None] = {"rss_kib": None, "vmhwm_kib": None}
    for line in status_text.splitlines():
        key, separator, rest = line.partition(":")
        if not separator or key not in ("VmRSS", "VmHWM"):
            continue
        match = re.search(r"\b(\d+)\s+kB\b", rest)
        if match:
            values["rss_kib" if key == "VmRSS" else "vmhwm_kib"] = int(match.group(1))
    return values


def read_proc_memory(pid: int) -> dict[str, Any]:
    status_path = Path("/proc") / str(pid) / "status"
    try:
        parsed = parse_proc_memory_status(status_path.read_text(encoding="ascii", errors="replace"))
        return {**parsed, "source": str(status_path), "status": "sampled"}
    except FileNotFoundError:
        return {
            "rss_kib": None,
            "vmhwm_kib": None,
            "source": str(status_path),
            "status": "process_disappeared",
        }
    except OSError as exc:
        return {
            "rss_kib": None,
            "vmhwm_kib": None,
            "source": str(status_path),
            "status": "unavailable",
            "error": str(exc),
        }


def parse_cuda_kv_allocation(stderr: bytes) -> float | None:
    text = stderr.decode("utf-8", errors="replace")
    for line in text.splitlines():
        match = CUDA_KV_MIB_RE.search(line)
        if match:
            value = float(match.group(1))
            if math.isfinite(value):
                return value
    return None


def parse_offloaded_layers(stderr: bytes) -> list[dict[str, int]]:
    text = stderr.decode("utf-8", errors="replace")
    return [
        {"offloaded": int(match.group(1)), "total": int(match.group(2))}
        for line in text.splitlines()
        if (match := OFFLOAD_RE.search(line)) is not None
    ]


def parse_cuda_device_info(stderr: bytes) -> dict[str, Any] | None:
    text = stderr.decode("utf-8", errors="replace")
    first_device = None
    for line in text.splitlines():
        match = CUDA_DEVICE_RE.search(line)
        if match:
            device = {
                "index": int(match.group(1)),
                "name": match.group(2).strip(),
                "compute_capability": match.group(3),
                "source": "stderr",
                "source_line": line[:500],
            }
            if device["index"] == 0:
                return device
            if first_device is None:
                first_device = device
    return first_device


def backend_tokens(value: str) -> set[str]:
    return {token.strip().upper() for token in value.split(",") if token.strip()}


def int_field_matches(row: dict[str, Any], key: str, expected: int) -> bool:
    value = row.get(key)
    return isinstance(value, int) and not isinstance(value, bool) and value == expected


def validate_result(
    stdout: bytes,
    case: dict[str, Any],
    args: argparse.Namespace,
    stderr: bytes = b"",
) -> tuple[dict[str, Any] | None, str | None]:
    if not stdout.strip():
        return None, "empty_stdout"
    try:
        payload = json.loads(stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"invalid_json: {exc}"
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
        return None, "expected_json_array_with_exactly_one_result_object"
    row = payload[0]
    mode = getattr(args, "mode", "decode")
    if mode == "prefill":
        n_prompt, n_gen, n_depth = case["depth"], 0, 0
    else:
        n_prompt, n_gen, n_depth = 0, args.gen, case["depth"]
    expected_ints = {
        "n_prompt": n_prompt,
        "n_gen": n_gen,
        "n_depth": n_depth,
        "n_batch": getattr(args, "batch", 512),
        "n_ubatch": getattr(args, "ubatch", 512),
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
    required_backend = getattr(args, "require_backend", "any")
    tokens = backend_tokens(backend)
    if required_backend == "cuda" and "CUDA" not in tokens:
        return None, f"backend_mismatch: CUDA required, got {backend!r}"
    if required_backend == "cpu" and "CPU" not in tokens:
        return None, f"backend_mismatch: CPU required, got {backend!r}"
    if required_backend == "cuda":
        cuda_kv_mib = parse_cuda_kv_allocation(stderr)
        if cuda_kv_mib is None or cuda_kv_mib <= 0:
            return None, "cuda_evidence_missing: no positive CUDA0 KV allocation found in stderr"
        ratios = parse_offloaded_layers(stderr)
        if not any(item["offloaded"] > 0 and item["offloaded"] == item["total"] for item in ratios):
            return None, "cuda_evidence_missing: no fully offloaded N/N layers record found in stderr"
    avg_ts = row.get("avg_ts")
    if isinstance(avg_ts, bool) or not isinstance(avg_ts, (int, float)) or not math.isfinite(avg_ts) or avg_ts <= 0:
        return None, f"field_mismatch: avg_ts expected finite positive number, got {avg_ts!r}"
    for key in ("samples_ts", "samples_ns"):
        samples = row.get(key)
        if not isinstance(samples, list) or len(samples) != args.repetitions:
            count = len(samples) if isinstance(samples, list) else None
            return None, f"sample_count_mismatch: {key} expected {args.repetitions}, got {count}"
        for index, value in enumerate(samples):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                return None, f"invalid_sample: {key}[{index}] expected finite positive number, got {value!r}"
    return row, None


def nvidia_smi_snapshot() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    captured_at = utc_now()
    if not executable:
        return {
            "captured_at": captured_at,
            "scope": "all devices visible to nvidia-smi; whole-device snapshot",
            "status": "unsupported",
            "reason": "nvidia-smi not found",
            "devices": None,
        }
    command = [
        executable,
        "--query-gpu=index,name,temperature.gpu,power.draw,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "captured_at": captured_at,
            "scope": "all devices visible to nvidia-smi; whole-device snapshot",
            "status": "unsupported",
            "command": command,
            "reason": str(exc),
            "devices": None,
        }
    stdout = result.stdout.decode("utf-8", errors="replace").strip()
    stderr = result.stderr.decode("utf-8", errors="replace").strip()
    if result.returncode != 0:
        return {
            "captured_at": captured_at,
            "scope": "all devices visible to nvidia-smi; whole-device snapshot",
            "status": "unsupported",
            "command": command,
            "return_code": result.returncode,
            "stderr": stderr[:1000],
            "devices": None,
        }
    names = ("index", "name", "temperature_c", "power_w", "utilization_percent", "memory_used_mib", "memory_total_mib")
    devices = []
    for values in csv.reader(io.StringIO(stdout)):
        if not values:
            continue
        devices.append({key: value.strip() for key, value in zip(names, values)})
    return {
        "captured_at": captured_at,
        "scope": "all devices visible to nvidia-smi; whole-device snapshot, not process memory",
        "status": "sampled",
        "command": command,
        "devices": devices,
    }


def nvidia_smi_process_memory(pid: int) -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {"status": "unsupported", "used_gpu_memory_mib": None, "reason": "nvidia-smi not found"}
    command = [
        executable,
        "--query-compute-apps=pid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "unsupported", "used_gpu_memory_mib": None, "command": command, "reason": str(exc)}
    if result.returncode != 0:
        return {
            "status": "unsupported",
            "used_gpu_memory_mib": None,
            "command": command,
            "reason": result.stderr.decode("utf-8", errors="replace").strip()[:1000],
        }
    matches = []
    for values in csv.reader(io.StringIO(result.stdout.decode("utf-8", errors="replace"))):
        if len(values) < 2:
            continue
        try:
            listed_pid = int(values[0].strip())
            used_mib = float(values[1].strip())
        except ValueError:
            continue
        if listed_pid == pid and math.isfinite(used_mib) and used_mib >= 0:
            matches.append(used_mib)
    if not matches:
        return {
            "status": "pid_not_reported",
            "used_gpu_memory_mib": None,
            "command": command,
            "reason": "nvidia-smi did not report the benchmark child PID",
        }
    return {
        "status": "sampled",
        "used_gpu_memory_mib": sum(matches),
        "matched_device_rows": len(matches),
        "command": command,
    }


def summarize_memory_monitor(
    args: argparse.Namespace,
    proc_samples: list[dict[str, Any]],
    gpu_samples: list[dict[str, Any]],
) -> dict[str, Any]:
    rss_values = [sample["rss_kib"] for sample in proc_samples if isinstance(sample.get("rss_kib"), int)]
    hwm_values = [sample["vmhwm_kib"] for sample in proc_samples if isinstance(sample.get("vmhwm_kib"), int)]
    gpu_values = [sample["used_gpu_memory_mib"] for sample in gpu_samples if isinstance(sample.get("used_gpu_memory_mib"), (int, float))]
    proc_supported = sys.platform.startswith("linux") and Path("/proc").is_dir()
    if not proc_supported:
        proc_status = "unsupported"
    elif proc_samples and (rss_values or hwm_values):
        proc_status = "sampled"
    else:
        proc_status = "no_valid_samples"
    gpu_statuses = {sample.get("status") for sample in gpu_samples}
    if not gpu_samples:
        gpu_status = "unsupported" if not shutil.which("nvidia-smi") else "not_sampled"
    elif gpu_values and gpu_statuses == {"sampled"}:
        gpu_status = "sampled"
    elif gpu_values:
        gpu_status = "partially_sampled"
    elif "unsupported" in gpu_statuses:
        gpu_status = "unsupported"
    else:
        gpu_status = "pid_not_reported"
    return {
        "requested": True,
        "sample_interval_ms": getattr(args, "sample_interval_ms", 100),
        "process_status": proc_status,
        "process_source": "/proc/<pid>/status" if proc_supported else None,
        "process_scope": "benchmark child PID only; not a process-tree peak",
        "sampled_peak_vmrss_kib": max(rss_values) if rss_values else None,
        "max_observed_vmhwm_kib": max(hwm_values) if hwm_values else None,
        "process_samples": proc_samples,
        "gpu_process_status": gpu_status,
        "gpu_process_scope": "matched benchmark child PID rows only; never whole-device memory",
        "sampled_peak_gpu_process_memory_mib": max(gpu_values) if gpu_values else None,
        "gpu_process_samples": gpu_samples,
        "interpretation": "Sampled process metrics are not full process-tree peaks and are not KV-specific.",
    }


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


def compact_memory_monitor(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    keys = (
        "requested", "sample_interval_ms", "process_status", "process_source", "process_scope",
        "sampled_peak_vmrss_kib", "max_observed_vmhwm_kib", "gpu_process_status", "gpu_process_scope",
        "sampled_peak_gpu_process_memory_mib", "interpretation",
    )
    compact = {key: value.get(key) for key in keys}
    compact["process_sample_count"] = len(value.get("process_samples", []))
    compact["gpu_process_sample_count"] = len(value.get("gpu_process_samples", []))
    return compact


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
        "mode": getattr(args, "mode", "decode"),
        "prompt_tokens": planned["depth"] if getattr(args, "mode", "decode") == "prefill" else None,
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
        "process_elapsed_seconds": None,
    }
    write_json(case_file, detail)
    stdout = b""
    stderr = b""
    error: str | None = None
    monitor_enabled = getattr(args, "monitor_memory", False)
    proc_samples: list[dict[str, Any]] = []
    gpu_samples: list[dict[str, Any]] = []
    gpu_snapshots = None
    process_start_monotonic = None
    process_started_at = None
    process_finished_at = None
    if monitor_enabled:
        gpu_snapshots = {"before": nvidia_smi_snapshot(), "after": None}
    try:
        if monitor_enabled:
            with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
                process_started_at = utc_now()
                process_start_monotonic = time.monotonic()
                child = subprocess.Popen(
                    planned["argv"],
                    cwd=planned["cwd"],
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_file,
                    stderr=stderr_file,
                )
                timed_out = False
                interval = getattr(args, "sample_interval_ms", 100) / 1000.0
                proc_supported = sys.platform.startswith("linux") and Path("/proc").is_dir()
                gpu_query_available = bool(shutil.which("nvidia-smi"))
                while child.poll() is None:
                    elapsed = time.monotonic() - process_start_monotonic
                    if proc_supported:
                        proc_samples.append({
                            "sampled_at": utc_now(),
                            "elapsed_seconds": round(elapsed, 6),
                            **read_proc_memory(child.pid),
                        })
                    if gpu_query_available:
                        gpu_samples.append({
                            "sampled_at": utc_now(),
                            "elapsed_seconds": round(elapsed, 6),
                            **nvidia_smi_process_memory(child.pid),
                        })
                    remaining = args.timeout - (time.monotonic() - process_start_monotonic)
                    if remaining <= 0:
                        child.kill()
                        child.wait()
                        timed_out = True
                        break
                    time.sleep(min(interval, remaining))
                detail["return_code"] = child.returncode
                process_finished_at = utc_now()
                detail["process_elapsed_seconds"] = round(time.monotonic() - process_start_monotonic, 6)
                if timed_out:
                    error = f"timeout_after_seconds: {args.timeout}"
                    detail["status"] = "timeout"
                elif child.returncode != 0:
                    error = f"nonzero_exit: {child.returncode}"
        else:
            process_started_at = utc_now()
            process_start_monotonic = time.monotonic()
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
            process_finished_at = utc_now()
            detail["process_elapsed_seconds"] = round(time.monotonic() - process_start_monotonic, 6)
            if completed.returncode != 0:
                error = f"nonzero_exit: {completed.returncode}"
    except subprocess.TimeoutExpired as exc:
        stdout = to_bytes(exc.stdout)
        stderr = to_bytes(exc.stderr)
        error = f"timeout_after_seconds: {args.timeout}"
        detail["status"] = "timeout"
        process_finished_at = utc_now()
        if process_start_monotonic is not None:
            detail["process_elapsed_seconds"] = round(time.monotonic() - process_start_monotonic, 6)
    except OSError as exc:
        error = f"launch_error: {exc}"
        detail["status"] = "launch_error"
        process_finished_at = utc_now()
        if process_start_monotonic is not None:
            detail["process_elapsed_seconds"] = round(time.monotonic() - process_start_monotonic, 6)
    if not monitor_enabled:
        stdout_path.write_bytes(stdout)
        stderr_path.write_bytes(stderr)
    if gpu_snapshots is not None:
        gpu_snapshots["after"] = nvidia_smi_snapshot()
    stdout = stdout_path.read_bytes()
    stderr = stderr_path.read_bytes()
    detail["finished_at"] = utc_now()
    detail["process_started_at"] = process_started_at
    detail["process_finished_at"] = process_finished_at
    detail["elapsed_seconds"] = detail["process_elapsed_seconds"]
    detail["kv_runtime_mib"] = parse_kv_runtime_sizes(stderr)
    detail["cuda_device_info"] = parse_cuda_device_info(stderr)
    detail["cuda0_kv_allocation_mib"] = parse_cuda_kv_allocation(stderr)
    detail["offloaded_layers"] = parse_offloaded_layers(stderr)
    detail["gpu_device_snapshots"] = gpu_snapshots
    detail["memory_monitor"] = summarize_memory_monitor(args, proc_samples, gpu_samples) if monitor_enabled else None
    detail["stdout_bytes"] = len(stdout)
    detail["stderr_bytes"] = len(stderr)
    if error is None:
        row, validation_error = validate_result(stdout, planned, args, stderr)
        if validation_error:
            error = validation_error
        else:
            detail["status"] = "success"
            detail["result"] = row
            detail["selected_settings"] = {
                key: row.get(key)
                for key in (
                    "backends", "type_k", "type_v", "n_gpu_layers", "n_threads", "flash_attn",
                    "n_prompt", "n_gen", "n_depth", "n_batch", "n_ubatch",
                )
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
        "mode": getattr(args, "mode", "decode"),
        "batch": getattr(args, "batch", 512),
        "ubatch": getattr(args, "ubatch", 512),
        "require_backend": getattr(args, "require_backend", "any"),
        "monitor_memory": getattr(args, "monitor_memory", False),
        "sample_interval_ms": getattr(args, "sample_interval_ms", 100),
        "measurement_count": len(completed_cases),
        "measurements": [
            {
                "case_id": case["case_id"],
                "depth": case["depth"],
                "kv_type": case["kv_type"],
                "mode": getattr(args, "mode", "decode"),
                "prompt_tokens": case["depth"] if getattr(args, "mode", "decode") == "prefill" else None,
                "avg_ts": case["avg_ts"],
                "stddev_ts": case["stddev_ts"],
                "process_started_at": case["process_started_at"],
                "process_finished_at": case["process_finished_at"],
                "process_elapsed_seconds": case["process_elapsed_seconds"],
                "selected_settings": case["selected_settings"],
                "kv_runtime_mib": case["kv_runtime_mib"],
                "cuda_device_info": case["cuda_device_info"],
                "offloaded_layers": case["offloaded_layers"],
                "memory_monitor": compact_memory_monitor(case["memory_monitor"]),
                "gpu_device_snapshots": case["gpu_device_snapshots"],
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
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.mode == "prefill" and any(depth == 0 for depth in args.depths):
        parser.error("prefill depths must be positive prompt token counts")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
