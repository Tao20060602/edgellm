#!/usr/bin/env python3
"""Validate and summarize repeated EdgeLLM llama-bench matrix outputs."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
EXPECTED_FRESH_RUNS = 3
EXPECTED_SAMPLES = 5
KV_ORDER = {"f16": 0, "q8_0": 1, "q4_0": 2}


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc


def finite_positive(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value)) and float(value) > 0
    except (OverflowError, ValueError):
        return False


def finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def discover_run_dirs(matrix: Path) -> list[Path]:
    candidates: set[Path] = set()
    for filename in ("plan.json", "summary.json", "failure.json"):
        candidates.update(path.parent for path in matrix.rglob(filename))
    for path in matrix.rglob("metadata.json"):
        metadata = read_json(path)
        if isinstance(metadata, dict) and "provenance" in metadata and "requested" in metadata:
            candidates.add(path.parent)
    return sorted(candidates, key=lambda item: str(item.relative_to(matrix)).lower())


def safe_case_path(run_dir: Path, relative: str) -> Path:
    candidate = (run_dir / relative).resolve(strict=False)
    root = run_dir.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"case_file escapes its run directory: {relative}") from exc
    return candidate


def allocation_source(entry: dict[str, Any]) -> str:
    line = str(entry.get("source_line", "")).upper()
    if "CUDA0" in line:
        return "CUDA0"
    if "CPU" in line:
        return "CPU"
    return str(entry.get("source", "unknown"))


def kv_allocations(case: dict[str, Any]) -> list[dict[str, Any]]:
    entries = case.get("kv_runtime_mib")
    if entries is None:
        return []
    if not isinstance(entries, list):
        raise ValueError(f"{case.get('case_id')}: kv_runtime_mib must be a list or null")
    result = []
    for entry in entries:
        if not isinstance(entry, dict) or not finite_number(entry.get("mib")):
            raise ValueError(f"{case.get('case_id')}: invalid KV allocation entry")
        result.append({
            "mib": float(entry["mib"]),
            "source": allocation_source(entry),
            "source_line": entry.get("source_line"),
        })
    return result


def read_memory(case: dict[str, Any]) -> dict[str, Any]:
    monitor = case.get("memory_monitor")
    if monitor is None:
        return {
            "sampled_peak_vmrss_kib": None,
            "max_observed_vmhwm_kib": None,
            "gpu_process_memory_mib": None,
            "gpu_process_status": "unsupported_or_not_requested",
            "process_scope": "benchmark child only; not a process-tree peak; not KV-specific",
            "gpu_process_scope": "matched child PID only; null does not mean zero",
        }
    if not isinstance(monitor, dict):
        raise ValueError(f"{case.get('case_id')}: memory_monitor must be an object or null")
    rss = monitor.get("sampled_peak_vmrss_kib")
    hwm = monitor.get("max_observed_vmhwm_kib")
    gpu = monitor.get("sampled_peak_gpu_process_memory_mib")
    for key, value in (("sampled_peak_vmrss_kib", rss), ("max_observed_vmhwm_kib", hwm), ("sampled_peak_gpu_process_memory_mib", gpu)):
        if value is not None and (not finite_number(value) or value < 0):
            raise ValueError(f"{case.get('case_id')}: invalid memory value {key}={value!r}")
    return {
        "sampled_peak_vmrss_kib": rss,
        "max_observed_vmhwm_kib": hwm,
        "gpu_process_memory_mib": gpu,
        "gpu_process_status": monitor.get("gpu_process_status", "unknown"),
        "process_scope": monitor.get("process_scope", "benchmark child only; not a process-tree peak"),
        "gpu_process_scope": monitor.get("gpu_process_scope", "matched child PID only; null does not mean zero"),
    }


def load_run(run_dir: Path, matrix: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    failure_path = run_dir / "failure.json"
    if failure_path.exists():
        raise ValueError(f"failed run found: {failure_path}")
    metadata_path = run_dir / "metadata.json"
    plan_path = run_dir / "plan.json"
    summary_path = run_dir / "summary.json"
    missing = [str(path) for path in (metadata_path, plan_path, summary_path) if not path.is_file()]
    if missing:
        raise ValueError(f"incomplete run at {run_dir}; missing {', '.join(missing)}")
    metadata = read_json(metadata_path)
    plan = read_json(plan_path)
    summary = read_json(summary_path)
    if not all(isinstance(value, dict) for value in (metadata, plan, summary)):
        raise ValueError(f"invalid run JSON object at {run_dir}")
    if metadata.get("status") != "success" or summary.get("status") != "success" or summary.get("planning_only"):
        raise ValueError(f"run is not complete and successful: {run_dir}")
    if metadata.get("planning_only"):
        raise ValueError(f"planning-only run cannot be summarized: {run_dir}")
    requested = metadata.get("requested", {})
    repetitions = requested.get("repetitions")
    if repetitions != EXPECTED_SAMPLES:
        raise ValueError(f"{run_dir}: expected repetitions={EXPECTED_SAMPLES}, got {repetitions!r}")
    mode = summary.get("mode", requested.get("mode", "decode"))
    if mode not in ("decode", "prefill"):
        raise ValueError(f"{run_dir}: unsupported mode {mode!r}")
    if requested.get("mode", mode) != mode:
        raise ValueError(f"{run_dir}: metadata/summary mode mismatch")
    measurements = summary.get("measurements")
    planned = plan.get("randomized_order")
    if not isinstance(measurements, list) or not isinstance(planned, list):
        raise ValueError(f"{run_dir}: summary measurements and plan randomized_order must be lists")
    if summary.get("measurement_count") != len(measurements) or len(planned) != len(measurements):
        raise ValueError(f"{run_dir}: planned, completed, and declared measurement counts differ")
    planned_ids = [item.get("case_id") for item in planned if isinstance(item, dict)]
    measured_ids = [item.get("case_id") for item in measurements if isinstance(item, dict)]
    if len(planned_ids) != len(planned) or len(measured_ids) != len(measurements) or set(planned_ids) != set(measured_ids):
        raise ValueError(f"{run_dir}: plan and summary case IDs differ")

    run_info = {
        "run_path": str(run_dir.relative_to(matrix)),
        "mode": mode,
        "created_at": metadata.get("created_at"),
        "finished_at": metadata.get("finished_at"),
        "seed": summary.get("seed"),
        "repetitions": repetitions,
        "batch": requested.get("batch"),
        "ubatch": requested.get("ubatch"),
        "require_backend": requested.get("require_backend", "any"),
    }
    extracted = []
    seen = set()
    for measurement in measurements:
        if not isinstance(measurement, dict):
            raise ValueError(f"{run_dir}: summary measurement is not an object")
        case_id = measurement.get("case_id")
        case_path = safe_case_path(run_dir, measurement.get("case_file", ""))
        case = read_json(case_path)
        if not isinstance(case, dict) or case.get("status") != "success":
            raise ValueError(f"{case_path}: case is not successful")
        if case.get("case_id") != case_id:
            raise ValueError(f"{case_path}: case ID does not match summary")
        result = case.get("result")
        if not isinstance(result, dict):
            raise ValueError(f"{case_path}: raw result object is missing")
        depth = measurement.get("depth")
        kv_type = measurement.get("kv_type")
        if not isinstance(depth, int) or isinstance(depth, bool) or not isinstance(kv_type, str) or not kv_type:
            raise ValueError(f"{case_path}: invalid depth or KV type")
        if not finite_positive(measurement.get("avg_ts")) or not finite_positive(result.get("avg_ts")):
            raise ValueError(f"{case_path}: avg_ts must be finite and positive")
        if not math.isclose(float(measurement["avg_ts"]), float(result["avg_ts"]), rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"{case_path}: summary avg_ts differs from raw result")
        samples = {}
        for name in ("samples_ts", "samples_ns"):
            values = result.get(name)
            if not isinstance(values, list) or len(values) != EXPECTED_SAMPLES:
                count = len(values) if isinstance(values, list) else None
                raise ValueError(f"{case_path}: {name} expected {EXPECTED_SAMPLES} samples, got {count}")
            if not all(finite_positive(value) for value in values):
                raise ValueError(f"{case_path}: {name} contains a non-finite or non-positive value")
            samples[name] = values
        case_mode = measurement.get("mode", mode)
        if case_mode != mode or case.get("mode", mode) != mode:
            raise ValueError(f"{case_path}: case mode does not match run mode")
        if case.get("depth") != depth or case.get("kv_type") != kv_type:
            raise ValueError(f"{case_path}: case settings do not match summary")
        if mode == "prefill" and case.get("prompt_tokens", depth) != depth:
            raise ValueError(f"{case_path}: prefill prompt token count differs from depth")
        key = (mode, depth, kv_type)
        if key in seen:
            raise ValueError(f"{run_dir}: duplicate measurement for {key}")
        seen.add(key)
        memory = read_memory(case)
        extracted.append({
            "case_id": case_id,
            "mode": mode,
            "depth": depth,
            "kv_type": kv_type,
            "avg_ts": float(measurement["avg_ts"]),
            "samples_ts": samples["samples_ts"],
            "samples_ns": samples["samples_ns"],
            "kv_runtime_mib": kv_allocations(case),
            "sampled_peak_vmrss_kib": memory["sampled_peak_vmrss_kib"],
            "max_observed_vmhwm_kib": memory["max_observed_vmhwm_kib"],
            "gpu_process_memory_mib": memory["gpu_process_memory_mib"],
            "gpu_process_status": memory["gpu_process_status"],
            "process_scope": memory["process_scope"],
            "gpu_process_scope": memory["gpu_process_scope"],
            "process_elapsed_seconds": case.get("process_elapsed_seconds"),
        })
    return run_info, extracted


def aggregate(matrix: Path) -> dict[str, Any]:
    run_dirs = discover_run_dirs(matrix)
    if not run_dirs:
        raise ValueError(f"no benchmark runs found under {matrix}")
    buckets: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    runs = []
    for run_dir in run_dirs:
        run_info, measurements = load_run(run_dir, matrix)
        runs.append(run_info)
        for measurement in measurements:
            key = (measurement["mode"], measurement["depth"], measurement["kv_type"])
            buckets.setdefault(key, []).append({"run_path": run_info["run_path"], **measurement})
    groups = []
    for (mode, depth, kv_type), members in sorted(
        buckets.items(),
        key=lambda item: (0 if item[0][0] == "decode" else 1, item[0][1], KV_ORDER.get(item[0][2], 100), item[0][2]),
    ):
        members.sort(key=lambda item: item["run_path"].lower())
        if len(members) != EXPECTED_FRESH_RUNS:
            raise ValueError(f"{mode}/{depth}/{kv_type}: expected {EXPECTED_FRESH_RUNS} fresh runs, got {len(members)}")
        unique_by_source: dict[str, set[float]] = {}
        for member in members:
            for allocation in member["kv_runtime_mib"]:
                unique_by_source.setdefault(allocation["source"], set()).add(allocation["mib"])
        unique_sorted = {source: sorted(values) for source, values in sorted(unique_by_source.items())}
        allocation_values = sorted({value for values in unique_sorted.values() for value in values})
        vmhwm_values_mib = [item["max_observed_vmhwm_kib"] / 1024 for item in members if item["max_observed_vmhwm_kib"] is not None]
        rss_values_mib = [item["sampled_peak_vmrss_kib"] / 1024 if item["sampled_peak_vmrss_kib"] is not None else None for item in members]
        groups.append({
            "mode": mode,
            "depth": depth,
            "depth_semantics": "prompt_tokens" if mode == "prefill" else "context_depth",
            "kv_type": kv_type,
            "fresh_process_count": len(members),
            "samples_per_process": EXPECTED_SAMPLES,
            "run_means_tps": [member["avg_ts"] for member in members],
            "median_of_run_means_tps": statistics.median(member["avg_ts"] for member in members),
            "min_run_mean_tps": min(member["avg_ts"] for member in members),
            "max_run_mean_tps": max(member["avg_ts"] for member in members),
            "unique_kv_allocation_mib": allocation_values,
            "unique_kv_allocation_mib_by_source": unique_sorted,
            "vmhwm_mib_values": vmhwm_values_mib,
            "vmhwm_mib_range": [min(vmhwm_values_mib), max(vmhwm_values_mib)] if vmhwm_values_mib else None,
            "sampled_vmrss_mib": rss_values_mib,
            "gpu_process_memory_mib": [member["gpu_process_memory_mib"] for member in members],
            "gpu_process_status": [member["gpu_process_status"] for member in members],
            "process_memory_scope": "benchmark child only; sampled VmRSS and observed VmHWM, not a process-tree peak and not KV",
            "gpu_process_scope": "matched benchmark child PID only; null does not mean zero; whole-device snapshots are separate",
            "runs": members,
        })
    if {run["mode"] for run in runs} != {"decode", "prefill"}:
        raise ValueError("matrix must contain successful decode and prefill runs")
    return {
        "schema_version": SCHEMA_VERSION,
        "matrix_path": str(matrix),
        "run_count": len(runs),
        "group_count": len(groups),
        "runs": runs,
        "groups": groups,
        "interpretation_limits": [
            "Three fresh-process means are summarized by their median and min/max; no confidence interval is calculated.",
            "Each fresh process retains its five llama-bench samples in samples_ts and samples_ns.",
            "Process memory values are child-only samples; GPU process memory requires a matching child PID.",
            "GPU-wide snapshots are not process memory; none of these memory figures are KV-specific unless labeled as KV allocation.",
            "Caller-provided quality context: F16 and Q8 retrieval passed 9/9; Q4 passed 2/9.",
            "Q4 throughput comparisons are diagnostic only and are not a recommendation; no mobile conclusion is drawn.",
        ],
    }


def fmt(value: Any, digits: int = 2) -> str:
    return "null" if value is None else f"{value:.{digits}f}"


def render_markdown(stats: dict[str, Any]) -> str:
    lines = [
        "# EdgeLLM M1 benchmark matrix",
        "",
        f"Source matrix: `{stats['matrix_path']}`  ",
        f"Successful fresh-process runs: {stats['run_count']}  ",
        f"Mode/depth/KV groups: {stats['group_count']}",
        "",
        "Each group contains three independent benchmark processes. The reported center is the median of those three process means; min/max are descriptive only. The five internal repetition samples for each process are preserved below. No confidence interval is calculated.",
        "",
        "| Mode | Depth meaning | Depth | KV | Run means (t/s) | Median | Min-max | Five samples per run (t/s) | Unique KV allocation (MiB) | Sampled VmRSS (MiB) | VmHWM range (MiB) | GPU child memory (MiB) |",
        "| --- | --- | ---: | --- | --- | ---: | --- | --- | --- | --- | --- | --- |",
    ]
    for group in stats["groups"]:
        means = ", ".join(fmt(value) for value in group["run_means_tps"])
        bounds = f"{fmt(group['min_run_mean_tps'])}-{fmt(group['max_run_mean_tps'])}"
        sample_text = "<br>".join(
            f"{member['run_path']}: [{', '.join(fmt(value) for value in member['samples_ts'])}]"
            for member in group["runs"]
        )
        alloc_text = "; ".join(
            f"{source}: {', '.join(fmt(value) for value in values)}"
            for source, values in group["unique_kv_allocation_mib_by_source"].items()
        ) or "null"
        rss_text = ", ".join(fmt(value) for value in group["sampled_vmrss_mib"])
        hwm_range = group["vmhwm_mib_range"]
        hwm_text = f"{fmt(hwm_range[0])}-{fmt(hwm_range[1])}" if hwm_range else "null"
        gpu_text = ", ".join(fmt(value) for value in group["gpu_process_memory_mib"])
        lines.append(
            f"| {group['mode']} | {group['depth_semantics']} | {group['depth']} | {group['kv_type']} | {means} | "
            f"{fmt(group['median_of_run_means_tps'])} | {bounds} | {sample_text} | {alloc_text} | {rss_text} | {hwm_text} | {gpu_text} |"
        )
    lines.extend([
        "",
        "## Interpretation limits",
        "",
        "- VmRSS and VmHWM are sampled for the benchmark child PID. VmHWM is shown in MiB after converting the recorded KiB value by 1024; neither metric is a process-tree peak or KV-specific measurement.",
        "- GPU child memory is matched by PID. `null` means the child PID was not reported or sampling was unsupported; whole-device before/after snapshots remain separate case evidence.",
        "- Caller-provided quality context: F16 and Q8 retrieval passed 9/9; Q4 passed 2/9. Treat Q4 throughput comparisons as diagnostic only, not a recommendation. No mobile conclusion is drawn.",
        "",
    ])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True, help="matrix root containing repeated bench output directories")
    parser.add_argument("--out", required=True, help="new output directory for stats.json and matrix.md")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    matrix = Path(args.matrix).expanduser().resolve(strict=False)
    out = Path(args.out).expanduser().resolve(strict=False)
    if not matrix.is_dir():
        parser.error(f"matrix directory does not exist: {matrix}")
    if out.exists():
        parser.error(f"output path already exists; refusing to overwrite: {out}")
    try:
        stats = aggregate(matrix)
    except ValueError as exc:
        parser.error(str(exc))
    out.mkdir(parents=True, exist_ok=False)
    with (out / "stats.json").open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(stats, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    (out / "matrix.md").write_text(render_markdown(stats), encoding="utf-8", newline="\n")
    print(f"matrix summary written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
