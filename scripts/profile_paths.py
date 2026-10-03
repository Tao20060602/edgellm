"""Capture M1 CUDA path evidence; profiler throughput is not a baseline result."""

import argparse
import subprocess
from pathlib import Path

from bench import cmake_cache_info, run_git, sha256_file, shared_library_info, utc_now, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "model", "upstream", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--nsys", default="nsys")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    metadata = {"status": "running", "started_at": utc_now(),
                "source_sha": run_git(args.upstream, "rev-parse", "HEAD"),
                "source_dirty": bool(run_git(args.upstream, "status", "--porcelain")),
                "model_sha256": sha256_file(args.model), "binary_sha256": sha256_file(args.binary),
                "script_sha256": sha256_file(Path(__file__)),
                "libraries": shared_library_info(args.binary), "build": cmake_cache_info(args.binary),
                "nsys_version": subprocess.check_output([args.nsys, "--version"], text=True).strip(),
                "scope": "whole fresh process: model load, warmup, depth fill and decode; no phase attribution",
                "cases": []}
    write_json(args.out / "metadata.json", metadata)
    try:
        for kv in ("f16", "q4_0"):
            out = args.out / kv
            out.mkdir()
            argv = [args.nsys, "profile", "--trace=cuda,nvtx", "--cuda-graph-trace=node", "--sample=none", "--cpuctxsw=none",
                    "--output", str(out / "capture"), str(args.binary),
                    "-m", str(args.model), "-p", "0", "-n", "32", "-d", "8192",
                    "-b", "512", "-ub", "512", "-ctk", kv, "-ctv", kv,
                    "-fa", "on", "-ngl", "99", "-t", "4", "-r", "1", "-v", "-o", "json"]
            record = {"kv_type": kv, "argv": argv, "started_at": utc_now()}
            metadata["cases"].append(record)
            write_json(args.out / "metadata.json", metadata)
            with (out / "profile.stdout.bin").open("wb") as stdout, (out / "profile.stderr.bin").open("wb") as stderr:
                subprocess.run(argv, cwd=args.upstream, stdout=stdout, stderr=stderr, check=True, timeout=180)
            record["trace_sha256"] = sha256_file(out / "capture.nsys-rep")
            record["reports"] = []
            for report in ("cuda_gpu_kern_sum", "cuda_api_sum", "cuda_gpu_mem_time_sum"):
                command = [args.nsys, "stats", "--force-export=true", "--report", report, "--format", "csv",
                           str(out / "capture.nsys-rep")]
                with (out / (report + ".csv")).open("wb") as stdout, (out / (report + ".stderr.bin")).open("wb") as stderr:
                    subprocess.run(command, stdout=stdout, stderr=stderr, check=True, timeout=180)
                record["reports"].append({"report": report, "argv": command})
            record["finished_at"] = utc_now()
        metadata.update(status="complete", finished_at=utc_now())
    except Exception as exc:
        metadata.update(status="failed", error=str(exc), finished_at=utc_now())
        write_json(args.out / "metadata.json", metadata)
        raise
    write_json(args.out / "metadata.json", metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
