"""Run the predeclared M1 depth matrix with sequential GPU subprocesses."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from bench import sha256_file, utc_now, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "model", "upstream", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = {"started_at": utc_now(), "script_sha256": sha256_file(Path(__file__)),
                "protocol": "docs/M1_PROTOCOL.md", "runs": [], "status": "running"}
    write_json(args.out / "matrix.json", manifest)
    for mode in ("decode", "prefill"):
        for index, seed in enumerate((20261003, 20261004, 20261005), 1):
            run_id = f"{mode}-{index:03d}"
            argv = [sys.executable, str(ROOT / "scripts/bench.py"),
                    "--binary", str(args.binary), "--model", str(args.model),
                    "--upstream", str(args.upstream), "--out", str(args.out / run_id),
                    "--depths", "2048,4096,8192,16384,32768",
                    "--kv-types", "f16,q8_0,q4_0", "--mode", mode,
                    "--gpu-layers", "99", "--threads", "4", "--repetitions", "5",
                    "--gen", "32", "--batch", "512", "--ubatch", "512",
                    "--seed", str(seed), "--require-backend", "cuda",
                    "--monitor-memory", "--sample-interval-ms", "100"]
            record = {"run_id": run_id, "argv": argv, "started_at": utc_now()}
            manifest["runs"].append(record)
            write_json(args.out / "matrix.json", manifest)
            with (args.out / (run_id + ".log")).open("wb") as stream:
                code = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT).returncode
            record.update(returncode=code, finished_at=utc_now())
            manifest["status"] = "failed" if code else "running"
            write_json(args.out / "matrix.json", manifest)
            print(json.dumps(record), flush=True)
            if code:
                return code
    manifest.update(status="complete", finished_at=utc_now())
    write_json(args.out / "matrix.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
