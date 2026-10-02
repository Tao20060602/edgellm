"""Fetch pinned upstream and optionally build. Requires Git and CMake; no pip packages."""

import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    subprocess.run([str(a) for a in args], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--cuda-arch", default="86", help="Use target device compute capability")
    parser.add_argument("--cuda-compiler", help="Optional full path to nvcc")
    parser.add_argument("--fetch-only", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    lock = json.loads((ROOT / "locks/upstream.json").read_text(encoding="utf-8"))
    source = ROOT / "external/llama.cpp"
    source.parent.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        run("git", "init", source)
        run("git", "-C", source, "remote", "add", "origin", lock["repository"])
        run("git", "-C", source, "fetch", "--depth", "1", "origin", lock["commit"])
        run("git", "-C", source, "checkout", "--detach", lock["commit"])
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip()
    if head != lock["commit"] or dirty:
        raise SystemExit("Upstream differs from lock or has changes. Preserve your work; use a new checkout.")
    if args.fetch_only:
        return
    build = source / ("build-" + args.backend)
    command = ["cmake", "-S", source, "-B", build, "-DCMAKE_BUILD_TYPE=Release",
               "-DGGML_CUDA=" + ("ON" if args.backend == "cuda" else "OFF"),
               "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF",
               "-DLLAMA_BUILD_SERVER=OFF", "-DLLAMA_BUILD_MTMD=OFF"]
    if args.backend == "cuda":
        command += ["-DCMAKE_CUDA_ARCHITECTURES=" + args.cuda_arch]
        if args.cuda_compiler:
            command += ["-DCMAKE_CUDA_COMPILER=" + args.cuda_compiler]
    run(*command)
    run("cmake", "--build", build, "--config", "Release", "--target", "llama-bench", "-j", args.jobs)


if __name__ == "__main__":
    main()
