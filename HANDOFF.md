# HANDOFF - read this first on a new system

Last updated: 2026-10-02 (Asia/Hong_Kong). This repository is the source of truth; previous chats and local agent memories are not required.

## User objective and confirmed choices

Build an explainable EdgeLLM project based on llama.cpp: KV quantization, memory behavior, actual runtime bottlenecks, and eventual real-device validation. It complements server-side NanoKV; it is an independent repository.

User authorized starting the project, creating GitHub repository and recording work remotely. Repository is public: https://github.com/Tao20060602/edgellm . Use Luna max for requested subagents. Actual phone: Xiaomi 13, 12 GB RAM / 256 GB storage. Target future development OS and phone software/driver remain unconfirmed.

## Current implementation

- `locks/upstream.json`: exact llama.cpp commit `46ca246de9bb1c35269722a6240d37d9dfd79cad`.
- `locks/model.json`: official Qwen3-0.6B Q8_0 GGUF, immutable HF revision, size/SHA256. Small dense-attention bring-up target, not final 4B claim.
- `scripts/bootstrap.py`: fetch pinned source, preserve unexpected/dirty checkouts, build CPU or CUDA with existing tools.
- `scripts/fetch_model.py`: independent HTTPS download, size/hash verification; no existing nano-vLLM model/environment dependency.
- `scripts/bench.py`: fixed-weight synthetic decode matrix, separate subprocess per KV/depth, randomized case order, hashes of harness/model/launcher/shared libraries, upstream SHA/dirty, CMake settings, raw logs, errors. No overwriting runs. Stops at first failure.
- `tests/test_bench.py`: success provenance/raw evidence, dry-run, overwrite rejection, timeout, invalid JSON, wrong parameters/nonpositive throughput. POSIX uses actual fake subprocess; Windows mocks the child process, so it does not prove native Windows llama-bench execution.
- `.github/workflows/check.yml`: offline Ubuntu/Windows Python 3.11/3.13 harness checks.

## Validated and pending

Local Windows Python 3.13 suite: 6 passed. Local WSL Python suite: 6 passed (real fake subprocess branch). Windows fixture suite was unusually slow (~974 s), while WSL suite took ~1.8 s; cause not established. Do not infer inference performance from test duration.

Upstream CPU Release build succeeded under Linux with existing CMake 3.28.3/G++13.3/Ninja. Official model size and SHA256 verified. Initial direct F16/Q4 checks and a six-case scratch run passed. Recorded smoke [003 report](results/cpu-smoke-20261002-003/report.md) and [002 report](results/cpu-smoke-20261002-002/report.md) each have six cases with three repetitions, exact raw streams and SHA256SUMS. Runner SHA256 matches committed script bytes; upstream/model identities match locks. Both show expected KV allocation reduction but noisy throughput, so no speedup claim. Public metadata redacts only hostname.

No runtime/kernel code modified, no mobile experiment, no CUDA inference baseline, no TTFT/quality/RSS/VRAM measurement yet. `llama-bench` synthetic token data cannot demonstrate model quality or end-to-end latency. A throughput difference in a small smoke run is not an optimization result.

## Important findings for the next agent

- CUDA already has quantized-KV FA paths, including in-kernel V dequant. Do not implement the proposed generic dequant+FA fusion as if absent.
- F16/Q8/Q4 KV comparison uses explicit FA on. Quantized V requires FA in the pinned version.
- Actual buffer allocation is padded; small contexts allocate at least 256 cells in the observed CPU run. Read log allocation rather than equating requested tokens with allocation.
- Pinned OpenCL `supports_op` deliberately rejects mixed/quantized FA paths on A7X/Adreno740 for compiler crash history and falls attention back to CPU. This matters to Xiaomi13. Keep the guard and verify execution path; see DEVICE_PLAN and RESEARCH.
- Shared libraries contain actual kernels; hashing only the small launcher would miss changes. Hashes of adjacent shared libraries and CMakeCache are recorded.

## Resume commands on Linux

```bash
git clone https://github.com/Tao20060602/edgellm.git
cd edgellm
python3 -m unittest discover -s tests -v
python3 scripts/bootstrap.py --backend cpu --jobs 4
python3 scripts/fetch_model.py
python3 scripts/bench.py \
  --binary external/llama.cpp/build-cpu/bin/llama-bench \
  --model models/Qwen3-0.6B-Q8_0.gguf \
  --upstream external/llama.cpp \
  --out results/local/new-host-cpu-001 \
  --depths 0,512 --kv-types f16,q8_0,q4_0 \
  --gpu-layers 0 --threads 4 --repetitions 3 --gen 32
```

Use a new output directory for every run. This rebuilds procedure/source/model identity; the resulting binary bytes/performance can differ with toolchain and hardware. Record the new environment.

## Next concrete task: M1

Active GitHub task: [Issue #1](https://github.com/Tao20060602/edgellm/issues/1). Read ROADMAP, EXPERIMENTS and SOURCE_MAP, then confirm target OS/device. On NVIDIA host with toolkit:

```bash
python3 scripts/bootstrap.py --backend cuda --cuda-arch 86 --jobs 4
python3 scripts/bench.py \
  --binary external/llama.cpp/build-cuda/bin/llama-bench \
  --model models/Qwen3-0.6B-Q8_0.gguf \
  --upstream external/llama.cpp \
  --out results/local/cuda-bringup-001 \
  --depths 0,512 --kv-types f16,q8_0,q4_0 \
  --gpu-layers 99 --threads 4 --repetitions 3 --gen 32
```

86 is for the observed RTX3080; change architecture for a different GPU. Use `--cuda-compiler /path/to/nvcc` if it is not in PATH. Check actual CUDA backend/placement in raw logs before calling it a GPU result. Add fixed quality datasets/scripts and thresholds before long-context/optimization conclusions. Then expand depths and prefill/TTFT/memory monitoring; profile before choosing a patch.

Mobile ARM CPU should follow after desktop baseline; do not start with an unvalidated OpenCL kernel on this phone. No paid compute is authorized.

## Original-machine locations (optional, not dependencies)

Windows repository was created in this chat workspace under `work/edgellm`. Linux source/build/model/scratch runs are under `/opt/edgellm` in the only installed WSL distribution `NanoVLLM-Ubuntu`. This choice reflects available tools, not a requirement on the next system. `/opt/nano-vllm` and its environment were not changed. Everything needed to rebuild is in this GitHub repository; large model/build files are fetched/rebuilt from the locks.

## Session closure checklist

Update this file with exact status and result links; record failures in WORKLOG; commit reviewed configs/scripts/selected raw evidence; push; verify remote HEAD and relevant CI. Keep unverified work labeled. Do not claim M1 or runtime optimization complete because M0 is runnable.
