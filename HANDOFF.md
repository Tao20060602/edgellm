# HANDOFF - read this first on a new system

Last updated: 2026-10-03 (Asia/Hong_Kong). This repository is the source of truth; previous chats and local agent memories are not required.

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
- `scripts/fetch_dataset.py` / `locks/quality.json`: immutable WikiText-2 identities and predeclared thresholds; corpus stays local.
- `scripts/quality.py`: PPL with materialized context/type/CUDA allocation/offload validation; fit probes and execution failures retained.
- `scripts/server_eval.py` / `eval/server_suite.json`: original synthetic retrieval and warm HTTP TTFT, no request reuse or RAM prompt cache; prompts/requests/SSE saved.
- `scripts/run_matrix.py` / `summarize_matrix.py`: three shuffled sweeps per prefill/decode mode, five repeats/case, raw distributions and median/min/max, PID memory sampling.
- `scripts/profile_paths.py` / `publish_evidence.py`: graph-node traces and small CSVs, public copied records with hostname redaction/hash manifest; originals retained.

## Validated and pending

M0 suites had6 tests; M1 has22. Linux fake-subprocess checks and GitHub Ubuntu/Windows jobs pass, not a proof of native Windows CUDA execution. Historical initial Windows fixture suite was unusually slow (~974s), cause not established. Do not infer inference performance from test duration.

Upstream CPU Release build succeeded under Linux with existing CMake 3.28.3/G++13.3/Ninja. Official model size and SHA256 verified. Initial direct F16/Q4 checks and a six-case scratch run passed. Recorded smoke [003 report](results/cpu-smoke-20261002-003/report.md) and [002 report](results/cpu-smoke-20261002-002/report.md) each have six cases with three repetitions, exact raw streams and SHA256SUMS. Runner SHA256 matches committed script bytes; upstream/model identities match locks. Both show expected KV allocation reduction but noisy throughput, so no speedup claim. Public metadata redacts only hostname.

M1a small-model CUDA baseline is complete: 90 formal case processes/450 internal repeats; separate prefill/decode, quality, warm TTFT, KV allocation, process host memory and profiler evidence. See [M1_REPORT](docs/M1_REPORT.md) and [public records](results/m1-cuda-20261003/). Q8 PPL15.2275 versus F16 15.2486 passes the2% gate; Q4 PPL48.3775 fails the10% gate. Retrieval is F16/Q8 9/9, Q4 2/9. Q4 speed is diagnostic only. Q8 saves46.875% of KV allocation but prefill is slower; decode has outliers. No stable speedup claim.

Explicitly disabling the server's separate RAM cache lowered observed F16 host VmHWM from about8.5GiB to930MiB with the same suite; this is a configuration effect. No upstream runtime/kernel code was changed. No mobile or4B result exists. Process GPU peak is unavailable under this WSL PID reporting; never substitute whole-device snapshots. Synthetic bench is not quality or request-latency evidence.

Remote recovery verification: freshly cloned public GitHub commit `32ee4bcbf6ffa643d5f200b79e7fb743f7317f3e` into an independent Linux directory. Six tests passed; all 49 files listed in the two SHA256SUMS verified unchanged after GitHub round trip; bootstrap `--fetch-only` fetched and checked out the exact locked upstream without using the original source clone. Full second build/model redownload was not repeated. All four CI jobs passed (Ubuntu/Windows x Python3.11/3.13) for this implementation commit: https://github.com/Tao20060602/edgellm/actions/runs/37031818499 . Later handoff-only commits do not change tested scripts or evidence.

## Important findings for the next agent

- CUDA already has quantized-KV FA paths, including in-kernel V dequant. Do not implement the proposed generic dequant+FA fusion as if absent.
- F16/Q8/Q4 KV comparison uses explicit FA on. Quantized V requires FA in the pinned version.
- Actual buffer allocation is padded; small contexts allocate at least 256 cells in the observed CPU run. Read log allocation rather than equating requested tokens with allocation.
- Pinned OpenCL `supports_op` deliberately rejects mixed/quantized FA paths on A7X/Adreno740 for compiler crash history and falls attention back to CPU. This matters to Xiaomi13. Keep the guard and verify execution path; see DEVICE_PLAN and RESEARCH.
- Shared libraries contain actual kernels; hashing only the small launcher would miss changes. Hashes of adjacent shared libraries and CMakeCache are recorded.
- Quantized types enable Hadamard rotation by default, F16 does not. Future runs should record non-secret overrides: LLAMA_ATTN_ROT_DISABLE, GGML_CUDA_DISABLE_GRAPHS, CUDA_VISIBLE_DEVICES, OMP_NUM_THREADS, OMP_PROC_BIND. Never dump API-key variables or the whole environment.
- Large-query CUDA attention can convert quantized K/V into temporary F16; the single-token vector path reads compressed KV internally. Profile the phase before choosing a fusion patch.
- llama-bench stores/restores a host context snapshot before timed decode; its VmHWM and D2H/H2D include tool costs, not necessarily online offload.
- cache_prompt=false means no request reuse; it does not disable the default8192MiB server RAM cache. The current runner also uses `--cache-ram 0` and verifies its log. Keep server-002 and server-003 distinct.

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

## Next concrete task: M2 path isolation

Completed baseline task: [Issue #1](https://github.com/Tao20060602/edgellm/issues/1). Read M1_REPORT and [M1_RUNBOOK](docs/M1_RUNBOOK.md) for the complete CUDA, quality, TTFT and profiler procedure. Read ROADMAP/EXPERIMENTS/SOURCE_MAP before changing an experiment. The command below is only short bring-up:

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

86 is for the observed RTX3080; change architecture for a different GPU. Use --cuda-compiler /path/to/nvcc if needed. Verify actual placement logs.

Next: isolate Q8 prefill conversion, rotation, attention and synchronization with explicit profiling ranges. Separately triage Q4 PPL/retrieval degradation with reference backend/rotation controls and relevant backend-op numerical tests. Whole-process dequant shares are only3.5-3.7%; do not promise large gains before phase attribution. A performance patch needs its own matched baseline and quality validation.

M1b is pending: official4B candidate identity and conservative memory/context limits in DENSE_MODEL_PLAN; not downloaded or locally hash-verified. Add explicit experiment-lock support rather than overwriting the0.6B lock, and predeclare new quality inputs/thresholds. Phone OS/driver, available RAM and future development OS remain unconfirmed.

Mobile ARM CPU should follow after desktop baseline; do not start with an unvalidated OpenCL kernel on this phone. No paid compute is authorized.

## Original-machine locations (optional, not dependencies)

Windows repository was created in this chat workspace under `work/edgellm`. Linux source/build/model/scratch runs are under `/opt/edgellm` in the only installed WSL distribution `NanoVLLM-Ubuntu`. This choice reflects available tools, not a requirement on the next system. `/opt/nano-vllm` and its environment were not changed. Everything needed to rebuild is in this GitHub repository; large model/build files are fetched/rebuilt from the locks.

## Session closure checklist

Update this file with exact status and result links; record failures in WORKLOG; commit reviewed configs/scripts/selected raw evidence; push; verify remote HEAD and relevant CI. M1a is a small-model baseline; M1b, a self-written optimization and mobile validation remain unfinished.
