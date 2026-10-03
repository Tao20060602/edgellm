# Work log

## 2026-10-03 - M1a small-model CUDA baseline

User said to continue. Reused two Luna max agents with separate ownership: benchmark/matrix aggregation and quality/dataset parsing. Root handled CUDA builds, actual sequential GPU experiments, server streaming, profiling, reports and GitHub handoff. No paid compute, model expansion or phone connection.

### Protocol and implementation

Published protocol before reading GPU/quality results at commit e9182b0. Quality lock and initial harness were published at7397142 before PPL. WikiText archive/member immutable hashes verified; no dataset text committed. Built locked source forSM86/CUDA12.8 with existing tools, targets llama-bench/perplexity/server; server needs MTMD enabled by its CMake. Full offload and actual K/V allocation checked rather than relying on CLI flags.

Scripts save new directories and preserve invalid execution separately from threshold failures. Added 100ms PID host-memory sampling, raw repetition samples, three shuffled sweeps for each of prefill/decode, warm native-completion SSE TTFT, original retrieval prompts and graph-node Nsight traces. All large trace binaries remain local; public copies contain CSVs, commands and original trace hashes. Structured hostname fields alone are redacted with original/public hash receipts.

### Problems kept in the record

1. quality-001 expected a total token-count log that this version does not output; PPL itself completed but the run was invalidated. Input token count now remainsnull, scored targets derive from actual chunks and pinned source. Default verbosity also omitted runtime proof; enabledverbose.
2. quality-002's verbose logs contained two initializations: common_params_fit creates a no-allocation sizing context, then the real context. Parser's one-offload assumption failed. Source verified; new parser retains consistent full-offload records and validates the final materialized context's KV type/nonzero CUDA allocation. Device-count banner also absent; it is not fabricated.
3. server-001 lacked placement logs; verbose fixed this. Complete server-002 then revealed that request cache_prompt=false still leaves global8192MiB RAM prompt snapshots enabled. Added `--cache-ram 0` for server-003, kept requests/thresholds unchanged, reran all cases. Host VmHWM drops markedly; retrieval result remains identical.
4. profile-001 default graph-level tracing omitted CUDA graph replay nodes. profile-002 collected nodes but stats stopped on SQLite export timestamp checks. profile-003 explicitly refreshes the derived export in its new owned directory and completes. profile-004 adds an exploratory Q8 trace after the performance data; it is not a new confirmatory performance trial.

The intended quality-first workflow was interrupted by parser errors: the retrieval screen completed before the formal matrix, then the final corrected PPL screen completed afterward. Thresholds/prompts were unchanged throughout; all Q4 performance is diagnostic, not selection of a quality-equivalent configuration. This order is recorded rather than silently presented as a clean first attempt.

### Results and limits

90 formal case processes/450 internal repeats completed. Q8 KV allocation is46.875% smaller thanF16, but prefill is9-12% slower and decode has large outliers. No stable speedup claim. Final PPL F16=15.2486±0.53842, Q8=15.2275±0.53733, Q4=48.3775±2.01248. Q8 passes its2% threshold; Q4 exceeds10% by a large margin. F16/Q8 retrieval9/9, Q4 2/9, both cache-on and cache-off. Full data and context scopes in M1_REPORT.

The state-snapshot mechanism in llama-bench contributes host memory and D2H/H2D outside timed decode. Server RAM snapshots contribute separate host memory unless disabled. CUDA single-token attention reads compressed KV internally, while large-query attention still materializesF16. Quantized types additionally use defaultHadamard rotation. Whole-process kernel shares cannot establish a decode bottleneck; conversion accounts for only3.5-3.7% of that whole-process kernel time.

M1a is a0.6B desktop baseline. Independent4B identity/context/memory plan is prepared but not downloaded or measured. ARM/Android validation and a self-written optimization remain pending. Next task: phase-specific profiling of quality-passing Q8 prefill, plus reference/rotation/numerical triage for Q4 before proposing a patch.

### Verification

Linux full suite has22 tests, with real fake-subprocess execution. Quality parser additionally parsed preserved real F16 logs offline; source identifies thefitprobe. Fresh GitHub clone atd86793e passed22 tests, all729 M1 hashes, historical M0 hashes, executed harness/lock byte checks and recalculation of all30 distributions. All four CI jobs passed at https://github.com/Tao20060602/edgellm/actions/runs/37095631337 . CI is portable-tool validation, not native Windows inference. A second full CUDA build/model download was not repeated. Exact limits and next Issue#2 are recorded in HANDOFF.

## 2026-10-02 - Repository bring-up

User authorized researching similar GitHub projects, planning and starting EdgeLLM, creating a GitHub repository and recording work so another system can resume without local/chat memory. User selected public visibility, requested Luna max subagents, and confirmed Xiaomi 13 12 GB / 256 GB.

### Research and decisions

Read upstream README, AGENTS, bench README/implementation, graph/context/KV allocation, CUDA attention and quant block source. Locked upstream `46ca246de9bb1c35269722a6240d37d9dfd79cad`. CUDA quantized attention already handles dequantization internally; do not call implementing that same feature novel. Related-repository review is in RESEARCH.md; code map is in SOURCE_MAP.md.

### Environment and setup

Live-checked original host: Intel i7-11800H, RTX 3080 Laptop 16 GiB VRAM, WSL Linux with existing CMake/Ninja/G++/CUDA12.8. EdgeLLM was initialized independently under `/opt/edgellm`; no changes to `/opt/nano-vllm` or its environment. Windows is used for repository management and portable stdlib harness tests.

Initial source was shallow-cloned on Windows for reading, then cloned with `--no-local` to `/opt/edgellm/external/llama.cpp` for Linux CPU build. Exact SHA is in lock; source fetch can be reproduced with bootstrap.py, without the original clone.

CPU build used Ninja, Release, `GGML_CUDA=OFF`, `LLAMA_BUILD_TESTS=OFF`, `LLAMA_BUILD_EXAMPLES=OFF`, `LLAMA_BUILD_SERVER=OFF`, `LLAMA_BUILD_MTMD=OFF`, target llama-bench, jobs=4. CMake warned OpenSSL headers/libraries were unavailable and disabled HTTPS in upstream. Model download used Python HTTPS independently and passed SHA256 verification. No extra packages installed.

Official Qwen3-0.6B-GGUF repository contained Q8_0 file at the locked revision; did not assume it contained a full F16/Q4/Q6 matrix. Downloaded 639446688 bytes and verified `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031`.

### Early checks and lessons

An initial f16, depth0, gen8, r1 run verified actual benchmark JSON fields. A q4 equivalent verified CPU quantized KV and FA startup. These exploratory checks are not formal matched measurements.

The default bench emitted no stderr allocation logs. Added `-v` to recorded runs so KV allocation/FA/backend evidence is available. Upstream context pads even tiny allocation to 256 cells; do not equate requested depth0/gen8 with just8 stored slots. The binary is a small launcher linked against libllama/ggml/bench shared libraries; archive library hashes as well as executable hash.

Windows PowerShell wildcard arguments passed to rg did not expand some file globs; switched to directory + `-g` filtering. One nested inline Python shell command failed quoting before execution; switched to direct commands/file-based scripts. An `option(...)`-only search initially missed the `set(...)` definition of GGML_CUDA_FA_QUANTS; fuller source reading corrected the inference before publishing. No performance inference was made from these read failures.

Pinned OpenCL dispatch was found to deliberately reject some FA kernels on A7X/Adreno740 because of compiler crash history. Device plan now explicitly starts ARM CPU and preserves fallback; GPU execution of quantized KV cannot be promised merely from Adreno branding.

### Validation and continuation

Final smoke, runner tests, fresh-clone and remote checks are recorded in HANDOFF.md and the linked results report. Next work is M1 CUDA baseline + independent quality gates; mobile ARM CPU follows the same locked experiment. No current kernel change, mobile run or paid GPU experiment.

Saved two six-case runs 002 and 003 after provenance improvements; both match the final harness hash, model lock and clean upstream. Depth512 KV allocations: F16 84.00 MiB, Q8 44.62 MiB, Q4 23.62 MiB. Throughput differed substantially across runs (including Q4 depth512 21.94 +/-12.29 vs41.52 +/-0.26 tok/s), so no optimization claim. Retained both small raw sets to expose variance instead of choosing a favorable result. Public copies redact hostname only; SHA256SUMS records exact public bytes.

Raw upstream logs contain trailing spaces and CMakeCache contains a blank EOF; initial full diff whitespace check reported these. Kept raw evidence bytes unchanged, marked raw streams binary and disabled whitespace lint for captured results only. Authored scripts/docs remain whitespace-checked.

Created own-repository Issue #1 for the next M1 task; a first body-file attempt used the wrong relative path and failed before issue creation, then was corrected. No upstream maintainer issue/comment/PR was created.

Initial commit `32ee4bcbf6ffa643d5f200b79e7fb743f7317f3e` pushed to public main and remote HEAD verified. Fresh GitHub clone in a separate Linux directory passed all six tests, verified all49 evidence hashes and fetched locked upstream with bootstrap.py --fetch-only. Original source/model paths were not needed for this recovery check. Full rebuild/redownload was not repeated. GitHub Actions checks Ubuntu/Windows on Python3.11/3.13; see HANDOFF for the run link.

GitHub Actions run37031818499 completed successfully: four matrix jobs (Ubuntu/Windows x Python3.11/3.13), each passing the six harness tests. Session closure updates only handoff/log text; tested implementation and raw evidence are unchanged. M0 bring-up/recovery is complete; M1 quality/CUDA/memory/TTFT work remains open in Issue#1.
