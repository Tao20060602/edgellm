# Work log

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
