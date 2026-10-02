# GitHub research notes

Checked on 2026-10-02. These are third-party source observations, not EdgeLLM measurements. Upstream experiment baseline is pinned at `46ca246de9bb1c35269722a6240d37d9dfd79cad`; other repositories/issue status can change.

## What the proposed project needs to correct

1. llama.cpp is a good runtime study target, but it has substantial existing quantized-KV/FA work. Do not assume Q4 KV always passes through a full FP16 intermediate tensor before attention.
2. An allocation at large context plus a short prompt is not successful full-length context inference.
3. Smaller KV does not imply faster decode, and coherent short output is not a quality gate.
4. Backend support must be checked per operator, type, head dimension, GPU/driver and source version; a README weight-type list is not quantized-KV support proof.

## llama.cpp: baseline and direct code study

[Repository](https://github.com/ggml-org/llama.cpp), [pinned bench README](https://github.com/ggml-org/llama.cpp/blob/46ca246de9bb1c35269722a6240d37d9dfd79cad/tools/llama-bench/README.md).

Learn graph construction, backend dispatch, buffer placement and quant block storage together. Its bench supports context depth and explicit K/V types, making it useful for controlled experiments. The synthetic bench deliberately excludes tokenizer/sampler time, so request TTFT and quality need separate tools.

Pinned CUDA FA code contains Q4/Q8 `dequantize_V` helpers called within the vector attention kernel. Dispatch can select other variants, so trace actual kernels. See SOURCE_MAP. `GGML_CUDA_FA_QUANTS` defaults to `q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16`; `GGML_CUDA_FA` enables FA, while `GGML_CUDA_FA_ALL_QUANTS` is a deprecated alias for selecting all combinations. Confirm CMakeCache and actual kernel dispatch rather than assuming a format is compiled.

## Adreno quantized-KV experiment

[Discussion #24109](https://github.com/ggml-org/llama.cpp/discussions/24109), [experiment fork](https://github.com/hyeoktae/llama.cpp), [functional snapshot 26c0831](https://github.com/hyeoktae/llama.cpp/tree/26c0831fc09bbb8a6cd18e66c115577438bad609).

The author tested OpenCL on RedMagic 10 Pro / 8 Elite / Adreno 830 with 16 GB. They added quantized `set_rows`, a dequant path in existing FA, and a KV state-view fix. Reported 64K Gemma KV allocations were 1054 MiB F16 versus 296 MiB Q4, while throughput stayed roughly 4.5–4.8 tok/s. This shows a useful memory experiment; it does not establish speedup. The author limited quality checks to simple prompts and explicitly distinguished large allocation from full 256K inference.

Learn both the write path (`set_rows`) and read path (FA), and state save/load/view offsets. A quantized format cannot be supported by changing only the attention kernel. The discussion linked a fork rather than a finished upstream contribution. A 2026-10-02 search found no upstream PR by the author; this is a search observation, not proof that no integration of related work exists. Pinned upstream now registers OpenCL Q4/Q8 KV write and FA kernels; the original experimental fork is not necessarily needed for those formats.

There is an important device exclusion: [pinned OpenCL dispatch](https://github.com/ggml-org/llama.cpp/blob/46ca246de9bb1c35269722a6240d37d9dfd79cad/ggml/src/ggml-opencl/ggml-opencl.cpp#L9234) declines mixed F32/F16 and symmetric Q8/Q4 FA on A7X (Adreno 740). The comment describes E031.41 compiler crashes; the guard checks the A7X generation and falls these attention operations back to CPU. Do not remove this guard simply to make a GPU benchmark run.

## Vulkan slowdown report

[Issue #24483](https://github.com/ggml-org/llama.cpp/issues/24483).

This is an AMD RDNA4/RADV large-context generation report, not an NVIDIA or phone benchmark. It also reports substantial repeated-run variance and experiments with KV formats. Proposed cache/split/driver explanations are hypotheses from the discussion. Closure due to stale activity does not establish that the runtime problem was fixed. Use it to design controls for context, driver, thermal state and run order; do not transplant the reported percentage to3080 or Xiaomi13.

## KIVI: quality and asymmetric quantization

[Official repository](https://github.com/jy-yuan/KIVI).

Study why K and V can need different quantization axes: per-channel K and per-token V, recent full-precision residuals and independent long-context evaluation. The repository includes passkey/LongBench evaluation workflows. Its algorithm and PyTorch/CUDA implementation differ from GGML q4_0/q8_0 block formats. We can borrow the evaluation discipline without claiming it is already implemented by llama.cpp or copying reported quality/speedups.

## Other runtime comparisons

| Project | What to learn | Role in this project |
| --- | --- | --- |
| [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp), [parameters](https://github.com/ikawrakow/ik_llama.cpp/blob/main/docs/parameters.md) | low-bit cache formats, K/V precision choices, Hadamard work and FA coverage | compare specific pinned algorithms/paths after upstream baseline; no wholesale engine migration |
| [MLC-LLM](https://github.com/mlc-ai/mlc-llm), [KV int8 request](https://github.com/mlc-ai/mlc-llm/issues/3236) | TVM/Relax compiled artifacts, paged KV, backend specialization | weight quantization is not KV quantization; checked code still had future-KV-quant TODO; later compiler direction |
| [ExecuTorch](https://github.com/pytorch/executorch), [LLM export](https://docs.pytorch.org/executorch/stable/llm/export-llm.html) | export/lowering, per-token int8 KV configuration, backend-specific .pte and delegates | later industrial deployment study; each CPU/GPU/NPU delegate/device needs independent validation |

These alternatives do not need installing for M0. A project with one well-measured optimization is more useful than trying three runtimes simultaneously.

## What current hardware can prove

Desktop i7/RTX3080 can validate x86 CPU/CUDA bring-up, quantized KV allocations, numeric tests and measured desktop dispatch. It cannot validate ARM NEON/OpenCL mobile performance. Xiaomi13 12GB/256GB is the user's actual device candidate: start ARM CPU; confirm GPU driver/backend/operator support before promising an Adreno kernel contribution. See DEVICE_PLAN.

## First chosen milestone

One official small dense model, fixed Q8 weights, F16/Q8/Q4 KV, explicit FA, fresh subprocess per case, logged context allocation and raw JSON. Then add quality, CUDA and larger contexts before choosing any runtime patch. See EXPERIMENTS and ROADMAP.
