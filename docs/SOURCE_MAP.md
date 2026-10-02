# Source map: one request to a kernel

Snapshot: [`46ca246de9bb1c35269722a6240d37d9dfd79cad`](https://github.com/ggml-org/llama.cpp/tree/46ca246de9bb1c35269722a6240d37d9dfd79cad). Links are pinned; symbol searches are preferable to stale line numbers.

| Layer | File / symbol | What to inspect |
| --- | --- | --- |
| benchmark request | [tools/llama-bench/llama-bench.cpp](https://github.com/ggml-org/llama.cpp/blob/46ca246de9bb1c35269722a6240d37d9dfd79cad/tools/llama-bench/llama-bench.cpp), `get_cparams`, `test_gen` | context allocation, depth prefill, random token workload, timed scope |
| execution | [src/llama-context.cpp](https://github.com/ggml-org/llama.cpp/blob/46ca246de9bb1c35269722a6240d37d9dfd79cad/src/llama-context.cpp), `llama_decode`, graph compute | ubatch graph build, backend scheduler, async/sync |
| attention graph | [src/llama-graph.cpp](https://github.com/ggml-org/llama.cpp/blob/46ca246de9bb1c35269722a6240d37d9dfd79cad/src/llama-graph.cpp), `build_attn_mha`, `ggml_flash_attn_ext` | FA vs non-FA graph, Q/K/V layout and mask |
| KV storage | [src/llama-kv-cache.cpp](https://github.com/ggml-org/llama.cpp/blob/46ca246de9bb1c35269722a6240d37d9dfd79cad/src/llama-kv-cache.cpp) | typed K/V tensors, n_stream, views, padded allocation, set_rows |
| backend scheduling | [ggml/src/ggml-backend.cpp](https://github.com/ggml-org/llama.cpp/blob/46ca246de9bb1c35269722a6240d37d9dfd79cad/ggml/src/ggml-backend.cpp) | supports_op, graph split, buffer placement, copy/sync |
| CUDA dispatch | [ggml/src/ggml-cuda/fattn.cu](https://github.com/ggml-org/llama.cpp/blob/46ca246de9bb1c35269722a6240d37d9dfd79cad/ggml/src/ggml-cuda/fattn.cu) | architecture, head dimension, GQA, query count, type dependent paths |
| in-attention quant handling | [fattn-common.cuh](https://github.com/ggml-org/llama.cpp/blob/46ca246de9bb1c35269722a6240d37d9dfd79cad/ggml/src/ggml-cuda/fattn-common.cuh), `dequantize_V_q4_0`, `dequantize_V_q8_0`, `get_vec_dot_KQ` | packed nibble/int8 loads, scale, dot products, V unpack |
| CUDA FA vector kernel | [fattn-vec.cuh](https://github.com/ggml-org/llama.cpp/blob/46ca246de9bb1c35269722a6240d37d9dfd79cad/ggml/src/ggml-cuda/fattn-vec.cuh), `dequantize_V` call | values dequantized within FA path; profile if this is selected |
| canonical block layout | [ggml/src/ggml-common.h](https://github.com/ggml-org/llama.cpp/blob/46ca246de9bb1c35269722a6240d37d9dfd79cad/ggml/src/ggml-common.h), `block_q4_0`, `block_q8_0` | scale bytes, block size, exact storage cost |

## Read in this order

1. Run tiny CPU smoke; compare a JSON row and its stderr. Explain what was timed and what was allocated.
2. Follow `llama_decode -> graph -> build_attn_mha -> ggml_flash_attn_ext -> backend`. Draw Q/K/V shapes for Qwen3-0.6B.
3. Read quant block layout and the CPU reference quant math. Explain where scales come from and which errors are introduced.
4. Follow CUDA dispatch and selected kernel in a trace. Having a dequant helper in source does not prove every backend/shape uses it.
5. Inspect the experimental Adreno fork separately; distinguish author measurement, upstream implementation and our own result.

量化 V 在 pinned context 初始化中要求 FA；auto会启用，explicit off会报错，block size还必须整除head dimension。不要把unsupported configuration当作“很慢的量化”。

Upstream AGENTS/CONTRIBUTING包含AI贡献要求。我们当前只读取/构建上游并开发自有实验工具，没有提交上游改动。未来希望贡献时，应重新阅读上游最新规则，先由用户理解问题和代码，再讨论范围。
