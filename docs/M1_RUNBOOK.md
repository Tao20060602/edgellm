# M1 reproduction on another Linux/CUDA host

Read HANDOFF and M1_PROTOCOL first. Use a fresh clone; all commands below use repository-relative paths. Existing runs are never overwritten. Change `86` and the compiler path for the new host, then record that host as a new experiment rather than assuming identical binary/performance.

```bash
python3 -m unittest discover -s tests -v
python3 scripts/bootstrap.py --backend cuda --cuda-arch 86 \
  --cuda-compiler /usr/local/cuda-12.8/bin/nvcc \
  --targets llama-bench llama-perplexity llama-server --jobs 4
python3 scripts/fetch_model.py
python3 scripts/fetch_dataset.py --out results/local/datasets/wiki.test.raw
```

The build needs an existing compiler/toolkit. No global dependency installation is performed. The server build also enables upstream MTMD, required by the pinned server CMake target. The WikiText download is verified against both archive and member hashes; do not commit the dataset.

Run quality first:

```bash
python3 scripts/quality.py \
  --binary external/llama.cpp/build-cuda/bin/llama-perplexity \
  --model models/Qwen3-0.6B-Q8_0.gguf --upstream external/llama.cpp \
  --dataset results/local/datasets/wiki.test.raw \
  --out results/local/quality-new-host-001
python3 scripts/server_eval.py \
  --binary external/llama.cpp/build-cuda/bin/llama-server \
  --model models/Qwen3-0.6B-Q8_0.gguf --upstream external/llama.cpp \
  --out results/local/server-new-host-001
```

Read the summaries even when the commands return nonzero. A complete quality threshold failure is a negative experiment, distinct from invalid execution, missing logs or truncation. Synthetic retrieval uses original prompts from `eval/server_suite.json`; gzip prompt bytes, request parameters and raw SSE responses are saved. This is a nine-question diagnostic, not a broad evaluation suite. Native `/completion` receives chat-template-formatted prompts with thinking disabled. Warm TTFT is HTTP POST to first nonempty content, with zero prompt-cache reuse; it excludes model load. Each format starts one new server with three warm requests, which is insufficient for a robust service-latency optimization claim.

The following captures diagnostic performance even for rejected formats. Only configurations passing the quality gates can become recommendations:

```bash
python3 scripts/run_matrix.py \
  --binary external/llama.cpp/build-cuda/bin/llama-bench \
  --model models/Qwen3-0.6B-Q8_0.gguf --upstream external/llama.cpp \
  --out results/local/matrix-new-host-001
python3 scripts/summarize_matrix.py \
  --matrix results/local/matrix-new-host-001 \
  --out results/local/matrix-summary-new-host-001
python3 scripts/profile_paths.py \
  --binary external/llama.cpp/build-cuda/bin/llama-bench \
  --model models/Qwen3-0.6B-Q8_0.gguf --upstream external/llama.cpp \
  --out results/local/profile-new-host-001
```

Run GPU experiments sequentially. The matrix performs three fresh-process sweeps per mode, five repetitions per case, independently shuffled seeds. Prefill counts prompt tokens; decode fills the requested KV depth before measuring 32 generated tokens. Sampling process memory adds overhead equally to all arms; `VmHWM` is the observed process high-water mark, while sampled `VmRSS` can miss a transient peak. WSL may not expose a matching compute-process GPU-memory row, so it remains null. Whole-device snapshots are neither process VRAM nor KV allocation.

Nsight Systems must already be installed for the optional profile command. The trace covers a whole process, including load, warmup and depth filling; its kernel shares must not be labeled decode-only. It provides kernel-name dispatch evidence. Instrumented tok/s are excluded from the performance matrix. Large `.nsys-rep` and SQLite traces stay local, with hashes and small CSV summaries available for publication.

For publication, review source paths and raw logs, then use `scripts/publish_evidence.py --runs <run directories> --out results/<new-public-directory>`. It copies evidence, redacts only the structured hostname field, records the redaction hashes, omits large profiler traces and generates SHA256SUMS. Preserve all original runs. Update the report and checksum manifest after adding derived reports. Verify checksums from a fresh GitHub clone before closing the task.
