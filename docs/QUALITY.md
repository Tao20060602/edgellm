# KV cache quality gate

This gate checks the quality impact of K/V cache storage precision for one fixed GGUF model. It uses llama.cpp's Wikitext-2 raw test split at the pinned `ggml-org/ci` dataset revision in [`locks/quality.json`](../locks/quality.json). The archive and the selected member are verified by SHA256 before use; the full corpus is fetched locally and is not committed.

## Predeclared experiment

The fixed matrix uses one unchanged model and runs K and V at the same type: `f16/f16`, `q8_0/q8_0`, and `q4_0/q4_0`. Every case enables Flash Attention, requests 99 GPU layers, uses context 2048, 8 chunks, batch and ubatch 512, and 4 threads. The runner verifies the reported model PPL, uncertainty, tokenized input count, actual chunk count, CUDA offload, and runtime K/V types. It saves exact argv and raw stdout/stderr with hashes for every case.

The measured gate is the point-estimate relative increase over the same-run F16/F16 baseline:

`100 * (candidate_mean_ppl / f16_mean_ppl - 1)`

Q8_0 must be at most 2.0%; Q4_0 must be at most 10.0%. The upstream absolute PPL uncertainty is recorded, but this predeclared gate does not combine uncertainty estimates. These are project acceptance thresholds, not claims that the differences are statistically significant. A valid threshold miss is reported as `evaluation_failed` with a complete summary and a nonzero exit status. Missing output, timeout, non-finite values, or runtime/configuration mismatch are `invalid_execution`; the runner preserves available evidence and stops.

## Scope and limits

At context 2048, the pinned upstream implementation scores 1023 next-token targets per chunk. Eight chunks therefore score 8184 targets. The program tokenizes the full supplied file but only evaluates its first 8 context chunks. This is a short quality smoke gate and does not establish quality over a 32K context or the complete test set. Long-context quality remains pending. No logits-dump file is written; upstream documents such files at multi-GiB scale.

The baseline is within-run and uses the same model, dataset bytes, binary, source revision, flags, and machine. This isolates KV cache type for this harness but says nothing about differences between tokenizers/models, backends, drivers, or devices. The standard output and error, invocation, hashes, parsed values, and provenance are retained for review.

## Dataset provenance

The source is the immutable Hugging Face dataset commit [`927b3642933080f1b0e811e2f916e14c292992f9`](https://huggingface.co/datasets/ggml-org/ci/commit/927b3642933080f1b0e811e2f916e14c292992f9), matching llama.cpp's [`get-wikitext-2.sh`](https://github.com/ggml-org/llama.cpp/blob/46ca246de9bb1c35269722a6240d37d9dfd79cad/scripts/get-wikitext-2.sh) URL and member path. The Git-LFS pointer at that revision gives the ZIP object's SHA256 and byte length; the member hash was independently computed from the downloaded immutable archive.
