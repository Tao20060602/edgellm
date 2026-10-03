# KV cache quality gate

This gate checks the quality impact of K/V cache storage precision for one fixed GGUF model. It uses llama.cpp's Wikitext-2 raw test split at the pinned `ggml-org/ci` dataset revision in [`locks/quality.json`](../locks/quality.json). The archive and the selected member are verified by SHA256 before use; the full corpus is fetched locally and is not committed.

## Predeclared experiment

The fixed matrix uses one unchanged model and runs K and V at the same type: `f16/f16`, `q8_0/q8_0`, and `q4_0/q4_0`. Every case enables Flash Attention, requests 99 GPU layers, uses context 2048, 8 chunks, batch and ubatch 512, 4 threads, and `-v` so backend/offload/KV logs are retained. The runner verifies the reported model PPL and uncertainty, plus the actual chunk/context/batch log, CUDA offload, and runtime K/V types. The pinned upstream logs tokenization activity but do not report the full tokenized input length, so the runner records `input_token_count: null` with an explicit status. It saves exact argv and raw stdout/stderr with hashes for every case.

Verbose logs can contain repeated initialization records. In the pinned source, `common_init_from_params`'s `common_init_result` may run `common_params_fit` first to measure device memory using a temporary no-allocation model/context; it then loads the real model and creates the context used by perplexity. The runner retains every reported offload and KV record, requires all offload summaries to agree, and validates the final pre-evaluation K/V types and nonzero CUDA KV allocation as the materialized PPL context. Earlier records are labeled as fit probes. This avoids treating a zero-byte sizing buffer as the evaluated cache. The pinned verbose run did not emit a CUDA device-count banner, so CUDA execution is established from the consistent full-layer offload summary and positive materialized `CUDA0` KV buffer; device-count lines are retained when available but are optional.

The measured gate is the point-estimate relative increase over the same-run F16/F16 baseline:

`100 * (candidate_mean_ppl / f16_mean_ppl - 1)`

Q8_0 must be at most 2.0%; Q4_0 must be at most 10.0%. The upstream absolute PPL uncertainty is recorded, but this predeclared gate does not combine uncertainty estimates. These are project acceptance thresholds, not claims that the differences are statistically significant. A valid threshold miss is reported as `evaluation_failed` with a complete summary and a nonzero exit status. Missing output, timeout, non-finite values, or runtime/configuration mismatch are `invalid_execution`; the runner preserves available evidence and stops.

## Scope and limits

At context 2048, the pinned upstream implementation scores 1023 next-token targets per completed chunk: `count += n_ctx - first - 1`, with `first = n_ctx/2`. The tool derives the scored-target total from the runtime-reported chunk count, not from the requested plan. The preserved CUDA run log reports `calculating perplexity over 8 chunks, n_ctx=2048, batch_size=512`; it does not report total tokenized input length, so that value remains unavailable rather than inferred. Eight completed chunks imply 8184 scored targets under this pinned source path. The program tokenizes the supplied file but only evaluates the first 8 context chunks. This is a short quality smoke gate and does not establish quality over a 32K context or the complete test set. Long-context quality remains pending. No logits-dump file is written; upstream documents such files at multi-GiB scale.

The baseline is within-run and uses the same model, dataset bytes, binary, source revision, flags, and machine. This isolates KV cache type for this harness but says nothing about differences between tokenizers/models, backends, drivers, or devices. The standard output and error, invocation, hashes, parsed values, and provenance are retained for review.

## Dataset provenance

The source is the immutable Hugging Face dataset commit [`927b3642933080f1b0e811e2f916e14c292992f9`](https://huggingface.co/datasets/ggml-org/ci/commit/927b3642933080f1b0e811e2f916e14c292992f9), matching llama.cpp's [`get-wikitext-2.sh`](https://github.com/ggml-org/llama.cpp/blob/46ca246de9bb1c35269722a6240d37d9dfd79cad/scripts/get-wikitext-2.sh) URL and member path. The Git-LFS pointer at that revision gives the ZIP object's SHA256 and byte length; the member hash was independently computed from the downloaded immutable archive.
