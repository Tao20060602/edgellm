# M1 predeclared measurement protocol - 2026-10-03

This protocol is fixed before reading CUDA/PPL/retrieval results. Original host: RTX3080 Laptop 16GiB, SM86, CUDA12.8, WSL. Same locked Qwen3-0.6B Q8_0 weights and llama.cpp source as M0. It is a baseline study; no runtime patch is being tested.

## Correctness and quality

- WikiText-2 test corpus: exact dataset revision, archive/member hashes and thresholds in locks/quality.json. Context2048, 8 chunks. Q8 relative PPL increase must be at most2%; Q4 at most10%, compared with F16 KV. This is a limited corpus PPL screen, not a claim of generation quality or32K quality.
- Independent synthetic retrieval probe: 9 prompts (target content lengths2048/8192/32768, needle positions10%/50%/90%). Original filler generated from a fixed phrase; a single explicitly named record contains a fixed six-digit access code. Fixed dataset specification in eval/server_suite.json; no external copyrighted text is embedded.
- Temperature0; thinking disabled via model chat template. Answer must contain the exact code as an isolated six-digit number. Record every output. Report absolute F16 baseline accuracy even if poor. Candidate gate: no more than one extra failure versus F16 across9 cases; insufficient F16 accuracy (<8/9) makes the probe inconclusive for an overall quality pass. Do not tune prompts/thresholds after observing results.
- Execution/parse/backend errors stop and preserve failure evidence. A quality threshold failure is a negative experiment result, not an excuse to discard raw data or adjust the gate.

## Throughput and memory

- Explicit FA on, full layer offload (`-ngl99` plus actual log proof), batch512, ubatch512, threads4. Compare symmetric F16/Q8/Q4 KV, fixed weights.
- First CUDA bring-up: depth0/512, decode32, three internal repetitions. Not used for formal speedup claims.
- Decode matrix: filled depths2048/4096/8192/16384/32768, generate32, 5 internal repetitions, three independent matrices in seeded shuffled order. Preserve all samples. Do not mix depth with allocated padded context.
- Prefill matrix: prompt lengths2048/4096/8192/16384/32768, depth0, gen0, five repetitions. Report explicitly as synthetic prefill, excluding tokenizer/sampler.
- Monitor Linux process VmRSS/VmHWM with100ms interval and device snapshots with common options across all arms. Distinguish sampled/high-water host RSS, runtime KV buffers, per-process GPU memory if available, and total device memory. CUDA/WSL may not expose process-specific GPU memory; report null rather than substituting total VRAM.
- Request latency: one freshly started, warmed localhost server per KV type, one slot, context33792, prompt cache disabled, fixed original-text chat prompt around2K tokens, three warm requests. Client TTFT is time from beginning HTTP POST to first non-empty streamed content, excludes model load/warmup but includes HTTP, queue, tokenizer and prefill. Report actual prompt token count and cache reuse; no synthetic bench TTFT inference.

## Profiling and interpretation

After uninstrumented runs, profile F16 and Q4 at8K filled depth with Nsight Systems. Record commands, profiler version, kernel summary and selected attention variants. Instrumented times are diagnostic and not part of uninstrumented speed comparison.

Inspect variance, sampled device temperature/power/utilization and workload boundaries. Current laptop runs do not lock GPU clocks/power and cannot establish mobile performance. If noise hides a difference, report that rather than choosing a favorable run.

M1 is complete only when the dataset/quality, CUDA baseline, prefill/decode, request timing, memory scope and profiler observations are recorded with failures and portable handoff. A long-context retrieval failure can be a valid negative result; it restricts any recommendation. M2 must select a concrete bottleneck before an optimization.
