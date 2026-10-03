# Decisions

## D001 - 2026-10-02: Independent repository with pinned upstream

The user authorized GitHub repository creation, planning, project bring-up and durable remote workflow records. They explicitly selected public visibility and Luna max subagents. This is an independent measurement/learning repository, not an upstream contribution submission.

Use `locks/upstream.json` rather than tracking moving master. Do not vendor the entire upstream repository. Later changes can be distributed as a small patch or an explicitly pinned fork after the optimization is chosen.

## D002 - Fixed Q8 weights, vary KV first

The initial model is official `Qwen/Qwen3-0.6B-GGUF`, immutable revision and SHA256 in `locks/model.json`. A small dense-attention model isolates setup risks; it is not the final 3B/4B performance target. Existing nano-vLLM model/runtime are kept independent.

Keep Q8_0 weights fixed while varying F16/Q8_0/Q4_0 K and V. Use explicit FlashAttention on in all matched arms because quantized V requires it in the pinned llama.cpp. Only add a separate weight-quantization experiment after the KV harness and quality gate are working. Do not requantize Q8 into Q4 for a final quality comparison: generate each weight format from the same original source/checkpoint.

## D003 - No predetermined fusion contribution

Pinned CUDA `fattn-common.cuh` and `fattn-vec.cuh` already contain Q4/Q8 V dequantization invoked inside attention. Other variants have their own paths. Determine selected path with profiling. Candidate improvements remain hypotheses, not committed implementations.

## D004 - Desktop first, device claims later

Original host was live-checked: RTX 3080 Laptop, 16 GiB VRAM; WSL Linux; Intel i7-11800H. CPU smoke establishes the experiment pipeline. User subsequently confirmed Xiaomi 13 with 12 GB RAM and 256 GB storage as the candidate Android device. Available RAM, OS/driver and future development OS remain unconfirmed. See DEVICE_PLAN; begin with ARM CPU, gate GPU support separately.

## D005 - GitHub is sufficient to resume

HANDOFF states completed work, exact next command, pending questions and evidence limits. WORKLOG records failed paths as well as successful work. Raw records live under results/ with provenance; generated local runs default to ignored results/local/. Promote selected small records with `git add -f`, review for sensitive paths, and link a run report. No model weights or machine-only dependency can be necessary for resuming.

## D006 - 2026-10-03: Keep negative quality and noisy performance results

M1 thresholds/prompts were published before inference. Q4 retrieval failed the gate (2/9 versus F16/Q8 9/9), including the rerun with RAM cache explicitly disabled. Preserve its throughput as diagnostic, not a quality-equivalent recommendation. Keep slow matrix runs and all internal samples; do not retrofit outlier exclusions or claim stable speedup from three means on an unlocked laptop GPU.

## D007 - Distinguish request reuse, RAM snapshots and benchmark state

`cache_prompt=false` leaves upstream's default 8192MiB server RAM cache enabled. M1 server-002 demonstrates this; server-003 adds `--cache-ram 0`, verifies the log, and repeats unchanged prompts/thresholds. Record both configurations separately. Also distinguish llama-bench's host context-state snapshot from actual service memory. Host VmHWM, sampled RSS, KV buffer allocation and whole-device VRAM are different metrics.

## D008 - Target a phase and preserve profiler limits

F16 and quantized KV follow different default rotation/attention paths. Trace CUDA graph nodes so replay is visible. The whole-process trace supports dispatch observation, not a decode-only time share. Quantized decode already reads compressed KV inside attention; observed large-query attention still has separate K/V-to-F16 conversion. The next task measures that Q8 prefill path before selecting a patch. Q4 numerical failure needs independent diagnosis, not a faster invalid configuration.

## D009 - Keep dense expansion separate

The current measured model is 0.6B. The official 4B revision/file hash and conservative context/memory budget are documented but not downloaded or verified locally. Add separate experiment locks and new quality evidence before expanding. Desktop results never establish Xiaomi13 performance; no paid compute is authorized.
