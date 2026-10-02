# cpu-smoke-20261002-003: CPU bring-up evidence

**A reproducible workflow smoke, not a runtime optimization result.** Qwen3-0.6B Q8_0 weights are fixed; only K/V format and filled depth vary. No tokenizer/sampler, quality or request TTFT measurement.

Intel i7-11800H, WSL Linux, CPU backend, 4 threads, explicit FA on, 32 generated synthetic tokens, 3 repetitions within each process, one fresh process per case. Case order seed 20261002 controls order only, not model sampling. CMake 3.28.3 / G++ 13.3 / Release / Ninja, CUDA disabled. Metadata records binary/shared-library/model/harness hashes and clean pinned upstream.

| Filled depth | KV format | Allocated KV MiB (runtime log) | Decode tok/s mean +/- SD |
| --- | --- | --- | --- |
| 0 | f16 | 28.00 | 55.06 +/- 1.85 |
| 0 | q4_0 | 7.88 | 54.89 +/- 2.05 |
| 0 | q8_0 | 14.88 | 53.80 +/- 1.30 |
| 512 | f16 | 84.00 | 38.96 +/- 4.37 |
| 512 | q4_0 | 23.62 | 41.52 +/- 0.26 |
| 512 | q8_0 | 44.62 | 35.95 +/- 5.51 |

Depth 0 allocated 256 slots; depth 512 plus generation allocated 768 slots. This is observed context padding, not a full 2K/32K context experiment. For depth 512, Q8/Q4 allocation is 44.62/23.62 MiB versus F16 84.00 MiB (roughly 53.1%/28.1%). These are KV buffers, not peak process RSS or GPU memory.

Throughput varies substantially between bring-up runs 002 and 003. They were not temperature/power/background controlled and do not establish a statistically valid speedup. The defensible result is that all configured CPU paths ran and produced the expected KV allocation differences. No real prompt quality was evaluated. CUDA/mobile/kernel optimization remains pending.

Raw per-case JSON files reference exact stdout/stderr byte files. This public copy redacts only environment.hostname in metadata; benchmark streams and measurements are unchanged. Original-machine absolute paths are provenance, while portable defaults are in the scripts/locks. Symlinked libraries may appear repeatedly with identical hashes.

For run 003, CMakeCache.txt, build-cpu.log and tests-linux.txt are additionally archived. Windows suite: 6 passed with child-process mocks; Linux suite: 6 passed using actual fake child executables. Neither fake-test throughput nor test duration is model performance.

See [experiment contract](../../docs/EXPERIMENTS.md) and [next task](https://github.com/Tao20060602/edgellm/issues/1).
