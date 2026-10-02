# Repository handoff rules

This file and HANDOFF.md are the durable project memory. Read them before work.

- Read `HANDOFF.md`, `docs/DECISIONS.md`, `docs/WORKLOG.md`, and the active issue. Do not rely on chat history or machine-local memories.
- This is an independent EdgeLLM project. Do not change `/opt/nano-vllm`, its environment, models, or existing experiments. On the original host the only installed WSL distro is NanoVLLM-Ubuntu; EdgeLLM uses its own `/opt/edgellm`. Other hosts may use native Linux, Windows or Android and should follow their own environment.
- Upstream and model versions are in `locks/`. Do not silently update them. Record the new revision and redo a baseline after an upgrade.
- Fetch upstream with `scripts/bootstrap.py`; preserve a dirty checkout, never reset it automatically. Read upstream instructions before modifying upstream code.
- Save each run in a new directory. Record exact argv, model/binary hashes, host, source SHA, dirty state, stdout, stderr, errors and scope of each metric.
- Keep raw performance data and unsuccessful experiments. Never turn `dry-run`, fake binaries, theoretical memory, or third-party measurements into project results.
- Do not use `llama-bench` tok/s as TTFT, quality, full request latency or mobile performance. Keep allocation, occupied KV depth and processed prompt length separate.
- Quantized KV needs a quality gate; output mismatch alone is not failure, simple fluent answers alone are not validation. Predeclare datasets/thresholds before comparing changes.
- CUDA already supports quantized KV within attention. Profile the actual graph/kernel before choosing an optimization. No unmeasured speedup claims.
- Free local experiments are authorized. Paid GPU/cloud runs need explicit purpose, commands, time/cost ceiling and renewed user approval. Do not make external upstream issues/PRs/comments without direct authorization.
- Use Luna with max reasoning when the user requests agents; assign non-overlapping file ownership. Do not undo others' work.
- At the end of substantive work, update HANDOFF, WORKLOG and the relevant issue, commit reproducible scripts/configs/evidence, and verify GitHub contains them. User requested this repository as the source of truth across systems.
- Never commit credentials, user-specific private data, model weights, build trees, caches or absolute local paths as portable defaults.

Recommended workflow: source reading -> hypothesis -> correctness gate -> paired measurement -> explanation -> handoff. Update failed attempts with their causes and remedies.
