# Measurement contract

## Initial experiment

固定一个 Q8_0 权重 GGUF；对 K/V 都选同一种 `f16`, `q8_0`, `q4_0`。CPU：`-ngl 0`；CUDA：配置 `-ngl 99` 并核实全 offload。所有匹配 case `-fa on`。线程默认显式设 4，不跟随机器默认。

当前 runner 命令形态：`llama-bench -m MODEL -p 0 -n 32 -d DEPTH -ctk KV -ctv KV -fa on -ngl 0 -t 4 -r 3 -o json`。

`-d` 是先填充的 token 数；`-n` 是本次测量 decode token 数；`-p 0` 禁止夹杂单独 prefill测试。upstream `n_ctx = n_prompt + n_gen + n_depth` 后还会按 context padding分配。`tg32 @ d2048` 不能称为 2048-token prompt 的 TTFT。

## Metric boundaries

| Metric | Source and interpretation |
| --- | --- |
| synthetic decode tok/s | llama-bench JSON `avg_ts`, `stddev_ts`, samples；不包括 tokenizer/sampler |
| allocated KV MiB | stderr KV buffer / cache-size日志，含实际 padding；不是推断公式，也不代表全部context已使用 |
| peak process RSS | 可选 Linux `/usr/bin/time -v`，含模型/host buffer等，绝不等同 KV内存；WSL不等同 Windows进程总RAM |
| GPU memory | 后续独立 device/process采样，记录采样间隔和漏峰风险；当前CPU smoke不提供 |
| model size | 实际文件字节；不等同resident weight buffers |
| TTFT | 后续热服务 streaming首个有效token事件的client wall clock；注明load/cold/warm、tokenization、prompt cache、queue是否计入 |
| quality | 固定dataset、tokenizer、prompt、评测脚本、hash、样本数；当前bench合成token无质量结论 |

## Theory, not measurements

传统dense attention单序列理想KV payload：`L * T * Hkv * D * (bytes_K + bytes_V)`。GQA用 KV heads，不能用 query heads或hidden_size推测head_dim。

Qwen3-0.6B config：28 layers、8 KV heads、head_dim 128（query heads 16，hidden_size 1024）；有效head维度不要简单取hidden_size/query_heads。F16 每元素2字节；Q8_0 每32元素34字节；Q4_0 每32元素18字节，包含scale。相同K/V格式下payload相对F16约53.125%和28.125%，不是准确50%和25%。实际buffer还需对齐/padding/布局。混合attention、SWA、recurrent state模型要单独推导。

## Failure and validity

失败时保存 argv/stdout/stderr/error 并停止；无 JSON、非零exit、timeout、参数不一致、无有效吞吐都不能产生成功报告。正式 CUDA 结果需日志确认 CUDA context和offload，只有 CLI `-ngl` 不足以证明GPU执行。

模型与binary全SHA256、upstream SHA/dirty、OS、thread/offload、输出raw都要记录。正式性能升级增加编译器/CMakeCache、CPU affinity/power、driver/CUDA、GPU温度/功耗。不要跨OS/backend直接做“优化前后”比较。

## Run promotion

`results/local/` 默认不上传。review具体一组run后复制到 `results/<run-id>/`，另写 report.md。只能上传真正必要、无凭据的日志/配置。每次 run directory必须新建；re-run不能覆盖旧数据。dry-run只生成计划，不算实验。
