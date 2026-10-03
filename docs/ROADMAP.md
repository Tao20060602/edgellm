# Roadmap and acceptance gates

第一目标是一个能测、能解释、能迁移的实验系统。最终想优化什么，必须由测量决定。

## M0 - Bring-up（完成）

锁定 upstream/model；构建 CPU；F16/Q8/Q4 KV 实际 smoke；保存原始输出与 provenance；通过 runner 的失败/超时测试；fresh clone 后能跑测试并找到所有恢复命令。这里的跑通不能称为 runtime 优化。

## M1a - Small-model matched baseline（完成；见 M1_REPORT）

1. 在目标系统构建 CUDA 或实际可用 backend，确认设备、编译选项和真实 offload 日志。
2. 固定 Qwen3-0.6B Q8 权重、batch/ubatch/thread/offload，先扫已填充 KV 深度 2048/4096/8192/16384/32768。每个深度比较三种 KV；只有通过质量门槛的配置进入最终选择。
3. 增加 prefill 测量、热服务请求 TTFT、模型/计算 buffer/KV/RSS/VRAM 的独立记录。现已实现分开的prefill/decode、uncached warm TTFT和process-memory监控；WSL无法提供的进程显存保留null。
4. 增加固定短文本 perplexity 数据（与 hash）和长上下文 retrieval 数据。先写阈值再看结果；报告相对 F16 的误差/失败率、实际 token 数和长度范围。短题 sanity 不替代质量。
5. Dense模型扩展归入独立的M1b，不能把0.6B结果当作4B结论。

门槛：至少 3 组独立 fresh-process A/B；内部重复至少 5 次用于正式性能；随机/交错顺序避免温度漂移；使用相同输入和有效 context；报告原始分布而非最好值。CPU只验证低成本路径，32K CPU 实验需先估计时长。

## M1b - Dense 4B expansion（计划，未下载/实测）

候选revision、文件hash、模型dimension与保守context预算见DENSE_MODEL_PLAN。新增独立实验锁和质量协议，先短context bring-up再扩大。现有runner默认锁仍为0.6B，需要显式支持多实验锁，不能绕过hash验证。

## M2 - Profiler-led selection

观察真实 backend dispatch、FA variant、KV update/set_rows、内存布局/连续化、H2D（若 hybrid）、CPU 调度与 GPU 同步。分别看 prefill/decode。证明某项占总耗时才修改；已有 fused 实现不再重复造。

候选而非承诺：某个 shape/type 的慢 dispatch；重复 layout conversion；小 batch 的调度/同步开销；量化 block 读取和 dequant 的局部优化；受限内存下 CPU/GPU layer 配置。每项必须注明目标 backend、设备、shape 和未覆盖范围。

## M3 - One optimization

一个独立 branch/patch。保留原版切换，运行 backend op 数值测试、质量评测、尾部/非整块/head维度/GQA 等 relevant shapes。执行匹配 A/B，比较 operator 与端到端，报告没有收益或退化的点。如果没有可信 kernel 瓶颈，可做有证据的 runtime配置工具，但不能包装成kernel贡献。

## M4 - Actual edge device

先记录真实手机/ARM设备、Android/NDK版本、RAM、GPU、driver、OpenCL/Vulkan支持。测量共享 RAM 与可用内存，不把桌面显存模型照搬到移动端。锁定模型、输入、backend、温度区间、电源与 sustained performance。CPU/GPU heterogeneity先以 layer placement为实验，不承诺通用NPU调度。

## 学习安排

先读一条调用链，而不是通读整个 llama.cpp。每个阶段写一个自己能讲明白的解释：直觉、公式、张量/量化 block 布局、源码路径、数据、限制。最终对照 NanoKV 的 CPU offload/sparse retrieval 与 EdgeLLM 的 compressed KV；只使用各自独立验证过的结论。
