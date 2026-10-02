# EdgeLLM

Memory-Efficient LLM Inference on Resource-Constrained Devices

研究问题：**KV Cache 量化节省的内存，何时能转化为更低延迟、更高吞吐？何时被反量化和执行路径成本抵消？**

以 llama.cpp 为实验基础，先在桌面 CPU/CUDA 上建立可解释的基线，再迁移到真实端侧硬件。桌面结果不能冒充 Android/ARM/Adreno 结果。

## 当前成果

这是已启动的研究项目，尚未有自研 runtime 优化或移动端性能结论。

- 锁定 llama.cpp commit 与一个官方 Qwen3-0.6B Q8_0 GGUF；权重、构建、模型均可重建。
- 独立的 KV 实验 runner：固定权重，比较 F16/Q8_0/Q4_0 KV，记录原始 JSON、日志和校验信息。
- 实验规划、源码地图、质量门槛、决策记录与跨系统交接均在仓库内。
- 已完成六种配置的真实 CPU smoke，每种三次重复；保存了两组原始记录。512 token 深度的 KV allocation 从 F16 的84.00 MiB降到Q8的44.62、Q4的23.62 MiB，吞吐波动较大，尚不能证明提速。见 [实测报告](results/cpu-smoke-20261002-003/report.md)。
- 已验证状态、运行数据和失败记录以 [HANDOFF.md](HANDOFF.md) 为准。
- GitHub 全新克隆的测试、49份证据校验与锁定源码拉取已通过；[Ubuntu/Windows 四组 CI](https://github.com/Tao20060602/edgellm/actions/runs/37031818499) 全部通过。下一步由 [Issue #1](https://github.com/Tao20060602/edgellm/issues/1) 跟踪。

## 从另一台系统接手

依次阅读 [HANDOFF.md](HANDOFF.md)、[AGENTS.md](AGENTS.md)、[实验规范](docs/EXPERIMENTS.md)。无需上一段聊天、Codex 本地记忆或原机器磁盘。

需要 Python 3.11+、Git、CMake 3.18+ 和 C++ 工具链；CUDA 构建另需匹配的 CUDA toolkit。使用自己的独立项目目录。脚本不会安装系统依赖。

```bash
git clone https://github.com/Tao20060602/edgellm.git
cd edgellm
python3 -m unittest discover -s tests -v
python3 scripts/bootstrap.py --backend cpu --jobs 4
python3 scripts/fetch_model.py
python3 scripts/bench.py \
  --binary external/llama.cpp/build-cpu/bin/llama-bench \
  --model models/Qwen3-0.6B-Q8_0.gguf \
  --upstream external/llama.cpp \
  --out results/local/cpu-smoke-001 \
  --depths 0,512 --kv-types f16,q8_0,q4_0 \
  --gpu-layers 0 --threads 4 --repetitions 3 --gen 32
```

下载约 610 MiB，不提交模型或构建产物。Windows 使用 `python`，并按生成器选择 `bin/Release/llama-bench.exe` 等实际位置；native Windows 流程尚待实机验证。Linux/WSL 的实际验证见交接。

## 项目顺序

| 阶段 | 交付与验收 |
| --- | --- |
| M0：可恢复实验基座 | 精确版本、脚本、单元测试、真实小模型 smoke、从 GitHub fresh clone 验证 |
| M1：CPU/CUDA KV 基线 | 固定权重，深度 2K/4K/8K/16K/32K；独立 prefill/decode，实测 KV/RSS/VRAM，生成质量门槛 |
| M2：定位实际瓶颈 | profiler 解释 dispatch、反量化、带宽、临时 buffer；用数据选一个小优化 |
| M3：一个可审查的 runtime 改动 | 独立分支/patch、正确性、匹配 A/B、统计分布、端到端验证和退化边界 |
| M4：真实端侧验证 | 在明确 Android/ARM/GPU 设备上重测，记录温度、持续性能、内存共享与驱动 |

完整 [路线](docs/ROADMAP.md)、[源码地图](docs/SOURCE_MAP.md)、[调研](docs/RESEARCH.md)、[工作日志](docs/WORKLOG.md)。

## 证据边界

`llama-bench` 使用合成 token，不测 tokenizer/sampler，也不是生成质量评测或请求 TTFT。KV allocation、实际填充深度和有效长上下文推理要分别记录。量化改变数值，不要求跨 KV 类型输出逐 token 完全一致，必须增加独立质量评测。

当前 CUDA 已有 attention 内的量化 KV 读取/反量化，不把再次实现相同 fusion 当作新贡献。任何性能改进都须先证明瓶颈，再证明正确性和收益。

项目构建与文档由用户授权的 AI 助手协助，包括 Luna max 子 agent 调研和实验工具实现。AI 协助不等于用户已掌握所有实现；源码阅读与人类复核是项目验收的一部分。
