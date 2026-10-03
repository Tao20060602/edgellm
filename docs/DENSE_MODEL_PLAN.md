# 下一阶段的 4B 模型预算（尚未下载或实测）

0.6B 用来验证实验工具和查执行路径，不能代表目标 3B/4B 的结果。正式扩大模型前要创建独立实验锁与质量协议，不能悄悄替换 `locks/model.json` 后沿用旧结果。

2026-10-03 只读检查官方 Hugging Face API：

| 项目 | 候选值 |
| --- | --- |
| 模型 | `Qwen/Qwen3-4B-GGUF` |
| GGUF revision | `bc640142c66e1fdd12af0bd68f40445458f3869b` |
| 文件 | `Qwen3-4B-Q8_0.gguf` |
| 下载字节 | 4,280,404,704（约 3.99 GiB） |
| 发布的 LFS SHA256 | `8c2f07f26af9747e41988551106f149b03eb9b5cb6df636027b6bf6278473300` |
| 原模型 config revision | `1cfa9a7208912126459214e8b04321603b3df60c` |
| 层 / Q heads / KV heads / head dimension | 36 / 32 / 8 / 128 |

来源：[固定 GGUF revision](https://huggingface.co/Qwen/Qwen3-4B-GGUF/tree/bc640142c66e1fdd12af0bd68f40445458f3869b)、[固定原模型 config](https://huggingface.co/Qwen/Qwen3-4B/blob/1cfa9a7208912126459214e8b04321603b3df60c/config.json)。上面的文件 hash 是发布元数据，尚未经过本地下载验证。

模型卡称 native context 为 32768，而 config 的 `max_position_embeddings` 为 40960、`rope_scaling=null`。先保守限制 **整个 prompt + generation ≤32768**，不要把最大 KV 深度直接设32768后继续生成。建议最大 decode depth31744、gen32，server ctx32768，retrieval内容目标30720并验证模板后的实际长度。更长范围必须另写协议、单独确认模型支持和质量。

固定 Q8 权重，与现有小模型保持 KV 实验思路一致。在 32000 allocated cells 下，F16 KV 理论值4500 MiB；Q8_0为2390.625 MiB；Q4_0为1265.625 MiB。它们不是总显存需求。权重下载体积也不等于 CUDA resident buffer。

原机16 GiB GPU可作为候选验证环境。先检查空闲内存，以 **12 GiB 估计预算上限**留出显示/driver/计算空间；真实 buffer 明细与进程峰值必须重新测。先 depth0/512 的 bring-up和质量，再扩大context；OOM保留失败，不自动切换offload而把结果混进原矩阵。不租卡。

手机12 GB是共享物理RAM，还需系统、后台服务与runtime。先在小米13上测0.6B ARM CPU，再选4B短context；桌面预算不能证明手机能在相同context运行。

实施顺序：先解释本次Q4质量失败和默认rotation影响，再为4B新增独立model/protocol锁；给runner增加显式实验锁参数并保留旧锁默认值。所有weight、KV、backend配置比较都需新数据。不会从现有Q8文件再次量化后拿来冒充同源weight-quality比较。
