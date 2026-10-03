# 从实际分配理解 KV 内存

直觉：每一层都为历史 token 保存 K、V。固定模型时，context 越长，保存的数据越多；低比特存储降低每个元素的字节数。但程序实际分配的容量通常大于当前已经填充的 token 数。

本次 Qwen3-0.6B 的真实模型日志报告 28 层、16 个 Q head、8 个 KV head、K/V head dimension 都为 128。内存公式使用 **8 个 KV head**：

```
KV bytes = allocated_cells × layers × kv_heads × head_dim × 2 × stored_bytes_per_element
```

最后的 2 表示 K 和 V。本实验两者使用相同类型；一般模型的 K/V dimension 不同时必须分别算。

F16 每元素 2 字节。Q8_0 每 32 元素包含 32 个 int8 和一个 2 字节 scale，因此是 34/32=1.0625 字节/元素；Q4_0 每 32 元素包含 16 字节 packed values 和 2 字节 scale，因此是 18/32=0.5625 字节/元素。它们分别是 F16 的 53.125% 和 28.125%，不是精确的“一半”和“四分之一”。参见锁定源码 [ggml-common.h](https://github.com/ggml-org/llama.cpp/blob/46ca246de9bb1c35269722a6240d37d9dfd79cad/ggml/src/ggml-common.h) 的 `block_q8_0` / `block_q4_0`。

decode benchmark 在已经填充 32768 token 后生成 32 token。实际日志报告 `n_ctx=33024`，而不是 32768。这是容量 padding，不能当作测过 33024 token 的检索质量。

| 类型 | 33024 cells 的公式值（MiB） | 实际 CUDA KV buffer（MiB，日志四舍五入） |
| --- | ---: | ---: |
| F16 | 3612.000 | 3612.00 |
| Q8_0 | 1918.875 | 1918.88 |
| Q4_0 | 1015.875 | 1015.88 |

这是 **KV buffer allocation**，不是 GPU 进程峰值，也不是手机总内存。模型 buffer、计算 buffer、CUDA context、CPU mapped pages 都另有开销。原始日志会分别报告 CPU_Mapped/CUDA0 model buffer 和 CUDA0/CUDA_Host compute buffer；不要把这些值机械相加后宣称进程物理峰值。

本次还发现一个容易漏掉的执行差异：锁定 upstream 会为符合 dimension 条件的量化 KV 默认开启正交 Walsh-Hadamard rotation。真实 Q8/Q4 日志中的 `attn_rot_k=1, attn_rot_v=1`，F16 中为 0。源码在 [llama-kv-cache.cpp](https://github.com/ggml-org/llama.cpp/blob/46ca246de9bb1c35269722a6240d37d9dfd79cad/src/llama-kv-cache.cpp#L315)，环境变量 `LLAMA_ATTN_ROT_DISABLE` 可以改变它。

因此本次是在比较各类型的默认 runtime 路径：存储大小、量化误差、rotation、attention dispatch 都可能变化。不能将所有速度差异直接归因于“反量化贵”，更不能凭质量失败断言是 kernel bug。下一阶段需要分开观察执行阶段，并单独固定 rotation 做诊断。原基线不回填环境字段；原始日志保留了实际 rotation 状态，运行期间没有人为设置这些 override。

与 NanoKV 的联系：服务器 offload 主要改变 KV 存放位置与搬运量；这里的量化主要改变表示与执行路径。两者都要验证数据流、质量和整体延迟，不能只看压缩比。
