# M1: CUDA KV baseline and memory-path findings

2026-10-03，固定 Qwen3-0.6B Q8_0 权重。**Q8节省KV内存，但prefill更慢；Q4未通过检索质量门。** 这是小模型基线和运行配置检查，尚未修改 llama.cpp runtime/kernel，尚未做手机实验或4B实测。

## 可复核的范围

- llama.cpp `46ca246de9bb1c35269722a6240d37d9dfd79cad`；模型与语料身份见 `locks/`。质量阈值和输入在读取结果前提交，见 [M1_PROTOCOL](M1_PROTOCOL.md)。
- RTX3080 Laptop 16GiB、SM86、i7-11800H、WSL Linux；CUDA toolkit12.8.93，Windows driver616.92。CPU/CUDA Release构建，没有锁GPU时钟。
- FA明确开启，GPU layers请求99、日志实际29/29，batch/ubatch512、threads4。量化KV默认开启Hadamard rotation；F16不启用，所以比较的是各类型默认执行路径。
- 正式矩阵：decode与prefill各三次fresh-process sweep，共90个case进程、450个内部重复。三个顺序seed为20261003/04/05。没有丢弃慢的样本。
- [原始证据目录](../results/m1-cuda-20261003/)包含命令、日志、原始样本、失败与SHA256SUMS。[完整分布](../results/m1-cuda-20261003/aggregate-20261003-001/matrix.md)及stats.json保留各组的三份均值和全部五次内部样本。

## 质量

| KV | WikiText-2 PPL ± upstream绝对不确定度 | 相对F16的PPL变化 | 预先阈值 | 检索正确 / 9 |
| --- | ---: | ---: | --- | ---: |
| F16 | 15.2486 ± 0.53842 | 基线 | 基线 | 9 |
| Q8 | 15.2275 ± 0.53733 | -0.138% | ≤2%，通过 | 9 |
| Q4 | 48.3775 ± 2.01248 | +217.259% | ≤10%，失败 | 2 |

完整 [quality-003 summary](../results/m1-cuda-20261003/quality-20261003-003/summary.json) 的状态是`evaluation_failed`：三种真实执行均完成、provenance校验通过，但Q4未通过阈值。Q8的小幅负变化在本报告中不解释为质量改善；gate仅按预先定义的点估计公式判定，上游不确定度保留但不代入阈值。

WikiText-2只评估ctx2048、8 chunks，8184个scored targets由实际chunk日志及锁定源码公式推导。总输入token数没有被该版本输出，保留null，不猜测。它不代表整套语料或32K质量。

固定synthetic retrieval在三种内容长度、三个needle位置各一题。实际模板后prompt为2094–2095、8238、32814–32815 tokens；无截断，生成最多64 tokens，所有请求`cache_n=0`。正式关闭RAM缓存的 [server-003](../results/m1-cuda-20261003/server-20261003-003/summary.json) 与先前server-002一致：F16 **9/9**、Q8 **9/9**、Q4 **2/9**。Q4多失败7题，超过允许的1题。它的性能只作诊断；这也不能证明错误必然来自kernel，而非量化误差等路径因素。9题通过也不能推广为一般长上下文质量。

## 吞吐：分别看decode和prefill

下表是三个独立进程均值的中位数（tok/s），不是最好值或置信区间。每个均值来自五次内部重复。decode的depth在计时前填充；prefill是独立的合成prompt处理。它们都排除tokenizer/sampler，不能当作完整请求延迟。

| KV depth / prefill tokens | Decode F16 | Decode Q8 | Decode Q4（质量失败） | Prefill F16 | Prefill Q8 | Prefill Q4（质量失败） |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 238.64 | 261.51 | 267.31 | 17511.18 | 15773.41 | 15715.79 |
| 4096 | 231.83 | 234.68 | 233.72 | 15309.85 | 13923.10 | 14394.99 |
| 8192 | 189.67 | 194.56 | 193.87 | 13056.42 | 11475.69 | 11885.67 |
| 16384 | 113.65 | 136.82 | 139.17 | 9697.51 | 8506.15 | 8702.63 |
| 32768 | 79.18 | 90.06 | 89.34 | 6255.67 | 5591.21 | 5825.78 |

Q8 prefill在全部长度下约慢9%–12%。32K decode的F16三份均值为83.58/79.18/56.75，Q8为90.06/90.18/88.11；虽然Q8中位数更高，F16波动很大，不能据此声称稳定13.7%提速。2K F16还有111.21的慢run，4K/16K和Q4也出现慢run，全部保留。温度/功耗/利用率有快照，时钟、后台争用原因未被隔离，不能把异常简单归因于其中一种。

## KV、host memory和显存是不同指标

| 已填充depth | F16 KV allocation MiB | Q8 MiB | Q4 MiB |
| ---: | ---: | ---: | ---: |
| 2048 | 252.00 | 133.88 | 70.88 |
| 4096 | 476.00 | 252.88 | 133.88 |
| 8192 | 924.00 | 490.88 | 259.88 |
| 16384 | 1820.00 | 966.88 | 511.88 |
| 32768 | 3612.00 | 1918.88 | 1015.88 |

Q8相对F16节省46.875%的KV buffer；Q4节省71.875%。32K decode实际allocated cells为33024，含generation预算和padding。[KV_MEMORY](KV_MEMORY.md)解释公式、scale开销、模型head维度及工具state备份。

`llama-bench`会把depth state序列化到host vector并在重复间恢复。因此32K decode的进程VmHWM：F16约4166MiB、Q8约2486MiB、Q4约1590MiB，包含实验工具的host快照；不能当作服务或手机的内存需求。prefill各配置约925MiB。100ms采样VmRSS和观察到的VmHWM都保留，进程GPU显存因WSL没有对应PID记录而为null。整卡快照单独保存，不能替代它。

### 一个实际runtime配置陷阱

初次完整server-002设置了请求`cache_prompt=false`，所有request reuse为0，但server仍默认开启 **8192MiB RAM prompt cache**。日志实际保存context快照，F16缓存增长到8082MiB。请求禁用reuse和禁用全局RAM缓存是不同设置。

随后统一加入`--cache-ram 0`，由日志验证关闭，原题目、seed、质量阈值、backend、KV设置不变，重跑完整server-003。F16 server观察到的VmHWM从约8739MiB降到约930MiB；Q8/Q4也约930MiB。两次均保留，不混成一份数据。这是已验证的配置效果，尚不是自研内核优化，也不是严格配对的服务延迟收益实验。用它说明总内存不能只从KV压缩比推算。

正式server-003的2087-token warm TTFT（HTTP开始到首个非空content，ms）为：

| KV | 三次TTFT ms | 中位数 ms |
| --- | --- | ---: |
| F16 | 139.26 / 139.76 / 131.58 | 139.26 |
| Q8 | 162.57 / 149.05 / 149.75 | 149.75 |
| Q4（质量失败） | 159.11 / 153.44 / 154.70 | 154.70 |

每种类型一个新server、预热后测三次。该值排除模型加载，包含HTTP/tokenizer/队列/prefill；首个content并不一定等于首个完整token。样本不足以确认服务延迟优化。

## Profiler找到了阶段相关的执行路径

Nsight Systems2026.5.1。正式 [profile-003](../results/m1-cuda-20261003/profile-20261003-003/)采F16/Q4，后续探索 [profile-004](../results/m1-cuda-20261003/profile-20261003-004/)追加Q8。深度8192、gen32、r1。CUDA graph tracing设node，否则默认graph粒度会漏掉replay的kernel活动。CSV覆盖整个进程，包括load/warmup/depth fill/decode，不能把总占比叫decode占比；instrumented tok/s没有进入正式矩阵。

- F16可见`flash_attn_ext_f16<128,128,32,2,...>`448次，以及单token方向的`<128,128,4,2,...>`924次。
- Q4/Q8可见同类F16大query attention448次，及直接读取量化KV的`flash_attn_ext_vec<128,1,type,type,...>`924次。Q4的type=2，Q8的type=8。
- Q4另有`dequantize_block_q4_0<__half>`896次，Q8有`dequantize_block_q8_0_f16`896次。权重始终Q8；Q4转换不是Q4权重转换。结合源码，896=28层×16个512-token块×K/V两份，与depth prefill转换相符。这是阶段关联推断，下一步仍需显式range隔离验证。
- 锁定 [fattn-common.cuh](https://github.com/ggml-org/llama.cpp/blob/46ca246de9bb1c35269722a6240d37d9dfd79cad/ggml/src/ggml-cuda/fattn-common.cuh#L1029) 在需要F16输入时把K/V转换到临时F16 storage；vector量化attention则在内部处理dequant。因此“CUDA已经有fusion”要具体到路径：这里的decode已有，prefill还有独立转换。

正式whole-process kernel-time中，Q4/Q8独立dequant约3.5%/3.7%，并不能解释全部prefill损失，也不足以承诺巨大端到端收益。rotation、dispatch、同步、临时storage访问仍待分开测。trace大文件留本地；公开小CSV、命令、版本和原trace SHA，可从任一新系统重新采集。

## 失败记录与下一步

质量001因预期的总token日志不存在被工具拒绝；002因假设只有一次offload摘要被拒绝。锁定版本会先做无分配的memory-fit probe，再初始化实际context；解析器现保留多次初始化，要求offload一致，并验证实际PPL context的类型和非零CUDA KV buffer。设备数banner也没有被PPL输出，CUDA依据来自真实allocation及full offload。这些是解析失败，不是质量阈值失败；003完整执行后才得到上表的有效负结果。server001默认verbosity缺少offload证明，添加verbose后成功。profile001默认graph粒度不完整；002采node成功但stats SQLite timestamp检查阻止后续导出；003明确刷新owned derived export后完整成功。原始文件未覆盖。

下一步先做 **Q8 prefill的阶段隔离：转换、rotation、临时storage与attention占比**，以已过质量屏障的候选路径选一个小优化；同时单独诊断Q4质量损失，不能拿质量失败换速度。4B候选和预算见 [DENSE_MODEL_PLAN](DENSE_MODEL_PLAN.md)，尚未下载/实测。小米13只确认型号与12GB/256GB，真实ARM/Android阶段仍待设备接入。
