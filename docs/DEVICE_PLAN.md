# Device plan: Xiaomi 13

用户已确认小米 13，12 GB运行内存、256 GB存储。Android/HyperOS版本、实际可用内存/存储、驱动或目标开发系统尚未确认。不要把标称12 GB当作可全部分配的预算，也不要以“手机两年前购买”判断是否能做实验。

小米官方规格确认 Snapdragon 8 Gen 2：[official specs](https://www.mi.com/global/product/xiaomi-13/specs/)。它是合理的 ARM CPU 端侧验证候选，实际 GPU backend需要单独验证。

Pinned llama.cpp [OpenCL文档](https://github.com/ggml-org/llama.cpp/blob/46ca246de9bb1c35269722a6240d37d9dfd79cad/docs/backend/OPENCL.md) 的验证列表包含 Adreno 750/810/830/840与X系列，未列出小米13的8 Gen 2。进一步的[源码证据](https://github.com/ggml-org/llama.cpp/blob/46ca246de9bb1c35269722a6240d37d9dfd79cad/ggml/src/ggml-opencl/ggml-opencl.cpp#L9234)明确对A7X/Adreno740的mixed F32/F16及对称Q8/Q4 FA返回不支持、由scheduler回退CPU；注释记录E031.41 compiler在clBuildProgram中崩溃。不能承诺本版本小米13的量化KV完全在GPU执行，也不能直接删guard规避。#24109 的 RedMagic 10 Pro/8 Elite 实验不能迁移成小米13结果。

## First device milestone

先在现有桌面完成实验规范，然后在手机运行小模型 ARM CPU baseline：F16/Q8/Q4 KV、小 context、低成本生成质量sanity。手机不用作为第一天必需依赖，也不需购买新设备。

在手机有可用终端/ADB后，只读记录：

```bash
adb shell getprop ro.product.model
adb shell getprop ro.build.version.release
adb shell getprop ro.build.version.sdk
adb shell cat /proc/meminfo
adb shell df -h /data
```

不要将全部 getprop输出或设备serial上传公开仓库。只记录必要型号、软件版本和内存；私有信息先审查。

两种构建选择：

- 初次CPU实验：参考 pinned [Android文档](https://github.com/ggml-org/llama.cpp/blob/46ca246de9bb1c35269722a6240d37d9dfd79cad/docs/android.md) 的Termux流程。项目Python runner适配情况需实机确认。
- 可重复部署：Android NDK cross-build，`arm64-v8a`, `GGML_NATIVE=OFF`, `GGML_OPENMP=OFF`, `GGML_LLAMAFILE=OFF`, `LLAMA_OPENSSL=OFF`；记录NDK与构建选项，不全局添加不适用的march。

此文为计划，尚未安装Termux/SDK/NDK，尚未连手机，也没有Android跑分。

## GPU gate

CPU成功后检查OpenCL/Vulkan driver、backend list devices与实际op支持；跑backend correctness测试后再对比。CPU/FA、GPU/FA、KV格式支持不能从README的权重量化类型表推断。若驱动不满足，就保留ARM CPU项目成果，标注限制后再选Vulkan或其他设备。

## Real edge measurements

记录物理RAM与memory extension区别、测试前可用内存、context实际长度、KV allocation、温度/功耗来源、重复间隔、电源/充电状态与降频。先小模型、短context，OOM/热限制按失败记录，不追逐手机最大可分配context。短prompt在大allocated cache下成功不等于完整长context成功。
