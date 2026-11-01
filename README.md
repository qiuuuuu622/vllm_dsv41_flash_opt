# DeepSeek-V4.1-Flash / vLLM 0.32.0 / H20 优化实验

基于 vLLM **v0.32.0**，基线 commit 为
[`5df63a7203cdb9328d824a27b682e77c7b9928be`](https://github.com/vllm-project/vllm/tree/5df63a7203cdb9328d824a27b682e77c7b9928be)。
目标是单机 **8×H20 96GB、TP8 + EP、DP1 / PCP1 / DCP1**。
EP 使用 `--enable-expert-parallel`，在该拓扑下专家组为 8。

本仓库已接入三项**默认关闭的实验实现**，提供不需要完整权重的测试与指标脚本。
已执行 macOS CPU 数值测试、逻辑通信模拟和通信成本模型；
**尚未执行 CUDA 编译、真实 H20/NVLink 测试、完整模型精度评测或端到端吞吐测试。**
不能据此宣称 H20 加速倍数、生产可用性或与原模型逐位等价。

原小 M MXFP8 GEMM 实验已移除：新增激活量化产生约 2.7% 的合成 GEMM 误差，且没有实测加速证据。
Marlin/Humming 已恢复 v0.32.0 原实现，不再保留额外权重副本；旧代码与结果保留在
[历史提交](https://github.com/qiuuuuu622/vllm_dsv41_flash_opt/commit/226b3c400da420a4ce9aac07a0961ae22faa0d01)。
旧配置若仍在 `VLLM_DSV41_H20_OPT` 中包含 `small_m_gemm`，会报未知特性，需删除该名字。

原项目介绍见 [README.upstream.md](README.upstream.md)，实施方案见
[docs/dsv41_h20_plan.md](docs/dsv41_h20_plan.md)。

## 三项改动

| 开关 | 原路径 | 新路径 | 主要边界 |
| --- | --- | --- | --- |
| `compact_indexer` | 全上下文打分，再过滤候选块 | 从 MXFP4 cache 直接读取候选位置并紧凑打分 | consumer 层；prefill / uniform decode / uniform verify；PCP1/DCP1 |
| `compressor_shard` | 每个 TP rank 投影完整 hidden 行 | 每卡投影约 1/8 行，再 all-gather FP32 输出 | TP8、DP1/PP1/PCP1/DCP1；默认 M≥2048 |
| `head_exchange` | 每卡 8 个有效 head 填充到 64 | Q all-to-all 改成每卡约 1/8 行、64 个有效 head；输出逆交换 | FlashMLA decode；全局 64 heads；默认 M≥128 |

三项可独立启停，未满足已知形状/硬件/拓扑条件时走原路径。
通信进入后发生的异常直接报错，不在部分 rank 已通信后偷偷回退。
目前要求 `--enforce-eager`，避免未经验证的编译和 CUDA graph 集成。

### 1. 候选块紧凑 indexer

原实现位于 [sparse_attn_indexer.py](vllm/model_executor/layers/sparse_attn_indexer.py)：
`fp8_fp4_mqa_logits` / `fp8_fp4_paged_mqa_logits` 先计算完整 logits，
consumer 再调用候选 mask，最后 top-k。候选很少时，许多打分结果算完就被丢掉。

新路径在同一位置调用 [indexer.py](vllm_dsv41_h20/indexer.py)。
候选块先排序、去重、展开成请求内位置；
[Triton kernel](vllm_dsv41_h20/kernels.py) 只读取这些位置的 packed E2M1 数据与 E8M0 scale：

```text
score(q, k) = Σ_head weight[head] × ReLU(dot(Q[head], K[k]))
```

prefill 使用 gathered K 的请求起始偏移；decode/verify 按 block table 找物理页。
因果边界之外、重复块、负数位置不参与结果；输出恢复为请求内 KV 索引。
候选 source 层仍走上游完整打分，负责生成候选块。

当前使用 MXFP4 解包后 **TF32×3 tensor-core dot**，不是蚂蚁的 FP8/Gluon kernel。
top-k 使用紧凑数组上的稳定排序，同分保留位置靠前者；不是蚂蚁的 top-k v2。
减少打分范围不保证加速：排序、解包成本可能抵消收益。
数值接近也不保证 top-k 边界完全相同，同分规则可能与上游不同。
非有限分数按无效值处理，不宣称异常数据语义与上游一致。

uniform verify 支持每行独立可见长度；ragged/padded decode 走原路径。
候选容量×行数超过预算也回退。预算限制的是元素数，**不是总工作区字节数**：
positions、scores、排序索引与临时数组都会占内存。

### 2. compressor 按行投影再汇聚

[attention.py](vllm/models/deepseek_v41/attention.py) 原先每个 rank 都执行：

```text
X[M,K] × W[N,K]^T → Y[M,N]，FP32 输出
```

PCP1 下 hidden 行已经在 TP ranks 间复制，压缩器权重由 `disable_tp=True` 保持完整。
新路径保留完整权重，令 rank r 只计算连续行分片：

```text
chunk = ceil(M/8)
rank r: X[r*chunk:(r+1)*chunk] × W^T → Y_r
all-gather(Y_0, …, Y_7) → 完整 Y；去掉尾部 padding
```

实现见 [distributed.py](vllm_dsv41_h20/distributed.py)。
不整除时各 rank 补齐相同大小，避免 collective 大小不一致。
汇聚后每卡仍得到完整 compressor 输出，后续压缩、KV 存储接口保持原布局。
默认按总行数 M≥2048 启用，也适用于满足阈值的混合批次；这是可调策略，不是已实测交叉点。
此路径执行期间禁用该 projection 组的辅助 stream 分发，保持通信顺序明确。

例如 M=16384，每卡 GEMM 行数变为 2048，名义 FLOPs 降为 1/8；
但新增 FP32 输出 all-gather。原 PCP1 路径在这里没有这次通信。
**所以不是“通信减少 8 倍”，也不是“打开 CP8”。**
这里未实现 copy engine/NVLink 对称内存，使用 vLLM 的 TP group all-gather。

### 3. decode attention 的 head↔row 交换

上游 [FlashMLA decode](vllm/models/deepseek_v41/nvidia/flashmla.py) 要求 64/128 个 Q heads。
全局 64 heads、TP8 时，每卡只负责 8 heads，却把 Q pad 成 `[M,64,512]`。

新路径 [attention.py](vllm_dsv41_h20/attention.py)：

```text
每卡: Q[M,8,512]
    all-to-all：交换 head 分片
每卡: Q[ceil(M/8),64,512]
    FlashMLA，读取本卡已有的复制 KV
每卡: O[ceil(M/8),64,512]
    逆 all-to-all
每卡: O[M,8,512] → 写回原 padded 输出
```

rank 顺序决定全局 head 顺序；每个 head 的 sink 也通过 all-gather 收集。
当前 rank 只切本地行对应的 SWA/压缩 KV indices 和 lengths；padding 行使用 `-1/0`。
独立的行分片 scheduler 与原 scheduler 分开，按原 decode step 的 scheduler 生命周期失效。

默认 M≥128 才尝试，要求 TP8、64 个全局 heads、BF16、FlashMLA、无 PCP/DCP。
KV 必须是 TP 复制布局。indexer 仍按自身路径运行，本项只重排 attention。
每卡 padded row-head 数在可整除时降为 1/8，但新增两次 all-to-all、sink all-gather、
打包和 scheduler 开销。它不是无通信的 8 倍加速。

## 如何安装和启停

本目录包含完整 vLLM 源码，`vllm_dsv41_h20` 被已有 `vllm*` 包发现规则收录。
Linux H20 上按 [上游源码安装文档](docs/getting_started/installation/gpu.md) 安装本 checkout；
不要直接使用不含这些 Python 修改的官方 wheel。
Mac 只安装后文 CPU 测试依赖，不要求安装完整 vLLM。

以下是**加入现有、已验证启动命令的实验参数**，不是经过实机验证的完整显存/调度配置：

```bash
# 所有 worker 使用相同环境变量；每次修改开关后重启进程。
export VLLM_DSV41_H20_OPT=compact_indexer

vllm serve /path/to/DeepSeek-V4.1-Flash \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --prefill-context-parallel-size 1 \
  --decode-context-parallel-size 1 \
  --enforce-eager
```

逐项 A/B，再测三项组合：

```bash
export VLLM_DSV41_H20_OPT=compact_indexer,compressor_shard,head_exchange
```

head exchange 接在 FlashMLA，其他 attention backend 不会进入本项。
每个进程首次真正进入路径时记录：`Experimental DSv4.1 H20 path active: <name>`。
仅设置环境变量而没有该日志，不应视为已启用某条运行路径。

| 环境变量 | 默认值 | 含义 |
| --- | --- | --- |
| `VLLM_DSV41_H20_OPT` | 空 | 逗号分隔特性名；未知名报错 |
| `VLLM_DSV41_H20_COMPRESSOR_MIN_ROWS` | 2048 | 投影分工最少总行数 |
| `VLLM_DSV41_H20_HEAD_MIN_ROWS` | 128 | decode 重排最少总行数 |
| `VLLM_DSV41_H20_INDEXER_MAX_SCORES` | 16777216 | compact score 元素上限；总显存更大 |

关闭全部实验：`unset VLLM_DSV41_H20_OPT` 后重启。
运行路径检查 SM90 和设备名 H20；独立 CUDA kernel 测试也可在 H100 SM90 上运行，
但 H100 测量不能充当 H20 测量。

## Mac 上已执行的测试与指标

环境：macOS ARM64、Python 3.12.13、PyTorch **2.13.0**（与上游 CUDA requirements 一致）、
单 CPU 线程、随机种子 20261031。使用合成数据，没有完整模型权重。

**24 passed，2 skipped**。2 项跳过需要 CUDA，分别验证连续和分页 cache 的多请求 indexer 打分。
原始报告：[CPU 指标](results/h20_cpu.json)、[JUnit](results/h20_tests.xml)、
[通信模型](results/h20_communication_model.json)。

另外已运行 **8 个独立进程的 CPU/Gloo 实际 collective**，直接调用新实现的
`project_rows`、`heads_to_rows`、`rows_to_heads`，不只是 Python 列表模拟。
[原始结果](results/h20_gloo_cpu.json)：M129/N1024/K512 投影最大误差0，head 往返完全一致。
该次 CPU dense 投影1.398 ms，分工加汇聚5.332 ms（0.262×，反而变慢），
两次 head 交换加打包3.211 ms。这是本机多进程通信测量，不能代表 NVLink。

| 检查 | 结果 | 可以说明什么 |
| --- | --- | --- |
| indexer：R8/H32/D128/L8192，候选容量512 | top-k 完全匹配；该样本 score 最大误差0 | CPU 参考数学与掩码一致；打分元素容量65536→4096，16×缩小 |
| compressor：M2049/N128/K256，8个逻辑 rank | 最大误差0 | 不整除行数和拼接顺序正确 |
| head 交换：M129/H64/D64 | 往返最大误差0 | padding、rank/head 顺序正确 |
| 带不同 sink、每行不同稀疏索引的 attention | FP64 参考在1e-12容差内一致 | 交换前后 attention 数学一致 |

JSON 中 CPU 时间仅用于本机复现，不能推算 H20 延迟。
16× 候选容量缩小、8× FLOPs/row-head 缩小也都不是延迟加速倍数。
尺寸是测试样例，不声称是所有模型层的真实尺寸。

复现：

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements/h20-cpu.txt

PYTHONPATH=. .venv/bin/python -m pytest \
  -c tests/kernels/dsv41_h20/pytest.ini \
  --confcutdir=tests/kernels/dsv41_h20 tests/kernels/dsv41_h20 -q \
  --junitxml=results/h20_tests.xml

PYTHONPATH=. .venv/bin/python benchmarks/kernels/dsv41_h20.py \
  --mode cpu --output results/h20_cpu.json
PYTHONPATH=. .venv/bin/python benchmarks/kernels/dsv41_h20.py \
  --mode communication --output results/h20_communication_model.json
```

独立 pytest 配置隔离上游根 conftest 中的 CUDA/服务依赖；不跳过本套 CPU assertions。

复现 Mac 多进程通信测试（显式 IPv4/loopback 避免本机 IPv6 名称解析问题）：

```bash
GLOO_SOCKET_IFNAME=lo0 PYTHONPATH=. .venv/bin/python -m torch.distributed.run \
  --nnodes=1 --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29571 \
  benchmarks/kernels/dsv41_h20_distributed.py \
  --device cpu --rows 129 --output results/h20_gloo_cpu.json
```

Gloo 使用实际进程组；CPU GEMM 分支将 BF16 输入转为 FP32 再乘，
CUDA 分支保持 `torch.mm(..., out_dtype=torch.float32)`。二者共享切片、汇聚和交换代码。
提交的 JUnit 只移除了本机 hostname/绝对路径，保留通过/跳过和耗时记录。

## 通信模拟如何解释

逻辑模拟使用与运行时相同的打包函数，按 source/destination rank 重排张量。
它证明数据布局关系，不模拟 NVLink 物理链路。
性能部分使用 alpha/bandwidth 模型，扫描**假设有效带宽50/100/200 GB/s**、
**每轮假设启动延迟2/5/10 μs**；这些都不是设备标称值或测量值。

例如 P=8、100 GB/s、5 μs/轮：

| 操作样例 | 数据量 | 模型估算的暴露通信时间 |
| --- | --- | --- |
| compressor：M16384、输出宽1024、FP32 | 完整结果64 MiB；每 rank 向其他 rank 发送56 MiB | 622.20 μs |
| head：M128、每卡8 heads、D512、BF16 | 每次交换每卡总发送缓冲1 MiB；共两次，加 sink 汇聚 | 123.35 μs |

模型不包含争用、拓扑路由、拷贝引擎、计算重叠、打包成本或 NCCL 具体算法选择。
盈亏判据：`原计算时间 - 分工后计算时间 - 打包等额外时间 > 暴露通信时间`。
没有 GPU 实测计算时间，不能从该表推导整体收益。

## 后续 GPU 验证，无需完整权重

单张 H20：在 Linux CUDA 环境按上游步骤安装本 fork 及匹配 PyTorch/Triton，
kernel 遵循仓库规范经 `vllm.triton_utils` 导入 Triton；运行上面的 pytest 命令，
2 个 CUDA 用例将实际执行。继续测合成 kernel 延迟：

```bash
PYTHONPATH=. .venv/bin/python benchmarks/kernels/dsv41_h20.py \
  --mode cuda --output results/h20_cuda.json
```

脚本输出新 kernel 正确性与延迟；没有将 PyTorch dense GEMM 冒充原 vLLM 性能基线。
真实加速比仍需对照同输入的原 DeepGEMM indexer、compressor 投影和 FlashMLA。

有2张或8张 GPU 后，运行真实 NCCL 投影汇聚、head 交换及逆交换：

```bash
PYTHONPATH=. .venv/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=8 \
  benchmarks/kernels/dsv41_h20_distributed.py --rows 2049
```

脚本验证真实 collective 的行/head 顺序并报告最慢 rank 延迟；
仍不等于完整 FlashMLA 多层集成测试。单卡无法验证 NVLink、多 rank 顺序错误或链路争用。
上线前还需 CUDA memory checking、不同 batch/context/verify 形状的上游 A/B、
峰值显存、完整模型 logprob/任务质量与 TTFT/TPOT/吞吐，以及另行完成 CUDA graph 集成。

## 与蚂蚁 SGLang 优化的关系

参考 [antgroup/sglang](https://github.com/antgroup/sglang) 的
[indexer](https://github.com/antgroup/sglang/commit/7a37d35)、
[CP](https://github.com/antgroup/sglang/commit/778bacd)、
[decode/mix TP](https://github.com/antgroup/sglang/commit/380e2d5) 优化方向。
本仓库是为 vLLM PCP1 适配的独立实验实现，不是该分支全部优化的等价移植。
未实现 copy engine 对称内存、Gluon/top-k v2、bounded replay 晚层跳过、prefill graph 等。
尤其 projector 分工在 PCP1 上**新增通信**，不能直接套用蚂蚁 CP8 的性能数据。

代码由 AI 辅助编写，Mac 运行记录随仓库提交；实机及模型评测边界如上。
沿用 Apache-2.0 许可证，上游版权与历史保留。
