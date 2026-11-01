# vLLM 0.32.0 · DSV4.1 H20 head↔row 交换优化

基于官方 **vLLM v0.32.0**（`5df63a7203cdb9328d824a27b682e77c7b9928be`），研究 DeepSeek-V4.1-Flash 在 H20 上的 head↔row 交换与融合 NVLink 通信。

**当前交付是可复现的 kernel／函数级实现，支持 TP2/4/8 的布局。两张 H20 已实测；TP8 通过 CPU 布局与数值测试，尚无八卡 GPU 性能或完整模型准确率结果。**

本模块通过 Python API 显式调用，不会自动修改 `vllm serve` 的 attention 路径。没有模型权重也能执行全部基准。上游介绍保留在 [README.upstream.md](README.upstream.md)。

## 代码范围与历史

- 主分支先重置到官方 v0.32.0，再提交本次 head↔row/通信实现。
- 旧主分支保存在 [`codex/archive-before-head-row-20261101`](https://github.com/qiuuuuu622/vllm_dsv41_flash_opt/tree/codex/archive-before-head-row-20261101)。此前 indexer、Q projection 等实验没有带入本次主分支。
- 不包含已撤销的小 M MXFP8 GEMM 实验，也没有引入新的激活量化。
- 已移除继承的 Dependabot 自动版本更新配置；对应的5个更新 PR 已关闭、分支已删除。

## 为什么交换 head 和 row

DSV4.1-Flash 总共64个query head。在 TP8 中，每卡只有8个有效head，而当前 SM90 FlashMLA 接口需要补齐到64个head。原版每卡处理全部 M 行、8个有效head。

head↔row 交换将同一批请求的8份head收集到行所属卡上：

| 配置 | 输入：每卡行数 × 有效head | 交换后：每卡行数 × 有效head | FlashMLA补齐head |
| --- | --- | --- | --- |
| 两卡子问题 | M × 8 | ceil(M/2) × 16 | 64 |
| 四卡子问题 | M × 8 | ceil(M/4) × 32 | 64 |
| TP8完整head布局 | M × 8 | ceil(M/8) × 64 | 64 |

**TP2测试固定每卡8个有效head，绝不是每卡32个head的完整模型TP2。** TP8布局是64个真实head，TP2/4用于部分问题验证。所有rank必须拥有相同请求顺序、相同可见KV数据，以及等价的KV地址/索引映射。本实验的KV在各卡复制，不是CP分片缓存。

以TP8、M=128为例：GPU0计算前16行的64个head，GPU1计算随后16行，以此类推。结果再交换回来，每卡恢复128行×8个本地head，供原有输出投影使用。有效数学计算没有改变；收益来自更合适的kernel形状与减少padding相关的执行浪费。

## 原版、NCCL实验与融合实现

原版 vLLM 没有这次交换：每卡直接运行自己的 padded FlashMLA。

此前NCCL实验：

```text
Q打包 → NCCL all-to-all → Q重排/补齐
      → FlashMLA
      → O打包 → NCCL all-to-all → O恢复布局
```

当前融合实现：

```text
Q融合peer push，直接写接收卡最终Q布局
      → 同一个FlashMLA
      → O融合peer push，直接写回原卡最终O布局
```

两个CUDA kernel替代两次NCCL交换与四次独立重排。初始化仍通过进程组交换信息、收集sink；不声称整个实验完全不使用NCCL。

### 最终布局直写

每个发送rank根据四个byte stride计算源分段、源行、目标发送rank分段和目标行的位置。每个片段为 `8 × 512 × 2 = 8192` 字节，通过SM发出的NVLink写操作直接落在对端最终布局中，没有中间接收张量的转置。**这是SM peer store，不是copy-engine搬运。**

Q/O持有独立的对称内存和同步区。不能整除world的尾部行写零，返回结果裁剪到真实M行。payload按字节复制，不改变BF16位模式。

### 两阶段跨卡同步

每个CTA有独立128字节同步区域：

1. 进入阶段确认所有rank都已结束使用上一轮缓冲，允许覆盖。
2. 写入线程经CTA同步后，由system-scope release发布数据；接收端acquire确认全部发送rank完成后返回。

轮次计数由GPU更新，CUDA Graph重放不需要host修改标志。信号与数据分离，不占用NaN或其他数值作为标记。等待有诊断超时；超时导致设备错误，不是可以继续执行的正常回退。

### warp分块和向量搬运

按warp分配tile，线程使用展开的4次16字节向量读写；相邻warp轮转目标rank。world与片段大小专门化，减少动态除法和地址计算。默认按消息量选择1–16个CTA。

实现借鉴 [蚂蚁 SGLang 的 segment_a2a](https://github.com/antgroup/sglang/blob/380e2d5bda9dbff32d924765c5a6c1c54bb9ad47/python/sglang/kernels/jit/csrc/distributed/segment_a2a.cuh) 与 [communicator](https://github.com/antgroup/sglang/blob/380e2d5bda9dbff32d924765c5a6c1c54bb9ad47/python/sglang/kernels/jit/include/sgl_kernel/distributed/communicator.cuh)，固定提交 `380e2d5bda9dbff32d924765c5a6c1c54bb9ad47`。改造说明及许可证见 [NOTICE](vllm_dsv41_head_exchange/csrc/NOTICE.md)。

## 两张H20实测

2026-11-01，2×H20 96GB、NV18互联；torch2.13.0+cu130、NCCL2.29.7、driver580.65.06，独立CUDA扩展用nvcc12.8编译。加载官方vLLM0.32.0 wheel的FlashMLA扩展，其依赖提交为 `0eee43b12f034b657133cf2afca6a72ebb6efccf`。这没有验证完整vLLM服务依赖栈。

下表为整理后模块的两卡复测，ratio=2、32K上下文、SWA128+512个稀疏位置、CUDA Graph，单位μs。取两rank各自P50中的较大值；30个样本、每样本20次调用。NCCL列来自同机器此前实验，是不同轮次的数据。

| Query行数 | 原版函数 | 之前NCCL方案 | 融合方案 | 相对原版加速 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 16.18 | 37.79 | 24.46 | 0.66× |
| 8 | 20.37 | 49.55 | 25.68 | 0.79× |
| 32 | 47.62 | 68.96 | 41.26 | 1.15× |
| 64 | 58.92 | 87.45 | 57.75 | 1.02× |
| 128 | 118.04 | 109.13 | 70.92 | 1.66× |
| 256 | 213.57 | 179.15 | 136.96 | 1.56× |
| 512 | 407.31 | 304.34 | 241.80 | 1.68× |

融合方案总时间包含两个方向的通信、布局写入、FlashMLA和结果恢复。单独测双向通信/布局：M128由40.67降至约11.97μs，M512由78.17降至约21.50μs。不能用独立计时项简单相加替代完整函数计时。

workspace初始化、sink汇聚、metadata准备、输入生成、模型投影、indexer和调度器不在计时范围。planner复用、重复输入、缓存已预热；数字不是端到端TTFT、TPOT或服务吞吐量。eager、ratio0/1和各rank完整样本也保存在JSON/CSV中。

**不是所有小行数都加速。** 8行仍更慢；64行只有约1–2%的改善，不能当成稳定收益。32行的Graph收益不能直接推广到eager。今后的服务端dispatch应按硬件、world、attention类型、行数及Graph/eager模式选择路径，不能把TP2阈值照搬TP8。

证据：

- [本次模块复测](results/head_exchange/release/fused_matrix.json)、[压力测试](results/head_exchange/release/stress.json)、[汇总CSV](results/head_exchange/release/comparison.csv)。
- [此前原型全部结果](results/head_exchange/20261101/)，含源码快照、NCCL对照、原始样本和检查日志。
- [SHA256清单](results/head_exchange/evidence.json)记录提交代码及结果文件；没有将两卡结果标成八卡结果。

## 正确性与测试边界

- Mac CPU：24项测试覆盖world2/4/8的字节布局、rank/head顺序、尾部和空rank、metadata刷新，以及带sink/空窗口的注意力还原。
- 两卡GPU：7种行数×eager/graph×2rank×200轮，共5600个rank往返；变化输入包含任意BF16位模式（NaN/Inf），以int16视图检查字节一致，并交替延迟rank。全部通过。
- 完整attention相对原版最大NRMSE约 **0.1875%**，与此前NCCL交换方案一致。交换改变FlashMLA行形状与归约调度，因此attention并非所有case逐位相同；通信本身精确复制，没有新增量化。
- 历史原型的定向Compute Sanitizer检查为两个rank均0个内存错误。首次全量检查记录了NCCL kernel-image和PyTorch fabric能力探测的API错误；随后仅检查交换kernel访存并关闭API错误报告，原始失败日志保留。不能称为完整程序sanitizer全通过，也不是跨进程race证明。
- 本次提交代码也通过同样范围的[定向访存检查](results/head_exchange/release/memcheck.txt)，两个rank均0个内存错误；rank配置不一致时的集体拒绝检查通过。
- 没有运行完整模型评测，没有实际八卡GPU测试；CPU的TP8验证只证明布局/数学结果，不能证明NVLink同步、拓扑兼容或性能。

## 文件与调用方式

| 文件 | 用途 |
| --- | --- |
| `vllm_dsv41_head_exchange/layout.py` | TP2/4/8行分片与byte stride契约 |
| `vllm_dsv41_head_exchange/workspace.py` | `FusedWorkspace`与NCCL对照，实现双向交换 |
| `vllm_dsv41_head_exchange/csrc/segment_exchange.cu` | 融合SM peer push与GPU同步 |
| `vllm_dsv41_head_exchange/build.py` | 显式编译，按源码hash缓存共享库 |
| `benchmarks/kernels/head_exchange/benchmark.py` | 原版与交换后的函数A/B |
| `benchmarks/kernels/head_exchange/verify.py` | 改变输入、Graph、慢rank及原始位模式检查 |
| `tests/kernels/head_exchange/test_layout.py` | CPU独立布局与数值oracle |

函数级接入顺序（所有TP rank一起执行）：

```python
from vllm_dsv41_head_exchange.workspace import FusedWorkspace

# 已初始化NCCL TP进程组；Q: BF16 [M, 1, 64, 512]，仅前8个head有效。
# sink: FP32 [64]。构造必须在graph capture之外，参数此后保持不变。
ws = FusedWorkspace(q, sink, group=tp_process_group)
row_indices = ws.metadata(indices, invalid=-1)
row_lengths = ws.metadata(lengths, invalid=0)

# 稳定执行阶段，可在预热后捕获：
row_q = ws.forward_q(q)
# 将row_q、ws.sink、row_indices/row_lengths传给FlashMLA，输出写到ws.row_o。
# 必须使用适配row形状的独立scheduler；不能复用原版M行的scheduler。
restored_output = ws.backward_o()
```

完整可执行的FlashMLA调用见benchmark。`metadata()`是视图或带尾部padding的副本：输入metadata更新后，必须重新准备/刷新副本，不能跨decode step复用陈旧数据。动态metadata准备成本未包含在当前性能数字中。

workspace绑定初始Q指针，返回缓冲在下一次调用时覆盖；每个workspace要求串行使用，所有rank以相同顺序和grid调用。支持传入TP subgroup，不能用含其他DP/EP rank的全局组替代TP组。Graph必须持有workspace与相关张量的生命周期。构造时会集体检查rank配置一致性，但这不替代完整服务端的生命周期管理。

## 复现：不需要模型权重

在Linux CUDA机器的仓库根目录中，使用独立虚拟环境。保留已安装的官方vLLM wheel，**不要对本仓库执行editable安装来替换它**；实验通过已安装wheel定位FlashMLA扩展，仓库模块由当前目录加载。

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python torch==2.13.0 numpy pytest \
  --index-url https://download.pytorch.org/whl/cu130 \
  --extra-index-url https://pypi.org/simple
uv pip install --python .venv/bin/python vllm==0.32.0 --no-deps

mkdir -p build/head_exchange
curl -fL https://raw.githubusercontent.com/deepseek-ai/FlashMLA/0eee43b12f034b657133cf2afca6a72ebb6efccf/flash_mla/flash_mla_interface.py \
  -o build/head_exchange/flash_mla_interface.py
.venv/bin/python -m vllm_dsv41_head_exchange.build --nvcc /usr/local/cuda/bin/nvcc

# 两卡原始位模式/同步检查。
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 \
  --module benchmarks.kernels.head_exchange.verify \
  --iterations 200 --output build/head_exchange/stress.json

# 原版attention与融合方案A/B（默认覆盖ratio0/1/2）。
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 \
  --module benchmarks.kernels.head_exchange.benchmark --transport fused \
  --rows 1 8 32 64 128 256 512 --context 32768 --repeat 30 --inner 20 \
  --interface build/head_exchange/flash_mla_interface.py \
  --output build/head_exchange/fused.json

# 同样参数、--transport nccl 可测NCCL对照。
# 有8张SM90卡并确认NVLink/P2P拓扑后，将nproc_per_node改为8即可测TP8；
# 每卡仍固定8个有效head，输出JSON自动记录真实world和交换后的head数。
```

可用 `DSV41_HEAD_EXCHANGE_LIBRARY=/absolute/path/library.so` 指定共享库位置；编译与运行要设置相同值。没有库时会报错，不会在graph capture中临时编译。

Mac/CPU只需torch与pytest，执行：

```bash
.venv/bin/python -m pytest -c tests/kernels/head_exchange/pytest.ini \
  --confcutdir=tests/kernels/head_exchange tests/kernels/head_exchange -q
```

## 下一阶段的服务端集成

仍需接入vLLM的buffer预分配、图捕获生命周期、dynamic metadata刷新、独立row scheduler、torch.compile/custom-op注册、拓扑检查和所有rank一致的dispatch。错误发生后不能让单个rank自行回退，而其他rank还在等待通信。

更进一步的“在row所属卡直接投影Q，省去第一次交换”需要重新考虑分片的Q投影权重与计算布局，未包含在本次实现中。全模型验证与真实TP8性能测试完成前，不把本模块标记为生产可用优化。
