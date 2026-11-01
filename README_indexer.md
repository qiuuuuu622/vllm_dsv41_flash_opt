# DeepSeek-V4.1-Flash · H20 FP8 候选 indexer

基于 vLLM 0.32.0（`5df63a7203cdb9328d824a27b682e77c7b9928be`）。本次新增
`vllm_dsv41_indexer`，提供候选块发布、FP8 WGMMA 打分和融合逻辑位置映射的 top-k。
这是显式调用的函数级模块，未自动接入 `vllm serve`，无需完整模型权重即可复现。

## 范围与 TP8 适配

当前处理 **decode/verify、ratio=1 的候选 consumer indexer**。V4.1 配置的候选
source 为第20层，后续24/28/32/36层的 indexer可消费候选。不是所有40层都执行
此优化，也不是文章中 prefill 的 FP4 解包/keys-major 路径。

vLLM H20 默认 indexer cache 是 FP8，不能直接运行上游要求 SM100 的 MXFP4 路径。
本实现沿用 FP8 Q/K 与原始 scales，没有增加量化步骤。

TP8 的关键不是把32个 index heads除以8。vLLM的 indexer Q投影和head权重投影均为
`ReplicatedLinear`，所以 **每个TP rank保留32个index heads×128维**。主attention的
每卡8个head属于另一套布局。

| 参数 | 本实现契约 |
| --- | --- |
| TP | 1/2/4/8；每rank完整32个index heads |
| EP8 | 不改变本模块的head、候选或KV布局 |
| PCP / DCP | 必须为1；拒绝CP分片KV |
| Q | FP8 E4M3 `[rows,32,128]` |
| head weights | FP32 `[rows,32]`，已包含Q scales |
| index cache | vLLM字节ABI `[pages,64,1,132]`；每页先64×128 payload，再64个FP32 scale |
| 候选 | 最多2,048个block，每block8个位置；source输出有效block唯一 |
| 输出 | `[rows,512]` int32逻辑位置，无序top-k；不足部分为-1 |

各rank使用自己的request→physical-page映射及本地cache，输出逻辑token位置。
不跨rank广播物理page ID，不增加all-reduce/all-to-all，也不消除原版的跨TP重复
indexer计算。所有rank仍需持有对应的完整逻辑上下文。

## 原实现与优化实现

原版 H20 FP8 consumer：

```text
对全部L个位置做DeepGEMM打分
→ 把候选块外的分数mask掉
→ 在全长分数行上运行vLLM top-k
```

优化路径：

```text
source生成候选block
→ 一次排序/计数，发布blocks和counts
→ 后续同ratio、同query顺序的consumer复用
    → 按候选block及本rank页表直接读取FP8 key
    → WGMMA只计算最多16,384个位置
    → Ant top-k v2 + 融合映射，直接输出逻辑token位置
```

### 候选准备：从约48.5μs降至5.8μs

早期移植每个consumer都“排序→去重→再次排序”，单请求时仅这一步就约48.5μs，
而打分只需约5μs，抵消了稀疏计算的收益。

本次采用Ant的source发布/consumer复用机制；对vLLM source已保证唯一的候选，
用新增CUB radix sort完成排序和有效位置计数。有效block范围小于65,535时只需
排序低16位，完整32位ID仍被保存；超出范围使用32位排序。无效项以INT_MAX排在末尾。
任意外部候选默认走单次排序+scan的去重路径，不能随意声明`source_unique=True`。

发布只在同一轮、相同query顺序/长度/ratio下共享。新decode step、请求重排或候选
变化必须重新发布，不能把旧候选状态跨轮缓存。

### 打分与 top-k

打分借鉴Ant candidate WGMMA调度，适配vLLM FP8页布局：key scales在FP32 dot结果
之后乘入；ReLU、head weighting、head sum保留FP32。每个query的候选可能不同，
所以仍按query独立打分。Ant全文verify分支的key-tile共享不能直接用于任意候选列表。

top-k直接编译固定版本Ant原始C++头文件，确定性处理同分。候选block充当大小为8的
逻辑页表，top-k内部直接映射为逻辑位置，省去独立映射kernel。可见分数最多16K，
仅使用register top-k；发布版关闭不可达的cluster路径，不需要运行cluster计划kernel。
分数尾部未初始化，所有读取严格受counts限制。

来源、适配差异、许可证与SHA见 [NOTICE](vllm_dsv41_indexer/NOTICE.md)。发布代码通过
TVM-FFI直接构建这些头文件，不依赖实验时的SGLang namespace加载器。

## 性能证据

### 本次提交代码复测

下表来自发布模块本身的正反两轮CUDA Graph测试，单张H20，完整consumer函数
包含发布成本，单位μs；两轮各自P50取均值，基线为未修改的官方vLLM0.32.0 wheel。

| 请求数 | 上下文 | 原版vLLM | 发布模块 | 加速 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 64K | 20.34 | 15.87 | 1.28× |
| 1 | 128K | 24.17 | 16.00 | 1.51× |
| 1 | 256K | 32.91 | 15.96 | 2.06× |
| 8 | 256K | 128.80 | 35.24 | 3.65× |
| 32 | 256K | 444.65 | 89.10 | 4.99× |
| 128 | 256K | 1602.77 | 245.43 | 6.53× |

发布版34次case执行、136次分数/集合对比均通过，最大nRMSE约7.23e-8，本轮top-k
集合100%一致。历史原型曾出现完全同分位置替换，因此不保证所有输入逐位相同。
11项CPU契约/源码检查、单GPU八逻辑rank物理页重排验证、真实双卡一致性验证通过。
这些验证不等于真实八卡运行或完整模型精度评测。

[复测汇总CSV](results/indexer/release/comparison.csv)、
[完整原始数据](results/indexer/release/)、
[验证边界和源码哈希](results/indexer/README.md)。

### 之前原型与瓶颈定位

历史实验：单H20，Torch2.13.0+cu130、Triton3.7.1、NVCC13.0.88；完整consumer函数
包含候选准备、打分、top-k和位置映射。CUDA Graph；每项20次预热、30个样本、每样本
10次，表中为正反顺序两轮各自P50的均值，单位μs。单卡结果不能作为TP8实测收益。

| 请求数 | 上下文 | 原版vLLM | 早期移植 | 优化原型 | 相对原版 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 64K | 20.22 | 60.85 | 15.89 | 1.27× |
| 1 | 128K | 24.17 | 60.88 | 16.05 | 1.51× |
| 1 | 256K | 32.98 | 60.92 | 16.06 | 2.05× |
| 8 | 256K | 128.75 | 80.87 | 35.21 | 3.66× |
| 32 | 256K | 443.66 | 148.60 | 90.22 | 4.92× |

同一轮四个consumer共用一次发布，单请求256K组合为130.31→46.51μs（2.80×），
包含一次准备成本。单请求六行verify、256K为76.75→29.67μs（2.59×）；该基线
按N=1展开，不能称为原生N=6融合实现。

历史记录、源文件快照、全部样本及误差见 [原型报告](results/indexer/20261101/REPORT.md)
和 [原始数据](results/indexer/20261101/)。原型36次case、144次分数/集合对比最大
nRMSE约7.23e-8；一个同分边界发生1/512位置替换，最低集合一致率99.8047%。
75组专项边界测试通过。128请求256K存在最高约22%的轮次波动，原始数据保留；
不能只挑最快数字。

发布模块的复测单独保存于 `results/indexer/release/`，避免把重构前的结果直接冒充
当前代码的结果。source候选选择、Q/K投影、KV写入和调度器不在上述计时中；没有
完整模型准确率、TTFT、TPOT或八卡硬件性能结果。

Ant原策略在至少16个候选窗口时启用compact，本配置对应256K；64/128K为强制
测试形状。实际服务应根据硬件/形状调度，不能仅凭这些结果全局开启。

## TP8 调用方式

```python
from vllm_dsv41_indexer import IndexerParallel
from vllm_dsv41_indexer.workspace import CandidateState, CompactIndexer

# 每个worker先设置自己的CUDA设备；传入TP subgroup，而非整个DP/EP全局组。
parallel = IndexerParallel.from_group(tp_group, pcp_size=1, dcp_size=1)
state = CandidateState(q.shape[0], q.device, parallel=parallel)
consumer = CompactIndexer(state, q, weights, local_fp8_cache,
                          local_block_table, request_indices)

# 每轮source产生新候选后执行一次；只有原生唯一候选可开启快速路径。
state.publish(candidate_blocks, sequence_lengths, source_unique=True)
logical_topk = consumer.forward()
```

同一rank的四个consumer可以共享state，各自绑定自己的Q/weights。构造/JIT/预热
必须在Graph捕获之前。捕获区需包含publish和forward；回放时原地更新输入，不能
替换已捕获的tensor存储。返回buffer在下一次forward被覆盖；同一workspace串行使用。
调用者保证request ID、page ID及lengths在本地cache可见范围内，并只传入有限分数。

真实服务还需接入source状态生命周期、metadata刷新、输出buffer、custom-op/
torch.compile和所有rank一致的路径选择。这里不修改模型加载或默认服务路径。

## 复现：不需要权重

使用独立Linux CUDA环境，保留已安装的官方vLLM0.32.0 wheel，不要用本仓库editable
安装替换基线。新包通过工作目录导入。所需版本及命令如下：

```bash
uv pip install --python .venv/bin/python apache-tvm-ffi==0.1.11 ninja
# 另需与已验证环境一致的torch2.13.0+cu130、triton3.7.1、vllm0.32.0。
# 设置CUDA_HOME到支持SM90的完整工具链，并把.venv/bin加入PATH。
.venv/bin/python -m vllm_dsv41_indexer.native

# 单物理GPU模拟8个TP rank，真实执行kernel并分别置换各rank物理页表。
.venv/bin/python -m benchmarks.kernels.dsv41_indexer.verify \
  --logical-tp8 --output build/indexer/logical_tp8.json

# 真正的双卡/八卡检查；有几张卡就填几张，不能把2卡标成8卡。
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 \
  --module benchmarks.kernels.dsv41_indexer.verify --distributed \
  --output build/indexer/tp2.json

.venv/bin/python -m benchmarks.kernels.dsv41_indexer.benchmark \
  --rows 1 8 32 128 --context 65536 131072 262144 \
  --output build/indexer/matrix.json
# --reverse-order反序复测；--rows 6 24 --verify 6测verify。
# benchmark的--tp-size=8仅检查单rank的TP8契约，不会自动启动8卡。
```

Mac只运行CPU契约与源码完整性检查：

```bash
.venv/bin/python -m pytest -c tests/kernels/dsv41_indexer/pytest.ini \
  --confcutdir=tests/kernels/dsv41_indexer tests/kernels/dsv41_indexer -q
```
