# H20 indexer 小批量回归定位与修复

结论：上一版小批量变慢主要来自移植适配层每个 consumer 重复做两次候选排序。对照蚂蚁源码后，改成 source 一次发布、后续层复用；进一步用 CUB radix sort 替换候选整理，并接入原生 Ant top-k v2 的位置映射。没有新增量化，也没有改变候选集合大小。

## 测试边界

- 单张 H20、vLLM 0.32.0、Torch 2.13.0+cu130、Triton 3.7.1；原版 Python 包源码 hash 校验通过。没有加载完整权重，没有测试服务或跨卡通信。
- 实际 H20 基线是 FP8 DeepGEMM。Ant 原路径的 FP4 cache 被适配成 vLLM FP8；Q/K、key scales、head weights 在两边相同，输出分数为 FP32。
- 测的是 late candidate consumer 的 decode/verify indexer（ratio=1），不是 prefill。32 个 indexer heads ×128 维，2,048 候选块×8 token，最终 top-k=512。
- Baseline=全上下文 DeepGEMM 打分→原版候选 mask→原版 C++ top-k；scheduler metadata 已在层外准备。两个路径共同的 Q/K 生成、cache 写入和 source 候选选择不计时。
- 单层优化计时**包含候选整理、plan、打分、top-k、输出逻辑位置映射**。四 consumer 测试包含一次整理/plan，并使用四份不同 Q/head weights；共用候选集合和 KV。不能直接换算成全模型加速。
- Decode 做正、反两轮测量；表格为两轮各自 P50 的均值，单位 μs。每项20次预热、30个样本、每样本10次；Graph和eager分别测，不混用。没有锁GPU时钟。

## 单层完整函数：CUDA Graph

| 请求数 | 上下文 | vLLM 原版 | 上一版移植 | 本次修复 | 相对原版加速 |
|---:|---:|---:|---:|---:|---:|
| 1 | 64K | 20.22 | 60.85 | 15.89 | 1.27× |
| 1 | 128K | 24.17 | 60.88 | 16.05 | 1.51× |
| 1 | 256K | 32.98 | 60.92 | 16.06 | 2.05× |
| 8 | 64K | 48.93 | 80.62 | 35.01 | 1.40× |
| 8 | 128K | 75.89 | 80.77 | 35.03 | 2.17× |
| 8 | 256K | 128.75 | 80.87 | 35.21 | 3.66× |
| 32 | 64K | 129.18 | 145.76 | 86.62 | 1.49× |
| 32 | 128K | 229.41 | 147.25 | 89.68 | 2.56× |
| 32 | 256K | 443.66 | 148.60 | 90.22 | 4.92× |
| 128 | 64K | 426.53 | 321.23 | 244.24 | 1.75× |
| 128 | 128K | 821.58 | 323.57 | 246.18 | 3.34× |
| 128 | 256K | 1611.67 | 327.26 | 267.17 | 6.03× |

顺序敏感性：1～8请求的新路径两轮差异低于0.4%；128请求256K出现21.93%差异（240.76/293.57μs），因此额外定点复查。复查正反两轮分别为241.02/255.24μs，仍有约5.9%差异。未锁频且未做专项cache/频率归因，最大形状应视为有波动；上表保留原两轮均值，不用复查较快值替换。

## Source 一次发布给四个 consumer：CUDA Graph

这是四层 indexer 的独立函数组合，准备成本只付一次，但并未把它移出计时。状态必须限定于同一轮、相同 query 顺序/长度/ratio/候选集合；新 decode step 或请求重排必须重新发布。源头候选选择仍是两边共同的未计时部分。

| 请求数 | 上下文 | 原版四次合计 | 优化四次合计（含准备） | 每层均摊 | 组合加速 |
|---:|---:|---:|---:|---:|---:|
| 1 | 64K | 79.27 | 46.18 | 11.54 | 1.72× |
| 1 | 128K | 95.56 | 46.71 | 11.68 | 2.05× |
| 1 | 256K | 130.31 | 46.51 | 11.63 | 2.80× |
| 8 | 64K | 193.95 | 122.06 | 30.52 | 1.59× |
| 8 | 128K | 301.46 | 122.61 | 30.65 | 2.46× |
| 8 | 256K | 513.86 | 122.73 | 30.68 | 4.19× |
| 32 | 64K | 515.74 | 324.14 | 81.04 | 1.59× |
| 32 | 128K | 922.50 | 328.99 | 82.25 | 2.80× |
| 32 | 256K | 1779.77 | 335.02 | 83.76 | 5.31× |
| 128 | 64K | 1713.83 | 926.26 | 231.57 | 1.85× |
| 128 | 128K | 3298.19 | 928.76 | 232.19 | 3.55× |
| 128 | 256K | 6754.88 | 931.74 | 232.93 | 7.25× |

## 单请求 256K：定位到哪里慢

各项为单独测得的 kernel/函数 P50，不能简单相加替代组合计时。

| 项目 | μs |
|---|---:|
| 旧版两次排序及去重 | 48.46 |
| 通用单排序+scan去重 | 16.02 |
| Ant式单排序/计数（Triton 8 warps） | 15.71 |
| Triton 32 warps排序 | 10.53 |
| 最终 CUB radix sort/计数 | 5.82 |
| 候选 WGMMA 打分（未改） | 4.94 |
| vLLM top-k | 6.42 |
| 旧版独立位置映射 | 1.86 |
| Ant top-k v2 + 融合映射 | 5.36 |

小 batch 的原版打分本来已很快；候选数量固定为16,384，少请求不会让两次2,048元素排序消失。原移植的固定开销把省下的扫描时间吃掉。修复后仍存在上下文越长收益越大的趋势，因为原版扫描随长度增长，候选打分的规模基本不变。

## 对蚂蚁实现的逐项核对

| 项目 | 蚂蚁 | 上一版移植 | 本次实现 |
|---|---|---|---|
| 候选状态 | source 发布排序 blocks、counts、top-k plan；同 ratio consumer 复用 | 每次 consumer 排序、去重、再排序 | 一次发布；生产 source 唯一性契约下用 CUB radix sort；通用输入保留单排序+scan 去重 |
| Top-k | 原生 top-k v2，确定性同分处理 | 继续用 vLLM top-k | 直接编译调用固定 commit 原生 top-k v2 |
| 位置映射 | finalize 映射，通用 top-k API也支持page transform | 独立 Triton kernel | 把候选块当作大小8的逻辑页表，融合进原生 top-k；输出仍是vLLM所需逻辑token位置 |
| verify key复用 | 全上下文 grouped分支共享；candidate WGMMA按row独立 | candidate WGMMA按row独立 | 保持；不能把不同query的候选列表当成相同列表共享 |
| prefill FP8/keys-major | 多query打分路径，FP4每请求解包一次 | 本次未测试 | 不把prefill的Torch基线收益套到vLLM decode DeepGEMM |

参考固定源码：[Ant decode状态复用](https://github.com/antgroup/sglang/blob/6e6d5d9e13915adb01db0856b15d6908eaaa2c48/python/sglang/srt/layers/attention/dsv4/v41_indexer/sm90_decode.py)、[候选发布](https://github.com/antgroup/sglang/blob/6e6d5d9e13915adb01db0856b15d6908eaaa2c48/python/sglang/kernels/ops/attention/dsv4/sm90_candidate.py)、[top-k接口](https://github.com/antgroup/sglang/blob/6e6d5d9e13915adb01db0856b15d6908eaaa2c48/python/sglang/kernels/ops/attention/dsv4/topk.py)、[grouped/candidate分支](https://github.com/antgroup/sglang/blob/6e6d5d9e13915adb01db0856b15d6908eaaa2c48/python/sglang/kernels/ops/attention/dsv4/sm90_fp4_indexer.py)。

CUB radix sort 是这次在 Ant 的状态复用机制上新增的适配优化，不是 Ant 原代码。2,048个唯一候选ID与无效padding一起排序/计数；有效block范围小于65,535时只需排序低16位，无效INT_MAX的低16位为最大值，保持完整32位输出。更大范围回退32位排序。没有压缩/截断ID，也没有降低分数精度。唯一性由source top-k保证；外部不可信候选应使用通用去重路径。

Ant 原策略只在上下文至少16个候选窗口时启用 compact，本例是256K；这里64K/128K为强制测试。新的结果仅支持这些已测形状，不应据此对所有长度/批次/平台无条件开启。

## 6行 verify

每请求6个query，候选列表可以不同。vLLM paged DeepGEMM原接口不支持N=6，因此基线按原准备策略展开成N=1。下表仅一轮，不能称为原生N=6 fused基线。

| 请求数 | Query行数 | 上下文 | 原版μs | 本次μs（含准备） | 加速 |
|---:|---:|---:|---:|---:|---:|
| 1 | 6 | 64K | 30.47 | 29.58 | 1.03× |
| 1 | 6 | 128K | 45.36 | 29.74 | 1.53× |
| 1 | 6 | 256K | 76.75 | 29.67 | 2.59× |
| 4 | 24 | 64K | 80.28 | 68.99 | 1.16× |
| 4 | 24 | 128K | 147.84 | 72.01 | 2.05× |
| 4 | 24 | 256K | 286.65 | 73.54 | 3.90× |

## 正确性和限制

- 最终版本36次case执行、每case四份不同Q/weights，共144次分数和top-k集合对比；最大分数nRMSE=7.233e-08，最大绝对误差=7.629e-06。沿用FP8基线；上一轮独立FP32参考已确认原版tensor core自身的舍入误差，本次未改打分kernel。
- 最低top-k集合重合率=99.80468750%。共有2条行级差异记录（含正反复测的重复记录）；是否全部为原始分数完全同分的替换：True。因此不能声称所有输出逐位一致。详见raw JSON的boundary_differences。
- 75组专项边界测试通过：空/短候选、部分尾块、重复/无效块、CPU集合与计数oracle、graph回放时更新候选和长度、全同分/多同分/负分数、不可见尾部填NaN。Top-k的16K register路径按低位置确定性处理同分。
- 有限Q/K/weights是本测试前提；未验证模型warmup产生NaN分数时的语义。无真实模型精度评测、无完整prefill/attention/解码速度、无TP8通信结果。
- Graph与eager都有原始结果；eager受Python提交和JIT wrapper开销影响较大，不拿本轮eager与上一轮进程跨进程比较。
- 全部修改在独立实验目录；未接入主线程vLLM仓库或服务。`ant_kernel_runtime.py`只用于新建的独立kernel测试进程，绕开serving包初始化，直接加载校验hash后的JIT/top-k源代码；不能在运行中的vLLM服务内替换sglang包namespace。

## 文件与复现

- `bridge_v2.py`：状态发布、唯一候选快速路径、重复候选回退。
- `candidate_publish.cuh`：CUB候选排序/计数，32位ID完整保存。
- `ant_kernel_runtime.py`：固定源码校验与kernel-only原生top-k加载。
- `indexer_stage_probe.py`：分项定位及warp/CUB对照。
- `indexer_v2_ab.py`：单层、四层组合、反序、verify、stress。
- `indexer_v2_edge_checks.py`：独立正确性oracle与graph更新检查。

在实验venv中设CUDA_HOME为其`lib/python3.12/site-packages/nvidia/cu13`，PATH包含venv/bin。Ant JIT链接需要该隔离工具链的lib64→lib和lib/libcudart.so→libcudart.so.13符号链接；不修改系统CUDA。解压ant-runtime.tar.gz至ant-runtime，然后运行：

```bash
python indexer_v2_edge_checks.py --ant-source ant-runtime
python indexer_v2_ab.py --ant-source ant-runtime --publication cub --output results/indexer_v2_final.json
python indexer_v2_ab.py --ant-source ant-runtime --publication cub --reverse-order --output results/indexer_v2_final_reverse.json
python indexer_v2_ab.py --ant-source ant-runtime --rows 6 24 --verify 6 --output results/indexer_v2_final_verify6.json
python indexer_v2_ab.py --ant-source ant-runtime --rows 1 8 --context 65536 262144 --stress --output results/indexer_v2_final_stress.json
```

这些命令会运行GPU；本地Mac只做py_compile和static_check.py。源文件hash、raw JSON与log保存在本目录的run-manifest.json及对应文件。
