# 测量证据

## release

本次提交的可复用模块在两张H20上的复测。`environment.json`包含实际运行的
Python/CUDA源码SHA256、共享库SHA256和环境版本。

- `fused_matrix.json`：21形状×2rank，每项包含eager/graph原始样本、误差和分项计时。
- `comparison.csv`：新融合测量与此前NCCL测量的对比，基线使用新一轮的原版函数测量。
- `stress.json`：5600个rank往返，包含任意BF16位模式；另外验证配置不一致时一起拒绝。
- `memcheck.txt` / `memcheck.json`：只对exchange_kernel检查访存，API错误报告关闭。
- `build.txt`：按最终CUDA源码hash选择的编译产物路径。

执行命令为仓库README中的两卡命令。定向memcheck执行：

```bash
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 \
  --no-python /usr/local/cuda/bin/compute-sanitizer --tool memcheck \
  --report-api-errors no --kernel-name kns=exchange_kernel --error-exitcode 99 \
  .venv/bin/python -m benchmarks.kernels.head_exchange.verify \
  --rows 9 129 --iterations 5 --output build/head_exchange/memcheck.json
```

`portable_tests.xml`记录Mac上的24项CPU测试。它不测CUDA同步或NVLink性能。

## 20261101

此前独立原型的原始数据，保持内容不变。`prototype_sources.tar.gz`保留当时
的最终原型源码、基线及运行脚本。旧脚本含当时远端绝对目录，复现当前模块应
使用仓库根目录README中的命令。

`fused_experiment_manifest.json`为当时完整实验目录的清单，部分中间版本和
编译产物没有重复放入本仓库；本仓库实际附带文件的校验值见上一级`evidence.json`。
旧日志的`.log`扩展名改为`.txt`，内容没有改写。

`13-fused-memcheck.txt`保留首次未过滤检查的非零退出记录；其后定向访存检查的
0错误不能覆盖、删除或解释为首次全量检查通过。

## 性能解读

所有GPU结果只有两个物理rank，每卡8个有效head，交换后16个有效head。
数据为预热、重复输入、复用planner的微基准。结果排除初始化、动态metadata
准备、模型投影、indexer与调度成本。不可当成端到端vLLM加速、TP8预测值或
模型精度验证。各rank的P50取最大值，不是跨rank逐次延迟的联合P50。
