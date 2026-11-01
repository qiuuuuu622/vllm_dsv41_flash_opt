# vLLM 0.32.0 · DeepSeek-V4.1-Flash H20 experiments

本仓库保留官方 vLLM 0.32.0 服务代码，新增显式调用的 kernel/函数级优化模块。

- [稀疏注意力 indexer：机制、TP8适配、性能与复现](README_indexer.md)
- [head↔row 交换与融合 NVLink 通信](README_opt.md)
- [MoE：W4A8 路径与输出后处理融合、机制及性能](README_moe.md)
- [上游 vLLM 介绍](README.upstream.md)

这些优化均不因导入模块而自动修改 `vllm serve`。单卡/双卡函数结果与完整模型、
八卡服务性能分开记录。已撤销的小 M MXFP8 GEMM 没有恢复。
