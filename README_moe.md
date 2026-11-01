# H20 MoE：W4A8 与输出后处理融合

基于官方 **vLLM 0.32.0 / Humming 0.1.17**，交付已验证的输出 kernel、显式适配器和既有性能数据。
不包含双 FP8 残差补算、实验驱动或测试脚本；本次整理没有重新运行实验。

**已有本地执行链测量：W4A16 → 普通 W4A8 + 输出融合，在 8–4096 token 样本中加速约 1.46–1.91×；如果已经使用 W4A8，融合本身减少约 1.47%–5.65% 延迟。**
这些是函数级结果，尚未把共享输出融合接入完整 MoERunner 调度与八卡通信。

## 配置与执行范围

- 目标 TP8 + EP8；MoE 专家内部 TP=1，每卡持有 48 个完整专家。
- 全局 E=384、top-k=6、H=5120、I=2304。W13 为 4608×5120，W2 为 5120×2304。
- 两张 H20 96GB 分别采样逻辑 rank 0、1，不是 TP2/EP2，也没有模拟其余六卡输出。
- 基线直接调用官方 `HummingGroupedExperts.apply()` 和官方权重转换函数；保留原版 GEMM 启发式配置。
- 所有输入、MXFP4 权重和共享 partial 均为合成数据，没有加载完整模型。

计时包含：路由整理、激活量化、W13 GEMM、SwiGLU/W2 输入处理、W2 GEMM、逆排列与专家加权和、routed scale、加已经就绪的 shared partial。
不包含 router GEMM、共享专家 GEMM、跨卡 dispatch/combine、attention、调度器或整模型服务。

## 改了什么

### 1. W4A16 → W4A8：使用已有的 Humming 路径

权重始终保留 MXFP4 存储。A16 使用 BF16 激活；A8 使用每行动态 E4M3 FP8 激活，沿用 Humming 的权重准备、scale 处理与 FP8 GEMM。
这是上游已有精度路径的选择，不是本模块新实现 GEMM；性能收益包含激活量化的数值变化。

vLLM 已提供 `VLLM_HUMMING_INPUT_QUANT_CONFIG`。对应本实验的输入配置为：

```json
{"dtype": "float8e4m3", "group_size": 0, "allow_fallback": false}
```

也可直接在构造 Humming 层时指定 `input_config={"dtype": "float8e4m3", "group_size": 0}`，与本实验一致。
上述环境变量只在实际进入 Humming 权重准备流程时生效，不负责选择 MoE 后端；全局配置可能影响其他 Humming 层。
本次没有验证整模型启动参数，不能将这段配置当作已验证的部署命令。加载后应检查两次 GEMM 的 `a_dtype=float8e4m3`、`b_dtype=float4e2m1` 和实际专家执行类。

### 2. 输出 kernel：二维并行与后处理融合

原版计算：

```text
W2 输出 → 逆排列/专家加权和 → BF16 输出
        → 乘 routed scale → BF16 输出
        → 加 shared partial → BF16 输出
```

融合计算：

```text
W2 输出 → 一个二维 kernel：逆排列/加权和 + scale + shared → 最终输出
```

`ops.py` 按 token 和 hidden 通道划分 CTA，增加小 token 下的并行度；融合后减少后续 kernel 启动和中间张量读写。
它保留 `FP32 加权和 → BF16 → scale → BF16 → shared → BF16` 的原始舍入边界，使用显式 PTX BF16 转换防止编译器跨边界合并计算。

两个入口不能混用：

- `moe_unpermute`：目标 EP8 的 Grouped 路径，按逆排列取回专家输出；跳过无效映射/接收尾界，空路由输出零。
- `moe_fused_mul_sum`：Indexed 路径，保留原版对全无效 ID 行和有效接收范围之外行的输出约定，不读取非本卡专家的无效输入。

共享输出必须处在与 routed 输出相同的归约阶段，且不能与输出缓冲重叠。本模块不移动 collective、不计算 shared GEMM。

### 3. 保守的可插拔适配器

`adapter.use_aligned_humming_finalize()` 在显式 context 内替换 Humming 输出函数，退出恢复，检查 H20 和版本。
只对输入 token≤128 使用二维 kernel，大 token 回到原版；它**不融合 shared、也不改变量化或 GEMM**。
这是保守替换接口，不能把完整融合的性能表当作这个适配器的收益。

```python
from vllm_dsv41_moe.adapter import use_aligned_humming_finalize

with use_aligned_humming_finalize():
    output = existing_humming_call()
```

这里的 `existing_humming_call` 表示调用者已有的执行函数。适配器修改进程内模块函数，只适用于串行实验或 worker 初始化；不要在并发请求之间反复切换。
在预热/图捕获前确定路径，导入本模块不会自动启用服务端 hook。

## 显式使用融合接口

在 Grouped 输出合并位置，调用：

```python
from vllm_dsv41_moe.ops import moe_unpermute

moe_unpermute(
    out,
    permuted_hidden_states,
    topk_weights,
    inv_permuted_idx,
    expert_first_token_offset,
    shared=shared_partial,
    routed_scaling_factor=1.5,
)
```

这段代码是合并位置的接口示意；张量由原有专家执行路径提供。完成后不要再重复乘 scale 或加 shared。
非单位 scale 的融合入口要求 BF16；`shared` 可在行方向非连续，但最后一维 stride 必须为 1。
不传 `shared` 时为普通输出合并，不能单独借该入口融合非单位 scale。
完整服务仍需验证共享专家就绪时机、EP 通信位置、DBO、图捕获生命周期和不同 dispatch 后端；本次不做未经验证的自动接线。

## 性能与数值结果

[完整性能表](docs/dsv41_moe/performance.md) 分开列出 A16、A8、A8+融合和收益；[CSV](docs/dsv41_moe/performance.csv) 保存精确中位数和来源。

- 同精度输出函数的既有 142 项 GPU 对照逐位一致；完整本地链在双方关闭 Stream-K 后也逐位一致。
- 性能测量恢复原版启发式；rank 1 单 token 的原版 A8 出现过约 0.00668 相对 L2 的重复运行差异，不宣称默认 Stream-K 完全确定性。
- 普通 A8 相对 A16 的独立 324 组精度复测：随机 FP4 和缩放正态权重的相对 L2 约 4.49%–4.72%，DeepGEMM `calc_diff` 约 0.00101–0.00111；未缩放正态权重为 6.76%–9.19%。严格均衡与等权重路由没有消除差异。
- 这是本地 routed 输出的量化差异，不是模型准确率下降百分比。融合本身的同精度对照没有新增误差。
- 尚无完整模型质量、八卡性能或完整部署评测。本次提交仅整理已验证的核心实现和已测结果，没有重新跑实验。

## 没有迁入的部分

- EP8 完整专家维度已经满足原有对齐，减少 padding 没有缩小本配置矩阵。
- vLLM 原路径已融合 SwiGLU 与输入量化，不重复计入收益。
- 关闭 Stream-K 没有稳定提速，不作为默认性能优化。
- 未把蚂蚁面向旧版 Humming、Indexed/专家内部 TP 的 overlay 或调优表直接套到本配置。
- 双 FP8 补偿会增加计算，本模块不包含它。

输出融合思路参考[蚂蚁 SGLang 的 Humming 优化与数值契约测试](https://github.com/antgroup/sglang/blob/6e6d5d9e13915adb01db0856b15d6908eaaa2c48/test/registered/kernels/ops/moe/test_humming_fp8_dispatch.py)。这里的 kernel 按 vLLM 0.32.0 的 Grouped/Indexed 接口实现；不包含旧 Humming 头文件或整份蚂蚁代码。
