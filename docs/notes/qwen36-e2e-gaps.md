# Qwen3.6 端到端还缺什么（vLLM 0.23 + AscendPlatform + AscendAttention，不含 TileInfer）

**目标链路**：`vllm 0.23.0` → `AscendPlatform`（引擎/平台）→ `AscendAttention`（算子）。
模型：`/home/models/Qwen3.6-35B-A3B`，`architectures = ["Qwen3_5MoeForConditionalGeneration"]`，
`model_type = qwen3_5_moe`，bf16（`quantization_config` 为 `None`），40 层 = 30 层 GDN +
10 层 full attention（`full_attention_interval = 4`），256 专家 top-8，`head_dim = 256`，
16 Q / 2 KV，`attn_output_gate = true`，MRoPE interleaved + `partial_rotary_factor = 0.25`，
`mamba_ssm_dtype = float32`。

## 结论

**算子数量够，但"算子层自己声明"这条路还没通**：13 个必需算子里 AscendAttention 的
`describe()` 只声明了 4 个（全部是 MoE 计算），**8 个 GDN 算子靠平台里硬编码的 `_LAYOUTS` 路径表
绑定**。所以"用户写自己的算子也能拉起平台"这个目标，对 Qwen3.6 目前只在 MoE 上成立，GDN 上不成立。

端到端**能跑起来的前置条件**只剩两类：一是 AscendAttention 补 8 个声明（他们改一个文件），
二是下面第 3 节的数值未验证项。模型侧、引擎侧、注意力后端都没有已知缺口。

## 1. 算子绑定（13 个必需算子）

| 分类 | 必需 | AscendAttention 声明 | 绑定途径 |
|---|---|---|---|
| MoE 计算 | `moe.routing_topk_softmax`、`moe.align_block_size`、`moe.experts`、`moe.swiglu_gate` | ✅ 4/4 | provider 契约 + `engine/abi` 的 in-place 适配器 |
| GDN（vLLM 的 FLA 名） | `gdn.chunk_gated_delta_rule`、`gdn.conv1d_prefill`、`gdn.conv1d_update`、`gdn.fused_sigmoid_gating`、`gdn.recurrent_decode`、`gdn.post_conv_prep`、`gdn.l2norm` | ❌ 0/7 | **只有** `contracts/ports._LAYOUTS` 的硬编码路径 |
| GDN 可选 | `gdn.chunk_local_cumsum` | ❌（故意 declined：签名不是 drop-in） | 不绑（链路内部自己调） |
| 平台自有 | `engine.moe_method` | —（平台 `engine/abi` 提供类工厂） | `custom_op_oot` |

实测（真 vLLM 0.23 符号 + AscendAttention `describe()`）：**11 bound / 1 declined / 1 off by switch，
`servable = True`**。也就是说"能绑上"，但 8 个 GDN 是**绕过 provider 契约**绑上的。

**缺的一步**：AscendAttention 在 `describe()` 里把上面 7 个 GDN 名字（连同 `gdn.chunk_local_cumsum`）
发出来，目标是它已有的 `ops/fla/*.py`、`ops/gdn/leaves.py`。做完之后 `_LAYOUTS` 可以只留
TileInfer/用户包的回落，甚至删掉。**在此之前，任何"用户自研算子包"都跑不了 Qwen3.6 的 GDN 部分**，
因为平台根本不会向它要这些名字。

## 2. 调用约定（已修，这次）

vLLM 的 `vllm._custom_ops.*` 是 **in-place** 的 C++ 包装：`topk_softmax(topk_weights, topk_ids,
token_expert_indices, gating_output, renormalize) -> None`（`_custom_ops.py:2494-2510`）、
`moe_align_block_size(topk_ids, num_experts, block_size, sorted_token_ids, experts_ids,
num_tokens_post_pad, expert_map=None) -> None`（`:2346-2363`）。AscendAttention 发的是函数式
（`topk_softmax(scores, top_k) -> (weights, ids)`），直接绑会在第一次路由时 `TypeError`。

现在由 `engine/abi/vllm_0_23.py::inplace_adapters()` 生成适配器，`install_ops(engine_impls=...)`
把它作为实际绑定的实现。实测（真 `vllm._custom_ops` 属性，vLLM 自己的调用协议）：

- `topk_softmax`：ids 与 `torch.topk` 逐位相同；权重与**重归一化**参考 fp32 完全一致（0.0e+00）、
  bf16 3.8e-03（输入量化，不是算子误差）；`token_expert_indices` 一并写入。
- `moe_align_block_size`：12 个真实 pair + 20 个 pad 槽，pad 值 = `numel`（vLLM 的 sentinel 约定），
  未产出的 expert block 尾部写 `-1`。

`moe.swiglu_gate` **不在** in-place 表里：vLLM 0.23 没有 `_custom_ops.silu_and_mul`（只有量化变体），
平坦版本是 `torch.ops._C.silu_and_mul`，只被 `SiluAndMul.forward_cuda` 调用；OOT 平台走
`forward_native`（纯 torch），没有可绑的模块属性。已从 `ENGINE_SYMBOL` 移到 `NO_ENGINE_SYMBOL`。

## 3. 还没验证的数值项（不是已知缺陷，是未知）

| 项 | 为什么可能出问题 | 怎么验（都要真机，且不抢显存） |
|---|---|---|
| bf16 logits 下的 top-k 权重 | C++ 版内部按 fp32 算，算子层按输入 dtype 算；ids 实测一致，但 logprob 末位可能差 | 32 token、`temperature=0`、`logprobs=5` 对拍 |
| GDN 状态 dtype = fp32 | 配置里 `mamba_ssm_dtype: float32`，而 q/k/v 是 bf16；FLA 移植是否接受混合 dtype 未测 | 单层 `chunk_gated_delta_rule` + `initial_state` fp32 微测 |
| `chunk_gated_delta_rule` 的 `scale` | 平台注入的 `forward_native(self, q, k, v, g, beta, initial_state, ...)` **没有** `scale`（上游签名就这样），算子的 `scale` 默认 `None`，内部默认值必须等于 `head_k_dim ** -0.5`（128 → 0.0884） | 传 `scale=None` 与显式 `0.0884` 比较单层输出 |
| `num_accepted_tokens`（投机解码） | 算子有该参数，但没测过非 `None` 分支 | 不属于基础端到端；接 MTP 时再验 |

## 4. 已确认不是缺口的东西

| 项 | 结论 |
|---|---|
| 架构注册 | `Qwen3_5MoeForConditionalGeneration`、`Qwen3_5MoeMTP` 都在 vLLM 0.23 的 registry 里，不需要平台注册 OOT 架构 |
| 量化 | 检查点是 bf16、`quantization_config = null` → 走 `UnquantizedFusedMoEMethod`，正是 `engine/abi` 类工厂覆盖的那条 |
| `attn_output_gate` | 在模型代码里应用（`qwen3_next.py:294-317`），注意力后端不需要知道 |
| MRoPE | `MRotaryEmbedding.forward_native` 是纯 torch，OOT 不碰 Triton 分支 |
| 注意力后端 / 混合 KV | p5 gate 已用同一条平台侧链路跑通到 logprob 逐位一致（那时算子层是 TileInfer） |

## 5. 运维事实：两个 platform plugin 不能同装

实测：同一 venv 里同时装了 `tileinfer` 和 `ascend_platform` 两个 `vllm.platform_plugins` 时，
任何 `import vllm` 都直接失败：

```
RuntimeError: Only one platform plugin can be activated, but got: ['tileinfer', 'ascend_platform']
```

这正是"不要依赖 TileInfer"的运维理由：目标链路必须用 `VLLM_PLUGINS=ascend_platform`（或卸载
tileinfer）。TileInfer 只在验证"用户自研算子包"这条路时才需要，且它是**被显式指定**的
（`ASCEND_PLATFORM_OPS=tileinfer`），不再参与自动发现——`contracts/ports.OPERATOR_ROOTS` 现在只有
`ascend_attention`。
