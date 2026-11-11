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

### 用户自研算子包怎么接进来（两种形式，任选其一）

这不是业界统一的"算子自声明"标准——**标准的是"扩展自己注册 + 纯数据 schema"这个形状**
（PyTorch 的 `torch.library`/`TORCH_LIBRARY`、Python 的 `importlib.metadata` entry point、
vLLM 自己的 `platform_plugins` / `general_plugins` / `CustomOp.register_oot` 都是这个形状）。
不标准的是"宿主里写死一张路径表"（本仓的 `_LAYOUTS`），所以它只作为最后回落。

形式 A（声明式，已有）：包根暴露 `describe()`，返回纯数据。

```python
def describe():
    return {
        "name": "my_ops", "version": "0.1.0",
        "ops": {
            "moe.routing_topk_softmax": "my_ops.moe:topk_softmax",
            "gdn.l2norm": "my_ops.gdn:l2norm_fwd",
        },
    }
```

形式 B（协议式，新增）：包根暴露 `resolve(name)`，**一个函数、不用抄平台的词表**。

```python
def resolve(name):              # callable 或 None
    if name.startswith("moe."):
        return _MY_MOE[name]    # 没有就返回 None，平台会记成 "unbound + 原因"
    return None
```

两种都不需要 import `ascend_platform`、也不需要认识 vLLM。选包有两种方式：`pip install` 后在
`pyproject.toml` 里注册 entry point 组 `ascend_platform.providers`（装完即生效，最标准），或者
`ASCEND_PLATFORM_OPS=<包名>` 显式指定。两者可同时用 `describe()` 与 `resolve()`：声明优先，
`resolve()` 补漏。

**所以回答"用户知道要声明吗"**：今天要读到这份文档才知道；补的是**可发现性**，不是机制。
下一步该做的是 `ascend-platform provider-template`（打印上面两个模板之一）和 doctor 在报缺算子时
直接打印"你的包缺这个语义名"。在那之前，这份文档 + `ascend-platform doctor` 的输出就是对接说明。

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

## 6. 算子侧 2026-11-11 那次实测的核对

他们报的：`TypeError: topk_softmax() takes 2 positional arguments but 5 were given`，加 5 行适配器后
model loaded 23.4s、生成正确、OK 29.1s；结论"阻塞不在本仓，缺平台 `engine/abi` 的两张 out-param 表"。

**核对结果：崩溃原因、机制、方向都对，两处需要收紧。**

| 他们的说法 | 核对 |
|---|---|
| 崩在 `topk_softmax`，因为 vLLM 是 out-param、我们发函数式 | ✅ 完全正确。vLLM 调 `(topk_weights, topk_ids, token_expert_indices, gating_output, renormalize)` 5 个位置参数（`router/fused_topk_router.py:17-30`），`(scores, top_k, *, renormalize)` 只收 2 个位置参数，报错就是这句 |
| 平台把我们的函数直接塞进那个符号 | ✅ 正确，`install.py` 的 `_attribute_takeover` 就是这个行为（现在被 `engine_impls` 覆盖） |
| 缺 `engine/abi` 里两张 out-param → 函数式表 | ⚠️ 方向对，但**只证明了第一张**。他们的链路里 `engine.moe_method` 已注入，走的是 `_MoEKernel.apply → fused_experts`，而 `_custom_ops.moe_align_block_size` 只在 vLLM 自己的 Triton 专家路径里被调（`fused_experts`/`_prepare_expert_assignment`，`fused_moe.py:1425-1465`、`triton_moe.py:235`），这条路径被绕开了。第二张表是为"MoE 方法没被注入"（例如量化检查点）准备的，属于预防不是必需 |
| GDN 8 个符号实测解到 `ops.fla.chunk` / `ops.gdn.leaves` / `ops.fla.causal_conv1d` | ✅ 与我们的绑定报告一致；但**它们不是经他们的 `describe()` 绑定**，是经平台 `_LAYOUTS` 的硬编码路径（本仓 `contracts/ports.py`）。所以"阻塞不在本仓"对**这次端到端**成立，对"用户自研算子包也能跑 qwen3.6"不成立——那是本仓 + 算子仓各一半 |
| attention（decode+prefill）、MoE 专家计算走本仓 | ✅ 与绑定报告一致（`attention.*` 是平台后端，`engine.moe_method` 是平台类工厂） |

两张表本仓都已在 `d43fe09` 落地（`engine/abi/vllm_0_23.py::INPLACE_ENGINE_SYMBOL` +
`inplace_adapters()`），并有用真实 `vllm._custom_ops` 属性、按 vLLM 调用协议跑的验证。

## 7. 本仓复跑（2026-11-11 14:24–14:43，NPU 空闲后）

`VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_PLUGINS=ascend_platform ASCEND_PLATFORM_OPS=ascend_attention`，
`LLM(model=/home/models/Qwen3.6-35B-A3B, max_model_len=2048, gpu_memory_utilization=0.85,
enforce_eager=True)`，两个提示、`temperature=0`、`max_tokens=16`、`logprobs=5`。

**结果：跑通，不需要算子侧的 5 行临时适配器**（`d43fe09` 之后本仓已带适配器）：

```
RESULT model_loaded_s=976.1 generate_s=140.0
PROMPT 'The capital of France is' -> ' Paris, a city renowned for its rich history and cultural
heritage. Paris is the'
PROMPT '1+1=' -> '2 是公理吗？…'
OK
```

链路上确认走的是两条本仓路径 + 算子仓：`registered AscendUnquantizedFusedMoEMethod as the out-of-tree
UnquantizedFusedMoEMethod`、`OOT MoE: using 'ascend_attention.ops.moe.compute' for the routed experts`、
`ops: 11 bound, 1 unbound (chunk_local_cumsum, declined), 1 off by switch | dense=off gdn=on moe=on`、
`profile=qwen3.6 hybrid=True block_size=2048`。与算子侧独立跑出的文本前缀一致
（`' Paris, a city renowned for its rich…'`），说明不是偶发。

**首次尝试失败过，失败与平台无关但值得记**：默认 V1 多进程（fork）下 torch_npu 直接拒绝初始化 ——
`RuntimeError: Cannot re-initialize NPU in forked subprocess. To use NPU with multiprocessing, you
must use the 'spawn' start method`。必须 `VLLM_ENABLE_V1_MULTIPROCESSING=0`（或
`VLLM_WORKER_MULTIPROC_METHOD=spawn`），本仓 [`docs/hardware-checklist.md`](../hardware-checklist.md) §0
早就写了这两个变量——是我没先读自己的检查单，不是新问题。

`model_loaded_s=976` 里绝大部分是 tilelang decode kernel 的分桶 JIT 编译（每个 bucket 约 105 s，
1/2/4/8/16/32/64/128/256 共 9 个），缓存目录 `tl_cache*` 命中后会显著下降；`generate_s=140` 是两条
提示的 prefill+decode（含 prefill kernel 首次编译）。这两项都随缓存状态变化，不能当性能基线。

### 回归基线（`temperature=0`，`logprobs=5`，本机复跑时记录）

| 提示 | 首 token | 前 5 个 logprob（token_id: logprob） |
|---|---|---|
| `The capital of France is` | ` Paris` | 11751: -0.569381, 264: -2.194381, 279: -3.006881, 3750: -3.631881, 271: -3.694381 |
| `1+1=` | `2` | 17: -0.806895, 18: -1.306895, 16: -2.244395, 15: -3.744395, 10992: -3.556895 |
