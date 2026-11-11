# 从 TileInfer 迁移硬件特性到 AscendPlatform

> **执行状态**：P0–P5 已落地，P6 的 CI 半已完成、**上机对比**待设备窗口。
> 逐阶段证据见下面的"阶段 → 证据"表；落地清单、搬运期间修正的既有错误、验收数字见
> [`../MIGRATION_LOG.md`](../MIGRATION_LOG.md)。本文其余部分是**方案与理由**（决策记录），
> 保持迁移当时的判断，不随进度重写——除这张状态表外，它是"当时为什么这么想"的存档。

### 阶段 → 证据

| 阶段 | 计划产物 | 现在的证据 | 偏差 |
| --- | --- | --- | --- |
| P0 冻结契约 | `device-profile.md` / `op-manifest.md` / `versioning.md` + `contracts/` | [`docs/device-profile.md`](device-profile.md)、[`docs/op-manifest.md`](op-manifest.md)、[`docs/versioning.md`](versioning.md)、[`src/ascend_platform/contracts/`](../src/ascend_platform/contracts) | 清单 schema 写在 `contracts/manifest.py` 里，没有单独的 `.json` 文件（同一份 schema 从代码导出，避免两份真相） |
| P1 骨架可安装 | `pyproject.toml` + 两个入口点 + doctor 骨架 | [`pyproject.toml`](../pyproject.toml)、`test_package_layout.py`（入口点可解析）、`ascend-platform --help` | 无 |
| P2 `DeviceProfile` 落地 | frozen dataclass + 950PR 首份 profile + 第二份可插 | [`src/ascend_platform/device/`](../src/ascend_platform/device)、`test_device_profile.py`（28 条，含"加第二张卡 = 一个模块 + 一行 import"的棘轮） | kernel 几何（128/2048）保留为常量并加防漂移断言，理由见 `device-profile.md` §6 |
| P3 平台语义搬运 | `platform/worker/graphs/doctor/plugin/compat` + `serving/` | [`src/ascend_platform/engine/`](../src/ascend_platform/engine)、[`src/ascend_platform/runtime/`](../src/ascend_platform/runtime)、`test_{platform,worker,attn,boot_rehearsal,doctor}_*.py` | 无 |
| P4 KV / 元数据 / attention 适配 | `attn/{backend,metadata,impl,kv_cache}` | [`src/ascend_platform/engine/attention/`](../src/ascend_platform/engine/attention)、`test_kv_cache_write.py`、`test_attn_contract.py` | 无 |
| P5 算子接口 + 名字重绑定 | 三函数 + 清单 + 启动期校验 + `prefill` 改走接口 | [`src/ascend_platform/contracts/ports.py`](../src/ascend_platform/contracts/ports.py)、[`contracts/manifest.py`](../src/ascend_platform/contracts/manifest.py)、[`src/ascend_platform/ops/`](../src/ascend_platform/ops)、`test_ports.py`、`test_contracts_manifest.py`、`test_ops_rebinding.py` | `prefill.py` **没有整体移出**：计划里它属于 AscendAttention，实际上"规划/元数据编排"这半留在平台（`engine/attention/prefill.py`），只有内核那半经端口调外部；见下 |
| P6 切流与 CI | 依赖方向 CI 检查 + 双跑对比 | 依赖方向：`test_layer_acl.py`（静态 + 子进程动态，78 条）；双跑对比：**待硬件**，步骤见 [`notes/p5-gate.md`](notes/p5-gate.md) | 上机对比需要设备窗口 |

**"一行不落地"现在有机器读的账.** 交付项 2 是关于**源仓**的断言，所以审计从源仓那一侧数：
`tileinfer/**/*.py` 共 107 个文件，每个必须落进"迁移 31 对 / 整棵树排除 59 / 单文件排除 17"三者之一，
排除理由不得是橡皮章，两份差异白名单必须恰好用尽。这条账顺带查出两件事：`vllm/ops/families/__init__.py`
搬过来了却没进比对清单，`vllm/ops/extern.py`（旧的算子桥）被交付项 4(b) 取代却没人写下——后者现在
落在一个写明"被取代"的条目里。细节见 `MIGRATION_LOG.md` §4.8–§4.10。

**§4 去向矩阵里 `attn/prefill.py` 那一行的偏差.** 计划是整文件去 AscendAttention，落地时按职责切开了：
`PrefillPath`（tile 规划、`_PACK_CACHE`、RaggedMetadata 编排）是**执行语义**，属于平台；
它调用的 `forward_prefill` / `_pack_indices` 是**内核**，属于算子层，经
`contracts/ports.py` 取（取不到就抛 `OperatorLayerUnavailable`，而不是 import 崩）。
这样"平台镜像里 grep 不到 kernels"照样成立，但平台仍然拥有 prefill 的**决策**部分——
这是原计划没有明确的一条判断，记在这里。

> 本文回答三件事：**AscendPlatform 是什么**（定位与边界）、**TileInfer 里有什么**（可迁移的素材）、
> **怎么搬**（阶段、接口、验收、风险）。
>
> 事实基线：TileInfer `339969a`（2026-11-10 15:59），AscendPlatform `bd72586`（2026-11-10 17:09），
> 单卡 Ascend950PR / CANN 9.3.0。所有 `file:line` 引用以这两个提交为准。
>
> 配套文件：[`hardware-facts.md`](hardware-facts.md) —— 逐条带出处的硬件与 vLLM 契约事实清单，
> 本文 §3 是它的分类摘要；搬运时用它对照"有没有搬丢"。

---

## 0. 结论先行

1. **迁移开始时 AscendPlatform 是"零代码契约仓"**：只有一份 README（当时还是未提交的工作区改动），没有
   `pyproject.toml`、没有源码、没有测试。它不是要"继续开发"的仓库，而是要从别处**收割**的仓库。
2. **TileInfer 是唯一有实现的单体**（37,941 行 Python，21,304 行在 `tileinfer/vllm/`），
   而且它内部**已经按"平台 / 算子 / 模型"切过一刀**——只是切在了包内，不是仓间。
3. 因此迁移的本质不是"把硬件特性重写一遍"，而是：
   **把 TileInfer 里已经长成平台语义的那 1 万余行连测试一起搬出来，并补上 TileInfer 里
   至今不存在的两样东西 —— `DeviceProfile` 与"平台↔算子层"的公共接口。**
4. 可行性有硬证据：`tileinfer/vllm/` 里 import 内核的只有 **3 处 / 2 个文件**
   （`attn/prefill.py` 2 处、`ops/fla/causal_conv1d.py` 1 处）。平台层对算子数学的依赖
   事实上已经断了，剩下的只是"没有正式接口"。
5. 建议 **7 个阶段、约 6 周纯工程 + 2 周上机验证**，关键验收门是两条：
   **"第二个 `DeviceProfile` 能装进来"** 与 **"平台层静态检查不出现 `kernels`"**。

---

## 1. AscendPlatform 的定位（README 逐条拆解）

### 1.1 一句话

> 面向昇腾的 **out-of-tree vLLM 平台插件**：提供设备与执行语义，**不提供算子数学**。

三个限定词都是排他性的，不能只读前半句：

| 限定词 | 排除的东西 |
|---|---|
| out-of-tree | 不改 vLLM 源码；只走 `vllm.platform_plugins` / `vllm.general_plugins` 两个官方入口 |
| 平台插件 | 不注册模型、不做服务端、不做 benchmark 框架 |
| 不提供算子数学 | 内核实现不属于本仓；本仓只决定"用谁的、能不能进图、形状对不对" |

### 1.2 README 九项职能的归属判定

| # | README 写的职能 | 判定 | TileInfer 现位置（行数） | 迁移动作 |
|---|---|---|---|---|
| 1 | 平台契约：设备探测、显存画像、能力协商 | 平台核心 | `vllm/platform.py`(1954) | 搬，并抽出 `DeviceProfile` |
| 2 | KV Cache：布局与形状、跨层共享 | 平台核心 | `vllm/attn/kv_cache.py`(279) + `platform.py` 的 hybrid 段 | 搬 |
| 3 | 批元数据：block table / seq len → 纯数据 | 平台 | `vllm/attn/metadata.py`(334)、`metadata.py`(469) | **拆两层**：类型归算子层，构造归平台 |
| 4 | 注意力后端适配（薄适配层） | 平台 | `vllm/attn/{backend,impl}.py`(466+1250) | 搬，并把直连内核的 2 处改走公共接口 |
| 5 | 跨来源策略：启动期一次绑定 | 平台 | `vllm/ops/{install,candidates,switch}.py`(755+440+129) | 搬**机制**，绑定表随模型族走 |
| 6 | 图捕获：CUDA 语义 → NPU 原语桥接 | 平台 | `vllm/graphs.py`(929) + `worker.capture_model` | 搬（**未完成状态一起搬**，见 §6 风险 3） |
| 7 | worker：设备初始化、可用显存、KV 初始化、预热 | 平台 | `vllm/worker.py`(2059) | 搬 |
| 8 | 自检与性能观测 | 平台 | `vllm/doctor.py`(808) + `serving/{phases,counters}.py` | 搬 |
| 9 | 模型事实：头维度、页策略、混合层比例 | 平台 | `vllm/profiles/`(936) | 搬 |

九项里没有一项需要内核实现——这就是"平台层"的可操作定义。

### 1.3 README 说了但没定义的三件事（迁移必须先补）

| 缺口 | 现状 | 为什么挡住迁移 |
|---|---|---|
| **C1 平台↔算子层接口** | README 只说"只要满足按形状算得出 + 可进图即可" | 没有函数签名，搬运时无法判断哪些调用算"合法" |
| **C2 纯数据清单 schema** | README 说"读取算子层导出的纯数据清单"，但无字段定义 | 跨来源策略要靠它做启动期绑定 |
| **C3 版本兼容区间** | README 说"两仓互相声明、启动期校验" | 迁移后两仓独立发版，没有它就只能靠人记 |

这三件事**不写代码也能先定**（P0 阶段就是干这个），成本最低、收益最大。

### 1.4 显式非目标 = 迁移的黑名单

README「非目标」列的：算子数学、搬入厂商 kernel、MLA、量化、TP/EP、完整投机解码。
迁移时**任何**以"顺便"开头的动作都在此列。特别注意：`tileinfer/kernels/`(6,413 行) 一行都不进本仓。

---

## 2. TileInfer 是什么

### 2.1 识别结论

一个 **vLLM 0.23 的原生昇腾后端**：`pip install vllm tileinfer` 之后，
`vllm serve Qwen3.6-35B-A3B` 在 1 张 950 上跑起来，**不安装 `vllm_ascend`**。
它把 vLLM 上游缺失的整层昇腾实现（平台、worker、attention 后端、图捕获、GDN/MoE 算子）自己扛了下来。

### 2.2 它内部已经分好层了

```
tileinfer/
├── vllm/        21,304   ← 引擎适配：平台语义 + 算子接线 + 模型适配   【迁移主体】
│   ├── platform.py worker.py graphs.py doctor.py plugin.py
│   ├── attn/      后端四件套
│   ├── profiles/  模型事实（936）
│   ├── compat/    vLLM 兼容补丁（370）
│   ├── ops/       算子接线与开关（1,946），另有 fla/moe/gdn 实现（~4.5k）
│   └── models/    dsv41 的批语义与状态槽（~470）
├── kernels/      6,413   ← TileLang 内核                      【→ AscendAttention】
├── models/       5,146   ← DeepSeek-V4.1 模型实现（"加模型"的样例）  【留 TileInfer】
├── serving/      1,452   ← 与引擎无关的会话语义：buckets/planner/step/phases/counters 【迁移】
├── attention/    1,288   ← plan/run 契约 + 后端注册表 (BatchAttention)  【→ AscendAttention】
├── metadata.py     469   ← 纯数据：PageTable / RaggedMetadata   【→ AscendAttention】
├── plan.py         646   ← 纯数据：AttentionPlan / LPT 调度     【→ AscendAttention】
├── testing/        743   ← 数值 oracle（离线可跑）              【→ AscendAttention】
└── utils.py        131
```

配套：`tests/` 91 个文件 **25,204 行**（代码:测试 ≈ 1.5:1），`docs/` 7,744 行。
模块 docstring 里普遍带 vLLM 0.23 的 `file:line` 依据——**这是最容易在迁移中丢掉、也最贵的资产**。

### 2.3 四条支撑"可以搬"的证据

**E1 平台层对内核的依赖事实上已断。** 全仓 `tileinfer/vllm/` 里只有 3 处 import 内核，
且**全部是函数内延迟 import**（不在模块加载路径上）：

| 位置 | 内容 | 处置 |
|---|---|---|
| `tileinfer/vllm/attn/prefill.py:263,279` | 直接 import `kernels.attention.paged_decode_ascend950`（`_pack_indices`、`forward_prefill`） | 改走算子层公共接口（P5） |
| `tileinfer/vllm/ops/fla/causal_conv1d.py:633` | 函数内延迟 import 卷积内核 | 本文件属于算子层，随族迁走 |

除这 3 处外，decode 路径**只经 `attention.api.BatchAttention.plan/run` 触达内核**
（`tileinfer/vllm/attn/impl.py:92,795`，后端名 `tilelang-ascend950`）——它已经是"合法入口"的样子了。

**E1b 厂商面比想象中窄。** `tileinfer/vllm/` 里**没有** `acl` / `aclnn` / `vllm_ascend` 的
API 调用（仅出现在注释里）；`torch_npu` 只在函数内 import 一次，目的是**注册 `torch.npu`
命名空间**，之后所有调用都走 `torch.npu.*`（`set_device` / `get_device_properties` /
`mem_get_info` / `memory_allocated` / `is_current_stream_capturing` / `NPUGraph` / `Stream` …）。
HCCL 从未被调用（`dist_backend` 为空）。这意味着平台层的移植面是 **torch 的 NPU 命名空间**，
不是某个私有 SDK——对"换卡"是好事。

**E2 平台层依赖的"外部"都是引擎无关核心。** `platform.py` → `serving.buckets`、`profiles`；
`attn/impl.py` → `attention.api.BatchAttention`、`serving.{buckets,counters,planner,step,phases}`；
`attn/metadata.py` → `serving.{buckets,counters}`。这些都不 import vLLM、都不含内核数学，
是干净的"会话语义层"——**归平台**。

**E3 契约测试已经存在。** `tests/test_platform_contract.py`、`test_worker_contract.py`、
`test_attn_contract.py`、`test_boot_rehearsal.py` 等把 vLLM 0.23 的 ABC 面钉住了。
迁移的验收可以直接复用，不用从零写测试。

### 2.4 缺口：迁移**不**解决的三件事

| 项 | 现状（TileInfer `docs/status.md`） |
|---|---|
| 性能 | TPOT 2.53×、TTFT 11.8× 于 vllm-ascend 基线，门是 1.10× |
| 图捕获（ACLGraph） | 未接上；`--enforce-eager` 是唯一稳定形态。**根因已定位为自身路径没接完，不是平台限制** |
| 投机解码（MTP） | 未开始；批语义目前只允许单 query token |

**迁移是"搬家不改行为"**。把未完成的东西搬过来，它还是未完成的——这点必须写在方案里，
否则会被当成"迁移后性能会变好"的承诺。

---

## 3. 硬件特性清单（到底要迁什么）

"硬件特性"不是一个抽象名词，落下来是 8 类可枚举的事实。每类的处置方式不同：

| # | 类别 | 代表事实 | TileInfer 位置 | 迁移后形态 |
|---|---|---|---|---|
| H1 | **设备探测与能力位** | 950PR：`multi_processor_count=56`、`cube_core_num=28`、`vector_core_num=56`；HBM 131072 MB；`get_device_name/properties` 两种拼写都要试 | `platform.py:1070-1160` | `DeviceProfile` 字段 + 探测函数（保留"探测失败就报错，不许猜"） |
| H2 | **显存画像与 KV 定容** | `NpuMemorySnapshot`：`mem_get_info` / `memory_stats` / `max_memory_allocated` 三种拼写都要兜底；`determine_available_memory` 走 profile_run 前后差值；cudagraph 显存估算；`gpu-memory-utilization` 0.75 | `worker.py:508-676,1533-1690` | 平台实现，profile 提供"画像口径" |
| H3 | **布局与对齐** | 引擎页 `HYBRID_BLOCK_SIZE=2048` → 内核页 `KERNEL_PAGE_SIZE=128`（整页可整除）；KV 形状 `(blocks, 2, page, heads, dim)` + 物理 stride 序 `(1,0,2,3,4)`（K/V 是两个连续半区，`block_dim=0`）；ND→NZ 模板要求 `ROWS % 16 == 0`、`dual_copy` 按 2 AIV 切 M ⇒ **M tile `% 32 == 0`**（所以 GQA group 8 补到 32）；**描述符粒度 32 B**（16 B 会触发 507035）；bf16/fp16 only | `platform.py:282`、`tileinfer/vllm/attn/backend.py:18,200-231`、`tileinfer/serving/buckets.py:57`、`tileinfer/kernels/attention/paged_decode_ascend950.py:19` | `BlockSizePolicy` + `KvShape` + 布局/对齐声明，作为 `DeviceProfile` 的一部分 |
| H4 | **执行模式 / 图捕获** | 把 `torch.cuda.*` 别名桥到 `torch.npu.*`：`CudaAlias` 表 **9 项（4 必选 + 5 尽力）**，另有 4 个**刻意不映射**并写明理由；`CUDAGraph`→`NPUGraph`（名字不是恒等）；捕获期**禁编译/禁分配/禁同步/禁建 plan**，7 个断言码 | `graphs.py:214-940` | 原样搬（这是最纯的"NPU 原语桥接"） |
| H5 | **环境与工具链** | CANN 9.3；`ACL_OP_INIT_MODE=1`（不设会**出错数而不是报错**）；libstdc++ 需 3.4.29/3.4.30；TileLang 0.1.15 + `TILELANG_CACHE_DIR` 必须预烧且与服务同目录；**编译会阻塞进程**（~70 s），所以必须离进程预烧 | `doctor.py`、`docs/hardware-checklist.md` | `doctor` 子命令 + `ENVIRONMENT.md`（表格化，每条带"失败长相"） |
| H6 | **算子名字重绑定** | 三种接管形态：`CustomOp.forward_oot`、模块属性重绑、`register_oot`；`TILEINFER_OPS` 按族开关；每次调用只做"免费判断" | `ops/{install,switch}.py` | **机制**留平台，**绑定表**随模型族（`ops/families/`） |
| H7 | **混合 KV** | `support_hybrid_kv_cache()=True`；mamba state page 补齐到**恰好一个 attention 页 = 4 MiB**（`mamba_page_size_padded=4194304`，attention 页 2(K+V)×2 kv_heads×256 head_dim×2 B = **2048 B/token**）；10 attn + 30 GDN 同池两页 | `platform.py:1200-1290,1454-1560` | 平台能力位 + 对齐算法 |
| H8 | **模型事实** | 头维度、GQA 组下限、页策略、`op_family`、拒绝清单 | `profiles/{base,qwen3_6}.py` | 原样搬，作为 `ModelProfile` |

> 判据提醒：**凡是"换个模型就变"的是 H8，而不是硬件；凡是"换个卡就变"的才是 H1–H5。**
> TileInfer 的 `profiles/base.py` docstring 已经在做这条区分（"哪些是模型选择、哪些是硬件事实"），
> 直接沿用即可。

**一个例子说明"布局是硬事实、不是可调参数"**（迁移时最容易被"顺手修好"的地方）：

> `worker.py:766 _init_kv_zero_meta` **主动拒绝** vLLM 的 `KVBlockZeroer`。
> 原因：zeroer 用 `kv.stride(block_dim) * element_size` 算每块字节数，而本后端的物理序是
> `(1,0,2,3,4)`、`block_dim=0`、缓冲是两个连续半区 `[K blocks][V blocks]`——那个 stride
> 恰好是**半页**。于是它会清零 K 半区、**永远不碰 V**：不报错、不崩溃，只是和引擎以为的
> 不一样。想"修"就得把 K/V 按块交错，而那正是分页内核读不了的排列。

同类"必须按硬件事实写、不能按直觉写"的地方还有三处，迁移时逐条保留原注释：

| 位置 | 事实 |
|---|---|
| `platform.py:1635` | 关掉 **7 个**只在 CUDA/XPU 下才 import 的融合 pass（多数是量化融合）——非 CUDA 上是 `NameError`，而 `--enforce-eager` 只是碰巧绕过了整个 pass manager |
| `worker.py:726` | `num_sms` 拿不到时保持 sentinel `1` 并告警，**不许猜** |
| `graphs.py:168` | 捕获结论是"patchable"，不是"已支持"；`torch.npu.CUDAGraph` **不存在**，只有 `NPUGraph` |

另外，`worker.py:208` 有一份 **`HARDWARE_CHECKLIST`（7 条）**：把"还没被硬件证实的问题"
写成了数据（每条带 `file:line` 依据与验证命令）。**这是迁移时最好的起点**——它本身就是
一份"硬件特性待办清单"，搬过去后由 `doctor` 打印。

---

## 4. 去向矩阵（文件级）

| TileInfer | 行数 | 去向 | 动作 |
|---|---|---|---|
| `vllm/platform.py` | 1954 | **Platform** | 搬 + 抽 `DeviceProfile` |
| `vllm/worker.py` | 2059 | **Platform** | 搬 |
| `vllm/graphs.py` | 929 | **Platform** | 搬 |
| `vllm/doctor.py` | 808 | **Platform** | 搬，改入口名 |
| `vllm/plugin.py` | 775 | **Platform** | 搬，`install()` 五步保留 |
| `vllm/profiles/**` | 936 | **Platform** | 搬 |
| `vllm/attn/{backend,metadata,impl,kv_cache}.py` | 2329 | **Platform** | 搬；`prefill.py` 拆出去 |
| `vllm/attn/prefill.py` | 543 | Attention | 内核调用改走公共接口 |
| `vllm/compat/patches.py` | 370 | **Platform** | 搬（README 允许的"兼容补丁"耦合点）。只有 **3 条**：`GPUModelRunner.load_model` 探测、`topk_topp_sampler.HAS_TRITON=False`、`torch.accelerator.empty_cache → torch.npu.empty_cache`，每条都带 `remove_when` |
| `vllm/ops/{install,candidates,switch}.py` | 1324 | **Platform** | 搬机制；`families/` 的表随族走 |
| `vllm/ops/families/gdn_moe.py` + `ops/{fla,moe,gdn,fused_moe_oot}.py` | 6,709 | Attention | 算子实现与绑定表（`fla/` 本身 5,023 行） |
| `vllm/models/{dsv41_batching,state_slots,req_ids}.py` | 366 | **Platform** | 批语义，随平台 |
| `vllm/models/dsv41.py`、`mmap_weights.py` | 1336 | TileInfer | 模型实现，留原地 |
| `serving/**` | 1452 | **Platform** | 搬（buckets/planner/step/phases/counters/flags） |
| `metadata.py`、`plan.py`（`tileinfer/` 根，不在 `vllm/` 内） | 1115 | Attention | 纯数据类型，作为算子层入参契约 |
| `attention/**` | 1288 | Attention | plan/run + 后端注册表 |
| `kernels/**` | 6413 | Attention | 内核 |
| `testing/**` | 743 | Attention | 数值 oracle |
| `models/**`（dsv41 实现） | 5146 | TileInfer | 留 |
| `scripts/ benchmarks/ tools/ legacy/ docs/` | — | TileInfer | 留（`hardware-checklist.md` 拆一半给平台） |
| `reference/vllm-0.23/` | — | **复制** | 原件留 TileInfer，平台仓另有副本（见下） |

**四个争议项与判据：**

1. `serving/` 归谁？→ **平台**。它是"会话语义"（bucket、plan 缓存、step guard、分相计时），
   不含数学，且 `attn/impl.py` 深度依赖它。放算子层会让算子库反向依赖会话概念。
2. `attention/api.py`（BatchAttention）归谁？→ **Attention**。README 说算子层对外只暴露
   "能力查询 / 规划 / 执行"三项，`plan/run` 正是这三项的实现。
3. `ops/install.py` 归谁？→ **拆**。重绑定"机制"（Patch/OpBinding/开关/报告）是引擎知识 → 平台；
   "哪张表绑哪些名字"是模型族数据 → 算子层（对外以纯数据清单导出）。
4. `models/dsv41.py` 归谁？→ **留 TileInfer**。它是模型实现（1,254 行前向），
   平台只需要它导出的"批语义 + 状态槽"（366 行）。

**文档也要按同一刀切**（7,744 行的知识库不能整体留在TileInfer）：

| 文档 | 去向 | 理由 |
|---|---|---|
| `hardware-checklist.md`(706) | **拆**：§1–§6、§10 → Platform；§7–§9（logprobs/GDN/MoE 对数）→ TileInfer | 前者是"平台上机点亮"，后者是"后端正确性" |
| `integration-vllm-native.md`(292) | **Platform** | 它就是平台的耦合地图 |
| `architecture.md` §6 环境段（源仓；落地为 [`hardware-facts.md`](hardware-facts.md)） | **Platform** | 工具链事实 |
| `known-issues.md`(435) | **拆**：KI-5（CANN 析构）、KI-2（入口点注册）→ Platform；其余 → TileInfer | 与"平台契约"直接相关的只有这两条 |
| `status.md` / `plan-vllm-native-backend.md` | TileInfer | 后端路线图，平台只继承其中的 M6 |

另一条硬要求：`reference/vllm-0.23/` 必须**随平台仓复制一份**。平台层的每一句
"这是 vLLM 0.23 的哪个方法、哪一行"都指向它；不复制，搬运时就只能凭记忆改注释。

---

## 5. 迁移方案

### 5.1 三条总原则

| 原则 | 含义 | 为什么 |
|---|---|---|
| **搬运不改行为**（moved, not rewritten） | 第一版逐字搬，只改 import 路径 | 行为不变才能用现有测试判"搬对了没有"；一旦同时改结构，失败无法归因 |
| **契约测试随迁** | 每个被搬的模块，其 `tests/test_*` 同批搬 | TileInfer 的测试是唯一的行为基线，测试留在原地等于没搬 |
| **双跑可回退** | 迁移期 TileInfer 保留旧路径，用环境变量切 | 单卡只有一张机器，不能"搬一半没得跑" |

### 5.2 阶段 P0–P6

| 阶段 | 目标 | 主要动作 | 产物 | 验收门（可执行） | 人日 |
|---|---|---|---|---|---|
| **P0** | 冻结契约 | 写 `docs/device-profile.md`、`docs/op-manifest.md`、`docs/versioning.md`；把 README 的 C1–C3 变成可引用的规范 | 3 份文档 + `contracts/` 目录 | 三份文档能被 TileInfer 侧的人读懂并复述；`op_manifest` 有 JSON Schema 文件 | 4 |
| **P1** | 骨架可安装 | `pyproject.toml`（两个入口点 + 版本区间）、`src/ascend_platform/` 包结构（包名待定）、空 `Platform` 子类、`assert-platform doctor` CLI 骨架、fake_vllm 夹具 | 能 `pip install -e .` 且 `python -c "import ..."` 通过 | `pytest` 有一个绿的空套件；入口点能被 vLLM 解析（可用离线假 vLLM） | 3 |
| **P2** | `DeviceProfile` 落地 | 把 §3 的 H1/H3/H7 收进一个 frozen dataclass；950PR 做第一份真实 profile；平台代码改为"只读 profile，不直接摸 torch_npu" | `profile_ascend950.py` + 契约测试 | 新增第二份假 profile（如 `ascend910b`）能让平台在**不装 NPU 的机器**上完成配置推导；`platform.py` 里不再出现设备常量 | 5 |
| **P3** | 平台语义搬运 | 搬 `platform/worker/graphs/doctor/plugin/compat`（6,895 行）+ `serving/`（1,452 行）；`install()` 五步保留 | 平台主体 **~8.3k 行** | TileInfer 的 `test_platform_contract` / `test_worker_contract` / `test_boot_rehearsal` 在新仓全绿（离线部分）；`graphs.capture_contract_ok` 的 7 个断言码逐条有测试 | 8 |
| **P4** | KV / 元数据 / attention 适配搬运 | 搬 `attn/{backend,metadata,impl,kv_cache}`（2,329 行）；`attn/metadata.py` 与 `metadata.py` 拆成"构造 vs 类型"；`prefill.py` 移出 | **~2.3k 行** | 混合 KV（2048 引擎页 / 128 内核页）在 CPU 单测下形状与对齐正确；`test_attn_contract` 绿 | 6 |
| **P5** | 算子层接口 + 名字重绑定 | 定义"能力查询/规划/执行"三函数与 `op_manifest`；`ops/install` 改为"读清单 + 启动期绑定"；把 `attn/prefill.py` 的内核直连改成接口调用 | 接口 + 适配层 | **静态检查：平台镜像里 grep 不到 `kernels`**；拔掉算子层能起服务（回退厂商算子）；清单版本不匹配时**拒绝启动** | 6 |
| **P6** | 切流与 CI | 版本区间校验接到启动路径；CI 加依赖方向检查（AST + `sys.modules`，复用 TileInfer `test_no_vllm_ascend.py` 的手法）；双跑对比脚本 | CI + 回归报告 | 单卡上"旧路径 vs 新仓路径"同 prompt logprobs 逐位一致；CI 能挡住"平台 import 内核"的提交 | 5 |

合计 **37 人日 ≈ 6 周**（单人），加 2 周上机验证与调稳，**8 周量级**。
P0/P1 可与 AscendAttention 的拆分并行；P3 与 P4 可并行（不同文件集）。

### 5.3 顺序依赖

```
P0 契约 ──┬─→ P2 DeviceProfile ──→ P3 平台语义 ──┬─→ P5 算子接口 ──→ P6 切流
          └─→ P1 骨架 ───────────→ P4 KV/元数据 ──┘
```

P5 必须等 P0（没有接口规范就只能照抄现状），
P6 必须等 P3+P4+P5（三块都搬完才有"新路径"可切）。

### 5.4 关键接口草案（P0 就要定下来）

**① `DeviceProfile`（补上 TileInfer 至今没有的东西）**

```python
@dataclass(frozen=True)
class DeviceProfile:
    name: str                          # "Ascend950PR_9579"
    vendor: str                        # "ascend"
    hbm_bytes: int                     # 131072 MiB
    compute_units: int                 # 56 (multi_processor_count)
    cube_units: int                    # 28
    vector_units: int                  # 56
    kernel_page_size: int              # 128  —— 编译期内核瓦片
    engine_page_multiple: int          # 2048 —— hybrid 引擎页
    dtypes: frozenset[torch.dtype]     # {bf16, fp16}
    layouts: frozenset[str]            # {"ND", "NZ"}
    capabilities: frozenset[str]       # {"graph_capture", "hybrid_kv", "mamba_state"}
    env_requirements: tuple[EnvReq, ...]   # ACL_OP_INIT_MODE=1 等，带失败长相
```

要点：**设备常量一条都不许散落在 `platform.py` 里**。P2 的验收就是"加一个假 profile 能让配置推导跑通"。

**② 算子清单（纯数据，算子层导出，平台只读）**

```json
{
  "schema": 1,
  "library": {"name": "ascend-attention", "version": "0.3.0",
              "requires_platform": ">=0.1,<0.2"},
  "ops": [{
    "name": "paged_decode_attention",
    "shapes": {"head_dim": [128, 256], "group": ">=2", "page": [128]},
    "dtypes": ["bf16", "fp16"],
    "kv_layouts": ["block_first"],
    "graph_capturable": true,
    "measured_ratio": 1.42,
    "bench": "benchmarks/bench_attention.py"
  }]
}
```

规则（沿用 TileInfer 已验证的纪律）：
没有 `measured_ratio` + bench 脚本的算子**不许**出现在清单里；
`graph_capturable: false` 的算子不得被绑进捕获区。

**③ 三项能力函数**

```python
query(op: str, shape: OpShape, device: DeviceProfile) -> Capability   # 能否 + 代价
plan(op: str, shape: OpShape, device: DeviceProfile) -> OpPlan       # 纯数据 + 静态形状
run(op: str, plan: OpPlan, tensors: list[torch.Tensor]) -> list[torch.Tensor]
```

约束：`run` 只接受张量与纯数据，**不接受任何引擎元数据对象**（AscendAttention README 已定）。

**④ 版本区间与启动期校验**

两仓各自在打包元数据里声明 `requires_platform` / `requires_ops`；平台在
`check_and_update_config` 里读一次，不匹配就 `raise`——**不静默运行**。这条 TileInfer 的
`platform.py` 里已有先例（不支持的配置在启动期拒绝），照抄语义即可。

对引擎侧同理：TileInfer 已经用 `plugin.SUPPORTED_VLLM_RANGE = ">=0.23,<0.24"`
（`tileinfer/vllm/plugin.py:184`）钉住了 vLLM 版本，平台仓要继承这条做法而不是发明新的。

---

## 6. 风险与对策

| # | 风险 | 触发条件 | 对策 |
|---|---|---|---|
| 1 | **契约只有一份实现，抽象无法被证伪** | 只有 950PR 一张卡 | P2 强制用"假 profile"跑配置推导；接口按"换卡要改哪几个文件"验收，不看文档写得多好 |
| 2 | **拆仓导致知识丢失**（docstring 里的 `file:line` 依据） | 搬运时"顺手精简注释" | 定规矩：P0–P5 只许改 import 行，注释与 docstring 逐字保留；`reference/vllm-0.23/` 随平台仓复制一份 |
| 3 | **把未完成的能力当成"已迁移"** | 图捕获、MTP 本来就没接完 | 在平台仓的 status 里显式写 `graph_capture: 未接通（根因：自身捕获路径未接完）`；迁移验收只判"行为等价"，不判"性能改善" |
| 4 | **平台层与算子层的共享核心归属不清** | `serving/`、`metadata.py`、`attention/api.py` 三处都可能被两边要 | §4 已给判据；P0 就把这些文件写进"归属表"并让两边签字 |
| 5 | **双仓同步开发成本** | 迁移期改一处要动两个仓 | 过渡期用 editable install + 同机双仓；P6 之后再切"发布版本依赖" |
| 6 | **vLLM 0.23 漂移** | 上游点版本新增 ABC 方法 | 沿用 TileInfer 的做法：`inventory_problems()` 从活基类反推 ABC 面，多一个方法就让测试红 |
| 7 | **测试套件本身不绿** | TileInfer 现状：整套 80 项失败（跨文件顺序污染），单文件全绿 | 迁移验收**按文件跑**，不按整套；这套问题的修复单独立项，不要混进迁移 |

---

## 7. 明确不做（防范围蔓延）

- 不搬 `kernels/`（6,413 行）——属于 AscendAttention。
- 不搬 `models/dsv41.py` 前向（1,254 行）——模型实现留在 TileInfer。
- 不搬 `benchmarks/`、`scripts/ti_trace.py`、`tools/`、`legacy/`。
- 不在迁移中做性能优化、不接图捕获、不做 MTP、不加第二块卡。
- 不"顺便"清理 TileInfer 的 80 项测试失败。
- 不引入新依赖（平台仓的依赖面应当只有 `torch` + 可选 `torch_npu` + 可选 `vllm`）。

---

## 8. 一页纸执行清单

```bash
# P1 之后每次搬运的固定动作
1. cp 源文件 → 目标仓（保留 docstring 与 file:line）
2. sed 改 import 根（tileinfer.vllm → ascend_platform）
3. cp 对应 tests/test_*.py，改 import
4. 逐文件跑：pytest tests/<file>.py      # 不跑整套（见风险 7）
5. 记录"搬了哪些、跳了哪些、为什么"到 MIGRATION_LOG.md
6. 平台仓 CI：AST 检查禁止 import kernels / 禁止出现设备常量
```

验收总门（P6 结束时必须同时成立）：

1. `import ascend_platform` 在**没有 NPU** 的机器上成功，且 `doctor` 能给出"缺什么"。
2. 平台仓静态检查：`grep -r "tileinfer.kernels\|ascend_attention.kernels" src/` 为空。
3. 拔掉算子层时服务能起（回退厂商原生算子），日志有明确归因。
4. 单卡上同 prompt 的 logprobs 与迁移前逐位一致（4 位小数口径）。
5. 新增第二份 `DeviceProfile` 只需加一个文件，不改平台代码。
