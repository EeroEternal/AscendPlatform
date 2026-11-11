# AscendPlatform

面向昇腾的 **out-of-tree vLLM 平台插件**：提供设备与执行语义，不提供算子数学。

## 定位：这一层负责什么

**一句话**：把"一张设备的执行语义"（设备事实、KV 布局、批元数据、能否进图、worker、引擎契约）
翻译成某个推理引擎能用的形式；**算子数学一行都不在本仓**。

两个仓库的边界用**两条可机器检查的判据**划，而不是靠读者理解：

| 判据 | 本仓（平台） | 算子仓 |
|---|---|---|
| 可以出现引擎的名字 | ✅ vLLM 的符号名、版本、补丁点、entry point | ❌ 出现即越界 |
| 可以出现内核与算子实现 | ❌ `tilelang.jit`、内核函数体、算子数值实现 | ✅ 这正是它的内容 |
| 可以出现设备事实 | ✅ `DeviceProfile`（引擎侧事实：页大小、显存） | ✅ 只读镜像（内核侧事实：tile 大小） |
| 决定"用哪个算子的哪个实现" | ❌ 这是算子库内部分派 | ✅ `spec/plan/run` 与候选实现 |

本仓的"不含算子数学"有测试钉住（[`tests/test_no_operator_math.py`](tests/test_no_operator_math.py)），
它查的是内核文件、内核 DSL、算子定义与端口转发，不靠人读文档。

## 仓库功能

- **平台契约**：设备探测、显存画像、能力协商（不支持的能力**启动期显式拒绝**）
- **KV Cache**：布局与形状（block-first、混合注意力页）、跨层共享
- **批元数据**：block table / 序列长度 → 纯数据描述
- **注意力后端适配**：把元数据翻译成算子调用（薄适配层，不含算子实现）
- **跨来源策略**：决定某个算子在"算子仓 / 厂商包 / 上游实现"之间选谁；**启动期一次绑定**，运行期只做免费分类
- **图捕获**：CUDA 语义到 NPU 原语的桥接与捕获可行性判定
- **worker**：设备初始化、可用显存、KV 初始化、启动预热
- **自检与性能观测**：环境检查；阶段 / 算子 / 每次发射 / 空转 / 异常的报告
- **模型事实**：按模型集中配置（头维度、页策略、混合层比例）

## 解耦设计

- **唯一合法入口**：平台插件与通用插件只由本仓库声明（平台插件是"第一个非空即胜"，只能有一个声明者）
- **平台不假设算子来自哪个库**：只要满足"按形状算得出 + 可进图"即可
- **算子层是可选的启动依赖、不是可选的计算依赖**：缺算子层时**能启动**（插件返回 True，`doctor`
  说明缺什么），但第一次前向会在调用点抛 `OperatorLayerUnavailable`——Qwen3.6 的 GDN/路由路径在
  vLLM 侧没有 NPU 实现，**没有厂商算子可以回退**。细节见 [`architecture.md`](docs/architecture.md) §4.4
- **选择时机**：开发期离线对比、启动期绑定、运行期只做分类；**禁止推理期现场测速**
- 引擎耦合只允许出现在两处：**平台覆盖** 与 **兼容补丁**

## 算子从哪来：provider 契约

算子不由本仓提供，也不由本仓去"猜"——由算子实现包**报名**。契约只有三件东西：

| 件 | 是什么 | 归谁 |
|---|---|---|
| 语义算子名 | 按"算什么"命名（`gdn.chunk_gated_delta_rule`、`moe.experts`），不按引擎符号 | 本仓定名词表；算子仓声明自己支持哪些 |
| 纯数据描述符 | 形状 / dtype / 布局 / 能否进图 / 实测比值 / 版本区间，见 [`op-manifest.md`](docs/op-manifest.md) | 算子仓发布，本仓启动期校验 |
| 发现入口 | 算子仓在 `pyproject.toml` 里声明一个 provider entry point（组名以本仓包名为前缀，常量放在 `contracts/provider.py`），值返回描述符并可按名字 `load()` | 算子仓声明，本仓读取 |

**引擎符号到语义算子的映射归本仓**（vLLM 的 `topk_softmax`、`ChunkGatedDeltaRule` 这类名字是
引擎知识，写在 `engine/abi/` 下并随引擎版本维护）。算子仓从头到尾不知道 vLLM 存在——这也是
"换引擎只改本仓"的含义。

**逐项生效，不搞全有全无**：每个算子有四种状态——`bound` / `unbound(原因)` / `declined(原因)` /
`disabled(开关)`；缺一个只影响它自己。**强制/可选分开**（`ops/requirements.py`）：可选算子缺失
（如 `gdn.chunk_local_cumsum`，我们刻意拒绝）照常服务；**强制**算子缺失时 `install_ops()` 记
`report.servable = False`，日志打 `CANNOT SERVE`，`doctor` 那一条变 FAIL，并且——**只要你给了算子层**——
`plugin.install()` 抛 `OperatorRequirementsUnmet` 直接拒绝启动（没给算子层时按"诊断模式"启动，让 `doctor`
能报缺什么）。这一条**不受别的异常处理兜底**：其余失败都退化成"vLLM 自己的实现还在"，而强制算子没有
可退化的实现——继续跑只会在第一次前向炸在一个不相关的堆栈里。

**现状（别按文档假定已实现）**：两块都通了，但都不是终态：

- **provider 发现已实现**：`ASCEND_PLATFORM_OPS` 指向的包（或端口解析出的算子层）若暴露
  `describe()`，启动期就会读它，校验 `requires_platform`，逐项 `load()`；其次才是 entry point 组。
  描述符是数据，坏一个只影响那一个算子（`contracts/provider.py`）。
- **语义名词表已实现**（`contracts/vocabulary.py`）：`moe.routing_topk_softmax` 这类名字是封闭集合，
  每个名字声明自己是 `compute` 还是 `class_factory`——**类别不符会在绑定期被拒**，而不是在模型
  加载时炸。
- **引擎 ABI 已独立成层**（`engine/abi/vllm_0_23.py`）：vLLM 符号表 + MoE 方法类工厂；MoE 那个槽由
  **引擎层注入**（`install_ops(engine_factories=...)`），算子层不 import 引擎。
- **还没做完的**：`_LAYOUTS` 仍是回落路径（语义名 → 引擎符号的替换只完成 12 个，`engine.moe_method`
  已走注入）；`unquantized_fused_moe` 之外没搬的 ABI 细节见
  [`docs/notes/repo-boundary.md`](docs/notes/repo-boundary.md) §4。

实测（真 vLLM 0.23 符号 + 拆分后的算子仓）：**13 个必需算子 → 11 bound / 1 declined / 1 off by switch**。

用户自带算子走同一条路：任何实现了上面三件的包都可以被 `ASCEND_PLATFORM_OPS=<name>` 选中，
本仓不区分"官方算子仓"与"用户算子包"。对接有两种形式：包根暴露 `describe()` 返回纯数据表
（声明式），或只暴露 `resolve(semantic) -> callable | None`（协议式，一个函数、不用抄词表）；
也可以在 `pyproject.toml` 里注册 entry point 组 `ascend_platform.providers`，装完即生效。
两种形式都不需要 import 本仓，也不需要认识 vLLM。自动发现只认**一个**根（`ascend_attention`），因为支持的
依赖方向是 `vllm` + 本仓 + `AscendAttention`；别的包（包括迁移源 TileInfer）都必须显式指定。

vLLM 的 `_custom_ops` 里有几个符号是**写进调用方预分配缓冲区**的（`topk_softmax`、
`moe_align_block_size`），而算子仓发布的是函数式实现。这种"调用约定"差异由本仓的
`engine/abi/vllm_0_23.py::inplace_adapters()` 抹平——它是引擎知识，不是算子数学，所以由平台承担，
算子仓继续发函数式算子。

## 支持的组合

| 组合 | 状态 |
|---|---|
| 本仓平台 + AscendAttention | ⚠️ 绑定层已通（13 个必需算子 11 bound / 1 declined / 1 开关关闭），MoE 的 in-place 调用约定已适配并在真机协议下验证；**端到端推理未验证**，欠什么见 [`docs/notes/qwen36-e2e-gaps.md`](docs/notes/qwen36-e2e-gaps.md) |
| 本仓平台 + 用户自研算子包 | ✅ 契约可用：包根暴露 `describe()` 即可被选中；⚠️ Qwen3.6 的 GDN 部分目前只能由平台内置布局表绑定，因为 AscendAttention 尚未声明那 7 个名字（同上文档 §1） |
| 本仓平台 + TileInfer（迁移源，**仅验证用**） | ✅ 曾在单卡跑通：16 token 的 logprobs 与源仓逐位一致（`MIGRATION_LOG.md` §6）。**不是依赖**：它不在自动发现里，只能 `ASCEND_PLATFORM_OPS=tileinfer` 显式指定，用来验证"用户自带算子包"这条路 |
| 本仓平台 + 无算子层 | ⚠️ 能启动、能自检、能报缺什么；**第一次前向失败**（无厂商回退） |
| 厂商平台 + 厂商算子（`vllm_ascend`） | 基线，用于对照；与本仓**不可同装**（vLLM 只允许一个平台插件） |

## 非目标

算子数学（含 **MoE 路由 / 分组 / 专家计算**、MLA、量化算子）、搬入厂商 kernel、TP/EP、
完整投机解码（先只做多 token 预测所需的最小批语义）。

上一条容易读错，写清楚：平台**不做**这些算子，只**绑定**它们。把"MoE 计算属平台层"写进算子仓
README 是上一轮的真实错误，代价见 [`docs/notes/repo-boundary.md`](docs/notes/repo-boundary.md)。

## 状态

昇腾单卡：平台、注意力后端、混合 KV、图捕获可行性判定、性能观测已成型，**CPU 上全绿**（两个环境，
计数见 [`MIGRATION_LOG.md`](MIGRATION_LOG.md) §6）。图捕获**只搬了判定逻辑**（`engine/graphs.py`），真机捕获不在本期范围
（[`MIGRATION_LOG.md`](MIGRATION_LOG.md) §9）。第二块卡：接口预留（`DeviceProfile` 一行注册），未实现。

本仓**不含**：算子实现、模型前向、benchmarks、tools、legacy。这条边界有测试钉住——
[`tests/test_no_operator_math.py`](tests/test_no_operator_math.py) 静态检查"没有内核源文件、没有内核 DSL、
没有算子实现定义、端口只转发"，并对着算子仓核对哪些子树没被复制过来。

## 仓库结构

```
src/ascend_platform/
  contracts/   契约：算子层端口（惰性）、版本范围、算子清单 schema、provider 描述符、语义名词表
  device/      设备画像：一张卡一条冻结记录 + 注册表 + 探测
  profiles/    模型事实：头维度、页策略、混合层比例
  runtime/     运行时骨架：页几何、规划器、步进记账、计数器
  ops/         算子接入：名字 → 实现的绑定表、来源切换、候选集
  engine/      引擎：vLLM 平台/worker/图/注意力/模型适配、两个插件入口
    abi/       引擎 ABI：vLLM 0.23 的符号表与类扩展点（`moe_method_factory`）
  doctor/      bring-up 自检（`ascend-platform`）
tests/         契约测试 + 架构棘轮测试
docs/          设计文档、硬件事实、迁移方案与日志
reference/     vLLM 0.23 源码副本（供 line-cited 契约测试核对）
```

设计说明：[`architecture.md`](docs/architecture.md)（分层与依赖方向）、
[`device-profile.md`](docs/device-profile.md)（加第二张卡）、
[`op-manifest.md`](docs/op-manifest.md)（算子仓怎么说话）、
[`versioning.md`](docs/versioning.md)（版本契约）、
[`env-vars.md`](docs/env-vars.md)（上机时会碰到的每个环境变量）；
上机手册：[`hardware-checklist.md`](docs/hardware-checklist.md)；
迁移进展与验收状态：[`MIGRATION_LOG.md`](MIGRATION_LOG.md)。

## 快速开始

```bash
pip install -e .            # 平台本体只需要 torch
pip install -e ".[vllm]"    # 需要跑引擎适配时
ascend-platform              # bring-up 自检（PASS/FAIL/SKIP + 修复提示；--json 可机读）
ascend-platform --help       # 用法

python -m pytest tests                    # 装了真 vLLM 的环境
python -m pytest tests/test_worker_contract.py tests/test_boot_rehearsal.py   # 用 fake vLLM 的契约
```

> 同一环境里不要同时装两个 out-of-tree 平台插件（vLLM 会拒绝启动）。迁移期两个仓库并存时，
> `tests/conftest.py` 会用 `VLLM_PLUGINS=ascend_platform` 固定这一个。
