# 架构：分层、依赖方向、两条扩展通道

> 本文回答三个问题：**代码分成哪几层**、**层与层之间允许谁依赖谁**、**第三方怎么加东西而不改本仓库**。
> 每条规则都对应 `tests/test_package_layout.py` 里的一条断言——写在这里的话不是口号，是会在
> 一秒内失败的测试。

---

## 1. 为什么需要分层

搬运之前的代码是能跑的，但它的结构与它的职责无关：算子调用、设备常量、页几何、vLLM 适配
混在同一批文件里。结果是三个具体问题：

1. **无法在没有算子的机器上启动**。想在只有 torch 的笔记本上验证"页几何算得对不对"，
   必须先把 tilelang 和算子仓装齐。
2. **换一个算子库要改平台**。语言模型里"某个名字由谁实现"是策略，但代码里它是 `import`。
3. **加第二张卡要改平台**。设备常量散落在三处，没有任何一处在语义上声明"这是硬件事实"。

分层就是为了这三件事，不是为了好看。

## 2. 层与职责

| 层 | 目录 | 拥有什么 | 明确不拥有 |
|---|---|---|---|
| 契约 | `contracts/` | 版本范围、算子清单 schema、算子层端口 | 任何实现、任何设备知识 |
| 设备 | `device/` | 一张卡的冻结记录 + 注册表 + 探测 | 引擎机制、模型事实 |
| 模型事实 | `profiles/` | 头维度、页策略、混合层比例（按模型） | 设备常量、算子实现 |
| 运行时骨架 | `runtime/` | 页几何、规划器、step 记账、计数器、开关 | 任何 vLLM 类型 |
| 算子接入 | `ops/` | 名字 → 实现的绑定表、来源切换、候选集 | 算子数学 |
| 引擎 | `engine/` | vLLM 平台/worker/图/注意力/模型适配、插件入口；`engine/abi/` 放随引擎版本走的符号表与类扩展点 | 算子数学、设备常量字面量 |
| 自检 | `doctor/` | 一条命令跑完 bring-up 清单 | 修复、隐式安装 |

`engine/` 是唯一允许 `import vllm` 的层（`engine/abi/vllm_0_23.py` 目前只用字符串持有符号路径，
所以它连 import 都不需要——测试钉住"import 它不会把 vllm 拉进来"）（`tests/test_package_layout.py` 里有一份审阅过的
白名单，新增文件必须先进白名单）。`runtime/`、`device/`、`contracts/` 只依赖标准库和 torch，
所以它们在 CPU 机器上完全可测。

## 3. 依赖方向（单向）

这份表在代码里有一份可读副本：[`src/ascend_platform/_acl.py`](../src/ascend_platform/_acl.py)
（`ALLOWED_IMPORTS` / `ENGINE_FREE`）。分层规则以它为准，文档、测试、`doctor` 都读这一份。
算子包的根不在这里：`OPERATOR_ROOTS` 归 `contracts/ports.py`，因为它的**顺序就是优先级**，而消费
顺序的是端口的 `_candidates()`。

```
doctor ──► engine ──► ops ──► contracts
             │        │        ▲
             ├────────┴────────┘
             ▼
         profiles ──► runtime ──► device
```

规则：

- `contracts` 与 `device` 是叶子：只能依赖标准库（`contracts/manifest.py` 例外，它依赖
  `contracts/version.py`）。
- `runtime` 不依赖引擎，也不依赖 vLLM。
- `ops` 允许依赖 `engine.compat`（它要报告补丁状态），仅此一处跨层。
- `engine` 可以使用下面所有层。
- `doctor` 可以使用所有层（它要检查所有层）。

**为什么 `ops` 能依赖 `engine.compat`**：绑定表需要说"这个替换生效了吗"，而补丁状态由
`engine/compat` 维护。把状态提升到 `runtime` 会让 `runtime` 知道 vLLM 补丁，代价更大。

## 4. 算子层端口：唯一合法的跨仓调用

平台**不 import** 任何算子包（`tileinfer`、`ascend_attention`），只通过
`ascend_platform.contracts.ports` 访问。这个模块有三个性质：

1. **惰性**：解析发生在第一次调用，不在 import 时。所以 `import ascend_platform` 在没有任何
   算子包的机器上成功——`tests/test_ports.py::test_the_package_imports_with_no_operator_layer_and_no_vllm`
   用一个干净子进程钉住这一点。
2. **可换**：`_LAYOUTS` 记录每种算子包的内部布局；`register_operator_layout()` 允许第三方
   注册自己的布局，不需要改平台。
3. **失败可读**：没有算子层时抛 `OperatorLayerUnavailable`，消息里带环境变量名、试过的候选、
   每个候选的失败原因。

端口的表面是**函数而不是类别名**：`BatchAttention(...)`、`RaggedMetadata(...)` 都是转发函数。
原因是别名会在 import 时就解析，正是上面第 1 条要避免的。

### 4.4 拔掉算子层会发生什么

"可回退"在昇腾上容易被读成"厂商算子接管"——**不是**。Qwen3.6 的 GDN/路由路径在 vLLM 侧没有 NPU 实现，
所以真正的行为分三层，每一层都是确定的、有测试的：

| 阶段 | 没有算子层时 | 依据 |
| --- | --- | --- |
| 启动 | **照常启动**。`plugin.install()` 返回 True，ops 步骤捕获 `OperatorLayerUnavailable` 并记进 `report.layer_error` | `plugin.py::_install_ops`、`ops/install.py::install_ops`；`test_boot_rehearsal.py::test_a_missing_operator_layer_degrades_to_a_report_instead_of_a_failed_boot` |
| 自检 | `ascend-platform doctor` 的 ops 段写明缺什么、去哪配（`ASCEND_PLATFORM_OPS`），退出码非 0 | `doctor/cli.py`、`docs/notes/p5-gate.md` |
| 第一次前向 | 在**调用点**抛出 `OperatorLayerUnavailable`，消息含候选根与最后一个 import 错误；不会静默算错 | `contracts/ports.py`、`engine/attention/{impl,prefill}.py` |

也就是说：平台对算子层的依赖是**软**的（import 阶段不炸），但对模型的正确性是**硬**的（跑起来必须有实现）。
把这两件事混为一谈，就会写出"看起来能回退、实际算错"的代码。

## 5. 三条扩展通道

### 通道 A：换/加一个算子实现（不改平台代码）

1. 算子仓写出实现，并提供一份 `manifest.json`（见 [`op-manifest.md`](op-manifest.md)）。
2. 在平台侧登记一条绑定：`ops/install.py` 的绑定表（或自己的 family 文件，
   `ops/families/` 下一个文件就是一个 family）。
3. 运行期按名字查表。切换由 `TILEINFER_OPS`（家族级，逐字沿用旧名，见 §7）控制，**每次调用**都
   重新判定，所以关掉一个算子不需要重启。

平台的承诺是"按形状算得出 + 可进图"，不是"必须是某个库"。

### 通道 B：加一张卡（不改平台代码）

新增 `device/<card>.py`，构造一个 `DeviceProfile` 并 `register_device_profile()`，再在
`device/__init__.py` 里加一行 `from . import <card>`（副作用就是注册）。探测按 `name_patterns`
匹配，平台其余代码不动。

那行 import 是刻意显式的、不是自动发现：依赖图保持静态，`_acl` 与 import 白名单的测试才读得到。
`tests/test_device_profile.py` 会检查"这个目录里的每个模块都被 `__init__` import 了、且都注册了
profile"，所以忘了那行会红，而不是变成一个"设备不认识"的运行期意外。见
[`device-profile.md`](device-profile.md)。

### 通道 C：换一个量化格式的模型（一个子类 + 一个环境变量）

同一个平台类不允许既"只吃 bf16/fp16"又"收 fp8/fp4"——那会把两条模型的契约混成一个 if。所以
第二个量化口径是**子类**：`engine/platform_dsv41.py` 只覆盖两个 classmethod

- `verify_quantization`：接受该 checkpoint 的格式（fp8-block-32 dense + fp4 专家），其余照旧拒绝；
- `support_hybrid_kv_cache`：返回 `False`（这个模型没有 GDN 层，vLLM 的 hybrid 协调器会断言
  "至少两个 attention group"）。

选择发生在 `engine/platform.py::register` 里，靠 `VLLM_DSV41=1`（`1/true/yes/on`，大小写不敏感）；
没设就返回 `TileInferPlatform`，Qwen3.6 那一支的行为一个字节都不变。注意顺序：**先探测设备**，
没有 NPU 时无论变量怎么设都返回 `None`（`tests/test_platform_contract.py` 钉住这两点）。

模型侧还有两件事不在本仓库：`DeepseekV41ForCausalLM` 这个 arch 由模型包通过
`engine/model_providers.py` 注册（vLLM 0.23 自己只认 `DeepseekV4ForCausalLM`），而 `dsv41` 的
`ModelProfile` 在本仓库（`profiles/dsv41.py`）。

### 三份"Qwen3.6 的 config"分别是什么（容易混）

上机时会在三个地方看到 Qwen3.6 的 config，它们不是同一件东西，谁都不该删：

| 位置 | 是什么 | 谁拥有 | 进不进本仓 |
| --- | --- | --- | --- |
| `/home/models/Qwen3.6-35B-A3B/config.json` | checkpoint 自己的 HF config（`text_config` 里 40 层 = 30 GDN + 10 full attention、16 头、2 KV 头、head_dim 256） | 模型仓 | 不进，也不该进 |
| `profiles/qwen3_6.py` | **模型事实**：arch 拼写、混合类型、页大小策略、KV 形状、头维度集合、op family、拒绝理由 | 本仓（`profiles/` 层） | 进——它是代码，不是数据副本 |
| `runtime/test-model-qwen36/config.json` | 测试用的**合成** config（无权重、无 tokenizer），形状与上表第一行一致，供 `--plan-only` 排演 | `tests/test_boot_rehearsal.py` | 不进：`/runtime/` 已 gitignore，且测试跑完会删掉自己建的那份 |

为什么需要第二份：平台的钩子（`determine_block_size`、`support_hybrid_kv_cache`、
`get_attn_backend`）在**引擎与权重存在之前**就要回答"这个模型长什么样"，而且 CPU box 上也得能回答。
所以事实必须是本仓的代码，不能靠读 checkpoint 的 `config.json`。

为什么需要第三份：套件不能依赖 67 GB 权重。合成的 config 只保留形状，形状对了，页算术与
profile 选择就能测。

## 6. 层不拥有什么（反面清单）

- 平台**不拥有模型实现**：`DeepseekV41ForCausalLM` 之类留在模型仓。平台只提供 hook
  （`engine/model_providers.py`），由模型包自己注册。
- 平台**不拥有部署配方**：启动脚本、端口、模型路径属于使用方。`scripts/` 不在本仓库。
- 平台**不拥有算子数学**：连兜底实现都不放。需要兜底时，兜底在算子仓里，用 manifest 声明。
- 平台**不拥有性能数字**：只消费别人测出来的 `measured_ratio`。

## 7. 强制手段

| 规则 | 测试 |
|---|---|
| 层依赖方向 | `test_layer_acl.py::test_the_dependency_direction_holds`（数据来自 `_acl.py`） |
| 只有端口能 import 算子包 | `test_layer_acl.py::test_only_the_port_imports_an_operator_package` |
| 包根 import 不拉重物 | `test_the_package_root_imports_nothing_heavy` |
| 无 vLLM 的层在干净子进程里真的能 import | `test_layer_acl.py::test_an_engine_free_layer_imports_in_a_clean_interpreter` |
| 不操作 `sys.path` | `test_no_module_manipulates_sys_path` |
| 每个包目录有 `__init__.py` 和真文档字符串 | `test_every_package_directory_has_an_init` / `test_each_layer_has_a_docstring_*` |
| vLLM import 白名单 | `test_the_operator_layer_is_never_a_hard_dependency_of_the_engine_import_path` |
| `os.environ` 的使用处必须 `import os` | `test_import_hygiene.py`（49 个模块级检查） |
| 两个 entry point 可解析 | `test_the_two_vllm_entry_points_are_declared_and_resolvable` |

## 8. 刻意不重命名的三样东西

迁移要给读者一个明确的"哪些名字是旧的、为什么"，否则每个看到 `TILEINFER_OPS` 的人都要重新判断一遍。

| 名字 | 为什么不改 | 依据 |
| --- | --- | --- |
| `TILEINFER_OPS` / `TILEINFER_OPS_CANDIDATES` / `TILEINFER_FLAG_DIR` 等环境变量 | 上机验收要拿同一组 shell 脚本、同一份 env 对照迁移前后的行为；改名的收益是美观，代价是验收表里多一个变量 | `docs/hardware-facts.md`、`MIGRATION_LOG.md` §6 |
| `__tileinfer_baseline__` / `__tileinfer_impl__` / `__tileinfer_candidates__` 标记属性 | 这是**跨仓协议**：算子仓与探针读它们判断"这个名字被谁替换了"，改名会让旧探针静默失效 | `ops/install.py`、`ops/candidates.py` |
| `TileInferPlatform` / `TileInferAttentionBackend` / `TileInferWorker` 类名 | 它们是 `test_migration_parity.py` 的键（`类名.方法名` 对照源仓库），改名会让 201 个方法"查不到来源"，除非给审计加一张映射表；功能上不需要改，`worker_cls` 已指向 `ascend_platform.engine.worker.NPUWorker` | `MIGRATION_LOG.md` §8 |

**源码引用保留原样.** 代码与文档里出现的 `tileinfer/vllm/...`、`tileinfer/kernels/...`、
`tileinfer/testing/...` 是**出处**（"这段来自哪里"），不是本仓路径；本仓自己的文件一律写成
`src/ascend_platform/...` 或相对模块路径。引用分四种写法，`tests/test_citation_anchors.py` 逐条核对：

| 被引用的是 | 写法 | 怎么核对 |
| --- | --- | --- |
| 已入库的 vLLM 摘录 | `v1/worker/gpu_worker.py:219`（唯一化到能定位） | `reference/vllm-0.23/` 里那一行存在且非空 |
| 源仓库 | `tileinfer/vllm/platform.py:1116-1118` | 源仓库里那一行存在（源仓库不在时跳过） |
| 未入库的依赖（torch） | `torch/cuda/memory.py::mem_get_info`——**只写符号，不写行号** | 装着的那份依赖里有这个符号 |
| 本仓 | `src/ascend_platform/engine/worker.py:123` | 本仓那一行存在 |

行号会随 wheel 变，符号不会：这是 torch 那类引用必须写成 `::symbol` 的原因。短路径（`registry.py:99`）
只有在**唯一**时才允许，否则读者有一半概率查错文件。用户照着做的**指示**（报错里的"去哪个目录注册"）必须指本仓，
`profiles/base.py` 的 `_no_match_error` 就是这么改的。

`ASCEND_PLATFORM_OPS` 是**另一个**东西：它指定算子层的 import 根（`contracts/ports.py`），不是开关。
两个名字长得像、含义无关——这也是上表存在的原因。
