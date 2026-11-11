# 迁移日志（P0–P4 已完成，P5 待硬件窗口）

> 基线：源仓库 TileInfer `339969a`。目标仓库：本仓库，包名 `ascend_platform`，版本 `0.1.0.dev0`。
> 方案与阶段划分见 [`docs/migration-from-tileinfer.md`](docs/migration-from-tileinfer.md)；
> 事实清单见 [`docs/hardware-facts.md`](docs/hardware-facts.md)。
>
> 本文只记**做了什么、改了什么、还差什么**。设计理由在
> [`architecture.md`](docs/architecture.md)、[`device-profile.md`](docs/device-profile.md)、
> [`op-manifest.md`](docs/op-manifest.md)、[`versioning.md`](docs/versioning.md)。

---

## 1. 落地形态

```
src/ascend_platform/
  __init__.py          版本与层次地图（import 不拉任何重物）
  _acl.py              目录 ACL 的唯一来源：谁能 import 谁、哪些层必须无 vLLM
  _version.py          0.1.0.dev0
  utils.py             无依赖的小工具
  contracts/           ports.py（算子层端口）version.py（版本范围）manifest.py（算子清单）
  device/              profile.py（DeviceProfile/注册表/探测）ascend950.py（参考卡）
  profiles/            base.py qwen3_6.py dsv41.py（模型事实）
  runtime/             buckets planner step phases counters flags（页几何与步进骨架）
  ops/                 install candidates switch families/gdn_moe（名字→实现的绑定表）
  engine/              platform platform_dsv41 worker graphs plugin model_providers
                       compat/patches attention/{backend,metadata,impl,kv_cache,prefill}
                       models/{batching,state_slots,req_ids}
  doctor/cli.py        bring-up 自检
tests/                 34 个文件：19 个搬迁套件 + 13 个新套件 + conftest/fake_vllm
reference/vllm-0.23/   源码副本，供 line-cited 的契约测试核对
docs/                  hardware-checklist hardware-facts + 4 篇设计文档 + 迁移方案
```

**定位自检**：`grep -r "from tileinfer\|import tileinfer" src/` 为空
（`tests/test_layer_acl.py::test_only_the_port_imports_an_operator_package`
把它变成断言）。算子层只通过 `contracts/ports.py` 访问，且是**第一次调用**才解析。

## 2. 搬运清单

| 源 | 目标 | 处理 |
|---|---|---|
| `tileinfer/vllm/platform.py` | `engine/platform.py` | 原样搬运；`get_name()` 仍为 `CUSTOM` |
| `tileinfer/vllm/platform_dsv41.py` | `engine/platform_dsv41.py` | 原样 |
| `tileinfer/vllm/worker.py` | `engine/worker.py` | 原样；`graphs` 引用改为包内导入 |
| `tileinfer/vllm/graphs.py` | `engine/graphs.py` | 原样 |
| `tileinfer/vllm/plugin.py` | `engine/plugin.py` | 模型注册改为 provider hook |
| `tileinfer/vllm/attn/*` | `engine/attention/*` | `BatchAttention`/`plan`/kernel 改为端口调用 |
| `tileinfer/vllm/models/dsv41_*` | `engine/models/{batching,state_slots,req_ids}.py` | 原样（模型实现留在源仓库） |
| `tileinfer/vllm/compat/patches.py` | `engine/compat/patches.py` | 原样 |
| `tileinfer/vllm/profiles/*` | `profiles/*` | 原样 |
| `tileinfer/vllm/ops/{install,candidates,switch}.py` | `ops/*` | 算子实现改为端口调用 |
| `tileinfer/vllm/ops/families/*` | `ops/families/gdn_moe.py` | 先搬族表（见 §3 未搬项） |
| `tileinfer/serving/*` | `runtime/*` | 原样 |
| `tileinfer/vllm/doctor.py` | `doctor/cli.py` | CLI 身份改名，入口脚本名随之变化 |
| `tileinfer/utils.py` | `utils.py` | 原样 |

**随之迁移的测试**（19 个文件，逐文件绿）：`test_platform_contract`、`test_worker_contract`、
`test_attn_contract`、`test_boot_rehearsal`、`test_doctor`、`test_kv_cache_write`、
`test_serving_buckets`→`test_runtime_buckets`、`test_serving_flags`→`test_runtime_flags`、
`test_serving_step`→`test_runtime_step`、`test_plan_cache`、`test_model_profiles`、
`test_dsv41_batching`→`test_models_batching`、`test_dsv41_req_ids`→`test_models_req_ids`、
`test_dsv41_state_slots`→`test_models_state_slots`、`test_patch_registry`、`test_import_hygiene`、
`test_op_coverage_note`、`test_vllm_prefill_path`、`test_no_vllm_ascend`、外加 `conftest`/`fake_vllm`。

**新写的模块**（源仓库没有对应物）：`contracts/{ports,version,manifest}.py`、
`device/{profile,ascend950}.py`、`engine/model_providers.py`、`_version.py`、
以及各层 `__init__.py`（带层次地图）。

**类名保持 `TileInfer*` 不变**：这是刻意的。搬运的第一原则是"行为不变"，改类名会让 diff
不可读，也会让 `reference/` 里的引用对不上。改名是一次独立的、机械的、可验证的提交，
放在测试全绿之后。

## 2.1 完整性审计：定义级 404/415 + 模块级 511/521 + 装饰器 456/456 + 调用级 1045

"一行不落地搬过来"是可证伪的说法，所以它被写成了三层测试
（[`tests/test_migration_parity.py`](tests/test_migration_parity.py)；源仓库不在时整文件跳过）。
三层是因为"漏搬"有三种形状，而它们需要三种不同的比对：

| 层 | 比的是什么 | 漏了会怎样 |
| --- | --- | --- |
| **定义级**（410 个函数/方法） | 函数体归一化成 token 流 | 逻辑被改写：改了分支、少了边界判断、拿错了变量 |
| **模块级**（513 个顶层项） | 类（含**类字段**与注解）、模块常量（含字面值）、模块级 `if` 块 | 少一个 dataclass 字段、常量值变了、`TYPE_CHECKING` 分支丢了 |
| **装饰器**（451 个定义） | `@staticmethod` / `@dataclass(frozen=True)` / `@cache` 之类的**列表本身** | 方法悄悄变成实例方法、可哈希性没了、缓存没了 |
| **调用级**（960 处调用，按文件求并集） | 文件里**还调不调**那些函数（源码出现过的调名，目标文件里必须仍出现） | 这一层专门补前面三层的**盲区**：一旦某个定义被登记进 `EXPECTED_DIFFERENCES`，它的函数体就**不再比对**，之后再改它没人看得见 |

四层缺一不可：函数体比对看不见类字段，类字段比对看不见装饰器，而这三样里任何一样丢失都**不会让
任何测试变红**——它只会让某条路径在真机上第一次被走到时表现不同。

第四层是给**审计自己**打的补丁。前三层的豁免机制是"登记一个函数名，它的差异就不再比对"，这等于
给那个函数开了一张永久免检证：本轮把 `install_ops` 的版本门放在族解析之前时，它的函数体变了，
而审计一声不响——因为上一轮已经把它登记过了。所以加了这条更粗但更耐用的不变式：**文件级调用
集合不许缩小**。代码可以在函数之间移动（本轮 `_resolve_families` 就是从 `install_ops` 里提出来的），
但不能消失。1045 处调用里只有 2 处被登记为例外，且两处都是**搬走**而非删除：`register_config` 改名成了
`register_configs()`（provider 机制），`_pack_indices` 从 `prefill.py` 搬进了算子端口——正是交付项 5
要求的那件事。

**方法.** 对 31 对文件，取每个顶层函数与每个方法，把函数体归一化成 token 流——保留节点类型、
标识符、参数名、数值常量，**丢掉字符串常量**（docstring、日志、报错文案允许改）并**把 import 目标
归一成 `<LAYER>`**（包名整个变了）。两条流相等 ⇔ 语句形状、变量名、调用形状一字未动。

**结果.**

| 项 | 数值 |
| --- | --- |
| 对比的文件对 | 31 |
| **定义级**：对比的定义（顶层函数 + 方法） | 415 |
| 其中逐 token 完全相同 | **404** |
| 其中登记了理由的差异 | 11（4 处解耦 + 3 处"多说了为什么" + 2 处"搬动弄坏的路径" + 1 处补文档 + 1 处"让横幅真的打出来"，逐条列在 `EXPECTED_DIFFERENCES`） |
| **模块级**：对比的顶层项 | 521 |
| 其中完全相同 | **511** |
| 其中登记了理由的取值差异 | 10（7 个类路径/入口点字符串 + 1 个横幅 + 1 个不予映射的名单 + 1 个字段，逐条列在 `EXPECTED_MODULE_LEVEL`） |
| **装饰器**：对比的定义 | 456 |
| 其中不同的 | **0** |
| 源文件里缺失的定义 | **0** |
| 源文件里缺失的顶层项 | **0** |
| 被裁掉的行数 | **0**（每个定义在目标里的行数 ≥ 源文件） |

10 处差异各自是什么（4 解耦：`PrefillPath.run`、`_reference_experts`、`bind`、`_install_models`；
3 处多说一句为什么：`install_ops` 的 `layer_error` + 版本门、`OpsReport.summary`、`doctor._run_check`
的第三个分支；2 处搬动弄坏的路径：`doctor._repo_root` 的深度、`_check_runner_source` 跟着改；
1 处补文档：`families.family_names`）：

| 定义 | 差异 | 为什么必须改 |
| --- | --- | --- |
| `plugin.py::_install_models` | 硬编码的 `("tileinfer.vllm.models.dsv41", "register_models")` → `model_providers()` 表 | 这是新增的扩展点（第二份模型族 = 一行表项），不是行为变化：默认表里就是同一个模块同一个函数 |
| `tileinfer/vllm/attn/prefill.py::PrefillPath.run` | 两处 `from tileinfer.kernels...` → 经端口取 `forward_prefill` / `pack_indices` / `RaggedMetadata` | 目标 §5：内核直连必须消失 |
| `gdn_moe.py::_reference_experts` | `reference_moe` / `_check_unquantized` 经端口解析 | 同上 |
| `gdn_moe.py::bind` | 8 个算子 import → `ops_module(...)` 查表 | 绑定表只**命名**实现，不 import 实现 |
| `ops/install.py::{install_ops, OpsReport.summary}` | 算子层缺失时把原因写进 `report.layer_error` 并进一行摘要 | 造这个改动的是一个真缺陷：没有算子层时摘要只说"0 bound"，读起来像"没有可绑定的"，而真实原因是"实现根本没装"；绑定逻辑本身未动 |
| `doctor.py::{_repo_root, _check_runner_source}` | `parents[2]` → 向上找 `pyproject.toml`；找不到时返回 `None` 并如实说"installed copy" | **搬动打坏了路径**：源仓 `tileinfer/vllm/doctor.py` 的 `parents[2]` 是仓库根，搬到 `src/ascend_platform/doctor/cli.py` 后成了 `src/`，于是 step 4 永远在找 `src/reference/`、永远 SKIP，还声称"这不是源码检出"——一个永远不会通过、也永远不会报警的检查 |

9 处模块级差异里最值得看的是那 7 个**类路径字符串**（`WORKER_CLASS_PATH`、`PLUGIN_ENTRY_POINT` 等）：
它们是 vLLM 用来 import 我们类的字符串。改名后漏改一个，后果不是"日志不好看"，而是启动时
`ModuleNotFoundError` 或者更糟——静默加载到**源仓的**那份实现。定义级比对永远看不见这类字符串
（字符串在 token 流里被丢掉了），所以必须由模块级比对来看：这类"必须改但不能乱改"的字符串，
正是两层审计的分界线。它们指向的对象另有 [`tests/test_wiring_strings.py`](tests/test_wiring_strings.py)
负责核对——审计只说"这个字符串变了"，那条测试说"变后的字符串确实指向一个存在的类"。

这条测试是个棘轮：以后谁再动一个搬过来的函数体、类字段或装饰器，它会指名道姓地报出第一个不同的
地方，除非那处改动被写进 `EXPECTED_DIFFERENCES` / `EXPECTED_MODULE_LEVEL` 并给出理由。

## 2.2 搬运本身打坏的东西：路径、命令与指示（6 处）

比"漏搬一个函数"更隐蔽的一类问题：代码搬过来了，但它对**自己在哪里**的假设没跟着改。
这类错误不会让测试变红（很多时候表现为 SKIP 或一句气话般的报错），只会让上机的人照着不存在的
文件跑。逐条都找到了、修了，并各留了一条测试。

| 症状 | 根因 | 修法 | 钉住它的测试 |
| --- | --- | --- | --- |
| `doctor` 的 step 4 永远 SKIP，还写"这不是源码检出" | `_repo_root()` 数父目录层数；文件从 `tileinfer/vllm/doctor.py` 搬到 `src/ascend_platform/doctor/cli.py` 深了一层，`parents[2]` 从仓库根变成了 `src/` | 改为向上找 `pyproject.toml`；找不到时返回 `None` 并如实说"installed copy" | `test_doctor.py::test_the_runner_source_check_finds_the_checkout_it_runs_from`（+ 一条 SKIP 措辞的） |
| "no model profile matches" 让人去 `tileinfer/vllm/profiles/` 注册 | 报错里的路径是源仓路径，本仓没有 | 改成 `src/ascend_platform/profiles/` | `test_model_profiles.py`（原断言钉的正是旧路径，一并改） |
| 包内多处 docstring 引用 `tests/test_serving_buckets.py` / `test_layer_deps.py` 等**已改名或未搬**的测试 | 机械改名只改了包名，没改文件名 | 改名的改成本仓文件名；没搬的标明"源仓/算子仓的" | `test_doc_references.py`（新增） |
| `hardware-checklist.md` 是本仓的**上机手册**，却让人跑 9 个本仓不存在的测试文件、`scripts/` 目录（本仓没有） | 整篇从源仓机械重写而来 | §3 重写为"本仓全绿、没有既存收集错误"；§6/§9 标注"算子仓步骤"；补上"两个仓的分工"前言 | `test_doc_references.py`（新增，扫 `docs/`） |
| 手册第 2 步的第一条命令是 `python -c "import ascend-platform.vllm.plugin"`——**连语法都不合法**（连字符不是标识符） | 机械改名把 `tileinfer.vllm` 逐字替换成 `ascend-platform.vllm`，没考虑包名里的连字符只属于**发行版名**，不属于模块名 | 全部改成 `ascend_platform.engine.*` / `ascend_platform.doctor.cli` / `ascend_platform.runtime` 等真实模块路径 | `test_checklist_commands.py`（对每条 `python -c` 与 heredoc 跑 `compile()`，并禁止文档里出现 `ascend-platform.<模块>`） <!-- not-a-citation --> |
| 手册第 2 步按 `e.name == "ascend-platform"` 找入口点，而 `pyproject.toml` 声明的名字是 `ascend_platform`，于是**装对了也打印 `MISSING`**，把人引向"只能有一个平台插件"的错误归因 | 同上：把发行版名（`ascend-platform`）当成了入口点名 | 手册改成 `ascend_platform`；期望输出块同步 | `test_checklist_commands.py`（入口点名/值与 `pyproject.toml` 逐条对照） |

**手册抽检（不是在搬运，是在验证搬运结果可执行）**：`hardware-checklist.md` 的 0–4 步在本机逐条跑过——
`doctor` 的四种拼写（`ascend-platform` / `doctor` / `check` / `status` / `--json`）退出码都是 1（本机缺
torch_npu，符合"有 hard 失败时非 0"），第 2 步在装了本包的 venv 里两个入口点都是 `loadable: True |
matches pyproject: True`，第 3 步两个环境全绿，第 4 步 `doctor` 直接给 PASS。5–11 需要设备或算子仓，
已在文末表里按「本仓 / 算子仓 / 上卡」三分类标注——**这台机器上只有三分之一的事是本仓能做的**，
没有卡时不能把它们判成失败。

新增的 [`tests/test_doc_references.py`](tests/test_doc_references.py) 把这条规则变成机械检查：
`src/` 与 `docs/` 里出现的路径引用必须能解析（仓库根 / `src/` / 包内三种基准），否则必须在同句里
标明它是**源仓**（`tileinfer/...`、"源仓"）还是**算子仓**（"算子仓"、`AscendAttention`）的出处。
它带一条"检查器本身不空转"的自检（至少 15 条本仓引用要能解析），并有一张显式的 `FOREIGN`
名单列出那些"长得像本仓、实际属于另两个仓"的路径——不用通配符，正因为它要防的就是"大致对"。

> 判据：**报错里的路径是承诺，注释里的路径是出处。** 前者必须指本仓，后者要写清属于谁。

## 2.3 引用审计：579 处代码引用 + 61 处手册引用，全部可打开

交付项 2 要求"保留原 docstring 中的 vLLM file:line 依据"。**引用只有在读者能打开时才是依据**，
而前三层审计看不见它（token 流把字符串丢掉了），所以它一直没被验过。本轮把 `src/` 里 579 处
`path.py:line` 抽出来逐条核对，设四类归属：

| 类别 | 数量 | 核对对象 |
| --- | --- | --- |
| vLLM 摘录 | 526 | `reference/vllm-0.23/` → 行号存在、区间非空 |
| 源仓出处 | 40 | `tileinfer/...` → 文件存在、行号在范围内（源仓不在时跳过） |
| 上游依赖 | 11 | 装着的那份 torch → 只查**符号**，见下 |
| 本仓 | 2 | `src/ascend_platform/...` |

结果：**0 处无法解析**。过程中修掉的是四类真问题：

| 问题 | 例子 | 处理 |
| --- | --- | --- |
| **引用的行号漂了** | "`TileInferPlatform.set_device` (`platform.py:993-995`)" —— 993 行其实是 `check_and_update_config` 里的 `return`，`set_device` 在 1116 | 改成 `tileinfer/vllm/platform.py:1116-1118`；同类还有 `enforce_eager`（1392→1744） |
| **路径有歧义** | `registry.py:99` 同时匹配 `v1/attention/backends/registry.py` 与 `model_executor/models/registry.py`，读者有 50% 概率查错文件 | 改成能唯一定位的写法（带目录或全路径） |
| **源仓文件没加前缀** | `attn/impl.py`、`serving/buckets.py`、`gating.py` —— 在本仓里并不存在这些路径 | 一律加 `tileinfer/` 前缀（`src/` 里现在共 40 处源仓引用），并顺手把 `fla/causal_conv1d.py` 这个写错的目录改成 `mamba/ops/` |
| **引用到未收录的 vLLM 模块** | 本轮 5 份：`compilation/passes/pass_manager.py`、`v1/cudagraph_dispatcher.py`、`utils/mem_utils.py`、`utils/system_utils.py`、`model_executor/models/config.py` | 按 README 的机制整份补录（verbatim），引用随即可解析；目录现共 112 份 |

上游依赖（torch）改成 `path.py::symbol` 形式，因为行号会随每个 wheel 变：`torch/cuda/memory.py:847`
在本机 torch 里是 822 行，而 `::mem_get_info` 永远指得到。新增
[`tests/test_citation_anchors.py`](tests/test_citation_anchors.py) 钉住这四类，且**类别之外一律失败**——
新写法不能悄悄溜进来。测试本身也做过注入式核对：把 `worker.py:21` 引用的 `gpu_worker.py:249-345`
改成 `:999991-999992`，报错立刻指名"`src/ascend_platform/engine/worker.py:21` cites
`gpu_worker.py:999991-999992`: the file has 1204 lines"。

同一套检查也用在**手册与本文**上（57 处带目录的引用；散文里的短名如 `platform.py:1121` 按上下文
指本仓模块，按设计跳过）。它当场抓到一个真错：本文原来引的
`tests/test_package_layout.py::test_no_module_reaches_an_operator_package_except_through_the_port` <!-- not-a-citation -->
**根本不存在**——那条断言早已改名并搬进 `tests/test_layer_acl.py::test_only_the_port_imports_an_operator_package`。
引用不核对，写的人只会越写越自信。

补录摘录的副作用被测试抓住了一次：`utils/platform_utils.py` 里的 `cuda_is_initialized()` 引入了
`torch.cuda._is_compiled` / `is_initialized` 两个名字，`tests/test_platform_contract.py` 的别名单
（把 `torch.cuda.*` 桥到 `torch.npu`）因此要求给它们定性。答案是不桥：`torch.npu.is_initialized`
**存在**，而桥过去会让 vLLM 以为 CUDA 在 NPU 上已经初始化并走 CUDA 专用路径——这正是"能桥所以桥"
会犯的错，两者记为 `UNMAPPED_TORCH_CUDA_NAMES` 并写明理由。

## 3. 刻意没有搬

| 没搬 | 原因 |
|---|---|
| `kernels/`、`ops/{fla,moe,gdn,fused_moe_oot}.py` | 算子数学属于算子仓（`AscendAttention`）；搬过来就破坏了本仓库的定位 |
| `models/dsv41.py`（1254 行）及 `models/` 其余部分 | 模型实现不是平台职责；平台只提供注册 hook |
| `benchmarks/`、`scripts/`、`tools/`、`legacy/` | 部署与测量属于使用方；脚本里有模型路径、端口等环境事实 |
| `vllm/config.py`（dsv41 的 HF config shim） | 跟着模型走 |
| 除 `gdn_moe` 之外的 op family 表 | 那些 family 的实现在算子仓，等边界的另一侧落地后再搬 |

因此 `tests/test_boot_rehearsal.py` 里的 15 条启动器断言被移除（文件末尾有说明）：
它们钉的是 `scripts/serve_qwen36.py`，而本仓库不发布 `scripts/`。

同理，`test_ops_install.py`（22 条）与 `test_op_candidates.py`（25 条）没有整体搬入：它们的断言对象
是 `ops/fla`、`ops/gdn`、`ops/moe` 这些**算子实现**的身份，而按边界这些实现不在本仓库。平台自己那半
（切换语义、候选门控、绑定表数据）改为新的 `tests/test_ops_rebinding.py`（16 条），不依赖任何算子代码；
端到端写名字/回滚仍由 `test_boot_rehearsal` 在装了算子层时覆盖。

## 4. 搬运期间修正的既有错误

搬过来的测试**不是全绿的**，源仓库整套跑本来就有失败（项目自己也记录了这一点）。逐条处理如下，
每一条都是"代码是对的、测试是旧的"或"测试假设了环境"：

| # | 位置 | 问题 | 处理 |
|---|---|---|---|
| 1 | `test_attn_contract::test_backend_identity_*` | 断言 `get_name() == "TILEINFER"`，但 vLLM 用这个字符串索引 `AttentionBackendEnum`，实现早就返回 `"CUSTOM"`（docstring 里记着这次硬件失败） | 断言改为 `CUSTOM` |
| 2 | `test_platform_contract::test_documented_counts_*` | 文档表写 10/6/3/56，实现已是 13/4/2/56 | 表与断言同步为 13/4/2 |
| 3 | `test_platform_contract::test_the_inherited_methods_still_raise` | 5 个"仍是基类实现"的断言里有 3 个（`manual_seed_all`、`get_current_memory_usage`、`num_compute_units`）已实现 | 只留真正未实现的 2 个 |
| 4 | `test_platform_contract::_fake_npu_module` | 假 `torch.npu` 按 `torch.cuda` 名字建键，而别名表是 `CUDAGraph → NPUGraph`，于是假对象里没有 `NPUGraph` | 按**对应物**名字建键 |
| 5 | `test_platform_contract::test_platform_helpers_fail_closed_*` | 用 `_npu_available()`（有没有设备）判断能否观察"没有 torch_npu"分支；本机 torch_npu 可 import 但无设备 | 改为判断 **import 是否可能** |
| 6 | `test_platform_contract::test_wrap_delegates_*` | `Platform` 没有 `graph_pool_handle`，CPU 机器上转发到 `torch.cpu` 得到 `None`，调用即 `TypeError` | 给当前平台类补上该 hook（NPU 平台本来就有） |
| 7 | `test_worker_contract::_snapshot` | 构造 `NpuMemorySnapshot` 漏了必填的 `device_`（该类故意不设默认值） | 补 `device_` |
| 8 | `test_worker_contract` | 硬编码 `tileinfer/vllm/worker.py` 等路径 | 改为 `src/ascend_platform/engine/*.py` |
| 9 | `test_boot_rehearsal::_install_fake_decode_backend` | 桩 `lambda: backend`，而工厂现在收 `dtype`（它决定编译哪个 kernel 变体） | 桩改为 `lambda dtype=None:` |
| 10 | `test_doctor::clean_state` | `plugin.install()` 会写 `torch.cuda → torch.npu` 别名（进程级），fixture 没撤销 → 下一文件看到"桥已存在" | teardown 里 `graphs.uninstall()` |
| 11 | `tests/conftest.py` | 双装（源仓库 + 本仓库）时 vLLM 报"只能有一个平台插件"，整套无法收集 | 默认 `VLLM_PLUGINS=ascend_platform` |

第 4–10 条是**测试隔离/环境假设**问题；第 1–3 条是**文档落后于实现**问题；第 11 条是拆仓带来的
新问题，属于迁移必须解决的部分。

### 4.1 搬运后按"新用户视角"走一遍流程时又修的两处

前面的检查都在验"文件、字符串、示例"，这一轮改成**照着文档从零做一遍**，于是碰到两处只有走流程
才会暴露的问题：

| 症状 | 根因 | 修法 |
|---|---|---|
| 没有算子层时，`doctor` 的 `ops.binding` 报 FAIL，`fix:` 却写着"这是 `ascend-platform` 的 bug，请报告 traceback" | `_run_check` 的兜底分支把所有异常都当成"本包的 bug"；而 `OperatorLayerUnavailable` 是一个**受支持的启动状态**，一条命令就能修 | `_run_check` 增加专门分支：FAIL 的 detail 保留异常原文，`fix:` 改成"装算子层或设 `ASCEND_PLATFORM_OPS`，然后重跑"（`architecture.md` §4.4 的语义） |
| 交付项 4(b) 承诺"版本区间启动期校验"，但 `SUPPORTED_OPERATOR_RANGE` **只有声明、没有执行**：清单那一侧有 `check_compatibility()`，而真正在用的导入根那一侧一个版本都不看 | 两半被写成了"一件事"，实际是两件事：清单校验回答"这个库宣称支持我的平台吗"，导入根校验回答"我 import 到的包是不是我以为的那个版本"。只做前者时，`ASCEND_PLATFORM_OPS` 指到一个 0.3.0 的旧目录仍然全绿 | `ports.check_operator_version()` + `install_ops()` 在**第一条绑定之前**调用它：超出区间就一条绑定都不做，原因写进 `OpsReport.layer_error`（与"没有算子层"共用同一个上报通道），引擎退回 vLLM 自带算子；`doctor` 新增 `ops.version` 检查，PASS 行同时打印版本与区间 |
| `TILEINFER_OPS` 拼错（如 `not_a_family`）时，安装器按设计**静默回落**到默认集合 `gdn,moe`，日志里只有一行 WARNING；上机时"我关掉了 moe"和"我拼错了"看起来一模一样 | 宽容解析是刻意设计的（拼错不许让机器停摆），但**没人告诉用户**；doctor 没有这一项 | 新增 `ops.spec` 检查：读用户写的原值，用严格解析器判断，FAIL 时点名拼错的词、列出合法值，并说明"安装器回落到了 `gdn,moe`" |
| 两个 out-of-tree 平台插件同时装着时，用户看到的是 `op unbound` / `Backend CUSTOM must be registered`，看起来像算子坏了 | vLLM 的真实拒绝（"Only one platform plugin can be activated"）发生在更早，而 doctor 没有这一项检查 | 新增 `plugins.unique` 检查（step 2）：列出两个插件名，给出 `VLLM_PLUGINS=ascend_platform` 的解法（vLLM 自带白名单，**不用卸载**）；顺手把 `entry_points(group=...)` 换成兼容 3.9 的写法（`requires-python = ">=3.9"`，3.9 上没有这个关键字参数） |

第二条的判据很具体：`plugins.unique` 在三种环境下分别给出 PASS / PASS / FAIL——
只装本仓（PASS，一个插件）、两个都装但设了 `VLLM_PLUGINS`（PASS，白名单生效）、两个都装且没设
（FAIL，点名 `ascend_platform, tileinfer`）。源码检出、未安装时给 SKIP 并说明原因。

### 4.2 交付后才暴露的打包错误：`.gitignore` 把整个 runtime 层忽略了

写 `.gitignore` 时写了 `runtime/`（意图是忽略测试在工作区里造的临时目录）。git 的规则里**没有前导
斜杠就没有锚点**，于是它同时匹配 `src/ascend_platform/runtime/`——**运行时层 7 个文件全被忽略**。

为什么一直没发现：这些文件在磁盘上真实存在，测试、doctor、acceptance 全绿；错误只在**克隆出来的
仓库**里出现（`ModuleNotFoundError: No module named 'ascend_platform.runtime'`），而克隆是我不会做的
动作。发现它的是"把仓库复制一份到别处再跑一遍全套"这一步：复制时按 `runtime` 排除，才看到 12 个
收集错误。修法是把根目录产物全部锚定（`/runtime/`、`/build/`、`/dist/`），并在
`tests/test_package_layout.py` 里加了两条断言——`git ls-files --others --ignored` 只允许缓存类路径，
且 `src/ascend_platform/runtime/step.py` 必须对 git 可见。把规则改回 `runtime/`，两条立刻红。

### 4.3 三个"广告了但不存在"的公开名字

给公开 API 补文档字符串时（新增 `tests/test_public_api.py`），机械检查先撞到三处**声称导出、实际
没有定义**的名字——`from x import *` 会直接 `AttributeError`，而读文档的人会以为它存在：

| 位置 | 症状 | 处置 |
| --- | --- | --- |
| `device/profile.py` 的 `__all__` | 列了 `profile_for_config`，但它定义在 `engine/platform.py`（配置期解析"在跑的模型用哪份 profile"） | 从 `__all__` 删掉；错的是导出表，不是函数 |
| `engine/models/__init__.py` 的 `__all__` | 列了 `dsv41`——那是**没有搬**的模型实现（留在模型仓） | 改成真实存在的三个模块：`batching` / `req_ids` / `state_slots` |
| 同上，模块文档字符串 | 说什么"每个模块拥有一个 `nn.Module`、数学在 `tileinfer/models/<arch>`"——这是早期计划的措辞，和现在的层完全不符（这里只有批语义，没有 `nn.Module`、没有内核） | 重写成这层实际拥有的三件事，并指向 `MIGRATION_LOG.md` §2 的逐行出处 |

第三条最值得记：**导出一张错的表比不导出更糟**，因为它把"这里有个模型实现"写进了读者的前提。

### 4.4 版本门会把 `+local` 后缀当成"不匹配"，然后以一个崩溃的形式炸出来

上机前把版本门对着**真实会出现的字符串**过了一遍，发现三类写法不是"被拒绝"，而是**抛出另一种异常**：

| 输入 | 改前 | 改后 |
|---|---|---|
| `0.23.0+cu123`（本地编译的 vLLM） / `0.1.0.dev0+git1234`（git 安装的算子层） | `InvalidVersion`（`ValueError`） | 接受——本地段不改变发布线 |
| `v0.23.0`（tag 驱动的构建脚本） | `InvalidVersion` | 接受 |
| `0.23.0-1` / `0.23.x` / `banana` | `InvalidVersion` | `IncompatibleVersion`，消息带原串与"refusing to start" |

前两行是**假阴性**：本该上机跑的组合被判成不兼容。第三行更麻烦——`install_ops` 与 `doctor` 都只
捕获 `IncompatibleVersion`，`ValueError` 会**穿过去**：启动阶段变成一个看不懂的 traceback，`doctor`
会把它记成本仓的缺陷（和之前 `OperatorLayerUnavailable` 被当成 bug 是同一类错误）。

同一类问题还有一处：清单里的 `requires_platform` 是**对方仓的数据**，写了 `~=0.1` 时同样会以
`ValueError` 穿透 `install_ops`。现在 `parse_manifest` 在**加载时**校验范围语法，坏清单抛
`ManifestError`（`docs/op-manifest.md` §4 加了一张"坏在哪 / 抛什么 / 谁去修"的表）。区分原则是
一致的：**对方的数据坏了 → 对方的错，抛我们文档里写明的类型；我们自己的常量坏了 → 我们的错**。

修法两条：解析忽略 `+local` 段与 `v` 前缀（`contracts/version.py::_parse`）；`check_range` 把**已安装
版本**解析失败包装成 `IncompatibleVersion`，而**范围常量**自身写错仍然抛 `InvalidVersion`——那是本仓的
bug，不该伪装成对方的版本问题。`docs/versioning.md` §3 加了一张"哪些写法算同一条线"的表。

### 4.5 "加第二张卡只要一个文件"——文档自己两处说法相反

`device/` 的包文档字符串与 `architecture.md` §5 通道 B 都写"新增一个模块即可，`__init__.py`
不需要改"；而 `device-profile.md` §3 的步骤写的是"然后在 `device/__init__.py` 里 import 一次"。
后者是对的：`__init__.py` 里有一行显式的 `from . import ascend950`。

真实后果不是文档难看，而是**静默**：只放一个 `device/ascend960.py`，它会坐在那儿没人 import，
`detect_device_profile()` 回落到参考卡的 HBM 与单元数，第二台机器用第一台的数字跑，全程无报错。

处置是两头都改：

- 说真话：目标改成"新增一个模块 + 一行 import"，并在 `architecture.md` §5 通道 B 解释**为什么
  不做自动发现**（依赖图要静态，`_acl` 与 import 白名单的测试才读得到）；
- 加棘轮：`tests/test_device_profile.py` 检查该目录**每个**模块都被 `__init__` import 且都调用了
  `register_device_profile()`。注入一个没人 import 的 `ascend960.py`，两条同时红——所以"忘了那行"
  现在是测试失败，不是运行期意外。

### 4.6 18 个旋钮没写在任何地方，1 个写了的旋钮不存在

把"代码里读的环境变量"与"文档里写的环境变量"对了一遍（现在这条检查是自动的）：

- **代码读、文档没写：18 个**，其中包括上机门最需要的那三个——`TILEINFER_PHASE_TIMING`（阶段耗时）、
  `TILEINFER_OP_COUNT`（每个接管目标被调了多少次）、`TILEINFER_PREFILL_TIMING`（tiles/batch 统计）。
  它们原本只写在模块文档字符串里；一个人在追 logprob 分叉时不会去翻 49 个文件。
- **文档写、代码没读：8 个**，全部是**出处**而不是旋钮：源仓当时的现场环境（`hardware-facts.md` 的
  "env vars in force"）与 vLLM 自己的开关。`TILEINFER_METRICS_PATH` 是其中最像陷阱的一个：手册里
  出现 4 次，但本仓没有任何代码读它（原生后端 v0 未接线，探针由遗留路径产出）。

新增 [`docs/env-vars.md`](docs/env-vars.md)：三张表（运行期开关 / 观测开关 / 只读的环境事实），
每条给出**默认值、读它的文件、作用**；§4 直接给出"定位问题先开哪三个"。`tests/test_env_vars.py`
双向校验（8 条）：漏写会红，写了没人读也会红，出处名单必须仍在文档里被解释。

### 4.7 机械改名在散文里留下的 `ascend-platform.vllm...` <!-- not-a-citation -->

`docs/notes/op-coverage-qwen36.md` 里三处把算子仓的模块写成 `ascend-platform.vllm.ops.moe.fused_experts`， <!-- not-a-citation -->
另有一句"只加了 `_ascend-platform_oot_*` 标记"。同一个病根：**连字符是发行版名的，不是模块名的**——
和 §4.1 里那条让 `python -c "import ascend-platform.vllm.plugin"` 语法不合法的错误一样，只是这次的 <!-- not-a-citation -->
受害者是散文而不是代码块，所以 `compile()` 检查抓不到。

修法：三处改成 `tileinfer.vllm.ops.moe.fused_experts`（那是**算子仓/源仓**的模块，本仓没有搬
`ops/fused_moe_oot.py`）；"加标记"那句改成源码里真实发生的事——注册的是**子类**
（`base_cls.register_oot(...)`），in-tree 类一个字节都没动。再补一条机械检查：文档里不许出现
`ascend-platform.<标识符>`（发行版名当模块用），`<!-- not-a-citation -->` 可以豁免"故意引用错误写法"
的那一行。注入回去立刻红。

### 4.8 `vllm/ops/extern.py`：被自己的交付项取代，却没人写下这件事

源仓有一个"单向依赖"的桥：`TILEINFER_OPS_LIB=ascend_attention` 打开它，import 期把
`chunk_local_cumsum` / `l2norm_fwd` 两个模块属性**换成**另一份实现，任何异常都静默回退到本地。
它其实就是交付项 4(b) 的**雏形**——方向对（不让平台反过来依赖算子仓），但三个地方与本仓的设计相反：

| 旧桥（`extern.py`） | 本仓（`contracts/ports.py` + `contracts/manifest.py`） |
| --- | --- |
| import 期改模块属性，改了就回不去 | 惰性按需取用，名字**不由平台重绑定到本地**，缺层就是缺层 |
| 失败静默回退到本地实现 | 缺层/版本不符**说出来**（`OperatorLayerUnavailable` / `IncompatibleVersion`），并给出修法 |
| 靠 `TILEINFER_OPS_LIB` 字符串约定 | 靠算子仓的纯数据 manifest + 版本区间在启动期校验 |

它没有出现在 `PAIRS` 里，也没有写在文档里——所以"全量迁移"这件事在它身上是**没有交代**的。
现在它落在 `EXCLUDED_FILES`，理由写清"被取代"，并指向本节。顺带说明：桥里那两个名字属于
**模型前向**路径（`ops/__init__.py` 在 import 期调用它），本仓不搬模型前向，所以那条路径连带
不进本仓——这一点也写进了同一个条目。

### 4.9 一条"已迁移但没被审计覆盖"的文件，和一条过期的白名单

补全"源文件 → 去留"的账时发现两件事：

1. `vllm/ops/families/__init__.py` **搬过来了**（`family_names`/`register_family`/`family_for` 都在），
   却不在 `PAIRS` 里。于是它的函数体从来没被逐 token 比对过——正是在它身上后来加了 `family_names`
   的 docstring，而审计一声不响。补进 `PAIRS`（第 31 对），并把那条 docstring 差异登记为理由。
2. `EXPECTED_DIFFERENCES` 里有一条**过期**的：`install.py::_resolve_families`。这个函数当初确实从
   `install_ops` 里提出来过，但后来对齐成了逐 token 相同——而白名单还留着。过期的白名单比没有更危险：
   它会把**未来**任何一次对该函数的改写静默吞掉。删掉它，并加一条自检：白名单必须**恰好**被用尽
   （`real == set(EXPECTED_DIFFERENCES)`，模块级同理）。

同时把 `MIGRATION_LOG.md` §3 结果表里的数字变成**测试算出来的**（`test_the_numbers_the_migration_log_quotes_are_these_numbers`）：
31 对 / 415 个定义 / 405 逐 token 相同 / 10 条登记；521 个顶层项 / 511 相同 / 10 条登记；456 个定义带装饰器、
0 处丢失。手抄的证据会腐烂，算出来的不会。

### 4.10 "全量迁移"第一次有了机器读的账

`PAIRS` 是手写清单，它只能证明**列出来的**文件没被改坏，证明不了**漏列**。而交付项 2 的
"一行不落地"是关于**源仓**的断言，不是关于清单的。所以审计改成从源仓那一侧数：枚举
`tileinfer/**/*.py`（107 个），要求每个路径落进三者之一——

| 类别 | 数量 | 说明 |
| --- | --- | --- |
| 迁移（`PAIRS`） | 31 | 逐 token 比对 + 模块级 + 装饰器 |
| 整棵树排除（`EXCLUDED_TREES`） | 59 | `kernels/`(18)、`models/`(14)、`testing/`(6)、`attention/`(7)、`integrations/`(2)、`vllm/ops/fla/`(12) |
| 单个文件排除（`EXCLUDED_FILES`） | 17 | 包 `__init__`（重写而非搬运）、`metadata.py`/`plan.py`（算子层数据类型）、`extern.py`（见 §4.8）、`config.py`、模型前向两文件、算子实现三文件 |

三条反向检查防止名单腐烂：排除项必须真的匹配到文件（防穿越）、理由不得短于 30 字（防橡皮章）、
同一个文件不许被分类两次。注入一个"谁也没决定过"的源文件、或一个不存在的目录前缀，都会立刻红。

### 4.11 上机门的第一步"看横幅"是空的：vLLM 只给它自己的 logger 装 handler

手册第 2 步的第一条命令是 `grep "ascend-platform vLLM plugin" serve.log`，文案写着"一眼看插件装没装"。
**上机实测：一个完全健康的 serve 进程里，这行一个字符都没有**（`grep` 计数 0，而 `ops: 11 bound`、
3 个补丁全打上了）。原因在 vLLM 自己的日志配置：

`vllm/logger.py` 的 `DEFAULT_LOGGING_CONFIG` 只给 **`vllm` 这一个 logger** 装了 handler，且
`propagate=False`；root logger 没有任何 handler。于是第三方 logger（本包的 `ascend_platform.*`）
的 `INFO` 被 `logging` 模块**直接丢弃**，`WARNING` 以上落到 `logging.lastResort`（裸 stderr，没有级别、
没有时间、没有 logger 名）。手动 `import vllm` 后调 `install()` 时看到的那些行，正是 `lastResort` 漏出来的。

修法（`engine/plugin.py`）：新增 `_logs_need_a_destination()` 与 `_ensure_visible_logs()`，在 `install()`
入口处调用一次——**仅当**本包 logger 与 root logger 都没有 handler 时，才给 `ascend_platform` 装一个
`StreamHandler`（级别取 `VLLM_LOGGING_LEVEL`，默认 INFO），并设 `propagate=False` 避免重复。
不动别人配好的日志：pytest、操作者自己的 `dictConfig`、`RANK0_LOG` 都保持原样。

修完再上机，同一位置是：

```
INFO 2026-11-11 00:20:02,118 [ascend_platform.engine.plugin] ascend-platform vLLM plugin installed |
vLLM 0.23.0 | strict=False | patches: 3 applied, 0 skipped (target gone), 0 failed | ...
```

这是一处**改了迁移函数**的修正，所以按纪律记进 `EXPECTED_DIFFERENCES`（`tileinfer/vllm/plugin.py::install`，理由同上）；`tests/test_plugin_install.py`（5 条）钉住三件事：该装时装、第二次不重复装、
别人配过就不动。

### 4.12 上机时另外两件必须写进手册的事

1. **`fork` 起不来**：vLLM 0.23 默认 fork，torch_npu 报
   `Cannot re-initialize NPU in forked subprocess`。`VLLM_ENABLE_V1_MULTIPROCESSING=0`（单进程）或
   `VLLM_WORKER_MULTIPROC_METHOD=spawn` 二者其一，已写进手册 §0 的四个变量旁边。
2. **`--max-model-len` 与 hybrid 页**：取 1024 时 decode 预规划会拒绝构建
   （`KV pool has a 2048-token page but cache_config.block_size is 1024`），服务能跑但首次 decode 懒编译，
   而懒编译不能发生在图捕获里。手册里现在直接给 ≥ 2048。

这两条都不是本仓的 bug，是"平台与引擎的契约在真卡上长什么样"，所以写进手册而不是代码。

## 5. 新增的测试（不是在搬运）

| 文件 | 钉住什么 |
|---|---|
| `test_device_profile.py`（28） | 记录自校验、注册表语义、探测不猜数字、参考卡数字、状态页 == 一个注意力页；**外加"加卡流程"的棘轮**：该目录每个模块都被 `__init__` import 且都注册了 profile（见 §4.5） |
| `test_contracts_manifest.py`（54） | 版本范围语义、清单 schema 与两条硬规则、启动期兼容性检查 |
| `test_ports.py`（20） | 惰性解析、布局可注册、缺失符号的消息、**无算子层也能 import**（干净子进程） |
| `test_package_layout.py`（21） | entry point 可解析、依赖清单刻意很小、vLLM import 白名单；**另外钉住"逐文件能跑"**：`pythonpath` ini 存在，且一个把 `PYTHONPATH` 摘掉、用真子进程跑 `pytest tests/test_wiring_strings.py` 的测试必须通过 |
| `test_layer_acl.py`（77） | 7 层依赖方向、5 个无 vLLM 层的静态与**子进程动态**证明、只有端口能碰算子包 |
| `test_wiring_strings.py`（16） | 引擎按**字符串**加载我们的类（`worker_cls`、backend slot、entry point）：每个 `*_CLASS_PATH` / `*_ENTRY_POINT` / `*_MODULE` 都必须解析到这个包里的真实模块与真实属性——静态解析，不需要 vLLM/torch/设备 |
| `test_no_operator_math.py`（5） | 边界的机器证明：`src/` 里没有内核源文件、没有内核 DSL、没有算子实现定义、`ports` 的转发函数体只有一句 `return`；并对着算子仓核对 `kernels/models/attention` 没被复制。三条检查都用注入式改动验证过"真的会红" |
| `test_docs_examples.py`（5） | 文档里的示例要被**执行**：`docs/op-manifest.md` 的 JSON 用本包的 `parse_manifest` 校验，并跑一次兼容性检查；版本区间在代码 / `pyproject.toml` / `versioning.md` 三处必须一致 |
| `test_plugin_install.py`（5） | 横幅的可打印性（§4.11）：只在"进程里没有任何 handler"时才给本包装一个；第二次调用不重复装；pytest / 操作者自己配过日志时不动；级别跟随 `VLLM_LOGGING_LEVEL` |
| `test_env_vars.py`（8） | 环境变量双向账本：`src/` 里读到的每个变量必须在 `docs/env-vars.md` 里（标注默认值与读它的文件），文档里的每个变量必须有人读或是显式登记的出处；出处名单不许无声消失。抓出了 §4.6 的两类问题 |
| `test_public_api.py`（10） | 公开面必须**有解释**：195 个公开类/函数逐个要有文档字符串（6 个从源仓逐字搬来、源仓本来就没有的记进 `VERBATIM_UNDOCUMENTED`，且该表不许腐坏——名字消失或后来补了文档都会红）；文档字符串不许是名字的复述；每个 `__all__` 的条目必须在本模块有定义（或确为子模块）、不许重复。这一条抓出了 §4.3 的三处 |
| `test_doc_references.py`（5） | 代码与文档里的路径引用必须能解析，否则要标明是源仓/算子仓的出处；含"检查器本身不空转"的自检。 文档/错误信息里的路径必须能解析，或**明说是谁的**（`tileinfer/…` 是出处，不是承诺）；新增一条检查 `§N` 指针：`README` 把"明确不做"指到 `architecture.md` §8（那是"不重命名"），本仓 §9 才是那张表——标题编号都存在，只有检查**目标**才抓得到 |
| `test_checklist_commands.py`（22） | 手册与 P5 门笔记里的每条 Python 片段必须能 `compile()`；入口点名/值必须与 `pyproject.toml` 一致；文档里出现的每个 `ascend_platform.…` 名字都要**能解析成模块或模块里的真实成员**（把 `engine.nonexistent_thing` 注入 `versioning.md` 立刻红）；另外禁止把发行版名当模块（`ascend-platform.vllm…`） | <!-- not-a-citation -->
| `test_utils_helpers.py`（21） | 设备助手：`quantize_capacity` 的 2 的幂（含"永不缩小 / 幂等"性质，它是编译键的来源）、`resolve_device` 在无 NPU 时回退 CPU 但**显式要 NPU 必须报错**、`device_name`/`default_dtype`/`synchronize` 的 CPU 行为 |
| `test_model_providers.py`（12） | provider 扩展点（out-of-tree 模型包）的接线：环境变量解析宽容、未安装的 provider 是**静默跳过**而非启动失败、`register_config` shim 按 `pkg.models.x → pkg.config` 约定查找、shim 抛异常也绝不外泄（general plugin 在**每个**进程里跑） |
| `test_migration_parity.py`（228） | 除了原有的逐 token / 模块级 / 装饰器 / 调用集合四层比对，新增**源仓侧的完整性账**（§4.10）：107 个源文件必须落进"迁移 31 / 整树排除 59 / 单文件排除 17"三者之一，排除理由不得橡皮章，两份白名单必须**恰好用尽**（§4.9），`MIGRATION_LOG` §3 的每个数字都由测试算出；§2.1 的标题数字与 §2.3 的两个引用数字也由测试对账（§4.9） |
| `test_p5_gate_comparator.py`（16） | 上机门的判据本身：逐位 PASS、1 ulp FAIL、长度不同 FAIL、top-k 不同 FAIL；**其中四条把 `docs/notes/p5-gate.md` 里的脚本原样抽出来当子进程跑**（避免"文档写一份、测试测另一份"），四条覆盖同一矩阵；另有**抓取半边**的两条：把 runbook 的 bash 块对假 `/v1/completions` 跑一遍（多行单引号 JSON、`python3 -c` 回读、落盘 tokens、"N tokens" 输出），以及对着**装着的 vLLM 0.23** 核对判据读的三个键（`tokens` / `token_logprobs` / `top_logprobs`）——引擎改名会让这条门先红，而不是在卡上比个空 |
| `test_citation_anchors.py`（763） | 579 处代码引用 + 61 处手册引用必须能打开（两个数字由测试算，不是抄的）：vLLM 摘录查行号、源仓查 `tileinfer/`、torch 只查符号（`::symbol`）、本仓查自身；类别之外一律失败，并有 ≥500/≥35/≥8 的反空转下限 |
| `test_migration_parity.py`（212） | 四层比对：定义名/函数体 token 流/行数、模块级项（类字段、常量、`if` 块）、装饰器列表、**文件级调用集合不缩小**（给已豁免的函数兜底）；差异必须登记理由（源仓库缺失时跳过） |
| `test_ops_rebinding.py`（17） | 切换是家族级且每次调用重读、未知名字严格报错/温和回退、候选门控把实测比值算进去、绑定表数据完整 |

这几个文件是"新架构"的实际断言：删掉它们，架构就只剩文档里的话。

**它们是怎么被找出来的**：用 stdlib 的 `trace`（这台机器没有 `coverage`，也不联网）量了一遍
`src/` 的行覆盖，按模块列出"从未执行的行"。结果里有一个整文件 0 行：`engine/platform_dsv41.py`
——DeepSeek-V4.1-Flash 那一支平台类，由 `VLLM_DSV41=1` 在运行期选中，**没有任何测试碰过它**。
同一轮还发现 `profiles/dsv41.py` 的 `_kv_shape`、`engine/model_providers.py` 的解析与 shim
查找、`utils.py` 的 2 的幂与设备回退都没被执行过。它们全是不需要设备就能测的纯逻辑，于是补了
上面三处断言（平台子类、profile 几何、provider 接线）与 `test_utils_helpers.py`。覆盖度不是目标，
**"整文件没人跑过"才是**。

## 5.5 许可证

`pyproject.toml` 一直声明 `license = { text = "Apache-2.0" }`，但仓库里没有 `LICENSE` 文件——声明没有
依据，打出来的 wheel 也没有许可证文本。本轮从源仓（`/home/userlist/zxs/TileInfer/LICENSE`，Apache-2.0
全文 202 行）原样复制一份，wheel 里因此出现 `ascend_platform-*.dist-info/licenses/LICENSE`，
METADATA 带 `License-File: LICENSE`。`tests/test_package_layout.py` 加了一条断言：声明、文件、内容三者
必须一致（把文字改成 MIT 或删掉文件都会红）。

## 6. 验收状态

| 门 | 状态 | 证据 |
|---|---|---|
| `pip install -e .` 可用 | ✅ | 一次性 venv（`--system-site-packages`，**连 torch 都没有**）里 `pip install --no-deps -e .` 成功；`importlib.metadata` 看到 3 个 entry point（`vllm.platform_plugins` / `vllm.general_plugins` / `console_scripts`）；`ascend-platform --help` exit 0；`pip check` 只报缺 torch（硬依赖，符合预期）；安装后删掉 `src/*.egg-info` 仍可 `import`（PEP 660 的 dist-info 在 venv 里），仓库不留第二份 distribution |
| 无 NPU / 无 vLLM 可 import | ✅ | `test_the_package_imports_with_no_operator_layer_and_no_vllm`；上述 torch-less venv 里 `import ascend_platform` 通过且 `sys.modules` 无 torch |
| doctor 能报缺什么 | ✅（**真卡 15/15**） | 真卡 + `VLLM_PLUGINS=ascend_platform` + `TILELANG_CACHE_DIR`：`15 check(s): 15 PASS, 0 FAIL (0 hard, 0 soft), 0 SKIP`，exit 0；裸 shell 里 12 PASS / 3 FAIL 且三条都打印修法。torch-less venv：`15 check(s): 5 PASS, 3 FAIL (3 hard), 7 SKIP`，exit 1，指向 step 1（torch/tilelang/tilelang_cache）；本仓 CPU 环境：4 hard FAIL（torch_npu/tilelang），exit 1；装了 vLLM 的环境：11 PASS / 1 FAIL（`tilelang_cache` 需设备环境变量） |
| 控制台脚本可用 | ✅ | `ascend-platform --help` exit 0；未知参数 exit 2 |
| 每个测试文件独立绿 | ✅ | 见下表 |
| `src/` 内无算子 import | ✅ | 布局测试 |
| 干净副本可跑 | ✅ | 把仓库（去掉 `__pycache__`/构建产物/临时目录）复制到 `/tmp/fresh`，`PYTHONPATH` 清空后 `python -m pytest tests`：**1883 passed / 18 skipped**——没有隐藏的绝对路径依赖（唯一的外部路径是源仓 `/home/userlist/zxs/TileInfer`，缺失时相关断言跳过） |
| 非 editable 安装 | ✅ | `pip wheel . --no-build-isolation --no-deps` 产出 `ascend_platform-0.1.0.dev0-py3-none-any.whl`（55 个文件，顶层只有 `ascend_platform/` 与 dist-info，无 `tests/`、无 `reference/`，三个 entry point 齐全；dist-info 里带 `licenses/LICENSE` 且 METADATA 有 `License-File: LICENSE`）；装进一个**全新空 venv**（无 torch、无 vLLM）后 `import ascend_platform` 通过、`ascend-platform --help` exit 0 |
| 每文件 pytest | ✅ | vLLM venv：**1821 passed / 28 skipped**；无 vLLM venv：**1883 passed / 18 skipped**；38 个测试文件 |
| 逐文件（交付项 3 的字面要求） | ✅ | 无 vLLM：**38/38 文件独立跑全绿**；有 vLLM：36 绿 + 2 个模块级设计内跳过；倒序与 3 个随机顺序全套跑，计数不变 |
| 第二个 `DeviceProfile` = 一个新文件 | ✅ | `test_a_second_card_is_one_registration` |
| 设备上 logprobs 逐位一致 | ✅ **2026-11-11 上机通过** | 1 × Ascend950PR_9579 / CANN 9.3.0 / vLLM 0.23.0，模型 `Qwen3.6-35B-A3B`，16 token、`temperature=0 / top_p=1 / logprobs=5`。四跑（源仓、本仓默认、`TILEINFER_OPS=off`、`all`）的 token 流、16 个采样 logprob 的 `float.hex()`、16 步 `top_logprobs` **全部逐位相同**。记录表与上机踩到的四件事见 [`docs/notes/p5-gate.md`](docs/notes/p5-gate.md) §5 |

两个环境的差异是**设计**而不是遗漏：`test_worker_contract`、`test_boot_rehearsal`、`test_models_*`、`test_runtime_*`
在无 vLLM 的环境里用 `tests/fake_vllm.py` 跑（钉 vLLM 0.23 的 API 形状），有真 vLLM 时整文件或大部分跳过；
`test_platform_contract`、`test_attn_contract`、`test_doctor` 需要真 vLLM。两边都绿，才是"两种环境都成立"。

## 7. 还差什么

1. **P5 硬件验收**（未做，需要设备窗口）——步骤、比对脚本与失败分类已写成
   [`docs/notes/p5-gate.md`](docs/notes/p5-gate.md)：
   - 只保留本仓库插件，`ascend-platform doctor` 全 PASS；
   - 同一请求在源仓库与本仓库各跑一次，logprobs 用 `float.hex()` **逐位**比较（4 位小数无法区分
     "语义不等价"与"差 1 ulp"）；
   - `TILEINFER_OPS=all` / `off` 两档各跑一次，确认接管是可替换的。
2. **算子层实现在算子仓落地后**：把 `test_ops_install.py` / `test_op_candidates.py` 中属于算子仓的
   断言随实现一起搬过去；平台侧已由 `test_ops_rebinding.py` + `test_boot_rehearsal` 覆盖。
3. **`DeviceProfile` 接管读点**：卡的事实（HBM、单元数、dtype、能力）已在 `engine/` 经
   `device_profile()` 读取；kernel 几何（128 / 2048）**故意**仍是常量，由 `test_device_profile.py`
   的两条防漂移断言钉住（理由见 `docs/device-profile.md` §6）。`engine/` 里剩余的 `128`
   已在复核后确认只出现在 docstring 与 `sub_pages(block_size)` 的派生物中，没有可执行字面量。
4. **其余 op family 表**：等在算子层侧落地后按同一形状补上。
5. **`runtime/` 是测试在工作区里的临时目录**：`.gitignore` 里写成 `/runtime/`（**必须锚定**——写成
   `runtime/` 会连带忽略 `src/ascend_platform/runtime/`，见 §4.2）。

## 8. 刻意不重命名的三样东西（与 `docs/architecture.md` §8 对齐）

- **类名不改** `TileInferPlatform` / `TileInferAttentionBackend` / `TileInferWorker` 等，两条理由：
  1. 它们是 [`tests/test_migration_parity.py`](tests/test_migration_parity.py) 的键——那张审计用
     `类名.方法名` 对照源仓库。改名会让 201 个方法"查不到来源"，除非再给审计加一张改名映射表，
     而那恰好是审计最不该有的东西；
  2. 上机对照时，`TileInferPlatform` 出现在日志里与源仓库一致，搜起来更省事（`worker_cls` 已经
     指向 `ascend_platform.engine.worker.NPUWorker`，功能上不需要改）。
- **`TILEINFER_*` 环境变量不改**：上机验收要拿同一组 shell 脚本对照迁移前后的行为。
- **`__tileinfer_*` 标记属性不改**：算子仓的探针按它们判断"这个名字被谁替换了"。

## 9. 明确不做的事

边界不是形容词，每条都指向一处能验证的地方（`docs/architecture.md` §6 解释"为什么层不拥有它"）。

| 不做 | 为什么 / 现在做到哪 | 依据 |
| --- | --- | --- |
| 算子实现 | 一个内核都不搬；`src/` 里没有内核源文件、没有内核 DSL、没有算子实现定义 | `tests/test_no_operator_math.py` |
| 模型前向 | `DeepseekV41ForCausalLM` 之类留在模型仓，本仓只提供注册 hook | `engine/model_providers.py`、`docs/architecture.md` §6 |
| benchmarks / tools / scripts / legacy | 一个文件都不进；性能数字只消费别人测出来的 `measured_ratio` | `ops/candidates.py`、`docs/op-manifest.md` |
| 性能优化 | 迁移期间不调参、不改策略；只把已有的选择逻辑搬过来 | `MIGRATION_LOG.md` §3 |
| 真机图捕获 | 只搬**判定**（能不能捕获、不能则为什么）；捕获本身不在本期 | `engine/graphs.py`、`engine/worker.py` 的 skip 路径 |
| MTP / 投机解码 | 只保留多 token 预测所需的最小批语义，不实现草案模型 | `engine/models/batching.py` |
| 第二块卡 | 接口预留（`DeviceProfile` 一个文件即可注册），未在真机验证 | `docs/device-profile.md` |
| 量化算子 | 只做验收口径校验（`verify_quantization`），不实现量化 kernel | `engine/platform.py`、`engine/platform_dsv41.py` |
