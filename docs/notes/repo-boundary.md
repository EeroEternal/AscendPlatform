# 仓库边界复盘：MoE 计算掉在了两条声明之间

这份文档只回答三个问题：之前怎么划的、为什么错、两个仓各自要修什么。划法本身写在两个仓的
README「定位」一节；这里是它的事故记录。

## 1. 旧边界写的是什么

三处原文（其中两处在算子仓，引用时只给文字，避免跨仓行号失效）：

- 算子仓 README「仓库功能」：**"MoE：token 对齐（slot 布局）与权重打包（ND → compact-NZ）；
  路由与专家计算属平台层，未实现。"**
- 算子仓 README「非目标」：**"设备语义、KV 分配、批元数据、图捕获、worker、量化、MLA、多卡通信
  ——这些属于平台层。"**
- 本仓 README「非目标」：**"算子数学、搬入厂商 kernel、MLA、量化、TP/EP、完整投机解码。"**

单独看每一句都自洽。合起来，**MoE 路由 / 块对齐 / 专家计算**被判给了"平台"，而本仓的
"不含算子数学"是**有测试钉住的**（`tests/test_no_operator_math.py`：内核文件、内核 DSL、算子定义、
端口转发四项静态检查）。于是这块地两个仓都不认领——不是谁偷懒，是两份声明互相把对方当成所有者。

同一轮里还有第二个错判：**MLA 与量化**被算子仓写进"属平台层"。它们是带形状/dtype/进图约束的数学，
与 FlashInfer 的 MLA、FP8/FP4 内核同类，属算子而不是平台语义。

要拆开两件事，否则会得出错误的结论：

- **"没迁 MoE 计算"不是违规。** 算子仓的迁移计划里，P3 批次本来就只承诺"先判决接不接，负结果只
  归档"，清册中 `prenz` / `slot_layout` 标"迁"，`align` 与 `grouped_gemm` 标"P3 再判决"。计划按
  批次推进，没做到的是"未轮到"，不是"漏了"。
- **真正的错是那句归属。** 把"路由与专家计算"写成"属平台层"，等于让 P3 的判决失去归属——即使
  判决"该做"，也没有仓库会去做。

所以教训不是"下次该补课"，而是**别替对方认领边界**：一句"属平台层"就足以让一条真实计算路径在
两个仓之间蒸发。MoE 计算今天升为 P0 的理由是"平台要绑的名字必须在本仓存在"，不是"上一轮偷懒"。

## 2. 为什么一直没被发现

三次都"差一点"被发现，但三次都被机制挡了回去：

1. 平台的模型族表把 13 个名字列成必需（`src/ascend_platform/ops/families/gdn_moe.py:308` 起），
   但表头是无条件 import——`ops_module("fused_moe_oot")`（旧写法；现在是 `src/ascend_platform/ops/families/gdn_moe.py::_Impl`，逐名解析、miss 记在该算子自己身上）。
   没有任何一处问过"这个模块该由谁提供"。
2. 清单机制是存在的（`src/ascend_platform/contracts/manifest.py:266` 的 `load_manifest`），
   但 `src/` 里**一个调用点都没有**。所以"算子仓声明了什么"从不参与启动判断。
3. 端口按**内部路径**解析（`src/ascend_platform/contracts/ports.py:74` 的 `_LAYOUTS`）。
   能 import 成功就等于"接上了"，恰好掩盖了"语义上该不该由它提供"这个问题。

## 3. 后果（实测，不是推断）

装上算子仓、把平台指向它之后启动，得到的是：

```text
WARNING ascend-platform ops: cannot resolve the model profile's operator family
(ModuleNotFoundError("No module named 'ascend_attention.ops.fused_moe_oot'"));
no takeover is planned. Check ModelProfile.op_family and ascend_platform/ops/families/.
families resolved: []
```

三件事同时发生，每一件都让问题更难看见：

- 13 个必需算子里有 8 个在算子仓里**确实存在**，但**一个都没绑**——构建表是全有全无的；
- 报错把读者指向 `op_family`，真实原因是缺一个模块；
- 服务**仍然能起来**：`_resolve_families` 吞掉异常返回空表（`src/ascend_platform/ops/install.py::_resolve_families`），
  错误被推迟到第一次 GDN prefill / MoE 前向，那时候的堆栈与根因已经隔了三层。

**改动后（同一台机器、同一组 vLLM 0.23 符号、同一个算子仓）**：

```
ops: 11 bound, 1 unbound (no vLLM name / no implementation), 1 off by switch | dense=off gdn=on moe=on
UNBOUND chunk_local_cumsum  :: declined - not a drop-in（刻意拒绝，FLA 的签名不同）
```

`unquantized_fused_moe` 由 `engine/abi` 的类工厂补上（算子仓只提供 compute），所以它从"unbound"
变成了 bound；剩下的那 1 个 `silu_and_mul` 只是 `dense` 族默认关闭。

## 4. 两个仓库各自要修什么

平台侧（本仓）。判据只有一条：**平台可以出现引擎的名字，不可以出现内核。**

| 优先级 | 修什么 | 落在哪 |
|---|---|---|
| ✅ | 逐项绑定：一个模块缺失只让该算子 `unbound`，不再清空整张表 | `ops/install.py` + `ops/families/`（已做，见 `MIGRATION_LOG.md` §4.13） |
| ✅ | provider 发现：`ASCEND_PLATFORM_OPS` 指向的包优先，其次声明式 entry point；描述符启动期校验版本区间 | `contracts/provider.py`（已做；算子仓的 `describe()` 已经被读到） |
| ✅ | 语义名词表 + `platform_name` 映射：`contracts/vocabulary.py` 是封闭集合，每个名字带 `kind`（`compute` / `class_factory`），类别不符在绑定期就拒 | `contracts/vocabulary.py` + `provider.ProviderOp.kind` + `gdn_moe._wrong_kind` |
| ✅ | 符号表由引擎层发布：`install_ops(engine_symbols=...)`，记录带 `platform_name`，与 `engine/abi` 不一致即报 drift 并拒绝绑定 | `ops/install.py::_targets_of` + `engine/plugin.py`（`tests/test_engine_abi.py`） |
| ✅ | vLLM ABI 适配上移：符号表与 `UnquantizedFusedMoEMethod` 的类替换都进了 `engine/abi/vllm_0_23.py`，由引擎层注入给 `install_ops(engine_factories=...)`；`unquantized_fused_moe` 现在能绑 | `engine/abi/vllm_0_23.py` + `engine/plugin.py::_engine_factories`（`tests/test_engine_abi.py`） |
| ✅ | 模型族要求改成"语义算子 + 强制/可选"两档：`ops/requirements.py` 声明每条要求的 `mandatory` 与理由；`report.servable` 给出判决，`doctor` 的绑定检查按它 FAIL | `ops/requirements.py` + `install.OpsReport.servable`（`tests/test_requirements.py`） |
| ✅ | 「`OPERATOR_ROOTS` 两份」是**重复声明**，不是行为分歧：`_acl.py` 那份只被分层测试按 `any(... for root in ...)` 读（顺序无关），活的那份在 `src/ascend_platform/contracts/ports.py:59`，被 `_candidates()` 消费——**顺序即优先级** | 已做：删掉 `_acl` 副本，测试改读端口那一份；`ports.py` 写明"顺序即优先级" |
| P2 | "每次调用选实现"的谓词下沉到算子库（它掌握形状与编译缓存），本仓只留总开关 | `ops/candidates.py` 的取舍 |

算子侧（AscendAttention）。判据同样一条：**算子仓可以出现算法签名，不可以出现引擎符号。**

| 优先级 | 修什么 | 落在哪 |
|---|---|---|
| P0 | MoE 计算算子（纯 torch 基线，`measured_ratio` 定义性 1.0）：路由 top-k、块对齐、专家计算、非量化专家方法。**不是补课**：P3 原本只承诺判决，今天升级为 P0 只因平台要绑的名字必须存在 | `ops/moe/` |
| P0 | provider 入口 + 描述符加 `platform_name` | `pyproject.toml` 的 entry point；`manifest.json` |
| P0 | 边界文字修正：MoE 计算/MLA/量化是算子，不是平台语义（本轮已改） | 算子仓 `README.md` |
| P0 | 未实现清单前置到 README，避免集成方按"应该支持"写代码（本轮已改） | 算子仓 `README.md` |
| P1 | 删掉 `vllm_adapter` 待办，改记为"由平台 `engine/abi/` 承担" | 算子仓 `docs/completion-audit.md` |

## 5. 目标契约（尚未实现，别按文档假定已存在）

三件东西，缺一不可：

| 件 | 内容 | 谁定 |
|---|---|---|
| 发现 | 一个 provider entry point（组名常量定义在 `contracts/provider.py`）；值是一个返回描述符、并能按语义名 `load()` 的工厂 | 算子仓声明 |
| 描述符 | 现有 schema（见 [`op-manifest.md`](../op-manifest.md)）**加一个 `platform_name`**：把本仓算子名映射到平台词表 | 算子仓发布，平台校验 |
| 绑定 | 引擎符号 → 语义名 → 实现对象；三种机制（`setattr` / `CustomOp.register_oot` / `torch.library`）藏在同一个接口后 | 本仓 |

四个状态必须在横幅与 `doctor` 里一致：`bound` / `unbound(原因)` / `declined(原因)` / `disabled(开关)`。
只有"模型族声明为强制"的算子缺失才拒绝启动——这条是"用户自带算子"能成立的前提。
