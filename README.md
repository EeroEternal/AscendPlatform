# AscendPlatform

面向昇腾的 **out-of-tree vLLM 平台插件**：提供设备与执行语义，不提供算子数学。

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

## 与算子层仓库的关系

- 依赖方向：本仓库 → 算子层（可选）；**不依赖**它的寄生适配器
- 读取算子层导出的纯数据清单来制定跨来源策略；**库内变体分派属于算子层**
- **算子名字重绑定属于本仓库**（"如何替换引擎里的名字"是引擎知识，不是算子能力）
- 两仓互相声明版本兼容区间，启动期校验；不匹配时拒绝启动，不静默运行

## 支持的组合

| 组合 | 状态 |
|---|---|
| 本仓平台 + 算子层 | ✅ 目标形态（单卡 Qwen3.6；logprobs 逐位一致待上机） |
| 本仓平台 + 无算子层 | ⚠️ 能启动、能自检、能报缺什么；**第一次前向失败**（无厂商回退） |
| 厂商平台 + 厂商算子（`vllm_ascend`） | 基线，用于对照；与本仓**不可同装**（vLLM 只允许一个平台插件） |

## 非目标

算子数学、搬入厂商 kernel、MLA、量化、TP/EP、完整投机解码（先只做多 token 预测所需的最小批语义）。

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
  contracts/   契约：算子层端口（惰性）、版本范围、算子清单 schema
  device/      设备画像：一张卡一条冻结记录 + 注册表 + 探测
  profiles/    模型事实：头维度、页策略、混合层比例
  runtime/     运行时骨架：页几何、规划器、步进记账、计数器
  ops/         算子接入：名字 → 实现的绑定表、来源切换、候选集
  engine/      引擎：vLLM 平台/worker/图/注意力/模型适配、两个插件入口
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
