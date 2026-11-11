# 设备画像：一张卡就是一条记录

> 目标：**加第二张卡 = 新增一个模块 + 一行 import**（`device/__init__.py` 里），平台其余代码零改动；并且"哪个数字是硬件事实、哪个是策略"
> 在类型上就能分开。

---

## 1. 三个概念，别混

| 概念 | 类型 | 回答什么问题 | 谁拥有 |
|---|---|---|---|
| 设备 | `DeviceProfile` | 这台机器是什么 | `device/` |
| 模型事实 | `ModelProfile` | 这个 checkpoint 要什么 | `profiles/` |
| 引擎机制 | worker / graphs / platform | 怎么把请求跑完 | `engine/` |

三者在搬运前的代码里是混的：`platform.py` 里有核数，`worker.py` 里有页大小，`graphs.py` 里
有"哪些 `torch.cuda` 名字有 NPU 对应物"。这种混法只有一张卡时无害，加第二张卡时无法回答
"改哪一行"。

## 2. 字段与出处

`src/ascend_platform/device/profile.py::DeviceProfile` 是冻结 dataclass。参考卡
（Ascend950PR）的值在 `device/ascend950.py`：

| 字段 | 值 | 怎么来的 |
|---|---|---|
| `hbm_bytes` | 131072 MiB | `torch.npu.get_device_properties(0).total_memory` |
| `compute_units` | 56 | `multi_processor_count`；vLLM 问的就是这个数 |
| `cube_units` / `vector_units` | 28 / 56 | `cube_core_num` / `vector_core_num` |
| `kernel_page_size` | 128 | 算子页 `BC`；引擎页必须是它的整数倍 |
| `engine_page_multiple` | 2048 | 混合 KV 池要求的引擎页 |
| `state_page_bytes` | 4194304 | 线性注意力状态页，**补齐到一个注意力页** |
| `dtypes` | bf16 / fp16 | 已编译的 kernel 变体 |
| `head_sizes` | 128 / 256 | 已编译的 head_dim |
| `min_gqa_group` | 2 | group=1 会读越界 |
| `capabilities` | graph_capture / hybrid_kv / mamba_state | 平台据此做启动期拒绝 |
| `name_patterns` | `950` | 探测时匹配 `torch.npu.get_device_name()` |

`state_page_bytes` 为什么必须正好一个注意力页：vLLM 在混合 KV 场景下会比较注意力组与状态组
的页大小，不相等会在启动期报错，而报错信息不会告诉你"是页大小的问题"。
`tests/test_device_profile.py::test_the_state_page_is_exactly_one_attention_page` 钉住这个等式。

## 3. 校验：注册即失败

`DeviceProfile.validate()` 拒绝：

- `kernel_page_size` 不是 2 的幂（内核按 2 的幂切页）；
- `engine_page_multiple` 不是 `kernel_page_size` 的整数倍（KV 池无法零拷贝重看）；
- 没有声明任何计算 dtype / head 尺寸；
- `min_gqa_group < 1`。

`register_device_profile()` 在**注册时**调用它，所以一条坏记录进不了注册表，
`tests/test_device_profile.py::test_registration_validates_before_storing` 断言了这点。

## 4. 探测：不猜数字

`detect_device_profile()` 的顺序：

1. 显式传入 `device_name`；
2. 否则 `torch.npu.get_device_name(0)`（失败就当没有设备）；
3. 按 `name_patterns` 匹配已注册的画像；
4. 匹配不到 → 返回一份**以 `DEFAULT_DEVICE_PROFILE` 为底**、名字改成实际卡名、`evidence`
   写明 `unprofiled` 的记录。

第 4 步是设计选择：宁可给一份"页几何已知、性能数字未验证"的记录，也不假装
"未知卡 = 参考卡"。装第二张卡的人看到的第一条信息应该是"这台机器还没有画像"，
而不是"跑起来看起来正常，数字全错"。

## 5. 加第二张卡

```python
# src/ascend_platform/device/<your_card>.py   (one file per card)
from .profile import DeviceProfile, register_device_profile

ASCEND960 = DeviceProfile(
    name="Ascend960",
    hbm_bytes=...,
    compute_units=...,
    kernel_page_size=128,
    engine_page_multiple=2048,
    state_page_bytes=...,
    dtypes=("bfloat16", "float16"),
    head_sizes=(128, 256),
    capabilities=("graph_capture", "hybrid_kv", "mamba_state"),
    name_patterns=("960",),
    evidence="measured on ... with torch_npu ...",
).validate()

register_device_profile(ASCEND960)
```

然后在 `device/__init__.py` 里 import 一次（它的副作用就是注册）。平台其余代码不需要动：
它们读 `device_profile(...)` 或探测结果，不读字面量。

## 6. 平台用画像做什么

| 用处 | 谁读 | 现状 |
|---|---|---|
| KV 池按 kernel page 重看 | `engine/attention/impl.py` | 已用 `runtime/buckets.py` 的常量（迁移期保留） |
| 混合 KV 页倍数认定 | `engine/worker.py` | 同上 |
| 能力拒绝（图捕获 / pin memory） | `engine/platform.py` | 迁移期仍读本地判定 |
| 版本/环境前置条件 | `doctor/cli.py` | 逐步改为遍历 `profile.env` |

迁移期的原则是**行为不变**：`DeviceProfile` 先作为"可校验的事实表"落地，读点逐个替换；
每替换一个读点，对应测试必须仍然绿。最终目标是没有一处硬件常量以字面量形式出现在 `engine/`。

### 两类数字，别混

字段看着都是"硬件常量"，其实分两类，处置方式不同：

| 类别 | 例子 | 怎么读 | 会不会随卡变 |
|---|---|---|---|
| **卡的事实** | `hbm_bytes`、`compute_units`、`dtypes`、`capabilities` | 运行期 `device_profile()` 查表 | 会。第二张卡就是新增一个模块 + 一行 import |
| **kernel 几何** | `kernel_page_size=128`、`engine_page_multiple=2048` | 记录在 profile 里，**断言相等** | 不会。它由算子编译决定，不是卡决定 |

对第二类做"按卡查表"是假间接：任何跑这套 kernel 的卡都是 128。所以它的防线不是替换读点，而是
**防漂移**——`tests/test_device_profile.py` 里两条断言把三处字面量钉在一起：

```
runtime/buckets.py::KERNEL_PAGE_SIZE
        == DeviceProfile.kernel_page_size                     （128）
engine/platform.py::HYBRID_BLOCK_SIZE
        == active_profile().block_size_policy.required_multiple(hybrid=True)
        == DeviceProfile.engine_page_multiple                 （2048）
```

改其中任何一处而不同步另外两处，测试立刻红——这比"全部改成查表"更能防住真正会犯的错
（把 sub-page 宽度从 128 改成 64 而忘了 kernel 的 BC tile）。
