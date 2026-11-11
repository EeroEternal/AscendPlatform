# 算子清单（manifest）：算子仓对平台说的话，全是数据

> 平台要在启动期回答两个问题：**这个名字现在由谁实现**、**它能不能进图**。
> 这两个问题必须在**没有装算子包、没有 NPU** 的进程里也能回答——所以答案是 JSON，不是 `import`。

---

## 1. 为什么是数据

如果平台 `import` 算子包才知道有什么算子，那么：

- 算子包的依赖（tilelang、torch_npu）会成为平台的依赖；
- "这个算子能不能进图"只能在第一次 decode 时才知道，那时错误已经变成错误数字；
- 第三方想加一个算子，必须让平台代码认识它。

manifest 把这三件事都变成"读一个文件 + 校验"。

## 2. 形状

```json
{
  "schema": 1,
  "library": {
    "name": "ascend-attention",
    "version": "0.3.0",
    "requires_platform": ">=0.1,<0.2"
  },
  "ops": [
    {
      "name": "paged_decode_attention",
      "shapes": {"head_dim": [128, 256], "group": ">=2", "page": [128]},
      "dtypes": ["bfloat16", "float16"],
      "kv_layouts": ["ND"],
      "graph_capturable": true,
      "measured_ratio": 1.42,
      "bench": "benchmarks/bench_attention.py"
    }
  ]
}
```

字段语义：

| 字段 | 含义 | 平台怎么用 |
|---|---|---|
| `schema` | 清单格式版本 | 不认识就拒绝启动 |
| `library.requires_platform` | 本库要求的平台版本范围 | 启动期校验，不匹配拒绝启动 |
| `ops[].shapes` | 形状约束（离散值列表，或 `">=2"` 这类表达式） | 规划期判定能否服务 |
| `ops[].dtypes` | 已编译的 dtype | 与设备画像求交 |
| `ops[].kv_layouts` | 能读的 KV 布局 | 与设备画像求交 |
| `ops[].graph_capturable` | **能否进图** | 决定该算子是否允许出现在捕获区 |
| `ops[].measured_ratio` | 相对基线的实测比值 | 人读；也是准入门槛 |
| `ops[].bench` | 产出该比值的脚本 | 复核用 |

`MANIFEST_SCHEMA`（`contracts/manifest.py`）是同一份东西的 JSON-Schema 形式，随包发布，
第三方可以不 import 本包就校验自己写的清单。

## 3. 两条硬规则

### 规则一：没有 `measured_ratio` + `bench` 的算子不能出现在清单里

> 仓库纪律：没有测量就没有主线位置。

校验在 `parse_manifest()` 里，错误信息原文是
`needs both measured_ratio and bench. An implementation without a recorded measurement may not be
advertised (repo rule: no ratio, no mainline)`。

这条规则的现实动机：搬运前的代码里出现过"实现了、默认开了、没人测"的算子，结果是线上
性能回退找不到责任人。把"测过"变成准入条件，回退就不会以"默默变慢"的形式出现。

### 规则二：`graph_capturable` 必须显式声明

不能省、不能是 `null`。理由是这个平台的具体失效模式：不可捕获的算子出现在捕获区里，
**不会报错，只会算错**。缺省值在这里是有害的——所以"没填"被当作错误，而不是当作 `false`。

## 4. 启动期兼容性检查

```python
manifest = load_manifest(path)
check_compatibility(manifest, platform_version=__version__, device=device_profile(...))
```

检查三件事：

1. `library.requires_platform` 是否包含当前平台版本（否则 `IncompatibleVersion`）；
2. 每个算子的 `dtypes` 与设备 `dtypes` 是否有交集；
3. 每个算子的 `kv_layouts` 与设备 `layouts` 是否有交集。

任何一条不满足都是**启动期失败**，消息里带算子名和具体集合。这不是洁癖：错配的 dtype
在昇腾上通常表现为"跑完了、数字不对"，定位成本以小时计。

两个异常类型的边界，是为了让"谁的数据坏了"一眼可见：

| 坏在哪 | 抛什么 | 谁该去修 |
|---|---|---|
| JSON 不合法、缺字段、`requires_platform` 不是可读的范围、`measured_ratio` 缺失或 ≤0 | `ManifestError`（`parse_manifest`，加载时） | 算子仓 |
| 清单本身合法，但要求的平台版本不包含本机 / dtype 或 layout 不匹配 | `IncompatibleVersion` 或 `ManifestError`（`check_compatibility`，启动时） | 装错版本的人，或需要升级本仓 |
| 平台自己的范围常量写错 | `InvalidVersion`（`ValueError`） | **本仓**——这是开发期 bug，不该被伪装成对方的版本问题 |

`requires_platform` 在**加载时**就校验一次语法，所以一个写了 `~=0.1` 的清单不会活到启动阶段才炸。

## 5. 算子仓怎么发布

1. 把清单放在包的根目录（例如 `ascend_attention/manifest.json`）；
2. 在 `pyproject.toml` 里把它当 package data 打进去；
3. 版本号在同一次提交里同时更新 `library.version` 和 `library.requires_platform`。

平台侧不需要知道清单在哪个文件里——`load_manifest(path)` 接受路径，`doctor` 打印检查结果。

## 5.1 算子仓怎么"报名"：`describe()` + entry point

清单是数据，数据需要一个入口。算子包在**包根**暴露一个 `describe()`，返回的映射里除了上面的
`library` / `ops`，还要给每个算子一个 `target`（`"包.模块:属性"`）——平台据此拿到可调用对象：

```python
# ascend_attention/__init__.py
def describe():
    return {"name": "ascend_attention", "version": "0.1.0",
            "requires_platform": ">=0.1,<0.2",
            "ops": {"moe.experts": {"target": "ascend_attention.ops.moe.compute:fused_experts"}}}
```

平台按两种顺序找它（[`contracts/provider.py`](../src/ascend_platform/contracts/provider.py)）：

1. `ASCEND_PLATFORM_OPS` 指向的包——"我把平台指向了这个包"是明确指令，优先；
2. `ascend_platform.providers` entry point 组——已安装的发行版自己声明。

**算子名是语义名**（`moe.experts`、`moe.routing_topk_softmax`），不是 `vllm._custom_ops.silu_and_mul`
那种引擎符号：把引擎符号写进算子仓，等于让引擎版本决定算子的名字。引擎符号到语义名的映射属于
本仓（`engine/`），`_LAYOUTS` 的路径只是**旧层兜底**，不是契约。

坏描述符、版本区间不匹配、`target` import 不到——都只让**那一个算子**变成 `unbound` 并带上原因，
启动继续；原因会出现在横幅、`doctor` 与 `install_ops().details[<op>]` 里。

## 6. 与端口的关系

| 关心的事 | 用 manifest | 用端口（`contracts/ports.py`） |
|---|---|---|
| 有哪些算子、约束是什么 | ✅ | |
| 谁能不能进图 | ✅ | |
| 拿到**实现对象**去调用 | | ✅ |
| 算子包内部布局不同（`attention.api` vs `api`） | | ✅ `register_operator_layout` |

一句话：**清单用来决定，端口用来调用**。清单永远不携带可执行对象。
