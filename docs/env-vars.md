# 环境变量：平台上机时真正会碰到的旋钮

> 每个变量都在代码里有一处 `os.environ.get(...)`。这份表存在的理由：**上机门失败时，你需要知道
> 有哪些开关能开**。之前它们只写在模块文档字符串里，而人在追一个 logprob 分叉时不会去翻 49 个文件。

两类，别混：

- **运行期开关**：决定跑哪条路径（算子接管、prefill 内核、图捕获、编译）。上机验收会用到。
- **观测开关**：只打印/计数，不改结果。定位用，默认全关（生产里保持关，否则日志量爆炸）。

`tests/test_env_vars.py` 双向检查这张表与代码：代码读的变量必须在表里，表里的变量必须有人读
（源仓遗留的变量除外，见 §3）。

---

## 1. 运行期开关

| 变量 | 默认 | 读它的地方 | 作用 |
|---|---|---|---|
| `TILEINFER_OPS` | `gdn,moe` | `ops/switch.py` | 哪些算子家族允许接管。`all` / `off` / 逗号列表。**每次调用重读**，所以关掉一个算子不需要重启 |
| `TILEINFER_OPS_CANDIDATES` | 空（不覆盖） | `ops/candidates.py` | `off` = 全局关闭**所有**非基线候选实现（进程级、逐调用重读） |
| `ASCEND_PLATFORM_OPS` | 空 → 依次试 `ascend_attention`、`tileinfer` | `contracts/ports.py` | 算子层的导入根，改变"哪一份算子层被绑定" |
| `ASCEND_PLATFORM_MODELS` | `tileinfer.vllm.models.dsv41:register_models` | `engine/model_providers.py` | 哪些包负责注册模型配置；格式 `pkg.mod:func`，逗号分隔。未安装的静默跳过 |
| `VLLM_DSV41` | 空 → Qwen3.6 平台类 | `engine/platform.py::register` | `1/true/yes/on` 选 DeepSeek-V4.1-Flash 子类（放行 fp8/fp4、关 hybrid KV）。**没设备时无论如何返回 `None`** |
| `TILEINFER_PREFILL` | 开 | `engine/attention/impl.py` | `0/off/false/no` 强制 prefill 走参考后端（同进程 A/B 的对照组） |
| `TILEINFER_ATTN_BACKEND` | `tilelang-ascend950` | `engine/attention/impl.py` | 注意力实现名；换名字等于选另一份实现（不存在则报错） |
| `TILEINFER_POOL_CAP` | 空（按模型算） | `runtime/buckets.py` | 钉住 KV 池页数上限。用于复现"池给大了/小了"的行为 |
| `TILEINFER_SPLIT_KV_PAGES` | `0`（关） | `engine/attention/impl.py` | split-KV 的页数；非 0 才启用。**注意它是策略选择，不是安全性**——编译病理见 `hardware-facts.md` |
| `TILEINFER_ACLGRAPH` | 关 | `engine/platform.py` | `1` 请求 decode 图捕获（前提是 `graphs.capture_path_ready()` 为真，否则只告警） |
| `TILEINFER_COMPILE` | `1`（开） | `engine/platform.py` | `0` 关 torch.compile 但保留图捕获 |
| `TILEINFER_ATTN_STEP_SHARE` | 关 | `engine/attention/impl.py` | 跨层共享同一步的注意力规划（省一次规划，不改语义） |
| `TILEINFER_DYNAMO_OPAQUE` | `1`（开） | `engine/attention/impl.py` | `0` 时让 dynamo 穿透算子调用（调试编译问题用；穿透后图会变碎） |
| `TILEINFER_VLLM_STRICT` | 关 | `engine/plugin.py` | `1` 时把补丁安装失败当启动失败（默认只告警：一个点版本的补丁失配不该让服务起不来） |
| `TILEINFER_VLLM_REGISTER_BACKEND` | `1`（开） | `engine/plugin.py` | `0` 不注册 `AttentionBackendEnum.CUSTOM` 槽位（平台钩子仍直接返回类路径，只有 `--attention-backend custom` 会失败） |
| `TILEINFER_VLLM_GRAPH_BRIDGE` | `1`（开） | `engine/plugin.py` | `0` 不装图桥接的补丁（用于确认"捕获判定"与"补丁"哪个出问题） |
| `VLLM_LOGGING_LEVEL` | `INFO` | `engine/plugin.py` | 本仓自建 handler 的级别（当进程里没有别的 handler 时）；`DEBUG` 会打开每层的绑定细节 |
| `VLLM_PLUGINS` | vLLM 自带白名单 | vLLM 本身（`doctor` 读它） | 同时装了多个 out-of-tree 平台插件时，指定只加载 `ascend_platform` |

## 2. 观测开关（默认全关）

| 变量 | 默认 | 读它的地方 | 输出 |
|---|---|---|---|
| `TILEINFER_PHASE_TIMING` | 空（关） | `runtime/phases.py` | `host` / `device` / `both` / `1`：每阶段耗时表 |
| `TILEINFER_PHASE_RESET_AFTER_S` | `0` | `runtime/phases.py` | 丢弃前 N 秒的样本（预热噪声） |
| `TILEINFER_PHASE_OPS` | 关 | `ops/install.py` | `detail` / `1`：每个算子绑定一行 |
| `TILEINFER_OP_COUNT` | 关 | `ops/install.py` | 每个接管目标的调用计数表（每 `TILEINFER_OP_COUNT_EVERY` 次打印一次，默认 2000） |
| `TILEINFER_OP_COUNT_EVERY` | `2000` | `ops/install.py` | 上一条的打印间隔 |
| `TILEINFER_PHASE_LEAF_LIMIT` | `600` | `engine/compat/patches.py` | 逐层计时的最大叶子模块数（防止超大模型把表撑爆） |
| `TILEINFER_PREFILL_TIMING` | 关 | `engine/attention/prefill.py` | `[tileinfer/prefill-timing]` 行：tiles/batch/pv1pass 统计 |
| `TILEINFER_PREFILL_KERNEL_BATCH` | `512`（下限 32） | `engine/attention/prefill.py` | prefill 内核的 batch 轴常量（影响编译键数量） |
| `TILEINFER_PREFILL_PV1PASS` | 关 | `engine/attention/prefill.py` | 单趟 PV 的 prefill 变体 |
| `TILEINFER_FLAG_DIR` | `/tmp/tileinfer_flag` | `runtime/flags.py` | 运行期开关目录（`<name>=1` / `<name>=0` 文件） |
| `TILEINFER_FLAG_WATCH` | 关 | `runtime/flags.py` | `1` 才真的读那个目录；关着时**不碰文件系统**（生产默认） |

## 3. 环境事实（只读，不是本仓的开关）

`doctor` 读它们来判断环境，不改变任何行为：`ASCEND_RT_VISIBLE_DEVICES`、`TILELANG_CACHE_DIR`、
`ASCEND_OPP_PATH`、`ASCEND_TOOLKIT_HOME`。

以下变量出现在文档里但**不是本仓的开关**，它们是源仓当时的现场记录（`hardware-facts.md` 的
"env vars in force"）或 vLLM 自己的开关，写在这里以免有人 export 了却不见效果：
`TILEINFER_ASCEND_TARGET`、`TILEINFER_SIMD_SOFTMAX`、`TILEINFER_DENSE_NN`、`TILEINFER_ALLOW_SPLIT_KV`、
`TILEINFER_SKIP_SPLIT_COMPILE_HANG`、`TILEINFER_METRICS_PATH`（探针路径，原生后端 v0 未接线）、
`VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE`、`VLLM_ENABLE_V1_MULTIPROCESSING`。

## 4. 上机门最可能用到的三个

定位"慢/不对"时，先开这三个，成本最低：

```bash
TILEINFER_PHASE_TIMING=host TILEINFER_OP_COUNT=1 TILEINFER_PREFILL_TIMING=1 <启动命令>
```

- `PHASE_TIMING=host`：每阶段耗时，回答"时间花在 prefill、decode 还是算子接管"；
- `OP_COUNT=1`：每个接管目标被调了多少次，回答"某层是不是根本没走到我们的实现"；
- `PREFILL_TIMING=1`：tiles/batch 统计，回答"prefill 是不是被 padding 成两倍工作量"。

三个都只打印，不改结果——所以可以开着跑 P5 的 A/B，只要两侧一致。
