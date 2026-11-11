# P5 门：单卡 logprobs 逐位一致怎么跑

**这条门要证明什么.** 迁移只搬了平台层（设备与执行语义），算子数学没动。于是有一个可以证伪的断言：
同一张卡、同一份权重、同一个 prompt、同一次贪心解码，`ascend_platform` 跑出来的 logprobs 必须与
`tileinfer` 的**每一位**相同（`float` 全等，不是"差在第 4 位以内"）。

**为什么不用 4 位小数.** `docs/hardware-checklist.md` §7 拿 4 位小数对冻结基线，那是给"首次点亮"
用的——它容忍数值路径差一点点。P5 要区分的是"语义等价"和"几乎等价"：如果 head layout 或 KV stride
错了，输出会看起来正常而 logprob 差在 1e-3；如果只是编译顺序变了，差在 1e-16。4 位小数把这两种
混成一类，逐位相等才把它们分开。

---

## 0. 两个仓不能同时在同一进程里

vLLM 拒绝两个 out-of-tree 平台插件（`Refusing to load two platform plugins`），所以：

1. 先跑 TileInfer，把 logprobs 落盘；
2. 换 venv（或至少换 `VLLM_PLUGINS` / `PYTHONPATH`），再跑 `ascend-platform`，落盘；
3. 用一个**只依赖标准库**的脚本比对两个 JSON。

不要在同一个 shell 里 `pip install -e` 两个包——那会让插件发现阶段直接失败，而且失败信息看起来像
本仓库的 bug。

## 1. 固定不变的东西（两侧必须一致，否则比对无意义）

| 项 | 要求 | 怎么核 |
| --- | --- | --- |
| 权重 | 同一份 checkpoint、同一 dtype（bf16） | `sha256sum` 一份大文件即可 |
| vLLM | 0.23.0（两侧同一个 wheel） | `python -c "import vllm; print(vllm.__version__)"` |
| CANN / torch_npu | 同一次 source 出来的环境 | `ASCEND_HOME_PATH`、`pip show torch_npu` |
| 采样 | `temperature=0`、`top_p=1`、`max_tokens>=16`、`seed` 固定 | 请求体，落盘时一起存 |
| 缓存目录 | `TILELANG_CACHE_DIR` 两侧各自独立（不要共用，否则一侧的编译产物会污染另一侧的 kernel 选择） | `env | grep TILELANG` |

`max_tokens` 至少 16：第一个 token 只用 prefill，decode 的页表映射、state 更新、MoE 路由都要多步才
暴露。反过来也不要太长——比对失败时你要能一眼看出从第几个 token 开始分叉。

## 2. 抓取（两侧各跑一遍，只改 `$LABEL`）

```bash
LABEL=tileinfer          # 第二遍改成 ascend-platform
OUT=/tmp/p5-$LABEL.json

curl -sS http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.6-35B-A3B","prompt":"The capital of France is","max_tokens":16,
       "temperature":0,"top_p":1,"seed":0,"logprobs":5}' \
  > "$OUT"

python3 -c "import json,sys; d=json.load(open('$OUT')); print(len(d['choices'][0]['logprobs']['tokens']), 'tokens')"
```

记下两侧服务启动日志里的 `ascend-platform vLLM plugin` / `TileInfer vLLM plugin` 行与 ops 绑定摘要
（同一个开关状态），一并存进 `/tmp/p5-$LABEL.meta`。

## 3. 比对：逐位，不是"接近"

```python
# /tmp/p5_compare.py  — 只用标准库，故意不 import 任何一个仓
import json, struct, sys

def flat(path):
    """[(token, token_logprob.hex(), ((alt, alt.hex()), ...)), ...] —— 逐位可比的形状。"""
    d = json.load(open(path))
    lp = d["choices"][0]["logprobs"]
    out = []
    for i, tok in enumerate(lp["tokens"]):
        alts = (lp.get("top_logprobs") or [])[i] or {}
        out.append((
            tok,
            float(lp["token_logprobs"][i]).hex(),
            tuple(sorted((k, float(v).hex()) for k, v in alts.items())),
        ))
    return out

a, b = flat(sys.argv[1]), flat(sys.argv[2])
ok = True
for i, (x, y) in enumerate(zip(a, b)):
    if x != y:
        ok = False
        print(f"第 {i} 个 token 分叉")
        print(f"  tileinfer      : {x}")
        print(f"  ascend-platform: {y}")
        break
if len(a) != len(b):
    ok = False
    print(f"长度不同: {len(a)} vs {len(b)}")
print("PASS: 逐位一致" if ok else "FAIL")
sys.exit(0 if ok else 1)
```

> **抓取那一段也有测试**（`test_the_runbooks_capture_block_fetches_and_counts`）：测试起一个假的
> `/v1/completions`，把下面第 2 步的 bash 代码块原样抽出来、换掉端口跑一遍，断言落盘 JSON 的
> `tokens` 与打印的 "N tokens" 都对得上。所以多行单引号 JSON、`python3 -c` 回读这两处不会在上机时
> 才暴露。请求体的四个采样参数（`temperature=0` / `top_p=1` / `logprobs=5` / `max_tokens>=16`）
> 也逐条断言——它们是这条门成立的前提。
>
> 判据本身有测试：[`tests/test_p5_gate_comparator.py`](../../tests/test_p5_gate_comparator.py)
> 用合成 payload 钉住了同一套语义（逐位相同 PASS、差 1 ulp FAIL、前缀相同但长度不同 FAIL、
> top-k 分布不同 FAIL）。**其中四条是把下面这个代码块原样抽出来、当子进程跑**——所以这段脚本
> 不会"文档里写着、测试测的是另一份实现"。另外两条在装了 vLLM 的环境里**对着 vLLM 0.23 的
> `CompletionLogProbs` schema 核对**判据读的三个键（`tokens` / `token_logprobs` / `top_logprobs`）
> ——引擎换个点版本把它们改名，这条门会先红，而不是在卡上比了个空。上机时如果它报 PASS 而你不信，先跑
> `pytest tests/test_p5_gate_comparator.py`。

`float.hex()` 是关键：`json` 把 `logprob` 序列化成 17 位有效数字，Python 解析回来是**同一个**
`float`，所以 `hex()` 相等 ⇔ 位模式相等。用 `round(x, 4)` 或字符串比较会把"差 1 ulp"和"差 1e-3"
混在一起。

## 4. 第二半：`TILEINFER_OPS=all` vs `off`

这一条比第 3 步更能定位问题：算子接管全开与全关，**平台侧**的输出必须一致——不一致说明某个接管的
语义与它的基线不等价（这是算子仓的 bug，但由这条门抓出来）。

```bash
# 同一个 venv，只改环境变量，服务各起一次，按第 2/3 步各抓一份
TILEINFER_OPS=all <serve>   -> /tmp/p5-all.json
TILEINFER_OPS=off <serve>   -> /tmp/p5-off.json
```

- `all` vs `off` 一致 → 接管是可替换的，算子层"透明"成立；
- 不一致 → 先看 `ascend-platform doctor --json` 的 `ops.binding` 段，确认两次运行的绑定集合就是开关
  要求的那个，再逐个 family 二分。

`off` 那一次会明显更慢（torch 回退），这是预期的，不要当成失败。

## 5. 记录表（**已填：2026-11-11 上机通过**）

| 项 | 结果 |
| --- | --- |
| 卡 / CANN / torch_npu / vLLM / tilelang | 1 × Ascend950PR_9579 / 9.3.0 / 2.13.0rc1 / 0.23.0 / 0.1.15 |
| 模型 | `/home/models/Qwen3.6-35B-A3B`（`Qwen3_5MoeForConditionalGeneration`，bf16，67 GB） |
| TileInfer 侧 logprobs 落盘 | `/tmp/p5-B.json`（16 token, `temperature=0, top_p=1, logprobs=5`） |
| `ascend-platform` 侧落盘 | `/tmp/p5-A.json`（默认 ops）、`/tmp/p5-off.json`（`TILEINFER_OPS=off`）、`/tmp/p5-all.json`（`all`） |
| **逐位一致（第 3 步）** | ✅ **PASS** —— 16/16 token、16 个采样 logprob 的 `float.hex()` 全部相同，16 步 `top_logprobs` 也逐项相同 |
| `TILEINFER_OPS=all` vs `off`（第 4 步） | ✅ **PASS** —— 与默认档、与源仓两侧都逐位相同 |
| `ascend-platform doctor` 退出码 | **0**——按它自己给的解法设好 `VLLM_PLUGINS=ascend_platform` 与 `TILELANG_CACHE_DIR` 后：`15 check(s): 15 PASS, 0 FAIL (0 hard, 0 soft), 0 SKIP`。裸 shell 里则是 12 PASS / 3 FAIL，三条都带着修法（`tilelang_cache` 未设、`plugins.unique` 双装、`ops.binding` 在 vLLM 进程外测得 0 bound），照着修就全绿——这本身是 doctor 的验收 |
| 备注 | 见下面的"上机时踩到的四件事" |

### 5.1 四跑一表（同一个 prompt，全部 `float.hex()` 相等）

| 跑 | 平台 | `TILEINFER_OPS` | 前 3 个 token | 前 3 个 logprob |
| --- | --- | --- | --- | --- |
| B（源仓） | `tileinfer` | 默认 | ` Paris`, `,`, ` a` | -0.533167, -0.657468, -0.824303 |
| A（本仓） | `ascend_platform` | 默认（`gdn,moe`） | 同上 | 同上 |
| off（本仓） | `ascend_platform` | `off` | 同上 | 同上 |
| all（本仓） | `ascend_platform` | `all` | 同上 | 同上 |

结论：**迁移没有改变数值，算子接管是否打开也没有改变数值**——这正是这条门要证明的两件事。

### 5.2 上机时踩到的四件事（都写回手册了）

1. **vLLM 0.23 默认用 fork 起 EngineCore，NPU 不能在被 fork 的子进程里重新初始化**：
   `RuntimeError: Cannot re-initialize NPU in forked subprocess. To use NPU with multiprocessing,
   you must use the 'spawn' start method`。两个可用解法：`VLLM_ENABLE_V1_MULTIPROCESSING=0`
   （单进程，本门用的这个）或 `VLLM_WORKER_MULTIPROC_METHOD=spawn`。
2. **`--max-model-len` 必须 ≥ 2048**（hybrid KV 的页大小）。取 1024 时 decode 预规划会报
   `the layer's KV pool has a 2048-token page but cache_config.block_size is 1024` 并跳过预规划——
   服务仍能跑，但每个 bucket 的首次 decode 会懒编译，而懒编译不允许发生在图捕获里。
3. **`gpu_memory_utilization` 要给权重留够**：35B bf16 权重 65.5 GB，0.55 × 131 GB 时
   KV 预算被算成 **-3.31 GiB**，平台的报错会把账目摊开（这是它该做的事）；0.85 时 KV 池
   683,212 token。
4. **横幅以前在 serve 日志里根本看不到**（已修，见 `MIGRATION_LOG.md` §4.11）：vLLM 的
   `dictConfig` 只给 `vllm` 这一个 logger 装 handler 且 `propagate=False`，第三方 logger 的 INFO
   被直接丢掉。修复后实测日志里有
   `INFO ... [ascend_platform.engine.plugin] ascend-platform vLLM plugin installed | vLLM 0.23.0 |
   strict=False | patches: 3 applied, 0 skipped (target gone), 0 failed | ...`——手册第 2 步那条
   `grep "ascend-platform vLLM plugin"` 现在真的有输出（修复前：0 行）。

## 6. 失败时先看哪三处

| 症状 | 最可能的原因 | 去哪核对 |
| --- | --- | --- |
| 第 0 个 token 就分叉 | prefill 路径（页表展开、tile 规划、RaggedMetadata） | `runtime/buckets.py::expand_page_table`、`engine/attention/prefill.py`；对比 `test_vllm_prefill_path` 的期望 |
| 前几个 token 相同，之后发散 | KV stride / 物理页序（`(1,0,2,3,4)`）或 state 页偏移 | `engine/attention/impl.py::get_kv_cache_shape` + `docs/hardware-facts.md` |
| 只在 `all` 与 `off` 之间不同 | 某个接管的语义与基线不等价 | `doctor --json` 的 `ops.binding`；按 family 二分 |
| 差 1 ulp（只有末尾位不同） | 归约顺序/编译选项，不是语义 | 记录，不阻塞；但要写进 `hardware-facts.md` |
