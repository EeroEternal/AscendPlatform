# 这些笔记去哪了

`src/` 的文档字符串里有若干引用指向源仓库（TileInfer）的笔记，例如
`docs/status.md`、`docs/performance.md`、`docs/plan-vllm-native-backend.md`、
`docs/notes/dsv41-facts.md`、`docs/notes/perf-comparison-rules.md`、
`docs/notes/dsv41-perf-plan.md`、`docs/notes/dsv41-s4-offload-plan.md`。

这些引用是**出处标注**，不是"本仓库还有这份文件"。它们的意思通常是：

> "这个常量 / 这个分支 / 这个 128 是被测出来的，测量记录在源仓库的某个文件里。"

保留它们的理由：一个没有出处的数字，下一个人只能选择相信或删掉。有出处，他可以去查。

迁移后这些文件仍在源仓库 `TileInfer/docs/`；本仓库只搬了与**运行期行为**直接相关的四份：

| 文件 | 为什么在这里 |
|---|---|
| [`hardware-checklist.md`](../hardware-checklist.md) | `ascend-platform doctor` 的清单就是它的可执行版本，两边必须同时改 |
| [`hardware-facts.md`](../hardware-facts.md) | 硬件与 vLLM 契约的逐条事实，附 `reference/` 里的源码副本可核对 |
| [`p5-gate.md`](p5-gate.md) | 上机门：logprobs 逐位一致的抓取、比对与失败分类；含 `float.hex()` 判据与 `TILEINFER_OPS=all/off` 对照 |
| [`op-coverage-qwen36.md`](op-coverage-qwen36.md) | 绑定表覆盖了 Qwen3.6 前向的哪些算子、哪些走回退、哪些 MISSING；`test_op_coverage_note.py` 拿它当账本核对 |

如果你在 `src/` 里看到一条指向别处的 `docs/...` 引用而本目录没有列出它，那是一个
**待办的出处迁移**，不是遗漏：等对应模块的测量在新仓库里重做之后，引用应指向新记录。
