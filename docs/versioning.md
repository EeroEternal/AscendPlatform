# 版本契约：不匹配就拒绝启动

> 平台与算子仓互相声明版本范围，**启动期**校验。不匹配不是警告，是拒绝。

---

## 1. 为什么要一次"拒绝启动"

搬运前的代码里，平台与算子仓同仓同版本，版本问题不存在。拆成两个仓之后，最常见的故障
会变成"装了一个能 import、但语义不同的算子包"——它在昇腾上通常不是崩溃，而是**数字错**。
所以这里的立场是：宁可启动失败并说清楚原因，也不要跑起来看起来正常。

## 2. 两侧声明

| 声明在哪 | 由谁写 | 内容 |
|---|---|---|
| `pyproject.toml` 的 `dependencies` / extras | 平台 | 引擎版本线（`vllm>=0.23,<0.24`） |
| `contracts/version.py::SUPPORTED_VLLM_RANGE` | 平台 | 同一件事的代码形式，便于运行期校验 |
| 算子仓 `manifest.json` 的 `library.requires_platform` | 算子仓 | 它能接受的平台范围 |
| `contracts/version.py::SUPPORTED_OPERATOR_RANGE` | 平台 | 平台能接受的**算子层版本**（导入根那一侧，不依赖 manifest） |

两侧都写，是因为它们回答的问题不同：`pyproject` 是**安装期**（pip 决定装不装），
`SUPPORTED_VLLM_RANGE` 是**运行期**（装上了也可能被人手动降级/升级）。

## 3. 范围语法

`VersionRange` 支持 `>=`、`>`、`<=`、`<`、`==`、`!=`，逗号是"与"：

```
">=0.23,<0.24"     # 0.23.x 全系列
">=0.1,<0.2"       # 平台 0.1.x
```

实现是 20 行正则 + 比较，不依赖 `packaging`——平台要能在只有标准库的机器上完成校验。
后缀（`rc1`、`.post4`、`.dev0`）不改变排序：`0.23.1rc1` 落在 `0.23.0 < x < 0.24` 之间。

**哪些写法被当作同一个版本线**（这条是说给上机的人听的：不要因为一个后缀去改常量的范围）：

| 写法 | 处置 | 为什么 |
|---|---|---|
| `0.23.0+cu123` / `0.1.0.dev0+git1234` | 接受（本地段被忽略） | 本地编译的 wheel、git 安装都会带 `+...`；它不改变发布线，为它拒绝启动是**假阴性** |
| `v0.23.0` | 接受 | 某些 tag 驱动的构建脚本会把 `v` 一起写进 `__version__` |
| `0.23.0rc1` / `0.23.0.dev0` / `0.23.0.post1` | 接受（后缀不参与比较） | 预发布版与同号正式版属于同一条线 |
| `0.23.0-1`（Debian 式修订）、`0.23.x`、`banana` | **拒绝**，但抛的是 `IncompatibleVersion` 而不是 `ValueError` | 两个调用点（`install_ops` 与 `doctor`）只捕获前者；解析不了的**已安装**版本必须表现为"不匹配"，否则它会以一个本仓缺陷的形式冒出来 |

注意最后一行与"常量写错"的区别：范围常量是本仓自己的代码，写错时仍抛 `InvalidVersion`（一个
`ValueError`）——那是**我们的** bug，不该被伪装成对方的版本问题。

## 4. 校验点

| 时机 | 代码 | 失败行为 |
|---|---|---|
| 安装 | `pyproject.toml` | pip 拒绝安装 |
| 进程启动（算子清单） | `check_compatibility()` | `IncompatibleVersion`，拒绝启动 |
| 进程启动（算子导入根） | `ports.check_operator_version()`（由 `install_ops()` 在第一条绑定之前调用） | `IncompatibleVersion`：**一条绑定都不做**，原因写进 `OpsReport.layer_error`，引擎退回 vLLM 自带算子 |
| `doctor` | `ascend-platform doctor` | 检查项 `ops.version` / `ops.binding` `FAIL`，exit 1 |
| 算子调用（兜底） | 端口 `operator_layer()` | `OperatorLayerUnavailable`，消息带候选与原因 |

`check_range(found, required, what=...)` 对 `found is None` 的处理是**失败**而不是放过：
"对方没声明版本"不等于"对方兼容"。

两条路都要校验，是因为它们**各自可能单独成立**：清单校验回答"这个算子库宣称支持我的平台吗"，
导入根校验回答"我 `import` 到的那个包，是不是我以为的那个版本"。只做前者，`ASCEND_PLATFORM_OPS`
指到一个旧版本目录时仍然全绿；只做后者，清单里声明的边界没人看。今天源仓 `tileinfer` 声明
`__version__ = "0.1.0.dev0"`，落在 `>=0.1,<0.2` 里；把同一个导入根换成 `0.3.0`，启动时会在第一条
绑定之前拒绝，并同时打印**装了什么**和**支持什么**：

```
operator layer 'tileinfer': version 0.3.0 is outside the supported range >=0.1,<0.2
```

## 5. 迁移期的双装陷阱

迁移期间同一个环境里可能同时装了源仓库与目标仓库，于是 vLLM 会看到**两个 out-of-tree
平台插件**，而 vLLM 的规则是"两个就是硬错误"。三种处理：

1. 测试套件：`tests/conftest.py` 默认 `VLLM_PLUGINS=ascend_platform`，让结果与会话里
   还装了什么无关；
2. 设备验收：只保留要测的那一个（`pip uninstall` 另一个），并在日志里记下当前装的是哪个；
3. 手工排查：`VLLM_PLUGINS=tileinfer` 显式指定另一个。

这一条不是"将来要处理的问题"，而是**现在就会撞到**的问题：装完 editable 包后
`src/*.egg-info` 本身就会让 vLLM 发现第二个同名发行版。所以 `*.egg-info/` 在 `.gitignore` 里，
测试前也不要把它留在 `src/` 下。

`doctor` 有一个 `plugins.unique` 检查专门盯这一条：它读的是**安装元数据**（不 import vLLM），
所以能在报错把自己伪装成算子问题之前就点出两个插件名，并给出 `VLLM_PLUGINS=ascend_platform`
（vLLM 自带的白名单，不必卸载）。

## 6. 升版本时改什么

| 场景 | 改哪里 | 别忘了 |
|---|---|---|
| 平台跟随 vLLM 新版本 | `SUPPORTED_VLLM_RANGE` + `pyproject` extra | `docs/hardware-facts.md` 里引用的 `reference/` 副本 |
| 平台自身破坏性变更 | `_version.py` 的 minor | 算子仓 manifest 的 `requires_platform` |
| 算子仓发布新版本 | 它自己的 manifest | 平台的 `SUPPORTED_OPERATOR_RANGE`——它是**执行中**的校验，不再是别名；改它等于宣布平台只能配这个区间的算子层 |
