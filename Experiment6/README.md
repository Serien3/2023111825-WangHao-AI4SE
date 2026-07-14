# 实验六：改进针对大模型生成代码的代码审查

在**只优化输入（上下文 + Prompt）、不改模型**的前提下，用假设驱动的
「**上下文阶梯 L0–L4 + Prompt 消融（P5/P6）**」弄清 AI 代码审查性能下降的真正来源，
并**修复两个失明指标**证明改进是否有效。设计规约见
`docs/specs/2026-07-10-exp6-ai-code-review-improvement-design.md`。

本实验是实验四/五的**下游消费者**：不重训、不重新采集数据、不重新识别 AI 标签、
不做 RAG/向量检索/clone 整仓库。

## 做了什么

| 维度 | 实验四/五 | 实验六的改进 |
|---|---|---|
| **上下文** | C1–C4 全是 diff 级浅上下文 | **单调递增 5 级深上下文阶梯** L0→L4（L2 拉改前后完整函数、L3 加 Issue+历史审查、L4 仓库级轻量词法检索） |
| **Prompt** | P1–P4（Zero/Few/CoT/Role） | 新增 **P5 Self-Reflection**（强制先列 blocking 缺陷再定论）与 **P6 多轮交互式** |
| **分类指标** | accuracy / merge-F1（失明：LLM 近乎全判 merge 也好看） | **per-class P/R + 混淆矩阵 + balanced accuracy**（头条），non-merge recall 作 Prompt 生效判据 |
| **生成指标** | BLEU/ROUGE（对开放式意见无区分度） | **LLM-judge（deepseek-v4-pro）多维打分**：relevance/actionable/correct/hallucination；BLEU/ROUGE 仅保留可比 |
| **对照** | — | 关键格（L4×P*、L4×P5）跑 matched_human_control，回答"改进后 AI 是否追平人类" |

**P\* 选取（R11，不硬编码）**：`select_pstar.py` 从实验五 AI 池预测指标经验选取旧最优
prompt——分类 `P*=P1`（均值 acc 最高）、生成 `P*=P3`（均值 BLEU 最高），结果落
`results/metrics/pstar_selection.json`，与 `config.PSTAR` 一致性自检。

## 实验矩阵（R12，非全网格）

每个任务（classify / generate）跑 10 个条件：

- **消融①（上下文阶梯）**：`{L0,L1,L2,L3,L4} × P*` —— AI 组（答"L4 是否弥合差距"）
- **消融②（Prompt）**：`L4 × {P*, P5, P6}` —— AI 组（P6 仅 L4，成本控制）
- **归因格**：`L0 × P5` —— AI 组（拆"上下文 vs prompt"功劳）
- **人类对照**：`L4×P*`、`L4×P5` 两格跑 control 组

`uv run python -m src.config` 打印完整矩阵。

## 复用边界（正确性纪律）

- **实验四（改造一处 = 接缝②）**：`llm_client.chat/_call_api/chat_semantic` 增加
  `model` 参数并把 **content_hash 纳入 model 维度**。`model=None` 时 hash 用
  `config.MODEL_ID`，与实验四/五既有行为**逐字一致**（回归，缓存不受影响）；裁判传
  `model="deepseek-v4-pro"` 与执行调用天然不同 key，互不覆盖。
- **零改动复用**：实验四 `context_builder`（C1/C4 块）、`prompts`（P1–P4）、
  `run_experiments.parse_decision/_extract_comment`、`evaluate._generation_metrics`；
  兄弟实验 `src` 由 `config.py` 以别名 `exp2src`/`exp4src` 载入。
- **LLM 缓存共享实验四** cache 目录；上下文/Prompt 改变 messages → content_hash 变 →
  不误命中旧缓存（R21）。
- **API 拉取全部落盘缓存**（`results/fetch_cache/`，R7）：文件@sha / issue / code search /
  目录列举，重跑命中不触网、失败也缓存以免反复重试（可复现）。

## 目录结构与产物

```
Experiment6/
├── src/
│   ├── config.py            # 路径、别名加载、L0–L4、P*/P5/P6、R12 矩阵、token 预算
│   ├── select_pstar.py      # R11：P* 选取（从实验五指标经验选取，可复现）
│   ├── github_fetch.py      # L2–L4 唯一触网层（复用实验一 GitHubClient）+ 落盘缓存
│   ├── data.py              # 复用实验四 data + 函数抽取/issue 引用/历史审查 join
│   ├── context_builder.py   # 接缝①：build_context(repo, number, level) L0–L4
│   ├── prompts.py           # P1–P4 复用 + P5 Self-Reflection + P6 多轮
│   ├── judge.py             # 接缝③：LLM-judge 多维打分（deepseek-v4-pro）
│   ├── run_experiments.py   # 主循环：遍历 R12 矩阵 → 执行模型 → 解析 →（生成额外裁判）
│   ├── evaluate.py          # per-class P/R+混淆矩阵+balanced acc / judge 聚合 / 基线 Δ
│   ├── visualization.py     # 5 张图对齐 6.9 五问
│   └── tests.py             # 接缝①②③ + 指标 + 解析兼容（离线桩，19 断言）
└── results/
    ├── fetch_cache/         # API 拉取缓存（文件/issue/检索）
    ├── predictions/         # {classify,generate}_predictions.parquet（每行一次调用）
    ├── metrics/             # classify_metrics / generate_metrics / exp5_baseline_delta
    │                        #   / pstar_selection.json
    └── figures/             # fig1_context_ladder … fig5_human_control.png
```

**指标产物字段**：
- `classify_metrics.json`：每条件含 `balanced_accuracy`（头条）、`per_class.{non_merge,merge}.{precision,recall,f1}`、
  `confusion_matrix.{tn,fp,fn,tp}`、`non_merge_recall`、`accuracy`、`merge_f1`（降级为参考）、`parse_error_rate`。
- `generate_metrics.json`：每条件含 `judge_{relevance,actionable,correct,hallucination}` 均值、
  `judge_n`、以及可比用的 `bleu`/`rouge1,2,L`。
- `exp5_baseline_delta.json`：与实验五 AI 基线（L0→C1、L1→C4 对应格）的改进 Δ。

## 如何运行

所有命令在 `Experiment6/` 目录下，用 uv（Python 3.12）。需 `.env` 内
`GITHUB_TOKEN=...`（L2–L4 拉取）与 `DEEPSEEK_API_KEY=...`（执行 + 裁判）。

```bash
cd Experiment6

# 0. 查看配置与实验矩阵
uv run python -m src.config

# 1. 复现 P* 选取（写 results/metrics/pstar_selection.json）
uv run python -m src.select_pstar

# 2. 离线测试（不触网、不花钱；接缝①②③ + 指标 + 解析兼容，19 断言）
uv run python -m src.tests

# 3.（可选）全链路冒烟（R13）：1 条样本跑通 L0→L4 + P5/P6 + 裁判
#    会触发 GitHub API（缓存）与 DeepSeek 执行 + 裁判调用
uv run python -m src.run_experiments --task classify --only L0 P1 --limit 1
uv run python -m src.run_experiments --task generate --only L4 P5 --limit 1

# 4. 全量运行 R12 矩阵（触网、有缓存、幂等重跑）
uv run python -m src.run_experiments --task classify     # 分类 10 条件
uv run python -m src.run_experiments --task generate     # 生成 10 条件（含裁判）
uv run python -m src.run_experiments --task all          # 两任务
#   常用开关：--only LEVEL PROMPT 单条件；--limit N 每条件前 N 样本；--no-judge 跳过裁判

# 5. 评估：修复失明指标 + 与实验五基线对比 Δ，末尾自动出图
uv run python -m src.evaluate                # --no-figures 只算指标；--task 选任务

# 6.（可选）单独出图
uv run python -m src.visualization
```

**推荐顺序**：先 `tests`（确认正确性）→ `run_experiments --only … --limit 1`（冒烟省钱）→
确认缓存命中与产物无异常后再全量 → `evaluate`。

## 结论叙事（对齐指导书 6.9 五问）

1. **L4 为何有用** → 图1 上下文阶梯增益曲线（balanced acc / non-merge recall / judge relevance）。
2. **哪种 Prompt 最适合** → 图2 Prompt 消融（预期 P5 拉高 non-merge recall）。
3. **Prompt vs 上下文各解决什么** → 图3 归因格 `L0×P5` vs `L4×P*` vs `L4×P5`。
4. **是否适用所有任务** → 分类可能仍受"标签噪声"天花板限制（未合并≠代码差，存在流程性拒绝），
   生成受益更明显。
5. **未来方向** → RAG / Agent / 工具调用（承接 6.10 与实验七）。

**已知威胁（写入报告）**：① 标签噪声是洞察不是 bug——部分 non-merge 是流程性拒绝，需用错误案例
区分"代码质量判断"与"合并结果预测"，不以硬提 merge-F1 为目标；② 裁判自偏——pro 评 flash 规避
"自评"但仍属同厂模型，列为 threat to validity。
