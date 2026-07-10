# 仓库速览（Repo Orientation）

> 给进入本仓库的 agent：读完这份就知道有什么数据、能复用什么、已经做到哪。
> 详细规范见 `CLAUDE.md`；各实验设计见 `docs/specs/` 与各 `Experiment*/docs/`。

## 项目一句话

智能软件工程课程项目，主题=**代码审查**。7 个实验组成一条流水线，贯穿两个任务：
**Merge Prediction**（分类，PR 是否被合并）与 **Review Comment Generation**（生成审查意见）。
实验一产出共享数据基座，后续实验依次消费。

## 环境

- `uv` 管理依赖（Python 3.12）。`.env` 需 `GITHUB_TOKEN`、`DEEPSEEK_API_KEY`（均 git-ignored）。
- 运行：`cd ExperimentN && uv run python -m src.<module>`。

## 进度

| 实验 | 状态 | 内容 |
|---|---|---|
| Exp1 | ✅ 完成 | 采集+清洗+EDA GitHub 代码审查数据集（数据基座） |
| Exp2 | ✅ 完成（已定稿，勿改） | 传统 ML（SVM/RF/XGB/LGBM）做 Merge Prediction |
| Exp3 | ⏭️ 跳过 | DL（Code2Vec/CodeBERT）—— 本项目不做 |
| Exp4 | ✅ 完成 | LLM + Prompt Engineering，两个任务的 16 条件网格 |
| Exp5 | 🚧 设计完成 | AI 生成代码的代码审查（复用 Exp2+Exp4 模型做泛化测试）见 `docs/specs/2026-07-08-exp5-*.md` |
| Exp6/7 | 未开始 | 上下文/Prompt 改进；VSCode 插件 |

## 共享数据（实验一产出）

路径 `Experiment1/results/processed/`，5 张表（`.parquet` + `.csv`）：

| 表 | 行数 | 关键列 |
|---|---|---|
| `prs` | 1706 | repo, number, title, body, is_merged, additions/deletions, num_reviews/reviewers/review_comments, base_sha/head_sha, **is_ai_code**, ai_code_signals |
| `files` | — | repo, pr_number, filename, status, language, has_patch, **patch**, previous_filename |
| `commits` | — | repo, pr_number, message |
| `reviews` | — | repo, pr_number, ... |
| `review_comments` | — | repo, pr_number, comment_id, path, **diff_hunk**, **body**, in_reply_to_id, created_at |

join 键：`prs.(repo,number)` ↔ 其他表 `(repo,pr_number)`。

### 关键数据事实
- **人类 vs AI**：`is_ai_code` 分布 = 人类 1412 / AI 294。AI 识别信号主要是 commit 尾注
  `co-authored-by: claude/copilot/cursor` + `ai slop`/`ready for maintainer review` 标签。
- **5 个仓库**：pandas-dev/pandas、home-assistant/core、apache/airflow、huggingface/transformers、django/django。
  各仓库合并率差异大（48%–90%），存在分布偏移——分析务必分仓库看。
- **AI PR 子集**：
  - 含 Python patch（可做 ML 特征）：**253** 条（187 merged / 66 non-merged，合并率 74%）。
    仓库分布 pandas 135 / home-assistant 52 / airflow 47 / transformers 19，**django 缺席**。
  - 含 ≥1 inline comment（生成任务真值）：**72** 条。
- **TF-IDF 词表未单独落盘**，但 50 个 `tfidf_*` 列名保存在 `Experiment2/results/features/features.parquet`，可恢复。

## 可复用资产

**Exp2（`Experiment2/src/`）**
- `feature_extraction.py`：`extract_ast_features` / `extract_cfg_features` /
  `extract_code_features_for_pr` / `extract_statistical_features` / `extract_text_features` /
  `reconstruct_post_change_code`；`build_feature_matrix()`（98 维，**内部写死过滤 is_ai_code==False，且 fit TF-IDF 后丢弃 vectorizer**）。
- `config.py`：`PRE_REVIEW_FEATURES`(93) / `FULL_FEATURES`(98)。
- `results/models/`：`{svm,rf,xgboost,lightgbm}_{pre,full}.pkl`、`scaler_{pre,full}.pkl`、`split_{pre,full}.pkl`。
- 复用纪律：下游用模型时，词表/scaler/权重一律**只 transform 不 fit**，否则特征列语义错位、预测无效。

**Exp4（`Experiment4/src/`）**
- `context_builder.py`（4 种上下文 C1–C4）、`prompts.py`（4 种 Prompt P1–P4）、
  `llm_client.py`（DeepSeek，唯一触网，含重试+计时+**缓存** `results/cache/`）、
  `run_experiments.py`、`ml_baseline.py`、`evaluate.py`（分类指标 + BLEU/ROUGE）、`sampling.py`。
- 矩阵：4 上下文 × 4 Prompt × 两任务；分类从 Exp2 test 抽 50，生成从 70 条真值抽 50。
- 生成真值口径：每 PR **首条顶层 inline comment**（最早、非回复），一 PR 一样本。
- 缓存 key = `(task, context, prompt, pr_key)`，重跑跳过已完成调用（断点续传 + 省钱）。

## 常见坑

- 运行 `uv run` 前确认当前目录：模块以 `cd ExperimentN && uv run python -m src.X` 方式跑。
- `read_parquet` 路径：Exp2 特征在 `results/features/features.parquet`（多一层 `features/`）。
- 复用 Exp2 模型预测新数据：必须沿用其词表 + scaler，禁止在新数据上 re-fit。
- 资源约束：WSL2 内存 ~7.6GB，Exp2 已做 OOM 防护（限制并行 worker/线程），大规模并行需注意。
