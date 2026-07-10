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
| Exp5 | ✅ 完成 | AI 生成代码泛化测试（复用 Exp2+Exp4 模型）。代码 `Experiment5/src/`，报告 `Experiment5/report/实验五结果分析.md` |
| Exp6 | 🚧 下一步 | 在 Exp5 AI 数据上改进上下文/Prompt（只优化输入，不改模型）。指导书 `Experiment1/docs/实验指导书_代码审查.md` §6 |
| Exp7 | 未开始 | VSCode 插件，集成 Exp2–6 模型 |

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

**Exp5（`Experiment5/src/`，纯下游适配层，零改动复用 Exp2/Exp4）**
- `config.py` 把兄弟实验 `src` 以别名 `exp2src` / `exp4src` 载入；LLM 缓存目录**指向 Exp4 cache**（共享，对照命中不重复付费）。
- `data.py`：AI 分类池 253 / AI 生成池 72 / 匹配人类对照。**对照只从 Exp2 held-out test 抽样**（`matched_human_control`），按 AI 的 `repo×is_merged` 分布匹配——早期从全人类池抽会 84% 命中 train/val，是效度陷阱，已堵。
- `ml_features.py`：复用 Exp2 `extract_*` + 用 features.parquet 里 50 词表构造 `TfidfVectorizer(vocabulary=...)`，在 Exp2 人类语料上 fit 恢复 IDF，对 AI/对照**只 transform**（自检 bit-level 重现，max diff 2e-16）。
- `report/实验五结果分析.md`：对齐指导书 5.9 五道思考题的完整结果分析。

## Exp5 关键结论（Exp6 的出发点）

- **ML 泛化确实下降**：pre-review F1，树模型在 AI 上 0.58–0.63 vs 人类 test 0.80+、无泄露对照 0.77–0.86，掉 0.17–0.23，非分布假象。
- **⚠️ 两个"AI 更好"是类别偏向假象，别被骗**：AI 分类池合并率 **74%**（正类多）。LLM 分类 AI 侧 F1（0.84–0.86）高于人类，但 **recall≈1.0 / precision≈0.75**——模型几乎"全判 merge"，撞上高正类率才虚高，判别力其实弱。ML 的 SVM 同理（AI recall 0.94）。**评 AI 分类务必看 precision/recall，别只看 F1/Accuracy。**
- **BLEU/ROUGE 在生成任务无区分度**：BLEU≈0.5–1.5（/100）、ROUGE-L≈0.04–0.06，人类/AI 互有高低。表面 n-gram 匹配不适合评开放式审查意见 → Exp6 需要语义/有用性层面的评价视角。
- **上下文敏感度（指导书假设"AI 更依赖完整上下文"）**：分类 C4−C1 增益 AI≈0（假设不成立）；生成 C4−C1 AI +0.005 而人类 −0.005（微弱成立）。**当前 C1–C4 上下文太弱**（只有 diff/PR 描述/commit），未触及仓库级/跨文件/Issue 上下文——这正是 Exp6 要补的。
- 错误案例：AI 假阳性 PR 多带 GH# 引用+测试+详尽注释，"表面完整度高"易骗过判别器；且"未合并≠代码差"（流程性拒绝是标签噪声）。

## Exp6 先验（改进 AI 代码审查，重点优化输入而非模型）

- **数据直接读 Exp5 的 AI 池**：分类 `Experiment5/results/samples/classify_ai.json`(50)、生成 `generate_ai.json`(72)；对照/人类锚点沿用 Exp5 口径。不要重新采集或重新识别 AI。
- **复用 Exp4 `llm_client`（共享缓存目录）**：新上下文/Prompt 只要改变 messages，语义 key 的 content_hash 会变，不会误命中旧缓存；未变的直接命中省钱。
- **Exp6 = 在 Exp5 基线上做上下文增强 + Prompt 优化**（指导书 6.4）。当前缺口即改进方向：
  - 上下文：加**修改前后完整代码 / 修改函数所在文件 / 调用关系 / Issue 描述 / 历史审查意见 / 仓库级上下文**（Exp4 的 C1–C4 只到 diff+PR描述+commit）。⚠️ Exp1 只存了 patch 不存完整文件，仓库级上下文需回仓库抓或用 base_sha/head_sha——这是 Exp6 的主要新增工作量。
  - Prompt：Exp4 已有 Zero-shot/Few-shot/CoT/Role，Exp6 需补 **Self-Reflection、多轮交互式**。
- **对比基线**：Exp6 的核心交付是"改进后 vs Exp5 基线"的性能对比，指标沿用 Acc/P/R/F1 + BLEU/ROUGE + 推理时间。**报告务必用 precision/recall 而非仅 F1**（见上"类别偏向假象"）。
- 生成任务真值口径与 Exp4/5 逐字一致：每 PR 首条顶层 inline comment。

## 常见坑

- 运行 `uv run` 前确认当前目录：模块以 `cd ExperimentN && uv run python -m src.X` 方式跑。
- `read_parquet` 路径：Exp2 特征在 `results/features/features.parquet`（多一层 `features/`）。
- 复用 Exp2 模型预测新数据：必须沿用其词表 + scaler，禁止在新数据上 re-fit。
- 资源约束：WSL2 内存 ~7.6GB，Exp2 已做 OOM 防护（限制并行 worker/线程），大规模并行需注意。
