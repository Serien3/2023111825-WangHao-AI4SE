# 实验五：针对 AI 生成代码的代码审查（泛化性测试）

实验二 / 四模型的**纯下游消费者**：不重训、不设计新 Prompt/上下文，只把已有模型与
条件搬到 AI 生成代码上，量化「人类 vs AI」性能差异并归因。设计规约见
`docs/specs/2026-07-08-exp5-ai-code-review-design.md`。

## 复用边界（正确性纪律）

- **实验二**（零改动）：`feature_extraction.extract_*`、`config.PRE/FULL_FEATURES`、
  saved 模型（`{svm,rf,xgb,lgbm}_{pre,full}.pkl`）、`scaler_{pre,full}.pkl`。
- **实验四**（零改动）：`context_builder`、`prompts`、`llm_client`、`run_experiments.run_condition`、
  `evaluate` 的指标函数。兄弟实验的 `src` 包由 `config.py` 以别名（`exp2src`/`exp4src`）载入。
- **TF-IDF**：实验二未落盘 vectorizer，本实验用 features.parquet 里的 50 词表构造
  `TfidfVectorizer(vocabulary=...)`，在**实验二人类语料**上 fit 以恢复一致 IDF，对 AI/对照文本
  **只 transform 不 fit**（自检确认 bit-level 重现，max diff 2e-16）。
- **LLM 缓存共享实验四** cache 目录：生成对照组复用实验四样本可直接命中；分类对照组为无泄露 test 匹配样本，未命中时会正常调用。

## 三个集合

| 集合 | 定义 | 规模 |
|---|---|---|
| AI 分类池 | AI PR 且含 python & has_patch 文件 | 253（187 merged / 66 non） |
| AI 生成池 | AI PR 且含 ≥1 顶层 inline comment | 72 |
| 匹配人类对照 | 按 AI 分类样本 repo×is_merged 分布，只从 Exp2 held-out test 集分层抽，避免 train/val 泄露 | 与 AI 抽样同量级 |

## Pipeline 与命令

所有命令在 `Experiment5/` 目录下运行。

```bash
cd Experiment5

# 0. 查看配置/矩阵
uv run python -m src.config

# 1. 抽样（AI 分类 50 / 生成全 72 / 匹配对照），固定种子落盘 results/samples/
uv run python -m src.sampling

# 2. ML 特征提取（AI + 对照）——瞬时、无成本，含 TF-IDF 自检
uv run python -m src.ml_features            # 全量
uv run python -m src.ml_features --limit 5  # 冒烟

# 3. ML 预测（实验二 4 模型 × pre/full，只 transform）——瞬时、无成本
uv run python -m src.ml_predict             # → predictions/ml_*.parquet, metrics/ml_metrics.json

# 4. LLM 编排（16 条件 × {ai, control} × {classify, generate}）——触网、有缓存
uv run python -m src.run_llm --task all --only C1 P1 --limit 3   # 冒烟（先验全链路）
uv run python -m src.run_llm --task classify --group ai          # 单组单任务
uv run python -m src.run_llm --task all                          # 全量（分类 16×50×2 + 生成 16×72+16×50）

# 5. 评估：LLM 逐组×条件指标 + human_vs_ai.json（人类锚点=实验四旧结果 + 实验二 test）
uv run python -m src.evaluate               # 末尾自动出图；--no-figures 只算指标

# 6. 分析：三条证据链（性能差值定位 / 上下文敏感度 / 错误案例归因）
uv run python -m src.analysis

# 7. 可视化：6 张图 → results/figures/
uv run python -m src.visualization
```

推荐顺序：`sampling → ml_features → ml_predict → run_llm(先 --limit 3 冒烟再全量)
→ evaluate → analysis → visualization`。

## 产物

```
results/
├── samples/      classify_ai / classify_control / generate_ai / generate_control .json
├── features/     ai_features.parquet, control_features.parquet（101 列，与实验二同 schema）
├── predictions/  ml_{group}_{fs}.parquet, {classify,generate}_predictions.parquet
├── metrics/      ml_metrics.json, llm_{task}_metrics.json, human_vs_ai.json,
│                 performance_gap_localization.json, context_sensitivity.json
├── cases/        error_cases.json, error_case_attribution_template.md
└── figures/      fig1..fig6 .png
```

## 成本与验证

- ML 侧瞬时、无成本。LLM 全量 ≈ 分类 16×50×2 + 生成 16×(72+50)；生成对照多数命中实验四缓存，分类对照因改为无泄露 test 匹配样本可能产生新增调用。
- 先 `--limit 3 --only C1 P1` 冒烟验证全链路（特征对齐、预测、解析、评估）再全量。
- 复现性：固定种子（42）+ 共享缓存 + 落盘抽样 key。
