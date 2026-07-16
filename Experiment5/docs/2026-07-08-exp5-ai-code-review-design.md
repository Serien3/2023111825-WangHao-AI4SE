# 实验五 设计说明：针对 AI 生成代码的代码审查（泛化性测试）

> 本文档定义实验五的设计、复用边界、实验矩阵与分析产物。
> 数据来源：实验一 5 张表（`Experiment1/results/processed/`）+ 实验二模型/划分（`Experiment2/results/models/`、`Experiment2/results/features/`）+ 实验四流水线（`Experiment4/src/`）。
> **本次范围 = 实验 + 结果 + 分析（含可视化），不含报告写作。** 分析产物以图表 + 结构化指标 JSON + 案例归因文件形式落盘。

---

## 0. 任务定位与边界

实验五是实验二 / 四模型的**纯下游消费者**：不重训、不设计新 Prompt/上下文，只把已有模型与条件搬到 AI 生成代码上，量化「人类 vs AI」的性能差异并归因。

- 跳过实验三 → **Merge Prediction** 用 实验二 ML + 实验四 LLM；**Review Comment Generation** 仅 LLM（无 DL）。
- 核心纪律：凡从训练数据学出的东西（TF-IDF 词表、scaler、模型权重、few-shot 示例）一律沿用上游，**只 transform 不 fit**。
- 尊重实验边界：不做仓库级/跨文件上下文增强（那是实验六）。

---

## 1. 数据与三个集合

实验一 `prs` 表含 294 条 `is_ai_code=True` 的 PR（主要靠 commit 的 `co-authored-by: claude/copilot/cursor` 尾注 + `ai slop`/`ready for maintainer review` 标签识别）。

| 集合 | 定义 | 规模 | 分布 |
|---|---|---|---|
| AI 分类池 | AI PR 且含 `language==python & has_patch` 文件 | 253 | 187 merged / 66 non-merged（合并率 74%）；4 仓库（pandas 135 / home-assistant 52 / airflow 47 / transformers 19，django 缺席） |
| AI 生成池 | AI PR 且含 ≥1 条 inline comment（真值必需） | 72 | 覆盖 5 仓库 |
| 匹配人类对照组 | 从人类池按 AI 分类池的 `repo × is_merged` 分布分层抽样 | 与 AI 抽样同量级 | 用于控制变量消融 |

**混杂因素**：AI 集合并率（74%）高于人类（64%），且仓库分布偏斜。匹配人类对照组用于堵住"性能差异其实来自仓库/类别分布"的效度漏洞。

---

## 2. 实验矩阵

| 任务 | 模型 | AI 侧规模 | 对照锚点 |
|---|---|---|---|
| Merge Prediction (ML) | 实验二 SVM/RF/XGB/LGBM × {pre, full} | 全部 253 | 实验二人类 test 旧指标 + 匹配对照重预测 |
| Merge Prediction (LLM) | 实验四 16 条件（4 上下文 × 4 Prompt） | 分层抽 50 | 实验四人类 50 旧结果 + 匹配对照重跑 |
| Review Comment Gen (LLM) | 实验四 16 条件 | 全部 72 | 实验四人类 50 旧结果 + 匹配对照重跑 |

- ML 主轴 **pre-review**（无泄漏、真实可部署），附带 **full** 消融（观察审查过程特征的泄漏增益在 AI 上是否变化）。
- LLM 分类抽 50 与实验四人类侧严格对齐；生成用全 72（>人类 50，语料级 BLEU/ROUGE 更稳）。
- LLM 生成真值 = 每 PR **首条顶层 inline comment**（时间最早、非回复），一 PR 一样本，与实验四逐字一致。
- 抽样按 `repo × is_merged` 分层、固定随机种子、落盘 `results/samples/`。

**调用量级**：分类 16×50=800 + 生成 16×72=1152 + 匹配对照增量（大部分命中缓存）。

---

## 3. 关键正确性约束（ML 特征）

实验二 `build_feature_matrix` 把「过滤人类」与「fit TF-IDF」焊死在一起，TF-IDF vectorizer 未落盘。但 50 个 `tfidf_*` 列名保存在 `Experiment2/results/features/features.parquet` 中，可恢复。

AI 特征提取必须：

1. import 复用实验二 `extract_ast_features` / `extract_cfg_features` / `extract_code_features_for_pr` / `extract_statistical_features` / `extract_text_features`，逐 PR 抽 AST/CFG/统计/文本特征。
2. TF-IDF 用从 `features.parquet` 读回的 50 词构造 `TfidfVectorizer(vocabulary=...)`，对 AI 文本只 `transform`，**绝不 fit**（重新 fit 会改变列语义，使预测无效）。
3. 按实验二 `PRE_REVIEW_FEATURES` / `FULL_FEATURES` 对齐列序。
4. 套实验二 `scaler_pre.pkl` / `scaler_full.pkl` 做标准化（只 `transform` 不 `fit`）。

实验二代码零改动；实验五做成纯下游适配层。

---

## 4. 复用与新增代码

**复用（零改动）**
- 实验二：`feature_extraction.extract_*` 函数、`config.PRE_REVIEW_FEATURES/FULL_FEATURES`、saved models、scalers。
- 实验四：`context_builder`、`prompts`、`llm_client`（含缓存 + 重试 + 计时）、`evaluate`（分类指标 + BLEU/ROUGE）。
- **LLM 缓存共享实验四 cache 目录** → 匹配对照组里凡实验四已跑过的 PR 直接命中，不重复付费；未命中的才新调用。

**新增薄封装**
- AI 特征提取（固定词表 transform + scaler transform + 列序对齐）
- AI / 对照抽样
- ML 预测 + 指标
- LLM 编排（16 条件 × {AI, 匹配对照}）
- 跨实验指标对比
- 分析 + 可视化

---

## 5. 目录结构（仿实验二 / 四约定）

```
Experiment5/
├── src/
│   ├── __init__.py
│   ├── README.md            # pipeline + 命令
│   ├── config.py            # 路径、AI 筛选、抽样参数、种子、矩阵定义（引用 Exp4）
│   ├── data.py              # 载 Exp1 数据、筛 AI 池、构造匹配人类对照组
│   ├── ml_features.py       # 复用 Exp2 extract_* + 固定词表 transform + scaler transform + 列序对齐
│   ├── ml_predict.py        # Exp2 saved models 对 AI + 对照预测，算指标
│   ├── sampling.py          # AI 分类抽 50 / 生成全 72 / 匹配对照抽样，落盘
│   ├── run_llm.py           # 复用 Exp4 pipeline，跑 16 条件 × {AI, 对照}，共享缓存
│   ├── evaluate.py          # 复用 Exp4 评估逻辑，算 ML+LLM 指标，输出人类vsAI 对比 JSON
│   ├── analysis.py          # 三条证据链：性能差值定位、上下文敏感度、错误案例抽取归因
│   └── visualization.py     # 图表
├── results/
│   ├── samples/             # AI 抽样 + 匹配人类对照 key（固定种子复现）
│   ├── features/            # AI + 对照的 ML 特征矩阵
│   ├── cache/               # LLM 调用缓存（共享/指向 Exp4 cache）
│   ├── predictions/         # ML + LLM 结构化预测（parquet）
│   ├── metrics/             # 各条件指标 + human_vs_ai 对比 JSON
│   ├── cases/               # 错误案例 + 人工归因（结构化）
│   └── figures/             # 可视化
└── docs/design-docs/        # 设计说明
```

---

## 6. 分析产物（三条证据链，全部数据驱动）

1. **性能差值定位**：人类 vs AI 逐指标下降幅度（ML pre/full、LLM 逐条件 + 匹配对照），量化"下降多少、在哪类条件下"，落 `metrics/human_vs_ai.json` + 对比图。
2. **上下文敏感度对比**：LLM 在 AI 代码上 C1(仅 diff)→C4(全上下文) 的增益 vs 人类，直接验证指导书"AI 代码更依赖完整上下文" → 折线/柱状图。
3. **错误案例归因**：从解析失败 / 误判 / 生成跑偏样本挑 3–5 例，人工归因（幻觉、局部对整体错、风格统一致信号弱等），落 `cases/`。

三条链覆盖指导书思考题 2、3、5，且全部复用已要跑的数据。

---

## 7. 图表清单

| 图 | 内容 |
|---|---|
| 人类 vs AI 分类性能对比 | ML 四模型 pre/full 在人类 test / AI / 匹配对照上的 Acc/F1 |
| AI 侧 16 条件热力图 | 4 上下文 × 4 Prompt 的 F1（分类）、BLEU/ROUGE-L（生成） |
| 上下文敏感度对比 | C1→C4 增益，人类 vs AI 叠加 |
| 匹配对照消融 | AI vs 匹配人类对照，证明差异非分布假象 |
| ML 泄漏增益变化 | pre vs full 的 Δ，人类 vs AI |
| 错误案例分布 | 误判 / 解析失败 / 生成跑偏的类型分布 |

---

## 8. 成本与验证

- LLM ≈ 800 + 1152 + 对照增量（多数命中缓存），几元级、缓存兜底。
- ML 侧瞬时、无成本。
- 先 `--limit 3` 冒烟验证全链路（特征对齐、预测、解析、评估）再全量。
- 复现性：固定种子 + 缓存 + 落盘抽样 key。

---

## 9. 对指导书思考题预答（5.9）

1. **AI vs 人类代码本质区别**：风格统一、重复多、局部正确整体可能错、易幻觉、更依赖上下文 —— 以本实验错误案例与上下文敏感度数据佐证。
2. **为何已有模型性能下降**：训练分布（人类代码）与 AI 代码分布偏移；局部 diff 信号在 AI 代码上更弱 —— 以人类 vs AI 差值 + 匹配对照消融量化。
3. **哪类模型泛化更好**：以 ML vs LLM 在 AI 上的相对下降幅度实测判断（预期 LLM 上下文理解强、下降更小，以数据为准）。
4. **是否需要新评价指标**：结合生成任务 BLEU/ROUGE 在 AI 上的表现讨论表面匹配指标的局限。
5. **AI 特有的审查挑战**：由错误案例归因直接产出。
