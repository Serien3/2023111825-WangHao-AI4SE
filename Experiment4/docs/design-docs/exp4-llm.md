# 实验四 设计说明：基于大语言模型 + Prompt Engineering 的代码审查

> 本文档说明实验四的设计思路、上下文构建、Prompt 设计、评估协议与成本控制。
> 代码位于 `Experiment4/src/`，运行命令见 `Experiment4/src/README.md`。
> 数据来源：实验一 5 张表（`Experiment1/results/processed/`）+ 实验二划分（`Experiment2/results/models/split_pre.pkl`）。

---

## 0. 与前序实验的依赖关系（为什么不需要先做实验三）

指导书 4.6/4.7.1 写"读取实验三构建的人类代码数据集"。经核实，这里的"数据集"= **人类代码 PR 集合 + train/test 划分**，而非实验三的模型产物。这份数据**实验二已经落盘**：

| 实验四需要 | 已就位来源 | 核实结果 |
|---|---|---|
| 人类代码 PR 集合 | 实验二 `features.parquet`（1036 PR，`is_ai_code=False`） | ✅ |
| 同一 test 集（供 LLM/ML 严格对比） | 实验二 `split_pre.pkl` 的 208 条 | ✅ 可回溯 repo+number，merge 分布 134/74 |
| 多粒度上下文原料 | 实验一 `prs`(body)、`files`(patch)、`commits`(message) | ✅ test 208 条 body/head_sha 全有 |

因此实验四**并列消费**实验一/二的产物，与实验三**无串行依赖**。代价：实验四报告第 7 项"与实验三 DL 对比"暂缺一列，锚定到实验二传统 ML 对比，待实验三补做后无损回填（同一 test 集）。

---

## 1. 任务定位

实验四在整门课中是 **Merge Prediction 与 Review Comment Generation 的 LLM + Prompt Engineering 方案**，与实验二（传统 ML）、实验三（DL）形成 ML → DL → LLM 的完整对比链条。核心方法论：**不训练模型，只优化输入**——通过设计不同的代码上下文与 Prompt，量化二者对代码审查质量的影响。

**两个任务都做**（严格遵循指导书 4.4.4）：
- **Merge Prediction**（分类）：LLM 判断 PR 是否会被合并
- **Review Comment Generation**（生成）：LLM 根据代码改动生成审查意见

---

## 2. 实验矩阵（指导书 4.7.2 + 4.7.3 硬要求）

**4 种上下文 × 4 种 Prompt × 50 抽样**，两个任务各跑一遍。

### 2.1 上下文（4 种，递增粒度）

| 编号 | 上下文 | 组成 |
|---|---|---|
| C1 | 仅 Diff | 拼接该 PR 所有 Python 文件的 patch（截断到 token 预算内） |
| C2 | Diff + PR 描述 | C1 + `prs.title` + `prs.body` |
| C3 | Diff + Commit Message | C1 + 该 PR 所有 `commits.message` |
| C4 | Diff + 全部 | C1 + PR 描述 + Commit Message（"Diff + 等其他信息"的最丰富组合） |

### 2.2 Prompt（4 种，指导书 4.7.3）

| 编号 | Prompt 策略 | 关键设计 |
|---|---|---|
| P1 | Zero-shot | 直接给任务指令，无示例 |
| P2 | Few-shot | 注入 2–3 个来自**实验二 train 集**的示例（防泄漏，见 §5） |
| P3 | Chain-of-Thought | 要求"先逐步分析、最后一行给结论" |
| P4 | Role-based | 设定角色（资深 reviewer / maintainer），再给任务 |

### 2.3 抽样（老师规约 N=50）

- **分类**：从实验二 test 208 条按 `repo × is_merged` 分层抽 **50** 条。
- **生成**：只有 70 条 test PR 有 ground-truth inline comment（BLEU/ROUGE 必需）。从这 70 条按 repo 分层抽 **50** 条。
- 两个任务的 50 条是**不同集合**（分类 50 ⊂ 208；生成 50 ⊂ 70），由数据本质决定（生成任务必须有人类审查意见作参照）。
- 抽样固定随机种子，落盘 `results/samples/` 供复现。

**调用量级**：每任务 4×4×50 = 800 次，两任务共 **1600 次** DeepSeek 调用（few-shot/CoT token 更多但次数不变）。

---

## 3. 模型与调用层

- **模型**：DeepSeek 在线 API，OpenAI 兼容 SDK，`base_url=https://api.deepseek.com`，默认 `model=deepseek-chat`（V3，快/省的一档，即"flash"档；实现时确认最终 id）。
- **Key**：`.env` 新增 `DEEPSEEK_API_KEY=...`（git-ignored，与既有 `GITHUB_TOKEN` 同文件）。
- **调用封装**（`llm_client.py`，唯一触网模块，仿实验一 `github_client.py`）：
  - 统一 `chat(messages, temperature, max_tokens)` 接口
  - 失败指数退避重试（最多 3 次），记录每次调用的 `latency`（供指导书要求的"推理时间"指标）
  - **响应缓存**：以 `(task, context, prompt, pr_key)` 为 key，结果 JSON 落盘 `results/cache/`；重跑跳过已完成调用（断点续传 + 省钱）
  - 分类用 `temperature=0`（判定要稳定），生成用 `temperature=0.3`

---

## 4. 分类输出解析（决策 D）

LLM 输出自由文本，需稳健解析为 merge/reject：

- Prompt 强制结尾输出 `DECISION: MERGE` 或 `DECISION: REJECT`（CoT 则"先推理，最后一行给 DECISION"）。
- 正则提取最后一个 `DECISION:` 标签。
- 解析失败重试 1 次；再失败记为**弃权**，计入 `parse_error_rate`（作为该条件的质量指标之一，不强行猜标签）。

---

## 5. Few-shot 示例来源（决策 E，防泄漏）

- Few-shot 示例**仅从实验二 train 集（662 条）选取**，绝不碰 test 的 50 条。
- 分类：选正/负例各 1–2 个（含其 diff 摘要 + 真实 `is_merged` + 一句理由）。
- 生成：从 train 集里挑 2 个优质 `(diff_hunk → 人类审查意见)` 对。
- 示例固定、跨所有 few-shot 调用复用，落盘 `results/samples/fewshot_*.json`。

---

## 6. 评估协议（指导书 4.7.5）

### 6.1 分类指标（对 50 条真值 `is_merged`）
- Accuracy / Precision / Recall / F1（正类=merged）
- 平均推理时间（每条件）
- `parse_error_rate`（弃权比例）
- 16 个条件（4 上下文 × 4 Prompt）各一组指标

### 6.2 生成指标（对 50 条人类审查意见）
- **BLEU**（sacrebleu，语料级 BLEU-4）
- **ROUGE**（rouge-score，ROUGE-1/2/L F-measure）
- 平均推理时间
- Ground-truth = 每 PR **首条顶层 inline comment**（时间最早、非回复），一 PR 一样本（决策已确认）
- 16 个条件各一组指标

### 6.3 对比锚点
- 分类：LLM 16 条件 vs 实验二 SVM/RF/XGB/LGBM（pre-review，同一 test 全集指标已存 `Experiment2/results/metrics/test_metrics_pre.json`）。注意 LLM 在 50 子集、ML 在 208 全集，报告需注明；如需严格同集，可用实验二模型在这 50 条上重新 predict（低成本，建议实现）。
- 生成：实验四内部横向对比（上下文间、Prompt 间）；与实验三 DL 对比暂缺（见 §0）。

---

## 7. 目录结构（仿实验二约定）

```
Experiment4/
├── src/
│   ├── __init__.py
│   ├── README.md              # pipeline + 命令
│   ├── config.py             # 路径、模型 id、矩阵定义、采样参数、种子
│   ├── sampling.py           # 从 Exp2 split 分层抽 50（两任务各一份）
│   ├── context_builder.py    # 4 种上下文拼装 + token 预算截断
│   ├── prompts.py            # 4 种 Prompt 模板（× 2 任务）
│   ├── llm_client.py         # DeepSeek 调用 + 重试 + 缓存 + 计时（唯一触网）
│   ├── run_experiments.py    # 主循环：16 条件 × 50 × 2 任务，写 predictions
│   └── evaluate.py           # 分类指标 + BLEU/ROUGE + 图表
├── results/
│   ├── samples/              # 抽样 PR key + few-shot 示例
│   ├── cache/                # 每次调用的原始响应（断点续传）
│   ├── predictions/          # 结构化预测结果（parquet）
│   ├── metrics/              # 各条件指标 JSON
│   └── figures/              # 对比热力图 / 柱状图
├── docs/design-docs/exp4-llm.md   # 本文档
└── report/                   # 实验报告
```

---

## 8. 图表清单（对应报告第 5、6 项）

| 图 | 内容 |
|---|---|
| 分类性能热力图 | 4 上下文 × 4 Prompt 的 F1（分类） |
| 生成性能热力图 | 4 上下文 × 4 Prompt 的 BLEU / ROUGE-L（生成） |
| 上下文对比柱状 | 固定 Prompt，4 上下文的指标对比（两任务） |
| Prompt 对比柱状 | 固定上下文，4 Prompt 的指标对比（两任务） |
| LLM vs 传统 ML | 分类：最佳 LLM 条件 vs 实验二四模型 |
| 推理时间对比 | 各条件平均 latency |

---

## 9. 成本与资源

- 1600 次调用，DeepSeek-chat 输入/输出单价极低，全矩阵预计成本 **数元人民币**。
- 缓存保证任何中断/重跑不重复付费。
- 无本地 GPU 需求（纯 API），8GB 显存/7.6GB RAM 约束不触发。
- 建议先 `--limit 3` 冒烟（每条件 3 条），验证解析/评估链路，再全量。

---

## 10. 指导书思考题预答（4.9）

1. **上下文为何影响推理？** 代码审查依赖改动之外的意图信息；描述/commit 提供"为什么改"，缩小 LLM 的歧义空间。
2. **Few-shot 优于 Zero-shot？** 示例锚定输出格式与判定尺度，减少 LLM 对任务的误解，尤其稳定 `DECISION` 解析。
3. **哪类上下文对 Merge Prediction 贡献最大？** 预期 C2（+PR 描述）增益明显——呼应实验二发现"引用上下文的 PR 合并率高"。以实测为准。
4. **Prompt 为何影响生成质量？** CoT/Role 改变推理路径与语气，影响审查意见的具体性与相关性（BLEU/ROUGE 敏感）。
5. **LLM vs DL 的优劣？** LLM 免训练、上下文强、可解释；但成本高、延迟大、输出需解析、结果有随机性。

---

## 11. 高质量标准

1. 16 条件 × 2 任务全部跑通并落盘，指标可复现（固定种子 + 缓存）。
2. 上下文/Prompt 的对比呈现清晰趋势，能支撑思考题回答。
3. 分类可与实验二严格对比（建议实现"同 50 子集"重评）。
4. 生成 BLEU/ROUGE 计算规范（sacrebleu 语料级 + rouge-score）。
5. 严格防泄漏（few-shot 只取 train 集）、严格控成本（缓存 + 冒烟）。
6. 尊重实验边界：不做 Repository 级/跨文件上下文（那是实验六）。
</content>
</invoke>
