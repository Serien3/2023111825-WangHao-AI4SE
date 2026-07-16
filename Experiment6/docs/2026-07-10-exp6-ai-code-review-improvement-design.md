# 实验六 设计规约：改进针对大模型生成代码的代码审查

> 本文件是实验六的**唯一权威设计依据**,后续开发一律参照执行。
> 生成日期 2026-07-10。上游依据:`Experiment1/docs/实验指导书_代码审查.md` §6、`docs/repo-orientation.md`、
> `Experiment5/report/实验五结果分析.md`。

---

## Problem Statement

实验五证明:已有代码审查方法(传统 ML、LLM)在 **AI 生成代码**上性能下降,而且现有评估被两个"失明"掩盖了真实情况——

1. **分类任务指标失明**:AI 分类池合并率 74%,LLM 近乎"全判 merge"(non-merge recall≈0),
   accuracy≈0.74、merge-F1≈0.85 看着漂亮却毫无判别力。
2. **生成任务指标失明**:BLEU/ROUGE 对开放式审查意见无区分度(BLEU 0.5–1.5/100,人类/AI 互有高低)。
3. **上下文假设未证**:Exp4 的 C1–C4 全是 diff 级浅上下文,分类 C4−C1 增益≈0、生成仅微弱正,
   "AI 更依赖完整上下文"因上下文太浅而**无法被检验**。

用户需要:在**只优化输入(上下文 + Prompt)、不改模型**的前提下,弄清 AI 代码审查性能下降的
真正来源,并用可被观测的指标证明改进是否有效。

## Solution

采用**假设驱动的"上下文阶梯 + Prompt 消融"**(而非 40 格全因子网格),配套修复两个失明指标:

- 构建**单调递增的 5 级深上下文阶梯**(L0 diff → L4 仓库级),检验"深度是否弥合 AI 差距"。
- 新增 **P5 Self-Reflection / P6 多轮**两种 Prompt,把"批判性审查姿态"作为压制 merge 偏向的主手段。
- 分类任务强制报 **per-class P/R + 混淆矩阵 + balanced accuracy**(测量仪器,非重采样)。
- 生成任务引入 **deepseek-v4-pro 作 LLM-judge 多维打分**(执行仍用 flash),修复 BLEU/ROUGE 失明。
- 全部结果与 Exp5 基线对比,并在关键格上跑人类对照,回答"改进后 AI 是否追平人类"。

结论叙事:先诚实地量(修复指标)→ 用 Prompt 改变推理姿态攻可攻的部分 → 对攻不动的"标签噪声"
(未合并≠代码差,流程性拒绝)给出有洞察的解释,而非硬提 merge-F1。

## Requirements

### 数据(功能性)

- **R1** 直接读 Exp5 AI 池:分类 `Experiment5/results/samples/classify_ai.json`(50),
  生成 `Experiment5/results/samples/generate_ai.json`(72)。**禁止**重新采集或重新识别 AI。
- **R2** 人类对照沿用 Exp5 口径 `matched_human_control`(仅从 Exp2 held-out test 抽,按 AI 的
  `repo×is_merged` 分布匹配),对照仅在关键格跑,不全铺。
- **R3** 生成真值口径逐字沿用 Exp4/5:每 PR **首条顶层 inline comment**(最早、非回复),一 PR 一样本。

### 上下文阶梯(功能性)

- **R4** 提供 5 级单调递增上下文,`build_context(repo, number, level)` 返回纯文本:

  | 级 | 内容 | 来源 |
  |---|---|---|
  | **L0** | 仅 Diff(拼接 PR 内 Python 文件 patch,超预算截断)=Exp4 C1 | 已有 patch |
  | **L1** | L0 + PR 描述(title+body) + 全部 commit message = Exp4 C4 | 已有 |
  | **L2** | L1 + **修改前后完整函数/文件**(patch 只给窗口,这里给全貌) | GitHub API 按 base_sha/head_sha 拉文件 |
  | **L3** | L2 + **Issue 正文**(解析 PR body 里 `#ref`/`Fixes #`) + **历史审查意见**(reviews / review_comments 表) | API + 既有表 |
  | **L4** | L3 + **仓库级轻量检索上下文**:被改符号在仓库内其它出现处(GitHub code search API 取路径片段)、同目录相关文件头部、该 repo 测试/贡献规范摘要 | API + 词法检索 |

- **R5** L4 采用**轻量词法检索,明确不做 RAG / 不做向量化 / 不 clone 整仓库**。检索范围按 PR
  动态圈定(改动文件 + 其 import + 同目录 + 符号出现处)。
- **R6** 每级都有**独立 token 预算与截断策略**,保证长上下文不超模型窗口;截断行为需可复现(确定性,
  不依赖随机)。
- **R7** 所有 API 拉取的文件/issue/检索结果**落盘缓存**,重跑命中不重复触网(防限流、可复现)。

### Prompt(功能性)

- **R8** 沿用 Exp4 P1–P4(Zero-shot / Few-shot / CoT / Role)。
- **R9** 新增 **P5 Self-Reflection**:单轮内两阶段——① 扮演挑剔 maintainer **主动找 blocking 缺陷**
  (不预设 PR 是好的),② 基于"是否存在 blocking 缺陷"反推 merge 判断 / 生成审查意见。P5 **全量**铺整条阶梯。
- **R10** 新增 **P6 多轮交互式**:分类"审代码→追问风险→定论";生成"初稿意见→自我批评→修订"。
  **仅在 L4 上跑一次**(成本控制,作 6.10 Agent 方向引子)。
- **R11** `P*` = 在 Exp4/5 AI 数据上表现最好的旧 prompt,**从预测指标经验选取,不硬编码**;
  选取过程与依据写入报告。

### 实验矩阵(功能性)

- **R12** 每个任务运行以下条件(非全网格):
  - 消融①(上下文):`{L0,L1,L2,L3,L4} × P*`
  - 消融②(Prompt):`L4 × {P*, P5, P6}`
  - 归因格:`L0 × P5`(拆"上下文 vs prompt"的功劳,答 6.9-Q3)
  - 人类对照:仅 `L4×P*`、`L4×P5` 两格跑 `matched_human_control`
- **R13** 先在少量样本(如 home-assistant 数条)跑通全链路 + 缓存 + judge 冒烟,再全量。

### 模型与评估(功能性)

- **R14** **执行模型** = `deepseek-v4-flash`;**裁判模型** = `deepseek-v4-pro`,裁判**仅评生成任务**。
- **R15** 分类指标:**per-class Precision/Recall + 混淆矩阵 + balanced accuracy 作头条**;
  accuracy/merge-F1 可列但不作为主结论依据。B(prompt 框定)生效的判据 = **non-merge recall 上升**。
- **R16** 生成指标:**LLM-judge 多维打分**——`relevance`(是否命中真值指出的问题)、`actionable`、
  `correct`、`hallucination`,各 1–5 分 + 理由;**保留 BLEU/ROUGE 仅为与 Exp5 表格可比**,并在报告点明其失明。
- **R17** 记录每条件**推理时间**(沿用 llm_client 计时)。
- **R18** 全部结果与 **Exp5 基线对比**,产出改进 Δ。

### 非功能性

- **R19** 复用纪律:下游用 Exp2 模型/词表/scaler 一律**只 transform 不 fit**(本实验若涉及)。
- **R20** 资源约束:WSL2 ~7.6GB 内存,API 拉取与检索需限流/限并发,避免 OOM 与 GitHub 限流。
- **R21** 成本约束:共享 Exp4 缓存目录;新上下文/Prompt 改变 messages 内容 → content_hash 变 → 不误命中旧缓存;
  未变的直接命中省钱。

## Implementation Decisions

> 后续开发的唯一权威上下文。

### 目录与加载

- 代码置于 `Experiment6/src/`,沿用 Exp5 的别名加载(`config.py` 把兄弟实验 `src` 以 `exp2src`/`exp4src`
  载入),LLM 缓存目录**指向 Exp4 `results/cache/`**(共享)。

### 接缝(seams,尽量复用 Exp4、接缝数最少)

| 接缝 | 形态 | 复用/新增 |
|---|---|---|
| **① 上下文** | `build_context(repo, number, level) -> str`(与 Exp4 同签名,新增 L2–L4 分支) | 复用签名 |
| **② LLM 调用** | `chat(messages, ..., model=None)`:model 穿透到 `_call_api`,且**缓存 content_hash 纳入 model 维度** | 改造 Exp4 |
| **③ 裁判** | 新模块 `judge.py`:`score(generated, reference, diff) -> {relevance,actionable,correct,hallucination,rationale}` | **唯一新接缝** |
| **④ 运行/评估** | `run_experiments.py` 遍历 R12 矩阵;`evaluate.py` 出 per-class P/R + 混淆矩阵 + judge 聚合 | 复用 Exp4 骨架 |

### `llm_client` 改造(接缝②,关键)

- 现状:`_call_api` 写死 `config.MODEL_ID`(=`deepseek-v4-flash`);`chat_semantic` 的
  `content_hash = sha1(config.MODEL_ID, messages, temperature, max_tokens)`。
- 改造:`chat`/`_call_api` 增加 `model: str | None`(None 时回退 `config.MODEL_ID`);
  **content_hash 用实际传入的 model**。裁判调用传 `model="deepseek-v4-pro"`,与执行调用天然不同 key,
  互不覆盖。
- 影响面:Exp4/5 现有调用 model=None 行为不变,缓存命中不受影响(hash 仍等于原 MODEL_ID)。

### 上下文构建关键算法

- **L2**:对 PR 每个改动的 Python 文件,按 `head_sha` 拉改后完整文件、按 `base_sha` 拉改前完整文件
  (`previous_filename` 处理重命名);优先只保留"改动 hunk 所在函数/类"的完整体,超预算再回退到文件级截断。
- **L3**:从 `prs.body` 正则抽 `#\d+` / `Fixes|Closes #\d+` 作 issue 号,API 拉 issue title+body;
  历史审查意见从 `reviews` / `review_comments` 表按 `(repo,pr_number)` join,排除"作为生成真值的那条 comment"以防泄漏。
- **L4**:对被改的顶层符号名,调 GitHub code search API 取仓库内其它出现文件+行片段(限 top-k);
  同目录同语言文件取头部 import/docstring;附该 repo 的 CONTRIBUTING/测试目录约定摘要(静态、可缓存)。
- 全程确定性截断:按固定优先级顺序裁剪,不用随机。

### Prompt 关键点

- P5 用 system 设"批判性 maintainer 角色 + 强制先列 blocking 缺陷清单再定论"的结构化格式;
  分类结论行、生成意见体与 Exp4 的 `*_FORMAT` 保持解析兼容。
- P6 用 `chat` 多次调用串起对话(把上一轮 assistant 回复回填进 messages),每轮独立缓存 key。

### 数据结构

- 预测落 `Experiment6/results/predictions/`,指标落 `results/metrics/`,图落 `results/figures/`。
- judge 输出结构:`{pr_key, condition, relevance, actionable, correct, hallucination, rationale, latency}`。

## Testing Decisions

好测试只验**外部行为**,不验实现细节。

- **上下文接缝①**:给定已知 PR(冒烟用的 home-assistant 样本),断言每个 level 的输出**包含应有的块**
  (L1 含 PR 描述串、L2 含改前/改后完整函数、L3 含 issue 正文串、L4 含检索片段标记),且 level 单调
  (高 level 是低 level 的超集,除截断)。
- **裁判接缝③**:构造"命中真值的意见"vs"明显无关/幻觉的意见"两个输入,断言 judge 给出**可区分的分档**
  (前者 relevance 明显高、hallucination 明显低);不断言精确分值。
- **LLM 接缝②**:断言传入不同 model 产生**不同 cache key**;model=None 时 key 等于原 Exp4 行为(回归)。
- **全链路冒烟(R13)**:1 条样本跑通 L0→L4 + P5 + judge,确认 API 拉取、缓存命中、指标产出无异常。
- 先例:参照 Exp4 `evaluate.py`(分类指标 + BLEU/ROUGE)、Exp5 `ml_features.py` 的 bit-level 重现自检风格。

## Out of Scope

- **不做 RAG / 向量检索 / clone 整仓库**(留作 6.10 拓展与 Exp7)。
- **不做真正的跨文件 AST 调用图**(L4 用轻量词法检索替代)。
- **不训练/微调任何模型**,不改 Exp2 权重、Exp3(本项目跳过)。
- **不重新采集数据、不重新识别 AI 标签**。
- **不做全 40 格因子网格**(仅 R12 的阶梯+消融)。
- **不用 judge 评分类任务**(分类靠 per-class P/R + 混淆矩阵)。
- P6 多轮**不铺满阶梯**,仅 L4 一格。

## Further Notes

- **核心结论叙事(对齐 6.9 五问)**:①L4 为何有用 → 消融①增益曲线;②哪种 Prompt 最适合 → 消融②
  (预期 P5 拉高 non-merge recall);③Prompt vs 上下文各解决什么 → 归因格 `L0×P5` vs `L4×P*`;
  ④是否适用所有任务 → 分类可能仍受"标签噪声"天花板限制、生成受益更明显;⑤未来方向 → RAG/Agent/工具调用
  (承接 6.10 与 Exp7)。
- **标签噪声是洞察不是 bug**:报告需用错误案例证明部分 non-merge 是流程性拒绝(未合并≠代码差),
  明确区分"代码质量判断"与"合并结果预测",不以硬提 merge-F1 为目标。
- **裁判自偏声明**:pro 评 flash 已规避"自评",但仍属同厂模型,报告需列为 threat to validity。
- **P\* 选取**需在开跑前从 Exp4/5 预测指标确定并记录,避免事后挑选偏差。
