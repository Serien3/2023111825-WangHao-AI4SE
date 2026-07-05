# 实验一：数据集设计说明

> 本文档说明实验一数据集的设计思路、获取策略、数据结构与探索性分析方法，
> 供后续实验（二～七）参考。代码位于 `Experiment1/src/`，运行命令见 `Experiment1/src/README.md`。

---

## 1. 设计思路：问题驱动而非数据驱动

实验一的核心任务看似是"爬数据"，但数据集的每一个设计决策都由**后续实验的具体需求**决定，而非单纯追求数据量或覆盖面。整门课程围绕两个任务展开：

- **Merge Prediction**：预测 Pull Request 是否最终被合并（分类任务）
- **Review Comment Generation**：根据代码改动自动生成审查意见（生成任务）

后续每个实验对数据有不同的隐含要求，如果实验一没有提前考虑，后面将无法进行：

| 后续实验 | 核心需求 | 对实验一的要求 |
|---|---|---|
| 实验二（ML/SVM/RF） | 构建 AST/CFG 特征 | 需要**函数级 diff（patch）**；语言最好统一，否则每种语言都要写单独解析器 |
| 实验三（DL/CodeBERT） | Review Comment Generation | 必须有 **(代码 hunk → 审查意见) 的配对**，即带 `diff_hunk` 的 inline review comment |
| 实验四/六（LLM + Prompt） | 多粒度上下文 | 需保存 PR body、commit message、`base_sha/head_sha`（可据此回仓库补抓文件内容） |
| 实验五（AI 生成代码审查） | 筛选 AI 代码样本 | 需要 `is_ai_code` / `has_ai_reviewer` 标签，且 AI 样本**不能太少** |
| 实验七（VSCode 插件） | 模型部署 | 无新增数据需求 |

因此，本数据集采用**关系型多表结构**（5 张表），而非单一扁平宽表——因为不同实验需要不同粒度的数据，一张大表既浪费空间又难以使用。

---

## 2. 目标仓库选择

### 选择标准

1. **Python 为主语言**：实验二需要用 tree-sitter 构建 AST/CFG；如果各仓库语言分散（Go/Java/TypeScript混杂），需要为每种语言单独实现解析器，成本巨大。Python 主导可以保证解析逻辑统一。
2. **PR 量大、审查文化成熟**：需要足够多的 inline review comment（带 `diff_hunk`），实验三生成任务才有训练数据。
3. **有 bot/AI 活动痕迹**：保证实验五能筛出足量 AI 生成代码 / AI reviewer 样本。
4. **覆盖不同开源社区**：呼应实验一"拓展目标"——分析不同社区代码审查流程的差异。

### 最终选定的 5 个仓库

| 仓库 | 领域 | 选择原因 |
|---|---|---|
| `django/django` | Web 框架 | 审查规范、reviewer 多、讨论充分；合并率约 48%，正负样本均衡 |
| `pandas-dev/pandas` | 数据科学 | inline comment 密集，label 体系丰富 |
| `huggingface/transformers` | 机器学习 | PR 量极大，社区活跃，有 AI 辅助痕迹 |
| `apache/airflow` | 基础设施 | CI/bot 生态活跃，AI reviewer 样本丰富 |
| `home-assistant/core` | IoT 平台 | PR 吞吐量极高，样本充足；`has_patch` 覆盖率高 |

---

## 3. 数据获取策略

### 3.1 只取"已关闭"PR

```
state = closed  （merged + unmerged 都包含）
不取 open PR
```

**原因**：Merge Prediction 的标签（`is_merged`）必须确定——开放中的 PR 结局未知，纳入会引入噪声。已关闭的 PR 结局已定：merged 为正例，closed-unmerged 为负例，标签天然干净。

### 3.2 每个 PR 抓取的 6 类子资源

每个 PR 被打包成一个 JSON 文件（`Experiment1/results/raw/<owner>__<repo>/pr_<编号>.json`），包含：

| 子资源 | 对应 API 端点 | 用途 |
|---|---|---|
| `detail` | `GET /repos/{owner}/{repo}/pulls/{n}` | 基本信息、合并状态、additions/deletions |
| `files` | `GET .../pulls/{n}/files` | 改动文件 + **patch（diff）** |
| `reviews` | `GET .../pulls/{n}/reviews` | 正式 review 决策（APPROVED/CHANGES_REQUESTED）|
| `review_comments` | `GET .../pulls/{n}/comments` | **inline 评论 + diff_hunk**（实验三命门）|
| `issue_comments` | `GET /repos/.../issues/{n}/comments` | PR 对话区评论（用于 AI reviewer 判定）|
| `commits` | `GET .../pulls/{n}/commits` | commit message + 作者（AI 代码信号）|

### 3.3 断点续传

每个 PR 独立存为一个 JSON 文件。`fetch.py` 在抓取前检查文件是否存在——**已存在则跳过**。因此抓取过程可以随时中断、续跑，不会重复消耗 API 额度。

### 3.4 API 限流处理

GitHub 认证后 core API 限额为 **5000 次/小时**，search API 为 **30 次/分钟**。`github_client.py` 实现：
- 核心限额耗尽时，读取 `X-RateLimit-Reset` 头，**精确睡眠到额度恢复**
- 二级限流（`secondary rate limit`）或 5xx 错误时，**指数退避重试**（最多 5 次）
- 自动翻页（`Link` 头），调用方无需关心分页

### 3.5 AI 定向补采

AI 生成代码的 PR 在自然分布中极为稀少（约占 2%～5%），若只靠随机抽样，1500 条 PR 里可能只有几十条，实验五样本不足。

补采策略：在基础 300 条/仓库之外，额外用 **Search API** 定向搜索带 AI 信号的 PR（每仓库上限 50 条），并在数据中标记 `oversampled=True`。实际补采 208 条，最终 `is_ai_code=True` 样本达到 **294 条**，`has_ai_reviewer=True` 达到 **323 条**，满足实验五需求。

---

## 4. 数据集结构

### 4.1 整体组织：关系型多表

数据集由 **5 张表**组成，通过 `repo + pr_number` 联结。

```
prs (主表，1706 行)
 ├── files           (1:N，14158 行，每个改动文件一行)
 ├── reviews         (1:N，每条正式 review 一行)
 ├── review_comments (1:N，3194 行，每条 inline 评论一行)
 └── commits         (1:N，4549 行，每个 commit 一行)
```

每张表存为 `.parquet`（后续实验读取用）和 `.csv`（人工查看用）两份。

---

### 4.2 表一：`prs`（主表，32 列）

**身份与基本信息**：`repo`、`number`（GitHub 原生编号）、`title`、`body`、`author`、`author_association`（OWNER/MEMBER/CONTRIBUTOR 等）、`pr_url`。

**预测标签**：`is_merged`（True=合并，False=关闭拒绝，**Merge Prediction 的目标 y**）、`review_decision`（APPROVED / CHANGES\_REQUESTED / COMMENTED / NONE）。

**统计特征**（实验二直接可用）：

| 字段 | 含义 | 注意 |
|---|---|---|
| `additions` / `deletions` / `changed_files` | 代码改动量 | — |
| `num_commits` | 包含的 commit 数 | — |
| `num_reviews` | 正式 review 次数 | 含同一人多轮审查 |
| `num_reviewers` | 参与审查的不同人数 | `num_reviews ≥ num_reviewers` |
| `num_review_comments` | inline 代码评论数 | 钉在具体代码行 |
| `num_issue_comments` | 对话区评论数 | 不针对代码行 |
| `num_labels` / `labels` | 标签数量与内容 | 竖线分隔 |

**代码定位**（实验六上下文增强用）：`base_sha`、`head_sha`、`merge_commit_sha`——可据此回仓库补抓完整文件内容。

**AI 标签**：`is_ai_code`、`ai_code_signals`（具体命中的信号）、`has_ai_reviewer`、`ai_reviewer_signals`、`oversampled`。

---

### 4.3 表二：`files`（14158 行，11 列）

每个 PR 的每个改动文件占一行。

| 字段 | 说明 |
|---|---|
| `filename` | 文件路径 |
| `status` | added / modified / removed / renamed |
| `additions` / `deletions` / `changes` | 文件级改动量 |
| `language` | 按扩展名识别（`.py` → python，`.yaml` → yaml 等）|
| `has_patch` | 是否有 patch；False 表示文件过大 GitHub 未返回 diff |
| `patch` | **原始 diff 文本，实验二/三/四的原料** |
| `previous_filename` | 重命名前的文件名（仅 renamed 时有值）|

实际语言分布：python 68.9%、rst 7.1%、other 6.6%、yaml 5.4%、json 3.4%、text 3.0%。

---

### 4.4 表三：`reviews`（正式 review）

| 字段 | 说明 |
|---|---|
| `review_id` | GitHub review ID |
| `reviewer` | 审查者账号 |
| `state` | APPROVED / CHANGES\_REQUESTED / COMMENTED / DISMISSED |
| `submitted_at` | 提交时间 |
| `body` | review 整体评论（可为空）|

---

### 4.5 表四：`review_comments`（3194 行，⭐实验三命门）

这是**实验三 Review Comment Generation 任务的训练数据来源**。每行代表一条钉在具体代码行上的 inline 评论。

| 字段 | 说明 |
|---|---|
| `path` | 评论所在文件路径 |
| `diff_hunk` | **被评论的代码片段（上下文 diff）** |
| `body` | **审查意见文本** |
| `position` / `original_position` | 在 diff 中的位置 |
| `in_reply_to_id` | 若为回复，指向父评论 ID |
| `created_at` | 评论时间 |

`(diff_hunk, body)` 配对即为生成任务的 `(input, target)` 样本。

---

### 4.6 表五：`commits`（4549 行）

| 字段 | 说明 |
|---|---|
| `sha` | commit hash |
| `message` | commit message（实验二文本特征 / AI 代码信号来源）|
| `author` | 提交者姓名（git config name）|
| `author_login` | GitHub 账号（可为 null，部分 commit 无关联账号）|

---

## 5. 数据清洗规则

`build_dataset.py` 在构建时执行以下清洗，所有操作均可追溯：

| 规则 | 处理方式 |
|---|---|
| 无文件改动的 PR | 直接丢弃（`changed_files == 0` 或 `files` 列表为空）|
| 重复 PR（同 repo+number 出现两次）| 去重，只保留一份 |
| 文件过大无 patch | 保留该文件行，设 `has_patch=False`；在 prs 表记录 `n_files_missing_patch` 计数 |
| 文件语言 | 按扩展名识别，未知扩展名归为 `other` |
| AI 标签 | 调用 `ai_labeling.label_pr` 逐条判定，结果写入 prs 表 |

清洗结果：原始缓存 1706 条有效 PR（小样测试时 0 条无效；全量中 AI 补采与基础样本重合者已去重）。

---

## 6. AI 标签设计

### 6.1 两个独立标签

AI 判定拆分为两个完全独立的标签，分别回答不同问题：

- **`is_ai_code`**：这个 PR 的代码是否由 AI 工具辅助生成？
- **`has_ai_reviewer`**：这个 PR 是否有 AI 进行代码审查？

实验五的两个子分析分别依赖这两个标签，互不干扰。

### 6.2 `is_ai_code` 判定逻辑

按强弱分四类信号，命中任意一条即判为 True：

| 信号类型 | 强弱 | 查找位置 | 信号记录格式 |
|---|---|---|---|
| PR 作者是代码生成 bot | **强** | `detail.user.login` ∈ 已知 bot 集合 | `author_bot:<login>` |
| commit 里的 Co-authored-by | **强** | 每个 commit message | `coauthor_commit:<pattern>` |
| PR body 里的 Co-authored-by | **强** | PR body 文本 | `coauthor_body:<pattern>` |
| body/title 文本提及 AI 工具 | 弱 | PR title + body | `text_hint:<hint>` |
| AI 相关 label | 弱 | PR labels 列表 | `label:<name>` |

匹配的具体模式（在 `config.py` 中定义，可扩充）：
- Co-author 模式：`co-authored-by: copilot`、`co-authored-by: claude`、`co-authored-by: cursor`、`co-authored-by: chatgpt`、`co-authored-by: devin` 等
- 代码生成 bot 账号：`devin-ai-integration[bot]`、`sweep-ai[bot]`、`cursor[bot]`、`copilot-swe-agent[bot]` 等

### 6.3 `has_ai_reviewer` 判定逻辑

扫描 PR 所有留言者（reviews / review_comments / issue_comments），判断 login 是否属于已知 AI 审查 bot 名单：

```
coderabbitai[bot]、sourcery-ai[bot]、ellipsis-dev[bot]、korbit-ai[bot]
copilot-pull-request-reviewer[bot]、gemini-code-assist[bot]
pr-agent[bot]、greptileai[bot]、bito-code-review[bot] …
```

### 6.4 标签的可靠性与使用建议

这是**启发式判定**，存在两类误差：

- **假阴性（漏判）**：AI 工具不留痕迹（如直接粘贴 Copilot 建议而不用 co-author）→ 实际 AI 代码比 294 只多不少
- **假阳性（误判）**：弱信号可能命中非 AI 代码（如 PR body 讨论了 copilot 但代码是人写的）

**使用建议**：后续实验若需高精度子集，通过 `ai_code_signals` 列筛选强信号行：

```python
# 高精度 AI 代码子集（只信强信号）
ai_strong = prs[prs['ai_code_signals'].str.contains('coauthor|author_bot', na=False)]

# 宽松子集（全部信号，样本更多）
ai_all = prs[prs['is_ai_code']]
```

---

## 7. 探索性分析（EDA）

对应实验一步骤五，由 `Experiment1/src/analyze.py` 生成，图表保存至 `Experiment1/results/figures/`。

### 7.1 分析维度与图表

| 图表文件 | 分析内容 | 对应指导书要求 |
|---|---|---|
| `merge_vs_nonmerge.png` | Merge / Non-merge 数量（柱状 + 饼图）| Merge 与 Non-Merge 数量统计 |
| `review_comment_dist.png` | inline review comment 数量分布（直方图，截尾至 p98）| Review Comment 数量分布 |
| `label_dist.png` | Top 15 标签频次（横向柱状）| Label 分布 |
| `reviewer_dist.png` | reviewer 数量分布（柱状）| Reviewer 数量分布 |
| `pr_size_dist.png` | PR 大小（additions+deletions）分布（对数直方图）| PR 长度分布 |
| `ai_vs_human.png` | AI 生成代码 vs 人类代码、有/无 AI reviewer 对比 | AI 与 Human PR 数量比较 |
| `cross_repo.png` | 跨仓库对比：merge 率 / 平均评论数 / 平均 reviewer 数 | 实验一拓展目标 |

### 7.2 关键观察

**类别不平衡（指导书思考题5）**

整体 merge 率 67.4%（1150/1706），正负样本比约 2:1。但各仓库差异显著：

| 仓库 | Merge 率 |
|---|---|
| home-assistant/core | 90% |
| apache/airflow | 80% |
| pandas-dev/pandas | 63% |
| huggingface/transformers | 56% |
| django/django | 48% |

仓库间差异意味着跨仓库训练/测试时需注意分布偏移。实验二/三训练集划分时应按仓库分层采样，或将仓库作为协变量。

**PR 大小的长尾分布**

- 中位数 32 行（极小改动为主）
- p90 = 339 行，p99 = 4809 行，最大 971560 行
- 对数直方图是正确的可视化选择；线性直方图会被极端值压缩

**inline comment 稀疏性**

- 中位数为 0（大多数 PR 没有 inline 评论）
- p90 = 5 条，p99 = 25 条，最大 202 条
- 对实验三：只有 `num_review_comments > 0` 的 PR 才能产生生成任务的训练样本，需要在后续实验中单独筛选

---

## 8. 数据集统计摘要

### 8.1 规模

| 表 | 行数 | 说明 |
|---|---|---|
| `prs` | 1706 | 有效 PR（基础 1498 + 净补采 208）|
| `files` | 14158 | 平均每 PR 8.3 个文件 |
| `review_comments` | 3194 | 平均每 PR 1.87 条（仅 inline）|
| `commits` | 4549 | 平均每 PR 2.67 个 commit |

### 8.2 Merge 分布

| 状态 | 数量 | 比例 |
|---|---|---|
| Merged | 1150 | 67.4% |
| Non-merged | 556 | 32.6% |

整体正负样本比约 2:1，存在轻度类别不平衡。各仓库差异较大（48%～90%），后续训练时需注意。

### 8.3 AI 标签分布

| 标签 | 数量 | 比例 |
|---|---|---|
| `is_ai_code=True` | 294 | 17.2% |
| `has_ai_reviewer=True` | 323 | 18.9% |
| 定向补采（`oversampled`）| 208 | 12.2% |

**AI 信号类型分布**（`is_ai_code` 命中信号）：
- `coauthor_commit`（强信号）：376 次 — 绝大多数
- `label`（弱信号）：79 次
- `coauthor_body`（强信号）：13 次
- `text_hint`（弱信号）：3 次

**AI reviewer 分布**：
- `copilot-pull-request-reviewer[bot]`：321 次（主力）
- `coderabbitai[bot]`：7 次

### 8.4 各仓库对比

| 仓库 | PR 数 | Merge 率 | 均 review_comments | 均 reviewers | 均 additions |
|---|---|---|---|---|---|
| apache/airflow | — | 80% | 1.49 | — | 223 |
| django/django | — | 48% | 2.45 | — | 97 |
| home-assistant/core | — | 90% | 1.18 | — | 873 |
| huggingface/transformers | — | 56% | 3.19 | — | 253 |
| pandas-dev/pandas | — | 63% | 1.12 | — | 3113 |

> `pandas-dev/pandas` 的 additions 均值极高（3113），因为该仓库 PR 常附带大量测试用例与文档改动。

---

## 9. 后续实验使用指引

| 实验 | 读哪张表 | 关键字段 | 注意事项 |
|---|---|---|---|
| 实验二（ML） | `prs` + `files` | `is_merged`, `additions`, `deletions`, `patch` | 筛 `is_ai_code=False`，只用人类代码 |
| 实验三（DL） | `prs` + `review_comments` + `files` | `diff_hunk`, `body`, `patch` | 筛 `num_review_comments > 0`；生成任务用 `(diff_hunk → body)` |
| 实验四（LLM） | `prs` + `files` + `commits` | `body`, `message`, `patch`, `base_sha`/`head_sha` | 多粒度上下文拼装 |
| 实验五（AI审查）| `prs` | `is_ai_code`, `ai_code_signals` | 建议同时给出强信号/宽松两个子集的结果 |
| 实验六（改进） | 全部表 + 回仓库补抓 | `base_sha`, `head_sha`, issue 链接 | 用 `base_sha/head_sha` 调用 GitHub API 抓完整文件 |
