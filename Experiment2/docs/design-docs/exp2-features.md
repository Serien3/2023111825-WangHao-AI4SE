# 实验二 设计说明：基于机器学习的 Merge Prediction

> 本文档说明实验二的设计思路、特征工程、模型选择与结果分析。
> 代码位于 `Experiment2/src/`，运行命令见 `Experiment2/src/README.md`。
> 数据来源为实验一构建的 5 张表（`Experiment1/results/processed/`）。

---

## 1. 任务定位：为什么用传统 ML，为什么只用 diff

实验二在整门课的定位是 **Merge Prediction 的传统机器学习基线**——用人工设计的特征
训练 SVM / Random Forest，为实验三（深度学习）、实验四（LLM）提供性能比较基准。
传统 ML 的价值不在于最高精度，而在于 **训练成本低、可解释性强**（特征重要性直接可读）。

### 1.1 代码来源：仅用实验一保存的 diff（patch），不回仓库抓完整文件

这是本实验最关键的设计决策。实验一保存的是 PR 的 **diff（patch）**，而非完整可解析的
Python 文件。我们经实证分析确认 **diff-only 足以高质量完成实验二**：

| 事实 | 数据 | 含义 |
|---|---|---|
| 有 patch 的 Python 文件行 | 9,177 | 特征提取原料充足 |
| `status=added`（patch 即完整新文件） | 1,357（14.8%） | AST/CFG 100% 准确 |
| `status=modified`（patch 是片段） | 7,747（84.4%） | 需容错解析 |
| hunk 头带 `def`/`class` 上下文 | 79.1% | 可恢复函数签名 |
| 含 Python 改动的人类 PR | 1,036 | 最终建模样本量 |

**为什么 diff-only 是正确选择**：

1. **指导书要的是聚合统计特征**（AST 节点数量、CFG 节点数量），不是完整语义分析。
   tree-sitter 对残缺片段容错解析（产生 ERROR 节点但仍能统计有效子树），完全够用。
2. **自包含、可复现**：无需联网、不消耗 GitHub API 额度、瞬时完成。
3. **尊重实验边界**：回仓库抓完整文件（`base_sha`/`head_sha`）是**实验六**明确保留的
   "Repository 级上下文增强"范围，在此实现会越界并模糊实验分工。
4. **Merge Prediction 关心的是"改动"**：diff 本身就是信号，完整文件反而引入无关噪声。

---

## 2. 数据筛选（步骤一）

```
实验一 prs 表（1706 行）
  ↓ is_ai_code == False           人类编写代码（指导书要求）
人类 PR（1412 行）
  ↓ 有 ≥1 个 language==python & has_patch 的文件
最终建模样本（1036 PR）
```

标签 `is_merged` 分布：Merged 666（64.3%）/ Non-merged 370（35.7%），正负比约 1.8:1，
轻度不平衡（与实验一整体 2:1 一致，人类子集略均衡）。

---

## 3. 特征工程（步骤二 + 三）

共 **98 维特征**，分四组。设计原则：每个特征服务 Merge Prediction，不堆砌。

### 3.1 AST 特征（tree-sitter，24 维）

从 patch 重建"改动后代码片段"（保留上下文行 + 新增行，丢弃删除行），用
tree-sitter-python 解析，逐 PR 聚合其所有 Python 文件：

- **总量与结构**：`ast_total_nodes`（节点总数）、`ast_max_depth`（最大树深）、
  `ast_avg_branching`（平均分支因子）、`ast_error_nodes`（解析错误节点数）
- **按类型计数**（20 种）：`function_definition`、`class_definition`、`if_statement`、
  `for_statement`、`while_statement`、`try_statement`、`call`、`import_statement`、
  `assignment`、`binary_operator`、`comparison_operator`、`return_statement` 等

重建逻辑（`feature_extraction.reconstruct_post_change_code`）：
```python
for line in patch.splitlines():
    if line.startswith('+'):   out.append(line[1:])   # 新增行
    elif line.startswith('-'): continue                # 删除行丢弃
    elif line.startswith(' '): out.append(line[1:])   # 上下文行保留
```

### 3.2 CFG 特征（基于控制流的轻量近似，4 维）

完整 CFG 需语义分析（def-use、跳转解析）。对 ML 统计特征而言，基于控制流语句计数的
近似已足够且更快：

- `cfg_nodes`：基本块近似（语句节点数 + 入口 + 出口）
- `cfg_edges`：控制转移近似（节点数 + 判定点 + 1）
- `cfg_complexity`：McCabe 圈复杂度近似 = 1 + 判定点数（if/for/while/try/and/or）
- `cfg_max_nesting`：最大嵌套深度

### 3.3 统计特征（11 维）

直接取自实验一 prs 表，其中区分两类（见第 5 节标签泄漏）：

- **提交时可得**：`additions`、`deletions`、`changed_files`、`num_commits`、
  `files_per_commit`、`churn_ratio`（新增占总改动比）
- **审查过程产生**：`num_reviews`、`num_reviewers`、`num_review_comments`、
  `num_issue_comments`、`review_density`

### 3.4 文本特征（9 + 50 维）

- **长度类**：`title_len`、`body_len`、`avg_commit_msg_len`
- **关键词命中**：`has_keyword_{fix,bug,test,refactor,add,remove}`（6 维 0/1）
- **TF-IDF**：PR 标题 + 正文，top 50 词（去英文停用词，min_df=3）

---

## 4. 模型训练（步骤四）

四个模型，全部用 `GridSearchCV` + 5 折交叉验证（scoring=f1）：

| 模型 | 调参空间 | 不平衡处理 |
|---|---|---|
| SVM (RBF) | C×gamma = 20 组合 | `class_weight='balanced'` |
| Random Forest | n_estimators×max_depth×min_split = 27 组合 | `class_weight='balanced'` |
| XGBoost | max_depth×lr×n_est×subsample = 54 组合 | `scale_pos_weight` |
| LightGBM | 同 XGBoost = 54 组合 | `is_unbalance=True` |

### 4.1 跨仓库分层划分

实验一 EDA 显示各仓库合并率 48%–90%，存在显著**分布偏移**。因此：
- train/val/test = 662/166/208，**按 repo 分层**（`stratify=repos`），保证每仓库都有代表
- 划分后验证三集索引**零重叠**，无数据泄漏
- 特征标准化（`StandardScaler`）仅在 train 上 fit，防止统计量泄漏

### 4.2 资源控制（OOM 防护）

本机 32 核但内存有限（WSL2 ~7.6 GB）。`GridSearchCV(n_jobs=-1)` 会一次 fork 与核心数
相等的 worker，每个复制一份数据，树模型内部又各自多线程 → 内存/线程双重超订，直接把
WSL2 打爆。故显式限制：外层 `N_JOBS=4`、模型内层 `MODEL_N_THREADS=2`，并在 import numpy
前锁定 `OMP/MKL/OpenBLAS` 线程为 2。实测内存峰值 ~3 GB，安全。

---

## 5. 标签泄漏分析（本实验核心方法论贡献）

**问题**：`num_reviews`、`num_reviewers`、`num_review_comments` 等特征只有在**审查过程
完成后**才可知，PR 刚提交时并不存在。用它们训练会造成**时间泄漏**——模型学到了未来信息，
在真实部署（PR 刚提交就要预测）场景下不可用。

**方法**：构建两套特征集，分别训练、对比：
- **Pre-review**（提交时可得）：AST + CFG + 提交时统计 + 文本 = 93 维
- **Full**（含审查过程特征）：Pre-review + 5 个审查过程特征 = 98 维

这既回答了指导书思考题 4（"哪类特征贡献最大"），又给出了**真实可部署性能**（pre-review）
与**理论上界**（full）的区间。

---

## 6. 实验结果与分析

### 6.1 主结果（测试集，208 PR）

**Full 特征集**（含审查过程特征，理论上界）：

| 模型 | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| SVM | 0.822 | 0.859 | 0.866 | 0.862 | 0.885 |
| **Random Forest** | **0.832** | 0.867 | 0.873 | **0.870** | **0.913** |
| XGBoost | 0.832 | 0.878 | 0.858 | 0.868 | 0.890 |
| LightGBM | 0.817 | 0.864 | 0.851 | 0.857 | 0.884 |

**Pre-review 特征集**（无泄漏，真实可部署）：

| 模型 | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| SVM | 0.740 | 0.767 | 0.858 | 0.810 | 0.769 |
| **Random Forest** | **0.774** | 0.872 | 0.761 | **0.813** | **0.834** |
| XGBoost | 0.750 | 0.831 | 0.769 | 0.798 | 0.820 |
| LightGBM | 0.745 | 0.796 | 0.813 | 0.804 | 0.809 |

### 6.2 标签泄漏消融（关键发现）

| 模型 | Pre-review F1 | Full F1 | Δ |
|---|---|---|---|
| SVM | 0.810 | 0.862 | +0.052 |
| RF | 0.813 | 0.870 | +0.057 |
| XGBoost | 0.798 | 0.868 | +0.070 |
| LightGBM | 0.804 | 0.857 | +0.053 |

**结论**：加入审查过程特征使 F1 提升约 **5–7 个百分点**，AUC 提升更明显（RF：0.834→0.913）。
这量化了"审查过程信息"的价值，但也警示：**报告 full 结果时必须声明其含时间泄漏**，
不能作为真实部署性能。Pre-review 模型 F1 仍达 0.81，说明**代码改动本身已携带主要预测信号**。

### 6.3 特征重要性（Random Forest）

**Full 特征集** top 特征：`num_reviews`(0.148) + `num_reviewers`(0.130) 合计占 28%，
印证审查过程特征主导——但这正是泄漏来源。

**Pre-review 特征集** top 特征（真实信号）：
1. 文本特征主导：`tfidf_github`、`tfidf_py`、`avg_commit_msg_len`、`title_len`、`body_len`
2. 改动量：`deletions`、`churn_ratio`
3. 代码结构：`ast_avg_branching`

**有趣发现**：`tfidf_github` 高居榜首。经查，50% 的人类 PR 正文含 github.com URL
（链接 issue/其他 PR），这类 PR 合并率 **81.5%**，远高于不含的 49%。这是一个合理的
**提交时质量信号**——引用了相关上下文的 PR 通常更规范、更易被合并，而非泄漏。

### 6.4 跨仓库性能（分布偏移验证）

Pre-review 特征集下各仓库 F1（RF）：

| 仓库 | Merge 率 | Pre-review F1 | Full F1 |
|---|---|---|---|
| home-assistant/core | 90% | 0.975 | 0.963 |
| apache/airflow | 80% | 0.825 | 0.877 |
| huggingface/transformers | 56% | 0.809 | 0.840 |
| pandas-dev/pandas | 63% | 0.720 | 0.769 |
| django/django | 48% | 0.500 | 0.800 |

**结论**：
- 高合并率仓库（home-assistant 90%）易预测——因类别本身极不平衡，模型偏向多数类即可。
- **django 是最难的仓库**（合并率 48%，正负最均衡）：pre-review F1 仅 0.50，
  加入审查特征后跃升至 0.80。说明对于"审查文化严格、结局最不确定"的仓库，
  代码改动本身不足以预测，审查过程信息至关重要。
- 这直接验证了实验一 EDA 的分布偏移警告：**不能只看整体平均，必须报告分仓库指标**。

---

## 7. 图表清单

| 图 | 文件 | 内容 |
|---|---|---|
| 图 1 | `feature_importance_rf.png` | RF top-15 特征重要性 |
| 图 2 | `roc_curves.png` | 四模型 ROC 曲线对比 |
| 图 3 | `confusion_matrices.png` | 四模型混淆矩阵 |
| 图 4 | `cross_repo_performance.png` | 跨仓库 F1 分解 |
| 图 5 | `label_leakage_ablation.png` | pre-review vs full 消融 |
| 图 6 | `training_time.png` | 训练时间对比 |

---

## 8. 对指导书思考题的回答

1. **为什么需要 AST/CFG 中间表示？**
   diff 是纯文本，AST/CFG 把代码的**语法结构与控制流复杂度**显式量化为数值特征
   （节点数、圈复杂度、嵌套深度），使传统 ML 能"看懂"代码结构而非仅字符串。

2. **为什么传统 ML 需要人工设计特征？**
   SVM/RF 只接受固定长度数值向量，无法直接处理变长代码文本。人工特征把领域知识
   （改动量、复杂度、审查强度）编码为模型可用的信号。这既是局限（依赖设计质量），
   也是优势（可解释）。

3. **SVM 与 RF 适用场景？**
   SVM 适合中小规模、特征维度适中、边界清晰的问题；RF 适合特征异构、含噪声、需要
   特征重要性分析的场景。本实验 RF 略优（F1 0.870 vs 0.862），且提供了可解释的重要性。

4. **哪类特征贡献最大？**
   审查过程特征（num_reviews/num_reviewers）贡献最大但含时间泄漏；剔除后，
   文本特征（尤其 PR 是否引用 github 上下文）与改动量特征贡献最大。

5. **相比实验一增加了哪些预处理？**
   新增：diff 重建为可解析代码、tree-sitter AST 解析、CFG 近似构建、特征标准化、
   TF-IDF 向量化、按仓库分层划分、类别不平衡加权、标签泄漏特征分组。
