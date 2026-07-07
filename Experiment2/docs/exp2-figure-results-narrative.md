# 实验二可视化图注与 Results 叙述

本文档用于将 `Experiment2/results/figures/` 中的实验二可视化结果插入实验报告或论文正文。每一节包含三部分：可直接使用的图注、正文结果叙述、以及可落入报告结论的小结。所有数值均来自 `Experiment2/results/metrics/`、`Experiment2/results/models/` 和重新评估得到的测试集输出；未使用手工改写数据。

## 结果叙述主线

实验二要回答的问题是：传统机器学习模型能否基于人工设计的 AST、CFG、代码修改统计与文本特征完成 Pull Request 的 Merge Prediction，并为后续深度学习和大语言模型实验提供可解释、低成本的基线。结果显示，在仅使用 PR 提交时可获得信息的 `pre-review` 设置下，Random Forest 取得最高 F1（0.813）和 ROC-AUC（0.834），说明人工特征已经能提供稳定的可部署预测信号；加入审查过程特征后的 `full` 设置进一步将 Random Forest F1 提升到 0.870、ROC-AUC 提升到 0.913，但这部分提升来自时间上不可提前获得的 review 信息，应解释为理论上界而非真实部署性能。

## 图表使用索引

同一逻辑图同时导出为 `svg`、`pdf`、`png` 和 `tiff`。报告排版优先使用 `svg` 或 `pdf`；需要位图预览时使用 `png`。旧版兼容文件 `feature_importance_rf.png`、`roc_curves.png`、`confusion_matrices.png`、`cross_repo_performance.png`、`label_leakage_ablation.png` 和 `training_time.png` 仍可引用，但正文建议引用下表中的逻辑图名。

| 逻辑图 | 推荐文件 | 对应报告问题 |
| --- | --- | --- |
| 模型总体性能矩阵 | `model_performance_matrix.svg` | SVM、RF、XGBoost、LightGBM 在 pre/full 两种特征集上的整体表现如何？ |
| 标签泄漏消融 | `label_leakage_ablation.svg` | 审查过程特征带来多少性能提升，为什么不能直接作为部署性能？ |
| Pre-review RF 特征重要性 | `feature_importance_rf_pre.svg` | 在无泄漏设置下，哪类特征贡献最大？ |
| Full RF 特征重要性 | `feature_importance_rf_full.svg` | 审查过程特征是否主导 full 设置？ |
| Pre-review ROC 曲线 | `roc_curves_pre.svg` | 无泄漏模型的排序/区分能力如何？ |
| Full ROC 曲线 | `roc_curves_full.svg` | 加入审查过程特征后的区分能力上界如何？ |
| Pre-review 混淆矩阵 | `confusion_matrices_pre.svg` | 可部署模型的错误类型是什么？ |
| Full 混淆矩阵 | `confusion_matrices_full.svg` | 审查过程特征是否减少误判？ |
| Pre-review 跨仓库性能 | `cross_repo_performance_pre.svg` | 模型在不同仓库上的泛化是否稳定？ |
| Full 跨仓库性能 | `cross_repo_performance_full.svg` | 审查过程特征对困难仓库是否更有帮助？ |
| 训练时间 | `training_time.svg` | 传统机器学习基线是否具有低训练成本？ |

## Figure 1. 模型总体性能矩阵

**建议图注。** Figure 1 | Test-set performance of traditional machine-learning baselines for Merge Prediction. The heatmaps compare SVM, Random Forest (RF), XGBoost and LightGBM under two feature settings. The pre-review setting uses only information available when a pull request is opened, whereas the full setting additionally includes review-process features. Values report accuracy, precision, recall, F1 and ROC-AUC on the same held-out test split of 208 pull requests.

**正文 Results 叙述。** 我们首先在相同测试集上比较了四类传统机器学习模型的整体预测性能。在 `pre-review` 设置下，RF 取得最高 F1（0.813）和最高 ROC-AUC（0.834），SVM 的 F1 为 0.810，LightGBM 的 F1 为 0.804，XGBoost 的 F1 为 0.798。虽然各模型 F1 差异不大，但它们体现出不同的误差偏好：SVM 的 recall 最高（0.858），而 RF 的 precision 最高（0.872），说明 RF 在合并预测中更少将未合并 PR 错判为可合并。

加入审查过程特征后，所有模型的 F1 均提升到 0.857 以上。其中 RF 仍为整体最优模型，F1 达到 0.870，ROC-AUC 达到 0.913；XGBoost 的 F1 为 0.868，SVM 为 0.862，LightGBM 为 0.857。该结果说明，审查过程中的互动信息确实携带了强预测信号，但这些信息发生在 PR 创建之后，因此只能作为 post-review upper bound，而不能作为真实插件或自动审查系统在 PR 提交时的可部署性能。

**可引用结论。** 在真实可部署的 pre-review 设置下，RF 是最稳健的传统机器学习基线；full 设置展示了审查过程信息可带来的性能上界，但不应与部署场景混淆。

## Figure 2. 标签泄漏消融

**建议图注。** Figure 2 | Label-leakage ablation between deployable and post-review feature settings. Lines connect each model's F1 score from pre-review features to full features. The right-side labels show the F1 gain introduced by review-process features.

**正文 Results 叙述。** 为量化审查过程特征带来的时间泄漏影响，我们对比了 `pre-review` 与 `full` 两组特征集。所有模型在加入 review 特征后均获得明显提升：SVM 的 F1 从 0.810 提升到 0.862（+0.053），RF 从 0.813 提升到 0.870（+0.057），XGBoost 从 0.798 提升到 0.868（+0.069），LightGBM 从 0.804 提升到 0.857（+0.053）。其中 XGBoost 的增幅最大，说明 boosting 模型能更充分利用 review 数量、reviewer 数量和评论密度等后验信号。

这一结果提供了一个重要的方法论边界：如果仅报告 full 特征集性能，会高估模型在真实代码审查前置预测中的表现。对后续实验三、实验四和实验七而言，应优先将 `pre-review` 结果作为可部署基线，而将 `full` 结果作为含时间泄漏的上界参考。

**可引用结论。** Review-process features raise F1 by about 5-7 percentage points, but the gain reflects future information rather than deployable pre-review predictive ability.

## Figure 3. Pre-review Random Forest 特征重要性

**建议图注。** Figure 3 | Feature importance of the deployable Random Forest model. The left panel ranks the top pre-review features by Gini importance; the right panel aggregates importance by feature family. Colors denote text, AST, CFG and change-statistics features.

**正文 Results 叙述。** 在无泄漏的 pre-review 设置下，RF 的重要性排序显示文本特征构成最主要信号来源。单个最重要特征为 `tfidf_github`（0.0595），随后是 `tfidf_py`（0.0414）、`avg_commit_msg_len`（0.0310）、`tfidf_pull`（0.0264）、`title_len`（0.0246）和 `body_len`（0.0228）。这说明 PR 标题、正文、commit message 以及文本中出现的上下文链接或关键词，对合并结果具有较强预测作用。

除文本特征外，代码修改量和结构复杂度也提供了辅助信号。`deletions`、`churn_ratio` 与 `additions` 进入前列，说明修改规模和修改方向会影响 PR 是否被接受；`ast_avg_branching` 和 `ast_max_depth` 进入重要特征列表，表明 AST 结构复杂度对传统 ML 模型仍有贡献。相比之下，CFG 特征在家族汇总中的贡献较弱，提示当前基于 diff 片段构建的轻量 CFG 特征更适合作为补充，而不是主要判别依据。

**可引用结论。** 在真实部署设置中，文本上下文和改动规模是 Merge Prediction 的主要人工特征信号，AST 结构特征提供次级解释力，CFG 特征贡献较小。

## Figure 4. Full Random Forest 特征重要性

**建议图注。** Figure 4 | Feature importance of the Random Forest model with full features. The plot separates review-process features from deployable text, structural and change-statistics features to show which signals dominate the upper-bound model.

**正文 Results 叙述。** 在 full 设置下，最重要的两个特征变为 `num_reviews`（0.1482）和 `num_reviewers`（0.1304），二者显著高于所有 pre-review 特征。其他 review-process 特征也进入前列，包括 `review_density`（0.0296）、`num_review_comments`（0.0290）和 `num_issue_comments`（0.0177）。与此同时，`tfidf_github` 仍保留一定重要性（0.0369），但其相对地位已被审查过程变量超过。

这一结果直接解释了 Figure 2 中 full 设置性能提升的来源。模型并非仅通过更好地理解代码改动而提升，而是利用了 PR 在审查过程中产生的交互强度。审查次数、审查者数量和评论密度很可能与维护者关注度、讨论充分性和最终合并决策高度相关，但它们在 PR 创建时不可获得，因此不能用于前置预测系统。

**可引用结论。** Full 模型的性能上界主要由 review-process 特征驱动；这验证了标签泄漏消融的必要性。

## Figure 5. Pre-review ROC 曲线

**建议图注。** Figure 5 | ROC curves of pre-review Merge Prediction models. Curves report true positive rate against false positive rate on the held-out test set using only deployable features.

**正文 Results 叙述。** ROC 曲线进一步比较了模型对 merged 与 non-merged PR 的排序能力。在 pre-review 设置下，RF 的 ROC-AUC 最高（0.834），XGBoost 次之（0.820），LightGBM 为 0.809，SVM 为 0.769。该排序与 F1 结果基本一致，说明 RF 不仅在固定分类阈值下表现最好，也在不同阈值下保持了更稳定的区分能力。

pre-review AUC 仍未达到 full 设置水平，说明仅依赖提交时特征时，模型对部分边界样本的排序存在不确定性。这与代码审查任务本身一致：PR 是否合并不仅由代码 diff 决定，也受到维护者反馈、项目计划、审查文化和后续修改等过程因素影响。

**可引用结论。** RF 在无泄漏条件下具有最好的整体判别能力，但 pre-review 特征无法完全捕获审查过程中的后验决策因素。

## Figure 6. Full ROC 曲线

**建议图注。** Figure 6 | ROC curves of Merge Prediction models under the full feature setting. The full setting includes review-process features and therefore represents an upper-bound analysis rather than a deployable pre-review protocol.

**正文 Results 叙述。** 在 full 设置下，所有模型的 ROC-AUC 均提升到 0.884 以上。RF 的 ROC-AUC 最高，为 0.913；XGBoost 为 0.890；SVM 为 0.885；LightGBM 为 0.884。与 pre-review 设置相比，RF 的 AUC 从 0.834 提升到 0.913，增幅约 0.079，说明审查过程特征不仅提高了固定阈值下的 F1，也增强了模型对样本风险排序的能力。

这种提升在解释上应保持谨慎。Full ROC 曲线展示的是包含未来信息后的预测上界，可用于分析审查过程信号的价值，但不能作为实验七 VSCode 插件在 PR 提交时预测的目标性能。

**可引用结论。** Full ROC 曲线说明 review-process features 显著增强模型区分能力，但其结果应作为上界而非部署基准。

## Figure 7. Pre-review 混淆矩阵

**建议图注。** Figure 7 | Confusion matrices of pre-review models on the held-out test set. Each cell shows the number of pull requests and the row-normalized proportion for actual non-merged and merged classes.

**正文 Results 叙述。** 混淆矩阵揭示了 pre-review 模型的具体错误类型。SVM 正确识别 115 个 merged PR，但将 35 个 non-merged PR 错判为 merged，表现出较强的合并倾向；这与其高 recall（0.858）和较低 precision（0.767）一致。RF 将 non-merged PR 的误判数降至 15 个，是四个模型中对负类最谨慎的模型，但同时漏掉 32 个实际 merged PR，因此 recall 低于 SVM。

XGBoost 和 LightGBM 的错误结构介于二者之间。XGBoost 产生 21 个 false positive 和 31 个 false negative；LightGBM 产生 28 个 false positive 和 25 个 false negative。若实验目标是减少将不可合并 PR 误判为可合并的风险，RF 更合适；若目标是尽可能召回可合并 PR，SVM 或 LightGBM 可作为补充参考。

**可引用结论。** Pre-review 模型存在 precision-recall trade-off；RF 更保守、更少误报，SVM 更偏向召回 merged PR。

## Figure 8. Full 混淆矩阵

**建议图注。** Figure 8 | Confusion matrices of full-feature models on the held-out test set. Compared with pre-review models, the full-feature setting reduces both false positives and false negatives for most classifiers.

**正文 Results 叙述。** 加入审查过程特征后，模型错误数整体下降。RF 在 full 设置下正确识别 56 个 non-merged PR 和 117 个 merged PR，仅产生 18 个 false positive 与 17 个 false negative；相比 pre-review RF，其 false negative 从 32 个降至 17 个，同时 false positive 仅从 15 个小幅升至 18 个。XGBoost 的 full 混淆矩阵也较均衡，false positive 为 16，false negative 为 19。

SVM 和 LightGBM 在 full 设置下同样提升明显。SVM 的 false positive 从 35 降至 19，false negative 从 19 降至 18；LightGBM 的 false positive 从 28 降至 18，false negative 从 25 降至 20。这些变化说明 review-process 特征使模型更容易同时识别 merged 与 non-merged 两类样本，但仍需注意其时间泄漏属性。

**可引用结论。** Full 特征减少了多数模型的分类错误，尤其降低了 pre-review 中较多的 false positive 或 false negative，但该改进来自后验审查信息。

## Figure 9. Pre-review 跨仓库性能

**建议图注。** Figure 9 | Repository-specific F1 under the pre-review setting. The left panel shows each repository's merge rate in the full modelling dataset; the heatmap reports per-repository F1 for each model on the test split.

**正文 Results 叙述。** 跨仓库分析显示，模型性能受到项目合并率和审查文化的显著影响。`home-assistant/core` 的整体合并率最高（92.0%），各模型 pre-review F1 均达到 0.963 以上，RF 达到 0.975。`apache/airflow` 的合并率为 80.7%，各模型 F1 处于 0.818-0.848 区间，仍表现稳定。

低合并率或类别更均衡的仓库更难预测。`django/django` 的合并率最低（46.7%），pre-review RF F1 仅为 0.500，LightGBM 为 0.651，SVM 为 0.609，XGBoost 为 0.541。`pandas-dev/pandas` 的合并率为 52.6%，RF F1 为 0.720，XGBoost 和 LightGBM 分别降至 0.621 和 0.563。该结果说明，整体平均指标可能掩盖仓库级分布偏移；报告传统 ML 基线时必须同时展示分仓库性能。

**可引用结论。** Merge Prediction 的难度具有强仓库依赖性；高合并率仓库更易预测，而合并率均衡或审查更严格的仓库需要更多上下文。

## Figure 10. Full 跨仓库性能

**建议图注。** Figure 10 | Repository-specific F1 under the full feature setting. Review-process features improve performance most strongly for repositories where pre-review prediction was difficult.

**正文 Results 叙述。** 在 full 设置下，困难仓库的性能提升最明显。`django/django` 中 RF F1 从 pre-review 的 0.500 提升到 0.800，XGBoost 从 0.541 提升到 0.808，SVM 从 0.609 提升到 0.784，LightGBM 从 0.651 提升到 0.792。对于 `huggingface/transformers`，各模型 full F1 也提升到 0.840-0.880 区间。

相比之下，高合并率仓库 `home-assistant/core` 在 full 设置下并未进一步提升，RF 从 0.975 小幅降至 0.963。这说明 review-process 特征最能帮助那些仅凭提交时信息难以判断的项目，而对于合并率高度偏向正类的项目，pre-review 特征和类别先验已经能达到较高 F1。

**可引用结论。** Review-process features 对低合并率、边界更不确定的仓库更有价值，这进一步说明仓库级上下文和审查过程信息是后续实验的重要改进方向。

## Figure 11. 训练时间

**建议图注。** Figure 11 | Training time of traditional machine-learning baselines. The dot plot uses a log-scaled x-axis because all models trained within a fraction of a second in the recorded run.

**正文 Results 叙述。** 训练时间结果表明，传统机器学习基线具有很低的计算成本。在记录的训练时间中，LightGBM 最快（0.021 s），SVM 为 0.046 s，XGBoost 为 0.095 s，RF 为 0.116 s。即使考虑不同运行环境的波动，四类模型都属于轻量级训练流程，适合作为后续 CodeBERT、LLM 和 VSCode 插件实验的快速可复现实验基线。

该结果也解释了传统 ML 在实验二中的定位：它未必能充分学习代码语义，但可以用较低成本提供可解释的性能参照。结合 Figure 3 和 Figure 4，RF 还提供了特征重要性分析，使实验报告不仅能比较预测效果，也能解释哪些人工特征驱动了预测。

**可引用结论。** 传统 ML 模型训练成本低、复现实验快，适合作为后续深度学习和 LLM 方法的可解释基线。

## 报告正文建议组织方式

建议在实验报告的“实验结果与分析”部分按如下顺序组织，而不是按文件名机械罗列。

1. **总体性能。** 先放 `model_performance_matrix`，说明 pre-review 是真实部署基线，full 是含审查过程信息的上界。
2. **泄漏消融。** 紧接 `label_leakage_ablation`，量化 full 相对 pre-review 的 F1 增量，并明确其时间泄漏边界。
3. **模型判别能力与错误类型。** 放 `roc_curves_pre/full` 与 `confusion_matrices_pre/full`，分别说明阈值无关的排序能力和具体 false positive / false negative 结构。
4. **可解释性分析。** 放 `feature_importance_rf_pre/full`，回答“哪类特征贡献最大”：pre-review 中主要是文本和改动规模，full 中主要是 review-process 特征。
5. **跨仓库泛化。** 放 `cross_repo_performance_pre/full`，说明整体平均性能受仓库合并率和审查文化影响，强调分仓库评估必要性。
6. **效率分析。** 最后放 `training_time`，作为传统 ML 低成本、可解释基线的补充证据。

## 可直接放入 Results 小节的整合段落

基于实验一构建的人类代码 Pull Request 数据集，我们在 208 个测试样本上评估了 SVM、Random Forest、XGBoost 和 LightGBM 四类传统机器学习模型。为避免时间泄漏，实验同时设置了仅包含提交时可获得信息的 pre-review 特征集，以及包含审查过程信息的 full 特征集。总体性能矩阵显示，在 pre-review 设置下，RF 取得最高 F1（0.813）和 ROC-AUC（0.834），说明由 AST、CFG、统计信息和文本信息构成的人工特征能够形成有效的 Merge Prediction 基线。Full 设置进一步将 RF 的 F1 提升到 0.870、ROC-AUC 提升到 0.913，但特征重要性分析显示，该提升主要由 `num_reviews`、`num_reviewers` 和 `review_density` 等审查过程特征驱动，因此应解释为 post-review upper bound。

消融结果进一步量化了这一边界：加入审查过程特征使四类模型 F1 提升约 5-7 个百分点，其中 XGBoost 的增幅最大（+0.069）。在无泄漏设置下，RF 的重要特征主要来自文本上下文和改动规模，例如 `tfidf_github`、`tfidf_py`、`avg_commit_msg_len`、`deletions` 和 `churn_ratio`；AST 结构复杂度提供辅助信号，而 CFG 特征贡献较弱。跨仓库分析表明，模型在高合并率仓库 `home-assistant/core` 上表现最好，而在合并率更均衡的 `django/django` 上明显下降，说明 Merge Prediction 存在仓库级分布偏移。综上，实验二建立了一个可复现、低成本且可解释的传统机器学习基线，同时明确了 review-process features 的上界价值和部署限制。
