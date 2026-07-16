# AI4SA-Exp6

[English](README.md) | 简体中文

## Overview

实验六在不重新训练或更换模型的前提下，改进大语言模型对 AI 生成代码的代码审查能力。实验复用实验五的 AI 代码样本和匹配人类对照样本，研究更丰富的软件工程上下文与 Prompt 优化能否改善合并预测和审查意见生成任务。

实验将上下文组织为 L0 至 L4 的单调阶梯，依次加入局部 diff、Pull Request 与 Commit 元数据、修改前后的完整函数或文件、关联 Issue 与历史审查意见，以及仓库级轻量词法检索。同时，实验将前序实验的最佳 Prompt 与 P5 Self-Reflection、P6 多轮交互进行比较。下面的概要图展示了主要实验结果。

![实验六结果概要图](results/figures/fig1_result_overview.png)

## Table of Contents

- [Key Feature](#key-feature)
- [Installation](#installation)
- [Requirements](#requirements)
- [Usage](#usage)
  - [1. 查看配置与实验矩阵](#1-查看配置与实验矩阵)
  - [2. 复现前序最佳 Prompt 选择](#2-复现前序最佳-prompt-选择)
  - [3. 运行离线测试](#3-运行离线测试)
  - [4. 运行小规模端到端样本](#4-运行小规模端到端样本)
  - [5. 运行完整实验矩阵](#5-运行完整实验矩阵)
  - [6. 评估预测结果](#6-评估预测结果)
  - [7. 重新生成图表](#7-重新生成图表)
- [Limitations](#limitations)

## Key Feature

- 直接复用实验五的 AI 生成代码样本和匹配人类样本，使实验六能够与前序性能差距分析进行对比。
- 构建从 L0 局部 diff 到 L4 仓库级上下文的五级上下文阶梯，并为每一级设置明确的字符和 token 预算。
- 新增 P5 Self-Reflection 和 P6 多轮交互 Prompt，同时保留根据前序实验结果选出的最佳 Prompt 作为基线。
- 每个任务采用聚焦的 10 条件矩阵，而不是完整笛卡尔积，覆盖上下文消融、Prompt 消融、改进归因和匹配人类对照。
- 使用 balanced accuracy、各类别 precision、recall 与 F1、混淆矩阵和 non-merge recall 评估合并预测，避免只依赖 accuracy 和 merge 类 F1。
- 使用独立的 `deepseek-v4-pro` 裁判模型，从相关性、可操作性、正确性和幻觉四个维度评估生成意见，同时保留 BLEU 和 ROUGE 以便与前序实验比较。
- 将 GitHub 拉取缓存与实验四共享的 LLM 缓存分离，并在 LLM 缓存键中加入模型标识，防止执行模型与裁判模型的响应相互覆盖。
- 在 `results/` 下生成预测表、机器可读指标、实验五基线差值、出版级图表及其源数据 CSV。

## Installation

建议使用 `uv` 复现实验环境，或参考 `pyproject.toml` 配置等价的 Python 环境。

在仓库根目录运行：

```bash
uv sync
```

模型推理和 L2-L4 上下文构建需要 API 凭据。请在仓库根目录创建或更新 `.env` 文件：

```bash
GITHUB_TOKEN=<your_github_token>
DEEPSEEK_API_KEY=<your_deepseek_api_key>
```

下面所有命令都应在实验六目录中运行：

```bash
cd /home/wzsyh/ai-software-engineer/Experiment6
```

## Requirements

- Python >= 3.12（已在 v3.12.3 测试）
- pandas >= 3.0.3 和 pyarrow >= 24.0.0 用于样本表与预测表处理
- openai >= 2.44.0 用于访问兼容 OpenAI 协议的 DeepSeek API
- PyGithub >= 2.9.1 和 requests >= 2.34.2 用于获取 GitHub 上下文
- scikit-learn >= 1.9.0 用于计算分类指标
- sacrebleu >= 2.6.0 和 rouge-score >= 0.1.2 用于计算生成指标
- matplotlib >= 3.11.0 和 seaborn >= 0.13.2 用于结果可视化

还需要具备以下输入和配置：

+- 系统路径中可用的 `uv`，用于复现实验环境
+- 实验五生成的 `Experiment5/results/samples/` 和 `Experiment5/results/metrics/`
+- 仓库根目录 `.env` 中的 `GITHUB_TOKEN`，用于 L2-L4 GitHub 上下文拉取
+- 仓库根目录 `.env` 中的 `DEEPSEEK_API_KEY`，用于执行模型和裁判模型调用

## Usage

### 1. 查看配置与实验矩阵

打印模型配置、L0-L4 上下文定义、Prompt 选择结果，以及每个任务使用的 10 个实验条件：

```bash
uv run python -m src.config
```

### 2. 复现前序最佳 Prompt 选择

根据实验五指标选择前序最佳 Prompt，并检查选择结果是否与 `config.PSTAR` 一致：

```bash
uv run python -m src.select_pstar
```

选择记录会写入：

```text
Experiment6/results/metrics/pstar_selection.json
```

### 3. 运行离线测试

在不访问网络、不产生付费 API 调用的情况下，验证上下文构建、Prompt 行为、执行模型与裁判模型的缓存隔离、输出解析和评估指标：

```bash
uv run python -m src.tests
```

### 4. 运行小规模端到端样本

正式运行完整矩阵前，分别对一个样本执行一个分类条件和一个生成条件：

```bash
uv run python -m src.run_experiments --task classify --only L0 P1 --limit 1
uv run python -m src.run_experiments --task generate --only L4 P5 --limit 1
```

L4 生成样本会覆盖 GitHub 上下文拉取、模型推理和 LLM 裁判流程。拉取的上下文缓存在 `results/fetch_cache/`，预测结果写入 `results/predictions/`。

### 5. 运行完整实验矩阵

可以分别运行两个任务，也可以依次运行全部任务：

```bash
uv run python -m src.run_experiments --task classify
uv run python -m src.run_experiments --task generate
uv run python -m src.run_experiments --task all
```

使用 `--only LEVEL PROMPT` 可运行单个条件，使用 `--limit N` 可限制每个条件的样本数，使用 `--no-judge` 可跳过生成任务的裁判调用。每个任务共评估 10 个条件，包括 5 个上下文阶梯条件、3 个 L4 Prompt 条件、1 个 L0 归因条件和 2 个人类对照条件，重复条件只计算一次。

### 6. 评估预测结果

计算分类与生成指标，将可比条件与实验五基线进行对比，并生成最终图表：

```bash
uv run python -m src.evaluate
```

使用 `--task classify` 或 `--task generate` 可只评估一个任务，使用 `--no-figures` 可只计算指标。输出目录为：

```text
Experiment6/results/metrics/
Experiment6/results/figures/
```

### 7. 重新生成图表

根据已有指标和预测结果，重新生成 6 组出版级图表及其源数据 CSV：

```bash
uv run python -m src.nature_viz
```

完整的实验结果解读见 [实验六结果分析.md](实验六结果分析.md)。

## Limitations

- AI 生成代码标签和匹配人类对照继承自实验五，因此启发式标签噪声或匹配误差也会传递到本实验。
- Merge 状态是软件开发流程结果，不是纯粹的代码质量标签。部分 PR 因流程原因未合并，因此需要结合具体案例解释 non-merge 预测错误。
- L4 使用轻量词法仓库检索，而不是语义检索、完整仓库克隆或完整依赖图，因此可能遗漏名称不同但行为相关的代码。
- GitHub 可能省略过大的 patch，关联 Issue 可能无法访问，API 请求失败也可能造成部分样本缺少高层上下文。落盘缓存提高了可复现性，但不能恢复本身不可用的数据。
- 生成任务裁判使用与执行模型相同供应商的更强模型。该设置减少了直接自评问题，但不能完全消除供应商相关的裁判偏差。
- P6 需要额外的模型轮次，生成任务裁判还会增加一次付费调用，因此应结合延迟和 API 成本判断质量改进是否值得部署。
- 实验采用聚焦矩阵和有限的 AI 代码样本。Bootstrap 置信区间能够展示不确定性，但不应将较小差异直接视为可广泛泛化的结论。
