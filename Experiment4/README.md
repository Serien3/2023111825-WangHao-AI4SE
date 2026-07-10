# AI4SA-Exp4

English | [简体中文](README-zh.md)

## Overview

Experiment 4 evaluates prompt-engineered large language models (LLMs) for code review of human-authored pull requests. It reuses the normalized GitHub code-review tables from Experiment 1 and the human-authored Python pull-request split saved by Experiment 2. Rather than fine-tuning an LLM, the experiment systematically changes the information and instructions supplied to the model.

The implementation combines four pull-request context settings—Diff, Diff + PR description, Diff + commit messages, and all three—with four prompting strategies: Zero-shot, Few-shot, Chain-of-Thought, and Role-based prompting. It evaluates both Merge Prediction and Review Comment Generation on fixed 50-PR test samples, with DeepSeek accessed through the OpenAI-compatible SDK. The overview figure below shows the Merge Prediction landscape across the 16 context-prompt conditions.

![Experiment 4 Merge Prediction landscape](results/figures/fig1_classify_landscape.png)

For the detailed design, evaluation protocol, and result interpretation, see [the Experiment 4 design document](docs/design-docs/exp4-llm.md).

## Table of Contents

- [Key Feature](#key-feature)
- [Installation](#installation)
- [Requirements](#requirements)
- [Usage](#usage)
  - [1. Build Reproducible Samples](#1-build-reproducible-samples)
  - [2. Smoke Test LLM Access](#2-smoke-test-llm-access)
  - [3. Run a Small Condition](#3-run-a-small-condition)
  - [4. Run the Full Experiment Matrix](#4-run-the-full-experiment-matrix)
  - [5. Evaluate and Visualize Results](#5-evaluate-and-visualize-results)
- [Limitations](#limitations)

## Key Feature

- Implements the course-wide Merge Prediction and Review Comment Generation tasks with an LLM, extending the traditional-ML and deep-learning stages of the code-review pipeline.
- Builds four progressively richer PR contexts from Python diff patches, PR title/body, and commit messages, while applying explicit input-length budgets.
- Compares Zero-shot, Few-shot, Chain-of-Thought, and Role-based prompts in a complete 4 context × 4 prompt matrix for each task.
- Uses fixed, reproducible 50-PR test samples: classification is stratified by repository and merge label; generation uses PRs with human top-level inline review comments.
- Prevents few-shot leakage by selecting all classification and generation examples only from the Experiment 2 training split.
- Calls `deepseek-v4-flash` through an OpenAI-compatible client with task-specific temperatures, JSON-structured outputs, retry handling, latency recording, and disk-backed response caching.
- Evaluates classification with Accuracy, Precision, Recall, F1, parse-error rate, and latency; evaluates generation with corpus BLEU-4, ROUGE-1/2/L, and latency.
- Re-evaluates Experiment 2 SVM, Random Forest, XGBoost, and LightGBM models on the same 50-PR classification sample for a fair traditional-ML comparison.
- Writes reusable samples, cached responses, structured predictions, metrics, and publication-style SVG/PNG figures under `results/`.

## Installation

It is recommended to reproduce the project environment with `uv`, or configure an equivalent Python environment based on the repository-root `pyproject.toml`.

From the repository root:

```bash
uv sync
```

Experiment 4 consumes processed tables from Experiment 1 and the `pre` feature split, scaler, and trained models from Experiment 2. If these artifacts do not exist yet, build them first:

```bash
cd /home/wzsyh/ai-software-engineer/Experiment1
uv run python -m src.build_dataset

cd /home/wzsyh/ai-software-engineer/Experiment2
uv run python -m src.feature_extraction
uv run python -m src.train --model all --feature-set pre
```

LLM inference requires a DeepSeek API key. Add it to the repository-root `.env` file (which is git-ignored):

```bash
DEEPSEEK_API_KEY=<your_deepseek_api_key>
```

All Experiment 4 commands below should be run from:

```bash
cd /home/wzsyh/ai-software-engineer/Experiment4
```

## Requirements

- Python >= 3.12
- `openai` and `python-dotenv` for the OpenAI-compatible DeepSeek client and `.env` loading
- pandas and pyarrow for reading source tables and writing Parquet predictions
- scikit-learn for classification metrics and the same-sample Experiment 2 baseline
- sacrebleu and rouge-score for review-comment generation evaluation
- matplotlib, seaborn, and tqdm for figures and progress reporting

The following inputs and tools must be available:

- `uv` for reproducing the project environment
- A reachable DeepSeek API endpoint and `DEEPSEEK_API_KEY` in the repository-root `.env` file
- Experiment 1 processed tables in `Experiment1/results/processed/`
- Experiment 2 `pre` artifacts in `Experiment2/results/features/` and `Experiment2/results/models/`

The default API-based workflow does not require a local GPU.

## Usage

### 1. Build Reproducible Samples

Create the fixed test samples and few-shot examples. The 50-PR classification sample is stratified by repository and merge label; the 50-PR generation sample contains PRs with a human inline review comment. Few-shot examples are selected from the training split only.

```bash
uv run python -m src.sampling
```

Sample definitions are written to:

```text
Experiment4/results/samples/
```

### 2. Smoke Test LLM Access

Verify that the DeepSeek key can be loaded and that an API request succeeds. The smoke-test cache entry is removed when the check finishes.

```bash
uv run python -m src.llm_client
```

### 3. Run a Small Condition

Before paying for the full matrix, run three classification PRs under one context-prompt condition:

```bash
uv run python -m src.run_experiments --task classify --only C1 P1 --limit 3
```

`C1` is Diff-only context and `P1` is Zero-shot prompting. The command writes or merges its output into:

```text
Experiment4/results/predictions/classify_predictions.parquet
```

### 4. Run the Full Experiment Matrix

Run both tasks across all 4 contexts × 4 prompts × 50 PRs:

```bash
uv run python -m src.run_experiments --task all
```

This schedules 800 calls per task and 1,600 calls in total before cache hits. Responses are cached under `results/cache/`, so an interrupted run can resume without repeating completed semantic calls. To run only one task, replace `all` with `classify` or `generate`.

Structured predictions are written to:

```text
Experiment4/results/predictions/classify_predictions.parquet
Experiment4/results/predictions/generate_predictions.parquet
```

### 5. Evaluate and Visualize Results

Calculate both task metrics, re-evaluate the Experiment 2 traditional-ML models on the same classification sample, and generate the four result figures:

```bash
uv run python -m src.evaluate
```

Metrics and figures are written to:

```text
Experiment4/results/metrics/
Experiment4/results/figures/
```

Use `--no-figures` to calculate metrics without regenerating figures. To render the figures again from existing predictions and metrics:

```bash
uv run python -m src.visualization
```

## Limitations

- The default workflow depends on an online third-party LLM service. Availability, latency, pricing, and model behavior can change; caches preserve completed responses but cannot eliminate this external dependency.
- The main evaluation uses fixed 50-PR samples per task. They make the 16-condition experiment affordable and reproducible, but do not replace a full-test-set or cross-project evaluation.
- Context is built from stored Python diff patches, PR metadata, and commit messages. Long inputs are truncated to fixed budgets, and repository-level, cross-file, and historical context are intentionally left for later experiments.
- Merge Prediction is a retrospective label and can be affected by project policy, reviewer availability, and other factors that are not fully observable in the input context.
- Review Comment Generation is evaluated against one earliest top-level human inline comment per PR. Valid comments can focus on different issues or use different wording, so BLEU and ROUGE are comparative signals rather than complete measures of review usefulness.
- The implementation is limited to human-authored PRs with usable Python patches and four predefined prompt strategies. It does not directly establish performance on AI-generated code, other languages, or alternative LLMs.
