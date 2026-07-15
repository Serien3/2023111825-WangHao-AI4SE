# AI4SE-Exp5

English | [简体中文](README-zh.md)

## Overview

Experiment 5 evaluates whether existing code-review models generalize from human-written code to AI-generated code. It is a downstream consumer of Experiments 1, 2, and 4: it does not retrain models or introduce new prompts or context strategies. Instead, it applies the established merge-prediction and review-comment-generation pipelines to AI-code pull requests, then compares their performance with a distribution-matched human control group.

The experiment studies two tasks: merge prediction with the Experiment 2 ML models and Experiment 4 LLM conditions, and review comment generation with the Experiment 4 LLM conditions. It produces model metrics, AI-versus-human comparisons, context-sensitivity analyses, error-case candidates, and six figures for the final analysis.

![Experiment 5 ML merge-prediction comparison](results/figures/fig1_ml_performance.png)

## Table of Contents

- [Key Features](#key-features)
- [Installation](#installation)
- [Requirements](#requirements)
- [Usage](#usage)
  - [1. Inspect the Configuration](#1-inspect-the-configuration)
  - [2. Create Reproducible Samples](#2-create-reproducible-samples)
  - [3. Extract ML Features](#3-extract-ml-features)
  - [4. Run ML Merge Prediction](#4-run-ml-merge-prediction)
  - [5. Smoke Test the LLM Pipeline](#5-smoke-test-the-llm-pipeline)
  - [6. Run the Full LLM Matrix](#6-run-the-full-llm-matrix)
  - [7. Evaluate and Visualize](#7-evaluate-and-visualize)
  - [8. Analyze Performance Gaps and Error Cases](#8-analyze-performance-gaps-and-error-cases)
- [Outputs](#outputs)
- [Limitations](#limitations)

## Key Features

- Reuses Experiment 2's saved SVM, Random Forest, XGBoost, and LightGBM merge-prediction models without retraining.
- Reuses the 16 Experiment 4 LLM conditions: four context levels (`C1`-`C4`) crossed with four prompts (`P1`-`P4`).
- Evaluates AI-generated code against human baselines and a human control group matched by repository and merge outcome.
- Keeps the ML inference boundary strict: recovered TF-IDF vocabulary and IDF statistics, scalers, and models are fit on upstream human data only; AI and control data are transformed only.
- Reuses Experiment 4's LLM cache, making reruns idempotent and allowing compatible human-control generation calls to reuse cached responses.
- Uses fixed seed `42` and persisted sample keys for reproducible LLM sampling.
- Produces metric JSON files, structured prediction files, candidate error cases, and six analysis figures.

## Installation

Use `uv` to reproduce the repository environment, or configure an equivalent Python environment from the root `pyproject.toml`.

From the repository root:

```bash
uv sync
```

LLM execution requires a DeepSeek API key in the repository-root `.env` file:

```bash
DEEPSEEK_API_KEY=<your_deepseek_api_key>
```

Experiment 5 consumes generated outputs from Experiments 1, 2, and 4. Before a clean reproduction, ensure these inputs are available:

- `Experiment1/results/processed/` for PR, file, commit, review, and inline-comment data.
- `Experiment2/results/features/`, `Experiment2/results/models/`, and `Experiment2/results/metrics/` for saved ML artifacts and human test metrics.
- `Experiment4/results/samples/` and `Experiment4/results/metrics/` for human LLM baselines; `Experiment4/results/cache/` is reused automatically for LLM responses.

All commands below should be run from the Experiment 5 directory:

```bash
cd /home/wzsyh/ai-software-engineer/Experiment5
```

## Requirements

- Python >= 3.12
- `uv` available on the system path
- Dependencies declared in the repository-root `pyproject.toml`, including pandas, pyarrow, scikit-learn, tree-sitter, xgboost, lightgbm, OpenAI-compatible client support, sacrebleu, rouge-score, matplotlib, and seaborn
- Completed or restored artifacts from Experiments 1, 2, and 4 as listed in [Installation](#installation)
- `DEEPSEEK_API_KEY` in the root `.env` file for `src.run_llm`; ML sampling, feature extraction, prediction, and local evaluation do not need API access

## Usage

### 1. Inspect the Configuration

Print the AI-code filtering configuration, model matrix, sample sizes, and shared LLM cache location:

```bash
uv run python -m src.config
```

### 2. Create Reproducible Samples

Create and persist four sample sets under a fixed seed:

- AI classification sample: 50 PRs stratified by repository and merge outcome.
- Matched human classification control: drawn from the Experiment 2 held-out test pool with the same stratum distribution.
- AI generation sample: all eligible PRs with a top-level inline-review-comment target.
- Human generation control: the Experiment 4 human generation sample.

```bash
uv run python -m src.sampling
```

Sample manifests are written to:

```text
Experiment5/results/samples/
```

### 3. Extract ML Features

Build Experiment 2-compatible feature matrices for the AI and matched-control classification pools. This restores the upstream fixed TF-IDF vocabulary and checks that the reconstructed human TF-IDF values match the Experiment 2 feature table.

```bash
uv run python -m src.ml_features
```

For a small local smoke test:

```bash
uv run python -m src.ml_features --limit 5
```

Feature matrices are written to:

```text
Experiment5/results/features/
```

### 4. Run ML Merge Prediction

Apply the four saved Experiment 2 models to both `pre` and `full` feature sets. The fitted scalers and models are used only for inference.

```bash
uv run python -m src.ml_predict
```

For a small local smoke test:

```bash
uv run python -m src.ml_predict --limit 5
```

Predictions and metrics are written to:

```text
Experiment5/results/predictions/ml_{ai,control}_{pre,full}.parquet
Experiment5/results/metrics/ml_metrics.json
```

### 5. Smoke Test the LLM Pipeline

Before the paid full matrix, verify one context/prompt condition on three samples. This performs LLM calls only when no compatible Experiment 4 cache entry exists.

```bash
uv run python -m src.run_llm --task all --only C1 P1 --limit 3
```

### 6. Run the Full LLM Matrix

Run the 16 conditions (`C1`-`C4` x `P1`-`P4`) for AI code and matched human controls. Classification uses 50 AI and 50 control samples; generation uses all eligible AI samples and the Experiment 4 human generation control sample.

```bash
uv run python -m src.run_llm --task classify --group ai
uv run python -m src.run_llm --task all
```

Useful switches:

```bash
uv run python -m src.run_llm --task classify --only C1 P1
uv run python -m src.run_llm --task generate --group control
uv run python -m src.run_llm --task all --limit 3
```

LLM outputs are written to:

```text
Experiment5/results/predictions/{classify,generate}_predictions.parquet
```

### 7. Evaluate and Visualize

Compute group-by-condition metrics, combine Experiment 2 and Experiment 4 human anchors with the AI and matched-control results, and generate available figures:

```bash
uv run python -m src.evaluate
```

To calculate metrics without rendering figures:

```bash
uv run python -m src.evaluate --no-figures
```

To rerun figure generation separately:

```bash
uv run python -m src.visualization
```

### 8. Analyze Performance Gaps and Error Cases

Generate the three evidence chains used in the experiment: performance-gap localization, `C1`-to-`C4` context sensitivity, and data-driven candidate error cases requiring manual attribution.

```bash
uv run python -m src.analysis
```

The recommended order is:

```text
sampling -> ml_features -> ml_predict -> run_llm (smoke test, then full matrix) -> evaluate -> analysis -> visualization
```

## Outputs

```text
Experiment5/results/
├── samples/
│   ├── classify_ai.json
│   ├── classify_control.json
│   ├── generate_ai.json
│   └── generate_control.json
├── features/
│   ├── ai_features.parquet
│   └── control_features.parquet
├── predictions/
│   ├── ml_{ai,control}_{pre,full}.parquet
│   └── {classify,generate}_predictions.parquet
├── metrics/
│   ├── ml_metrics.json
│   ├── llm_{classify,generate}_metrics.json
│   ├── human_vs_ai.json
│   ├── performance_gap_localization.json
│   └── context_sensitivity.json
├── cases/
│   ├── error_cases.json
│   └── error_case_attribution_template.md
└── figures/
    └── fig1_ml_performance.png ... fig6_error_cases.png
```

The principal output is `metrics/human_vs_ai.json`. It records Experiment 2/4 human anchors, AI-code results, matched-control results, and AI-versus-human or AI-versus-control metric differences. The error-case attribution template is deliberately a review aid: its automatic descriptions are hypotheses, not confirmed root causes.

## Limitations

- AI-code identification is inherited from Experiment 1's heuristic labels. Visible co-author markers, bot identities, labels, and textual signals can miss AI-assisted code or include weak-signal false positives.
- Merge labels describe repository outcomes, not code quality alone. Workflow policies, maintainer availability, and project priorities can cause a pull request to remain unmerged.
- AI and human pools differ in their natural repository and class distributions. The matched human control addresses repository and merge-outcome composition, but cannot remove every temporal, author, or project-level confounder.
- The `full` ML feature set includes review-process information. It is retained as an ablation for leakage gain, not as a realistic pre-review deployment setting; `pre` is the primary deployment-oriented result.
- BLEU and ROUGE measure lexical overlap with a single historical inline comment. They cannot fully assess correctness, usefulness, coverage, or alternative valid review comments.
- LLM output, availability, and latency depend on the external API. Shared caching improves reproducibility and cost control, but uncached reruns can still differ if the underlying service changes.