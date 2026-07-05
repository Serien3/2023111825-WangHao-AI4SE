# AI4SA-Exp1

<p align="center">
  <a href="README.md">English</a> |
  <a href="README-zh.md">简体中文</a>
</p>

## Overview

Experiment 1 builds a GitHub code-review dataset for the follow-up Intelligent Software Engineering experiments. It collects closed pull requests from five active Python-centered repositories, extracts pull request metadata, changed files, commits, formal reviews, inline review comments, and discussion comments, then normalizes them into relational tables for later modeling.

The experiment also performs exploratory data analysis on merge outcomes, inline review density, pull request size, reviewer activity, author association, AI participation, repository differences, and pull request lifetime. The overview figure below summarizes the final dataset.

![Experiment 1 dataset overview](results/figures/eda_overview.png)

## Table of Contents

- [Key Feature](#key-feature)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Smoke Test GitHub Access](#1-smoke-test-github-access)
  - [2. Fetch a Small Sample](#2-fetch-a-small-sample)
  - [3. Fetch the Full Dataset](#3-fetch-the-full-dataset)
  - [4. Build Normalized Tables](#4-build-normalized-tables)
  - [5. Run Exploratory Data Analysis](#5-run-exploratory-data-analysis)
- [Limitations](#limitations)

## Key Feature

- Provides a reusable code-review dataset foundation for later experiments, especially merge prediction and review-comment generation.
- Produces normalized tables
- Preserves raw diff patches in `files.patch`, which can support AST/CFG feature extraction, CodeBERT input construction, and LLM prompt context.
- Extracts `(diff_hunk -> review comment)` pairs from inline review comments for review-comment generation tasks.
- Stores merge labels through `prs.is_merged`, using only closed pull requests so the target label is known.
- Adds heuristic AI labels, including `is_ai_code`, `has_ai_reviewer`, and the underlying signal columns for audit and filtering.
- Generates reusable figures and summary statistics under `results/figures/`.

## Installation

The Python environment is managed at the repository root and is shared by all experiments. Keep `pyproject.toml`, `uv.lock`.

From the repository root:

```bash
uv sync
```

GitHub API collection requires a token. Create or update the root `.env` file:

```bash
GITHUB_TOKEN=<your_github_token>
```

All commands below should be run from the repository root:

```bash
cd /home/wzsyh/ai-software-engineer/Experiment1
```

## Usage

### 1. Smoke Test GitHub Access

Check whether the token can be loaded and whether the GitHub API rate limit is readable:

```bash
uv run python -m src.github_client
```

### 2. Fetch a Small Sample

Use a small sample before running the full collection pipeline:

```bash
uv run python -m src.fetch --repo django/django --limit 20 --no-ai
```

Raw PR cache files are written to:

```text
Experiment1/results/raw/
```

### 3. Fetch the Full Dataset

Collect the configured five-repository dataset with AI-signal oversampling:

```bash
uv run python -m src.fetch
```

This step is resumable. If a PR JSON file already exists, it is skipped on the next run.

### 4. Build Normalized Tables

Convert raw JSON files into normalized Parquet and CSV tables:

```bash
uv run python -m src.build_dataset
```

Outputs are written to:

```text
Experiment1/results/processed/
```

### 5. Run Exploratory Data Analysis

Generate statistics and visualization figures:

```bash
uv run python -m src.analyze
```

Figures are written to:

```text
Experiment1/results/figures/
```

## Limitations

- AI-code and AI-reviewer labels are heuristic. They rely on visible signals such as bot accounts, co-author markers, and text hints, so false negatives and weak-signal false positives are possible.
- Inline review comments are sparse. Many PRs have no `review_comments`, which limits the amount of direct `(diff_hunk -> comment)` training data for later generation experiments.
- Repository behavior differs strongly. Merge rate, review density, and PR size vary across communities, so later models should use repository-aware splits or report per-repository metrics.
- Large PRs and missing patches can affect downstream parsing. Some changed files may not include `patch` because GitHub omits oversized diffs.
- The sample is biased toward Python-centered repositories and GitHub-hosted open-source workflows, so conclusions may not generalize to private projects or other programming ecosystems.
- AI oversampling improves sample size for Experiment 5, but it changes the natural prevalence of AI-related PRs. Analyses that estimate population-level AI participation should account for the `oversampled` flag.
