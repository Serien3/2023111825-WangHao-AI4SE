# Exp 2 (traditional ML for Merge Prediction)

Guidance for `Experiment2/src/`.

## Purpose

Exp 2 builds the **traditional-ML baseline** for Merge Prediction — SVM & Random Forest
(+ XGBoost/LightGBM as extensions) trained on hand-crafted AST/CFG/statistical/text
features. It provides the performance baseline that Exp 3 (deep learning) and Exp 4 (LLM)
are compared against.

Key design decisions (see `docs/design-docs/exp2-features.md` for full rationale):

- **Human code only** — filters `is_ai_code == False` (AI code is Exp 5's subject).
- **Diff-only, no repo re-fetch** — AST/CFG are built from Exp 1's `patch` (diff), not
  full files. Verified sufficient: 79% of hunks carry `def`/`class` context, tree-sitter
  parses fragments error-tolerantly, and the guidebook only needs aggregate structural
  counts. Full-file re-fetch is Exp 6's scope, not ours.
- **Label-leakage analysis** — review-process features (`num_reviews`, `num_reviewers`,
  …) are only known *after* review, so we train two feature sets (`pre` = deployable,
  `full` = upper bound) and compare.

## Commands

```bash
# 1. Feature extraction: Exp1 tables → 98-dim feature matrix (one row per PR)
uv run python -m src.feature_extraction                 # full (1036 PRs, ~15s)
uv run python -m src.feature_extraction --limit 50      # small sample for testing

# 2. Train (two feature sets × 4 models each). GridSearchCV + 5-fold CV.
uv run python -m src.train --model all --feature-set full   # with review features
uv run python -m src.train --model all --feature-set pre    # pre-review (no leakage)
uv run python -m src.train --model svm --feature-set full   # single model

# 3. Evaluate + figures (per feature set)
uv run python -m src.evaluate --feature-set full
uv run python -m src.evaluate --feature-set pre

# 4. Label-leakage ablation (needs both sets evaluated first)
uv run python -m src.evaluate --ablation
```

## Pipeline

Four stages, each writing to `Experiment2/results/` (git-ignored, regenerable):

```
Exp1 processed/*.parquet
  ──feature_extraction──▶ results/features/features.{parquet,csv}   (1036 PR × 98 feat)
  ──train──▶ results/models/{svm,rf,xgboost,lightgbm}_{full,pre}.pkl + split/scaler
  ──evaluate──▶ results/metrics/*.json + results/figures/*.png
```

Non-obvious design decisions:

- **`config.py` is the single source of truth** for paths, feature groups
  (`PRE_REVIEW_FEATURES` vs `REVIEW_PROCESS_FEATURES`), param grids, and resource limits.
- **OOM protection is deliberate, not incidental.** This machine is 32-core but
  memory-constrained (WSL2 ~7.6 GB). `GridSearchCV(n_jobs=-1)` would fork 32 workers ×
  each copying data × each model spawning its own threads → memory/thread oversubscription
  crashed the whole WSL2 VM. So `config.N_JOBS=4`, `config.MODEL_N_THREADS=2`, and
  `train.py` pins `OMP/MKL/OpenBLAS` thread env-vars to 2 **before importing numpy**.
  Measured peak ~3 GB. Tune `N_JOBS` down to 2 if your machine has less RAM.
- **Stratified-by-repo split.** Merge rates span 48–90% across repos (Exp 1 EDA), so
  train/val/test (662/166/208) are stratified by `repo`; splits verified zero-overlap.
- **Scaler fit on train only** — `StandardScaler` statistics never see val/test.

## Feature groups (98 dims)

| Group | Count | Examples | Source |
|---|---|---|---|
| AST | 24 | node counts by type, tree depth, branching, error nodes | tree-sitter on reconstructed post-change code |
| CFG | 4 | nodes, edges, McCabe complexity, max nesting | control-flow keyword approximation |
| Statistical | 11 | additions, deletions, num_commits, num_reviewers* | Exp1 prs table |
| Text | 59 | title/body/commit lengths, keyword flags, TF-IDF top-50 | PR title + body + commit msgs |

\* `num_reviews`, `num_reviewers`, `num_review_comments`, `num_issue_comments`,
`review_density` are **review-process features** (post-hoc); excluded from the `pre` set.

## Results summary

Random Forest is the best model. Test-set F1 (208 PRs):

| Feature set | SVM | RF | XGBoost | LightGBM |
|---|---|---|---|---|
| Full (with review feat.) | 0.862 | **0.870** | 0.868 | 0.857 |
| Pre-review (deployable) | 0.810 | **0.813** | 0.798 | 0.804 |

Review-process features add ~5–7 F1 points (and boost RF AUC 0.834→0.913) but carry
temporal leakage. Pre-review F1 of 0.81 shows the code change alone carries most signal.
Full analysis and per-repo breakdown in `docs/design-docs/exp2-features.md`.

## Files

- `config.py` — paths, feature-group definitions, param grids, resource limits.
- `feature_extraction.py` — diff reconstruction, AST/CFG/stat/text extraction, TF-IDF.
- `train.py` — stratified split, scaling, GridSearchCV for 4 models, timing.
- `evaluate.py` — metrics (Acc/P/R/F1/AUC), per-repo breakdown, ablation, 6 figures.
