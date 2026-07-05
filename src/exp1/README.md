# Exp 1 (dataset construction)

Guidance for `src/exp1/`.

## Purpose

Exp 1 acquires, cleans, and does EDA on a GitHub code-review dataset that is the **shared foundation for all later experiments**. Its schema is deliberately shaped by downstream needs — preserve them when changing anything here:

- Exp 2 (ML on AST/CFG) → needs function-level `patch`; repos are Python-dominant so parsing is uniform.
- Exp 3 (Code2Vec/CodeBERT generation) → needs `(diff_hunk → review comment)` pairs.
- Exp 4/6 (LLM context) → needs PR body, commit msg, and `base_sha`/`head_sha` to re-fetch repo context later.
- Exp 5 (AI-code review) → needs the `is_ai_code` / `has_ai_reviewer` labels.

## Commands

```bash
uv run python -m src.exp1.fetch                                          # full: 5 repos × 300 closed PRs + AI oversampling (~1.5-2h)
uv run python -m src.exp1.fetch --repo django/django --limit 20 --no-ai  # small sample for testing
uv run python -m src.exp1.build_dataset                                  # raw JSON cache → 5 normalized tables
uv run python -m src.exp1.analyze                                        # EDA: stats + 7 figures
uv run python -m src.exp1.github_client                                  # smoke-test token + rate limit
```

## Pipeline

Three stages, each writing to `data/` (git-ignored, regenerable):

```
GitHub API ──fetch──▶ data/raw/<owner>__<repo>/pr_<n>.json ──build──▶ data/processed/*.{parquet,csv} ──analyze──▶ data/figures/*.png
```

Non-obvious design decisions:

- **`github_client.py` is the only code touching HTTP.** It handles auth, primary rate-limit (sleeps until `X-RateLimit-Reset` when remaining hits 0), secondary-limit / 5xx exponential backoff, and Link-header pagination. Other modules only call `get_json` / `paginate`.
- **Resumable fetch**: one JSON per PR; PRs whose cache file already exists are skipped, so interrupting and re-running is safe. Each PR bundles 6 sub-resources: `detail`, `files`, `reviews`, `review_comments`, `issue_comments`, `commits`.
- **Closed PRs only** — a closed PR's fate is decided, giving a clean `is_merged` label. Open PRs are excluded.
- **AI oversampling**: beyond the base 300/repo, `fetch.py` uses the Search API to pull PRs with AI signals (flagged `_oversampled`), because AI-generated-code PRs are rare and Exp 5 needs enough of them.

## Dataset schema

`build_dataset.py` normalizes raw JSON into **5 relational tables** joined by `repo` + `pr_number` (not one flat table — a PR has many files/comments/commits). Build-time cleaning: calls `ai_labeling.label_pr` per PR, detects language by extension, drops PRs with no file changes, dedups, flags oversized diffs (`has_patch=False`, `n_files_missing_patch`).

- **`prs`** — one row per PR. Carries `is_merged` (the label), `num_*` stats, AI labels + signal columns, and `base_sha`/`head_sha`/`merge_commit_sha` for later repo-context re-fetching. Note `num_reviews` (review actions) ≠ `num_reviewers` (distinct people); the three comment counts differ: `num_reviews` (formal reviews with a verdict), `num_review_comments` (inline, pinned to code), `num_issue_comments` (conversation-tab chatter).
- **`files`** — the `patch` column is the raw diff for Exp 2/3/4.
- **`review_comments`** — `path` + `diff_hunk` + `body` is the (code → comment) pair. **Without this, Exp 3's generation task has no data.**
- **`reviews`**, **`commits`** — review verdicts and commit messages.

Every table is written as **`.parquet` (canonical, used by all downstream code) and `.csv` (human inspection only)**. CSV multi-line fields are RFC-4180-quoted and correct — never parse them with `wc`/`grep`/`awk`; use pandas or read the parquet.

## AI labeling (heuristic — noisy)

`ai_labeling.py` + dictionaries in `config.py` produce two independent labels by matching traces AI tools leave, and **record which specific signal fired** into `ai_code_signals` / `ai_reviewer_signals` for auditability:

- `is_ai_code`: bot author or `Co-authored-by: Copilot/Claude/...` (strong signals); AI text hints / labels (weak signals).
- `has_ai_reviewer`: any reviewer/commenter login in `AI_REVIEWER_BOTS` (coderabbitai, copilot reviewer, sourcery, …).

Lossy: false negatives when a tool leaves no trace; weak signals can false-positive. When precision matters, filter to strong-signal rows via the `*_signals` columns rather than trusting the boolean.

## Configuration

`config.py` centralizes the 5 target repos, `PRS_PER_REPO`, `AI_OVERSAMPLE_PER_REPO`, rate-limit params, and the AI-signal dictionaries. Change behavior here, not in logic code.
