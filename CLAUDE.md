# AI4SE

This file provides guidance to AI Agent when working with code in this repository.

## Project context

Multi-experiment course project (Intelligent Software Engineering) on **code review**. The full spec is in `docs/实验指导书_代码审查.md` — **read it before making design decisions**. The project is problem-driven: every choice in an earlier experiment must serve the later ones, so treat the experiments as one pipeline, not independent tasks.

There are 7 experiments. Two tasks run through all of them: **Merge Prediction** (classification) and **Review Comment Generation**.

- **Exp 1** (built, `Experiment1/src/`): acquire + clean + EDA a GitHub code-review dataset — the shared data foundation for every later experiment. See `Experiment1/src/README.md`.
- **Exp 2**: traditional ML (SVM/RF) on AST/CFG + statistical features.
- **Exp 3**: DL (Code2Vec/CodeBERT) — needs `(diff_hunk → review comment)` pairs.
- **Exp 4/6**: LLM + prompt engineering — needs multi-granularity context.
- **Exp 5**: review of AI-generated code — needs the AI labels.
- **Exp 7** (built, `Experiment7/`): local FastAPI + native Web code-review workbench deploying DeepSeek and the Experiment 2 pre-review models. This intentionally replaces the originally proposed VS Code extension UI while preserving the HTTP model-service architecture. See `Experiment7/README-zh.md`.

## Environment

Uses **uv** for env/deps (Python 3.12). A GitHub token must be in `.env` as `GITHUB_TOKEN=...` (git-ignored).

```bash
uv add <pkg>                          # add a dependency (updates pyproject + uv.lock)
cd Experiment1 && uv run python -m src.<module>  # experiment 1 modules
cd Experiment7 && uv run python -m src.app serve --repo <git-worktree>
cd Experiment7 && uv run pytest tests -q        # fully offline, includes Playwright
```
