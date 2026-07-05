# exp1

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

Multi-experiment course project (Intelligent Software Engineering) on **code review**. The full spec is in `docs/实验指导书_代码审查.md` — **read it before making design decisions**. The project is problem-driven: every choice in an earlier experiment must serve the later ones, so treat the experiments as one pipeline, not independent tasks.

There are 7 experiments. Two tasks run through all of them: **Merge Prediction** (classification) and **Review Comment Generation**.

- **Exp 1** (built, `src/exp1/`): acquire + clean + EDA a GitHub code-review dataset — the shared data foundation for every later experiment. See `src/exp1/CLAUDE.md`.
- **Exp 2**: traditional ML (SVM/RF) on AST/CFG + statistical features.
- **Exp 3**: DL (Code2Vec/CodeBERT) — needs `(diff_hunk → review comment)` pairs.
- **Exp 4/6**: LLM + prompt engineering — needs multi-granularity context.
- **Exp 5**: review of AI-generated code — needs the AI labels.
- **Exp 7**: VSCode plugin deploying the above models.

## Environment

Uses **uv** for env/deps (Python 3.12). A GitHub token must be in `.env` as `GITHUB_TOKEN=...` (git-ignored).

```bash
uv add <pkg>                          # add a dependency (updates pyproject + uv.lock)
uv run python -m src.<exp>.<module>   # always run modules with -m (package-relative imports)
```

## Conventions

- `data/` is entirely git-ignored and regenerable; never commit it.
- Config/tunable knobs live in a `config.py` per experiment, not scattered in logic code.
- Entry scripts call `load_dotenv()` themselves; `config` modules read the token only after that.
