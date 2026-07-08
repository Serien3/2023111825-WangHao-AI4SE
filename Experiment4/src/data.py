"""共享数据访问层：加载实验一 processed 表与实验二 split/features。

- 把实验二 split_pre 的 train/test 行索引回溯到 (repo, number)。
- 提供按 PR 取 patch / body / title / commit messages / 顶层审查意见的接口。
所有加载做 lru_cache，避免主循环里重复读盘。
"""
from __future__ import annotations

import pickle
from functools import lru_cache

import pandas as pd

from . import config


# --------------------------------------------------------------------------- #
# 实验一原始表
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def load_prs() -> pd.DataFrame:
    return pd.read_parquet(config.EXP1_DATA_DIR / "prs.parquet")


@lru_cache(maxsize=1)
def load_files() -> pd.DataFrame:
    return pd.read_parquet(config.EXP1_DATA_DIR / "files.parquet")


@lru_cache(maxsize=1)
def load_commits() -> pd.DataFrame:
    return pd.read_parquet(config.EXP1_DATA_DIR / "commits.parquet")


@lru_cache(maxsize=1)
def load_review_comments() -> pd.DataFrame:
    return pd.read_parquet(config.EXP1_DATA_DIR / "review_comments.parquet")


# --------------------------------------------------------------------------- #
# 实验二 split + features → (repo, number) 回溯
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def load_split() -> dict:
    with open(config.EXP2_SPLIT_PKL, "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def load_features() -> pd.DataFrame:
    return pd.read_parquet(config.EXP2_FEATURES)


def _split_prs(part: str) -> pd.DataFrame:
    """把 split 的 X_{part} 行索引映射回 (repo, number, is_merged)。

    part ∈ {'train','val','test'}。返回 DataFrame，index 保留原始 features 行号
    （供实验二模型在同一子集上重新 predict）。
    """
    split = load_split()
    feat = load_features()
    X = split[f"X_{part}"]
    y = split[f"y_{part}"]
    out = feat.loc[X.index, ["repo", "number"]].copy()
    out["is_merged"] = y.values
    return out


@lru_cache(maxsize=1)
def test_prs() -> pd.DataFrame:
    return _split_prs("test")


@lru_cache(maxsize=1)
def train_prs() -> pd.DataFrame:
    return _split_prs("train")


# --------------------------------------------------------------------------- #
# 按 PR 取内容
# --------------------------------------------------------------------------- #
def pr_meta(repo: str, number: int) -> pd.Series:
    prs = load_prs()
    hit = prs[(prs["repo"] == repo) & (prs["number"] == number)]
    if hit.empty:
        raise KeyError(f"PR not found in prs table: {repo}#{number}")
    return hit.iloc[0]


def pr_python_patches(repo: str, number: int) -> list[tuple[str, str]]:
    """返回该 PR 所有有 patch 的 Python 文件 [(filename, patch), ...]。"""
    files = load_files()
    sub = files[(files["repo"] == repo) & (files["pr_number"] == number)]
    py = sub[(sub["language"] == "Python") & (sub["has_patch"])]
    if py.empty:  # 回退：语言字段缺失时按扩展名兜底
        py = sub[sub["filename"].str.endswith(".py") & sub["has_patch"]]
    return [(r.filename, r.patch) for r in py.itertuples() if isinstance(r.patch, str)]


def pr_commit_messages(repo: str, number: int) -> list[str]:
    commits = load_commits()
    sub = commits[(commits["repo"] == repo) & (commits["pr_number"] == number)]
    return [m for m in sub["message"].tolist() if isinstance(m, str) and m.strip()]


def pr_top_level_comments(repo: str, number: int) -> pd.DataFrame:
    """该 PR 的顶层 inline 审查意见（非回复），按时间升序。"""
    rc = load_review_comments()
    sub = rc[(rc["repo"] == repo) & (rc["pr_number"] == number)]
    top = sub[sub["in_reply_to_id"].isna()].copy()
    top = top[top["body"].apply(lambda b: isinstance(b, str) and b.strip() != "")]
    if "created_at" in top.columns:
        top = top.sort_values("created_at")
    return top
