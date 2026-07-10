"""数据层：载实验一表、筛 AI 池、构造无泄露的匹配人类对照组、按 PR 取内容。

三个集合（见设计 §1）：
- AI 分类池：is_ai_code=True 且含 language==python & has_patch 文件 → 253 条。
- AI 生成池：is_ai_code=True 且含 ≥1 条顶层 inline comment → 72 条。
- 匹配人类对照组：只从实验二 held-out test 集按 AI 分类样本的 repo × is_merged
  分布分层抽样，避免 train/val 泄露。

按 PR 取内容的接口直接复用实验四 data.py（pr_python_patches / pr_commit_messages /
pr_top_level_comments 等），保持与实验四逐字一致。
"""
from __future__ import annotations

import importlib
import pickle
from functools import lru_cache

import pandas as pd

from . import config

# 复用实验四数据访问接口（按 PR 取 patch/commit/comment）
exp4_data = importlib.import_module("exp4src.data")


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


# 复用实验四按 PR 取内容的函数（同一份实验一数据，口径一致）
pr_meta = exp4_data.pr_meta
pr_python_patches = exp4_data.pr_python_patches
pr_commit_messages = exp4_data.pr_commit_messages
pr_top_level_comments = exp4_data.pr_top_level_comments


# --------------------------------------------------------------------------- #
# 候选池筛选
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _python_patch_pr_keys() -> pd.DataFrame:
    """含 language==python & has_patch 文件的 (repo, pr_number) 去重集合。"""
    files = load_files()
    py = files[(files["language"].str.lower() == "python") & (files["has_patch"])]
    return py[["repo", "pr_number"]].drop_duplicates()


@lru_cache(maxsize=1)
def ai_classify_pool() -> pd.DataFrame:
    """AI 分类池：is_ai_code=True 且含 Python patch 的 PR（预期 253）。"""
    prs = load_prs()
    ai = prs[prs["is_ai_code"] == True]  # noqa: E712
    keys = _python_patch_pr_keys()
    pool = ai.merge(
        keys, left_on=["repo", "number"], right_on=["repo", "pr_number"], how="inner"
    )
    return pool[["repo", "number", "is_merged"]].reset_index(drop=True)


@lru_cache(maxsize=1)
def human_classify_pool() -> pd.DataFrame:
    """人类分类池：is_ai_code=False 且含 Python patch 的 PR（预期 1036，= 实验二全集）。"""
    prs = load_prs()
    human = prs[prs["is_ai_code"] == False]  # noqa: E712
    keys = _python_patch_pr_keys()
    pool = human.merge(
        keys, left_on=["repo", "number"], right_on=["repo", "pr_number"], how="inner"
    )
    return pool[["repo", "number", "is_merged"]].reset_index(drop=True)


@lru_cache(maxsize=1)
def exp2_split_prs() -> dict[str, pd.DataFrame]:
    """把实验二 pre split 的 train/val/test 行索引映射回 (repo, number, is_merged)。"""
    feat = pd.read_parquet(config.EXP2_FEATURES_PARQUET)
    with open(config.EXP2_MODELS_DIR / "split_pre.pkl", "rb") as f:
        split = pickle.load(f)
    out = {}
    for part in ("train", "val", "test"):
        X = split[f"X_{part}"]
        y = split[f"y_{part}"]
        df = feat.loc[X.index, ["repo", "number"]].copy()
        df["is_merged"] = y.values.astype(bool)
        out[part] = df.reset_index(drop=True)
    return out


@lru_cache(maxsize=1)
def human_test_pool() -> pd.DataFrame:
    """实验二 held-out test 人类 PR 池；匹配对照只能从这里抽，避免训练/调参泄露。"""
    return exp2_split_prs()["test"].copy()


@lru_cache(maxsize=1)
def ai_generate_pool() -> pd.DataFrame:
    """AI 生成池：is_ai_code=True 且含 ≥1 条顶层 inline comment 的 PR（预期 72）。

    附首条顶层 comment 作为 ground-truth target（时间最早、非回复），
    与实验四生成真值口径逐字一致。
    """
    prs = load_prs()
    ai = prs[prs["is_ai_code"] == True]  # noqa: E712
    rows = []
    for r in ai.itertuples():
        top = pr_top_level_comments(r.repo, int(r.number))
        if top.empty:
            continue
        first = top.iloc[0]
        rows.append({
            "repo": r.repo,
            "number": int(r.number),
            "is_merged": bool(r.is_merged),
            "target_comment": first["body"],
            "target_path": first.get("path"),
            "target_comment_id": int(first["comment_id"]),
        })
    return pd.DataFrame(rows).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 分层抽样工具（复用实验四 sampling 的算法：占比向下取整 + 最大剩余补足）
# --------------------------------------------------------------------------- #
def stratified_sample(df: pd.DataFrame, strata_cols: list[str], n: int,
                      seed: int) -> pd.DataFrame:
    """按 strata_cols 组合分层，按占比分配名额，固定种子可复现。"""
    total = len(df)
    if total == 0 or n <= 0:
        return df.head(0)
    n = min(n, total)
    groups = list(df.groupby(strata_cols, sort=True))
    alloc, remainders, assigned = {}, [], 0
    for key, g in groups:
        exact = len(g) / total * n
        base = int(exact)
        alloc[key] = base
        assigned += base
        remainders.append((exact - base, key))
    for _, key in sorted(remainders, key=lambda x: x[0], reverse=True):
        if assigned >= n:
            break
        alloc[key] += 1
        assigned += 1
    parts = []
    for key, g in groups:
        k = min(alloc[key], len(g))
        if k > 0:
            parts.append(g.sample(frac=1, random_state=seed).head(k))
    out = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    return out


def matched_human_control(ai_pool: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """从实验二 held-out test 集按 ai_pool 的 repo × is_merged 分布抽匹配对照。

    这样 control 与 human_test 一样都是样本外，避免把实验二 train/val 中模型见过的 PR
    放入对照组造成泄露。若某个 AI 分层在 test 中容量不足（例如 pandas merged），则先
    取该层全部样本，再把剩余额度按 AI 分布的最大剩余法分配给其它仍有容量的层，尽量保持
    repo × is_merged 分布接近 AI 样本。
    """
    human = human_test_pool()
    if ai_pool.empty or n <= 0:
        return human.head(0)

    ai_dist = ai_pool.groupby(["repo", "is_merged"]).size().to_dict()
    ai_total = sum(ai_dist.values())
    layers = {
        key: human[(human["repo"] == key[0]) & (human["is_merged"] == key[1])]
        for key in ai_dist
    }

    alloc = {key: min(int(cnt / ai_total * n), len(layers[key])) for key, cnt in ai_dist.items()}
    remaining = n - sum(alloc.values())
    while remaining > 0:
        candidates = []
        for key, cnt in ai_dist.items():
            capacity = len(layers[key]) - alloc[key]
            if capacity <= 0:
                continue
            desired = cnt / ai_total * n
            candidates.append((desired - alloc[key], cnt, key))
        if not candidates:
            break
        _, _, key = max(candidates, key=lambda x: (x[0], x[1], str(x[2])))
        alloc[key] += 1
        remaining -= 1

    parts = []
    for key, k in alloc.items():
        if k > 0:
            parts.append(layers[key].sample(frac=1, random_state=seed).head(k))
    if not parts:
        return human.head(0)
    out = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    return out


def _dist_str(df: pd.DataFrame) -> str:
    if df.empty:
        return "(空)"
    return df.groupby(["repo", "is_merged"]).size().to_string()


if __name__ == "__main__":
    clf = ai_classify_pool()
    gen = ai_generate_pool()
    hum = human_classify_pool()
    print(f"AI 分类池: {len(clf)} | merged 分布: {clf['is_merged'].value_counts().to_dict()}")
    print(f"  按仓库: {clf['repo'].value_counts().to_dict()}")
    print(f"AI 生成池: {len(gen)} | 按仓库: {gen['repo'].value_counts().to_dict()}")
    print(f"人类分类池: {len(hum)}")
    ctrl = matched_human_control(clf, config.N_CLASSIFY_SAMPLE, config.SEED)
    print(f"匹配对照（n={config.N_CLASSIFY_SAMPLE}）: {len(ctrl)}")
    print(_dist_str(ctrl))
