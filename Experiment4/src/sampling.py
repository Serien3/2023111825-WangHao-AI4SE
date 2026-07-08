"""切片 1：抽样 + few-shot 示例。

- 分类：从实验二 test 208 条，按 repo × is_merged 分层抽 N_SAMPLE(50)。
- 生成：从 test 中有顶层 inline comment 的 PR（约 70 条）按 repo 分层抽 50，
  每 PR 取首条顶层 comment 作为 ground-truth target。
- few-shot：仅从实验二 train 集选，绝不碰 test（防泄漏），落盘复用。

产物：
  results/samples/classify_50.json
  results/samples/generate_50.json
  results/samples/fewshot_classify.json
  results/samples/fewshot_generate.json
"""
from __future__ import annotations

import json

import pandas as pd

from . import config, data


# --------------------------------------------------------------------------- #
# 分层抽样工具
# --------------------------------------------------------------------------- #
def _stratified_sample(df: pd.DataFrame, strata_cols: list[str], n: int,
                       seed: int) -> pd.DataFrame:
    """按 strata_cols 组合分层，按各层占比向下取整分配名额，余额按最大剩余补足。

    固定种子；每层内 sample(frac=1) 打乱后取前 k 条，保证可复现。
    """
    total = len(df)
    groups = list(df.groupby(strata_cols, sort=True))
    # 先按占比算每层目标数
    alloc = {}
    remainders = []
    assigned = 0
    for key, g in groups:
        exact = len(g) / total * n
        base = int(exact)
        alloc[key] = base
        assigned += base
        remainders.append((exact - base, key))
    # 余额分配（最大小数优先）
    for _, key in sorted(remainders, reverse=True):
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


# --------------------------------------------------------------------------- #
# 分类抽样
# --------------------------------------------------------------------------- #
def sample_classify() -> list[dict]:
    test = data.test_prs()
    sampled = _stratified_sample(test, ["repo", "is_merged"], config.N_SAMPLE, config.SEED)
    records = [
        {"repo": r.repo, "number": int(r.number), "is_merged": bool(r.is_merged)}
        for r in sampled.itertuples()
    ]
    # 防泄漏断言：全部来自 test
    test_keys = set(zip(test["repo"], test["number"]))
    assert all((r["repo"], r["number"]) in test_keys for r in records)

    dist = sampled.groupby(["repo", "is_merged"]).size()
    print(f"[classify] 抽样 {len(records)} 条，分层分布：")
    print(dist.to_string())
    print(f"  merged 分布: {sampled['is_merged'].value_counts().to_dict()}")
    return records


# --------------------------------------------------------------------------- #
# 生成抽样
# --------------------------------------------------------------------------- #
def _generation_pool() -> pd.DataFrame:
    """test 中有 ≥1 顶层 inline comment 的 PR，附首条 comment 作 target。"""
    test = data.test_prs()
    rows = []
    for r in test.itertuples():
        top = data.pr_top_level_comments(r.repo, int(r.number))
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
    return pd.DataFrame(rows)


def sample_generate() -> list[dict]:
    pool = _generation_pool()
    print(f"[generate] 候选池（有顶层 comment 的 test PR）: {len(pool)} 条")
    n = min(config.N_SAMPLE, len(pool))
    sampled = _stratified_sample(pool, ["repo"], n, config.SEED)
    records = sampled.to_dict(orient="records")

    test_keys = set(zip(data.test_prs()["repo"], data.test_prs()["number"]))
    assert all((r["repo"], r["number"]) in test_keys for r in records)

    dist = sampled.groupby("repo").size()
    print(f"[generate] 抽样 {len(records)} 条，按 repo 分层：")
    print(dist.to_string())
    return records


# --------------------------------------------------------------------------- #
# few-shot 示例（仅 train 集）
# --------------------------------------------------------------------------- #
def _truncate(text: str, budget: int) -> str:
    if not isinstance(text, str):
        return ""
    return text if len(text) <= budget else text[:budget] + "\n... [truncated]"


def _diff_summary(repo: str, number: int, budget: int) -> str:
    patches = data.pr_python_patches(repo, number)
    joined = "\n".join(f"--- {fn} ---\n{p}" for fn, p in patches)
    return _truncate(joined, budget)


def build_fewshot_classify() -> list[dict]:
    """从 train 集选 1 正 1 负例（各含 diff 摘要 + 真值 + 一句理由）。"""
    train = data.train_prs()
    examples = []
    for label in (True, False):
        cand = train[train["is_merged"] == label]
        # 选有 python patch 的第一条（固定种子打乱）
        cand = cand.sample(frac=1, random_state=config.SEED)
        for r in cand.itertuples():
            diff = _diff_summary(r.repo, int(r.number), config.FEWSHOT_DIFF_CHAR_BUDGET)
            if diff.strip():
                examples.append({
                    "repo": r.repo,
                    "number": int(r.number),
                    "diff": diff,
                    "is_merged": bool(label),
                    "reason": ("代码改动完整、符合项目规范，可合并。" if label
                               else "改动存在问题或未达合并标准，暂不合并。"),
                })
                break
    return examples


def build_fewshot_generate() -> list[dict]:
    """从 train 集挑 2 个优质 (diff_hunk → 人类审查意见) 对。"""
    train = data.train_prs()
    train_keys = set(zip(train["repo"], train["number"]))
    rc = data.load_review_comments()
    top = rc[rc["in_reply_to_id"].isna()].copy()
    top = top[top["body"].apply(lambda b: isinstance(b, str) and 20 <= len(b) <= 400)]
    top["k"] = list(zip(top["repo"], top["pr_number"]))
    top = top[top["k"].isin(train_keys)]
    top = top[top["diff_hunk"].apply(lambda d: isinstance(d, str) and d.strip() != "")]
    top = top.sample(frac=1, random_state=config.SEED)

    examples = []
    seen_prs = set()
    for r in top.itertuples():
        if r.k in seen_prs:
            continue
        seen_prs.add(r.k)
        examples.append({
            "repo": r.repo,
            "number": int(r.pr_number),
            "diff_hunk": _truncate(r.diff_hunk, config.FEWSHOT_DIFF_CHAR_BUDGET),
            "comment": r.body,
        })
        if len(examples) >= 2:
            break
    return examples


# --------------------------------------------------------------------------- #
# 落盘
# --------------------------------------------------------------------------- #
def _dump(obj, name: str) -> None:
    path = config.SAMPLES_DIR / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"  → 落盘 {path.relative_to(config.PROJECT_ROOT)} ({len(obj)} 条)")


def main() -> None:
    print("=" * 60)
    classify = sample_classify()
    _dump(classify, "classify_50.json")
    print("=" * 60)
    generate = sample_generate()
    _dump(generate, "generate_50.json")
    print("=" * 60)
    fs_c = build_fewshot_classify()
    print(f"[fewshot-classify] {len(fs_c)} 个示例（train 集）")
    _dump(fs_c, "fewshot_classify.json")
    fs_g = build_fewshot_generate()
    print(f"[fewshot-generate] {len(fs_g)} 个示例（train 集）")
    _dump(fs_g, "fewshot_generate.json")
    print("=" * 60)
    print("抽样完成。")


if __name__ == "__main__":
    main()
