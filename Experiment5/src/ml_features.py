"""ML 特征适配层：复用实验二 extract_* 为任意 PR 集构造特征矩阵。

正确性约束（设计 §3）：
- import 复用实验二 extract_ast/cfg/code/statistical/text 函数逐 PR 抽特征。
- TF-IDF **不在 AI 文本上 fit**：实验二未落盘 vectorizer，但 50 个 tfidf 列名保存在
  features.parquet 中。本模块用这 50 词构造 TfidfVectorizer(vocabulary=...)，并在
  **实验二的人类训练语料**（1036 条，按 features 行序重建 title+body）上 fit —— 这只恢复
  与实验二逐字一致的 IDF 权重（IDF 只依赖该语料的文档频率，与 AI 文本无关），
  随后对 AI / 对照文本仅 transform。已验证可 bit-level 重现实验二 tfidf 值（max diff 2e-16）。
- 按实验二 PRE_REVIEW_FEATURES / FULL_FEATURES + tfidf 列对齐列序，供 scaler.transform。

产物：results/features/{group}_features.parquet（含 repo/number/is_merged + 全部特征列）。
"""
from __future__ import annotations

import importlib
from functools import lru_cache

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from . import config, data

# 复用实验二特征提取（零改动）
exp2_fe = importlib.import_module("exp2src.feature_extraction")
_PARSER_LANG = exp2_fe.PY_LANGUAGE
from tree_sitter import Parser  # noqa: E402


# --------------------------------------------------------------------------- #
# TF-IDF：固定词表 + 人类语料 fit → 只 transform
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _tfidf_vocab() -> list[str]:
    """从实验二 features.parquet 列名恢复 50 词 TF-IDF 词表（保持列序）。"""
    cols = pd.read_parquet(config.EXP2_FEATURES_PARQUET, columns=None).columns
    return [c[len("tfidf_"):] for c in cols if c.startswith("tfidf_")]


@lru_cache(maxsize=1)
def _fitted_vectorizer() -> TfidfVectorizer:
    """用固定词表在实验二人类训练语料上 fit，恢复与实验二一致的 IDF。"""
    feat = pd.read_parquet(config.EXP2_FEATURES_PARQUET)
    prs = data.load_prs().set_index(["repo", "number"])
    texts = []
    for r in feat.itertuples():
        row = prs.loc[(r.repo, r.number)]
        texts.append(str(row["title"] or "") + " " + str(row["body"] or ""))
    vec = TfidfVectorizer(vocabulary=_tfidf_vocab(), stop_words="english")
    vec.fit(texts)  # 仅恢复 IDF，绝不在 AI/对照文本上 fit
    return vec


# --------------------------------------------------------------------------- #
# 逐 PR 特征
# --------------------------------------------------------------------------- #
def _pr_feature_row(repo: str, number: int, parser: Parser) -> dict:
    """对单个 PR 抽 AST/CFG/统计/文本特征 + 原始文本（供 TF-IDF）。"""
    meta = data.pr_meta(repo, number)
    patches = [p for _, p in data.pr_python_patches(repo, number)]
    commit_msgs = data.pr_commit_messages(repo, number)

    row = {}
    row.update(exp2_fe.extract_code_features_for_pr(patches, parser))
    row.update(exp2_fe.extract_statistical_features(meta))
    row.update(exp2_fe.extract_text_features(meta, commit_msgs))
    row["_text"] = str(meta.get("title", "") or "") + " " + str(meta.get("body", "") or "")
    return row


def build_feature_matrix(pr_df: pd.DataFrame) -> pd.DataFrame:
    """对 pr_df（含 repo/number/is_merged）逐 PR 抽特征 + TF-IDF transform，
    返回含 meta 列 + 全部特征列（AST/CFG/统计/文本/tfidf_*）的 DataFrame。
    """
    parser = Parser(_PARSER_LANG)
    rows = []
    for r in tqdm(pr_df.itertuples(), total=len(pr_df), ncols=80, desc="  extract"):
        feat = {"repo": r.repo, "number": int(r.number), "is_merged": bool(r.is_merged)}
        feat.update(_pr_feature_row(r.repo, int(r.number), parser))
        rows.append(feat)
    df = pd.DataFrame(rows).fillna(0)

    # TF-IDF：只 transform（用人类语料 fit 好的 vectorizer）
    vec = _fitted_vectorizer()
    mat = vec.transform(df["_text"].fillna("")).toarray()
    tfidf_df = pd.DataFrame(
        mat, columns=[f"tfidf_{w}" for w in vec.get_feature_names_out()], index=df.index
    )
    df = pd.concat([df.drop(columns=["_text"]), tfidf_df], axis=1)
    return df


# --------------------------------------------------------------------------- #
# 列序对齐（供 scaler.transform）
# --------------------------------------------------------------------------- #
def align_to_feature_cols(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """按 scaler 期望的 feature_cols 顺序取列；缺失列补 0（应不发生，做防御）。"""
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        for c in missing:
            df[c] = 0.0
        print(f"      [警告] 补齐缺失特征列 {len(missing)} 个: {missing[:5]}...")
    return df[feature_cols].copy()


# --------------------------------------------------------------------------- #
# 主流程：为各受试组落盘特征矩阵
# --------------------------------------------------------------------------- #
def build_and_save(group: str, pr_df: pd.DataFrame) -> pd.DataFrame:
    print(f"[{group}] 构造特征矩阵：{len(pr_df)} PR")
    df = build_feature_matrix(pr_df)
    out = config.FEATURES_DIR / f"{group}_features.parquet"
    df.to_parquet(out, index=False)
    print(f"  → 落盘 {out.relative_to(config.PROJECT_ROOT)}（{df.shape[0]}×{df.shape[1]}）")
    return df


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="实验五 ML 特征提取（AI + 匹配对照）")
    ap.add_argument("--limit", type=int, default=None, help="每组只取前 N 个 PR（冒烟）")
    args = ap.parse_args()

    ai = data.ai_classify_pool()
    ctrl = data.matched_human_control(ai, config.N_CLASSIFY_SAMPLE, config.SEED)
    if args.limit:
        ai, ctrl = ai.head(args.limit), ctrl.head(args.limit)

    # 自检：TF-IDF 重建应 bit-level 重现实验二值
    _selfcheck_tfidf()

    build_and_save("ai", ai)
    build_and_save("control", ctrl)
    print("特征提取完成。")


def _selfcheck_tfidf() -> None:
    """断言重建的 vectorizer 能重现实验二 tfidf 值（防回归）。"""
    import numpy as np
    feat = pd.read_parquet(config.EXP2_FEATURES_PARQUET)
    tfidf_cols = [c for c in feat.columns if c.startswith("tfidf_")]
    prs = data.load_prs().set_index(["repo", "number"])
    texts = [str(prs.loc[(r.repo, r.number)]["title"] or "") + " "
             + str(prs.loc[(r.repo, r.number)]["body"] or "") for r in feat.itertuples()]
    mat = _fitted_vectorizer().transform(texts).toarray()
    diff = np.abs(mat - feat[tfidf_cols].reset_index(drop=True).values).max()
    assert diff < 1e-6, f"TF-IDF 重建偏差过大: {diff}"
    print(f"  [自检] TF-IDF 重建 max diff={diff:.2e} ✓（与实验二一致）")


if __name__ == "__main__":
    main()
