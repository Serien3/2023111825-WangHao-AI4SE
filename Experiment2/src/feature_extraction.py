"""实验二 步骤二 + 三：从实验一的 diff 数据提取 AST / CFG / 统计 / 文本特征。

设计要点（见 docs/design-docs/exp2-features.md）：
- 代码来源是实验一保存的 patch（diff），不回仓库抓完整文件——这是实验六的范围。
- patch 是"改动"而非完整文件，因此用 tree-sitter 的**容错解析**：从 patch 重建
  "改动后代码片段"（保留上下文行 + 新增行，丢弃删除行），残缺片段产生 ERROR
  节点仍可统计有效子树。指导书要的是 AST/CFG 的**聚合统计特征**（节点数量），
  这种近似完全够用。
- 每个 PR 聚合其所有 Python 改动文件的特征，得到一行特征向量。

用法：
    uv run python -m src.feature_extraction              # 全量
    uv run python -m src.feature_extraction --limit 50   # 小样测试
"""
from __future__ import annotations

import argparse
import re

import pandas as pd
import tree_sitter_python as tspython
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from tree_sitter import Language, Parser

from . import config

PY_LANGUAGE = Language(tspython.language())


# --------------------------------------------------------------------------- #
# diff 重建：从 patch 得到"改动后代码片段"
# --------------------------------------------------------------------------- #
def reconstruct_post_change_code(patch: str | None) -> str:
    """从 unified diff 的 patch 重建改动后的代码片段。

    保留：上下文行（' ' 前缀）+ 新增行（'+' 前缀）；丢弃：删除行（'-'）与 diff 元信息。
    这样得到的是"改动被应用后"的代码近似，用于统计 AST/CFG 结构特征。
    """
    if not patch:
        return ""
    out: list[str] = []
    for line in patch.splitlines():
        if line.startswith("@@") or line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            out.append(line[1:])
        elif line.startswith("-"):
            continue
        elif line.startswith(" "):
            out.append(line[1:])
        else:
            # diff 中偶见无前缀行（如 "\ No newline at end of file"），跳过
            continue
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# AST 特征：tree-sitter 解析 + 节点计数
# --------------------------------------------------------------------------- #
def _walk_ast(node, counts: dict, stats: dict, depth: int = 0) -> None:
    """递归遍历 AST，累计节点类型计数、最大深度、分支因子。"""
    counts[node.type] = counts.get(node.type, 0) + 1
    stats["total"] += 1
    stats["max_depth"] = max(stats["max_depth"], depth)
    n_children = len(node.children)
    if n_children > 0:
        stats["branch_sum"] += n_children
        stats["internal"] += 1
    if node.type == "ERROR" or node.is_missing:
        stats["errors"] += 1
    for child in node.children:
        _walk_ast(child, counts, stats, depth + 1)


def extract_ast_features(code: str, parser: Parser) -> dict:
    """解析代码片段，返回 AST 结构特征。空/无法解析时返回全 0。"""
    feats = {f"ast_{t}": 0 for t in config.AST_NODE_TYPES}
    feats.update({
        "ast_total_nodes": 0,
        "ast_max_depth": 0,
        "ast_avg_branching": 0.0,
        "ast_error_nodes": 0,
    })
    if not code.strip():
        return feats

    tree = parser.parse(bytes(code, "utf8"))
    counts: dict = {}
    stats = {"total": 0, "max_depth": 0, "branch_sum": 0, "internal": 0, "errors": 0}
    _walk_ast(tree.root_node, counts, stats)

    for t in config.AST_NODE_TYPES:
        feats[f"ast_{t}"] = counts.get(t, 0)
    feats["ast_total_nodes"] = stats["total"]
    feats["ast_max_depth"] = stats["max_depth"]
    feats["ast_avg_branching"] = (
        stats["branch_sum"] / stats["internal"] if stats["internal"] else 0.0
    )
    feats["ast_error_nodes"] = stats["errors"]
    return feats


# --------------------------------------------------------------------------- #
# CFG 特征：基于控制流关键词的轻量近似
# --------------------------------------------------------------------------- #
# 说明：完整 CFG 需要语义分析（def-use、跳转解析）。对 ML 统计特征而言，
# 基于控制流语句计数的近似（节点=基本块、边=控制转移、复杂度=McCabe 近似）
# 已足够且更快。用 tree-sitter AST 节点类型统计控制流结构。
_CFG_BRANCH_TYPES = {
    "if_statement", "for_statement", "while_statement",
    "try_statement", "with_statement", "elif_clause", "except_clause",
    "boolean_operator",  # and/or 短路 → 额外分支
}


def extract_cfg_features(code: str, parser: Parser) -> dict:
    """从 AST 近似构建 CFG 特征：节点数、边数、圈复杂度、最大嵌套深度。"""
    feats = {"cfg_nodes": 0, "cfg_edges": 0, "cfg_complexity": 0, "cfg_max_nesting": 0}
    if not code.strip():
        return feats

    tree = parser.parse(bytes(code, "utf8"))

    # 圈复杂度 McCabe 近似：1 + 判定点数量
    decision_points = 0
    max_nesting = [0]
    n_stmt_nodes = [0]

    def visit(node, nesting=0):
        nonlocal decision_points
        if node.type in _CFG_BRANCH_TYPES:
            decision_points += 1
        # 基本块近似：语句节点数
        if node.type.endswith("_statement") or node.type.endswith("_clause"):
            n_stmt_nodes[0] += 1
        # 嵌套深度：控制流结构增加嵌套
        new_nesting = nesting
        if node.type in {"if_statement", "for_statement", "while_statement",
                         "try_statement", "with_statement"}:
            new_nesting = nesting + 1
            max_nesting[0] = max(max_nesting[0], new_nesting)
        for child in node.children:
            visit(child, new_nesting)

    visit(tree.root_node)

    complexity = 1 + decision_points
    # CFG 节点近似为基本块（语句节点）+ 入口出口；边近似为节点数 + 判定点（分支多一条边）
    n_nodes = n_stmt_nodes[0] + 2  # +入口 +出口
    n_edges = n_stmt_nodes[0] + decision_points + 1

    feats["cfg_nodes"] = n_nodes
    feats["cfg_edges"] = n_edges
    feats["cfg_complexity"] = complexity
    feats["cfg_max_nesting"] = max_nesting[0]
    return feats


# --------------------------------------------------------------------------- #
# PR 级特征聚合
# --------------------------------------------------------------------------- #
def extract_code_features_for_pr(patches: list[str], parser: Parser) -> dict:
    """对一个 PR 的所有 Python patch 聚合 AST + CFG 特征（求和 / 取最大）。"""
    ast_accum: dict = {}
    cfg_accum: dict = {}
    max_keys = {"ast_max_depth", "cfg_max_nesting", "ast_avg_branching"}

    for patch in patches:
        code = reconstruct_post_change_code(patch)
        ast_f = extract_ast_features(code, parser)
        cfg_f = extract_cfg_features(code, parser)
        for k, v in ast_f.items():
            if k in max_keys:
                ast_accum[k] = max(ast_accum.get(k, 0), v)
            else:
                ast_accum[k] = ast_accum.get(k, 0) + v
        for k, v in cfg_f.items():
            if k in max_keys:
                cfg_accum[k] = max(cfg_accum.get(k, 0), v)
            else:
                cfg_accum[k] = cfg_accum.get(k, 0) + v

    return {**ast_accum, **cfg_accum}


def extract_statistical_features(pr_row: pd.Series) -> dict:
    """从 PR 元数据提取统计特征。"""
    additions = float(pr_row.get("additions", 0) or 0)
    deletions = float(pr_row.get("deletions", 0) or 0)
    changed_files = float(pr_row.get("changed_files", 0) or 0)
    num_commits = float(pr_row.get("num_commits", 0) or 0)
    num_review_comments = float(pr_row.get("num_review_comments", 0) or 0)

    return {
        "additions": additions,
        "deletions": deletions,
        "changed_files": changed_files,
        "num_commits": num_commits,
        "files_per_commit": changed_files / (num_commits + 1),
        "churn_ratio": additions / (additions + deletions + 1),
        # review-process features (含时间泄漏，单独分组)
        "num_reviews": float(pr_row.get("num_reviews", 0) or 0),
        "num_reviewers": float(pr_row.get("num_reviewers", 0) or 0),
        "num_review_comments": num_review_comments,
        "num_issue_comments": float(pr_row.get("num_issue_comments", 0) or 0),
        "review_density": num_review_comments / (additions + deletions + 1),
    }


def extract_text_features(pr_row: pd.Series, commit_msgs: list[str]) -> dict:
    """从 PR 标题/正文/commit message 提取文本特征。"""
    title = str(pr_row.get("title", "") or "")
    body = str(pr_row.get("body", "") or "")
    avg_msg_len = (
        sum(len(m) for m in commit_msgs) / len(commit_msgs) if commit_msgs else 0.0
    )
    text_lower = (title + " " + body).lower()

    feats = {
        "title_len": float(len(title)),
        "body_len": float(len(body)),
        "avg_commit_msg_len": float(avg_msg_len),
    }
    for kw in config.TEXT_KEYWORDS:
        feats[f"has_keyword_{kw}"] = float(bool(re.search(rf"\b{kw}", text_lower)))
    return feats


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def build_feature_matrix(limit: int | None = None) -> pd.DataFrame:
    """读取实验一数据，筛人类代码 PR，逐 PR 提取全部特征，返回特征矩阵。"""
    print("[1/4] 读取实验一数据 ...")
    prs = pd.read_parquet(config.EXP1_DATA_DIR / "prs.parquet")
    files = pd.read_parquet(config.EXP1_DATA_DIR / "files.parquet")
    commits = pd.read_parquet(config.EXP1_DATA_DIR / "commits.parquet")

    # 步骤一：数据筛选 —— 仅保留人类编写代码
    human = prs[prs["is_ai_code"] == False].copy()  # noqa: E712
    print(f"      人类代码 PR: {len(human)} / 全部 {len(prs)}")

    # 只保留有 Python patch 的 PR（AST/CFG 需要可解析代码）
    py_files = files[(files["language"] == "python") & (files["has_patch"])].copy()
    py_pr_keys = py_files[["repo", "pr_number"]].drop_duplicates()
    human = human.merge(
        py_pr_keys, left_on=["repo", "number"], right_on=["repo", "pr_number"], how="inner"
    )
    print(f"      含 Python 改动的人类 PR: {len(human)}")

    if limit:
        human = human.head(limit)
        print(f"      [测试模式] 仅处理前 {limit} 个 PR")

    # 预分组，避免循环内重复过滤
    patches_by_pr = (
        py_files.groupby(["repo", "pr_number"])["patch"].apply(list).to_dict()
    )
    msgs_by_pr = (
        commits.groupby(["repo", "pr_number"])["message"].apply(list).to_dict()
    )

    parser = Parser(PY_LANGUAGE)
    rows = []
    print("[2/4] 逐 PR 提取 AST / CFG / 统计 / 文本特征 ...")
    for _, pr in tqdm(human.iterrows(), total=len(human)):
        key = (pr["repo"], pr["number"])
        patches = patches_by_pr.get(key, [])
        commit_msgs = msgs_by_pr.get(key, [])

        feat = {"repo": pr["repo"], "number": pr["number"], "is_merged": bool(pr["is_merged"])}
        feat.update(extract_code_features_for_pr(patches, parser))
        feat.update(extract_statistical_features(pr))
        feat.update(extract_text_features(pr, commit_msgs))
        # 保留原始文本供 TF-IDF
        feat["_text"] = (str(pr.get("title", "") or "") + " " + str(pr.get("body", "") or ""))
        rows.append(feat)

    df = pd.DataFrame(rows).fillna(0)

    # 步骤三补充：TF-IDF 文本特征
    print(f"[3/4] 生成 TF-IDF 文本特征（top {config.TFIDF_TOP_K}）...")
    tfidf = TfidfVectorizer(
        max_features=config.TFIDF_TOP_K, stop_words="english", min_df=3
    )
    tfidf_mat = tfidf.fit_transform(df["_text"].fillna(""))
    tfidf_df = pd.DataFrame(
        tfidf_mat.toarray(),
        columns=[f"tfidf_{w}" for w in tfidf.get_feature_names_out()],
        index=df.index,
    )
    df = pd.concat([df.drop(columns=["_text"]), tfidf_df], axis=1)

    print(f"[4/4] 特征矩阵完成：{df.shape[0]} PR × {df.shape[1]} 列")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="实验二特征提取")
    ap.add_argument("--limit", type=int, default=None, help="仅处理前 N 个 PR（测试用）")
    args = ap.parse_args()

    df = build_feature_matrix(limit=args.limit)

    out_parquet = config.FEATURES_DIR / "features.parquet"
    out_csv = config.FEATURES_DIR / "features.csv"
    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)
    print(f"已保存：{out_parquet}")
    print(f"已保存：{out_csv}")

    # 打印标签分布与特征概览
    print("\n标签分布 (is_merged):")
    print(df["is_merged"].value_counts())
    print(f"\n特征列数（不含 repo/number/is_merged）：{df.shape[1] - 3}")


if __name__ == "__main__":
    main()
