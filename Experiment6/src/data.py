"""实验六数据访问层：复用实验四 data（prs/files/commits/review_comments 表 + 按 PR 取内容），
新增 L2/L3 所需——改动 hunk 所在函数抽取、PR body 里的 issue 引用解析、历史审查意见 join。

零改动复用实验四 data 的加载函数（lru_cache）；本模块只加派生逻辑，不重复读盘。
"""
from __future__ import annotations

import importlib
import re

import pandas as pd

from . import config

exp4_data = importlib.import_module("exp4src.data")

# 直接透传实验四 data 的按 PR 取内容接口（口径逐字一致）
pr_meta = exp4_data.pr_meta
pr_python_patches = exp4_data.pr_python_patches
pr_commit_messages = exp4_data.pr_commit_messages
pr_top_level_comments = exp4_data.pr_top_level_comments
load_prs = exp4_data.load_prs
load_files = exp4_data.load_files


# --------------------------------------------------------------------------- #
# reviews 表（实验四未用，L3 历史审查意见需要）
# --------------------------------------------------------------------------- #
from functools import lru_cache


@lru_cache(maxsize=1)
def load_reviews() -> pd.DataFrame:
    return pd.read_parquet(config.EXP1_DATA_DIR / "reviews.parquet")


# --------------------------------------------------------------------------- #
# L2：改动 hunk 所在函数/类抽取
# --------------------------------------------------------------------------- #
_HUNK_HEADER_RE = re.compile(r"^@@\s*-\d+(?:,\d+)?\s+\+(\d+)(?:,(\d+))?\s*@@")
# Python 顶层/缩进的 def/class 起始行
_DEF_RE = re.compile(r"^(\s*)(async\s+def|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)")


def changed_line_ranges(patch: str) -> list[tuple[int, int]]:
    """从 patch 解析改后文件里被触及的行号区间（1-based，改后坐标）。"""
    ranges: list[tuple[int, int]] = []
    if not isinstance(patch, str):
        return ranges
    new_ln = 0
    cur_start = None
    for line in patch.splitlines():
        m = _HUNK_HEADER_RE.match(line)
        if m:
            new_ln = int(m.group(1))
            continue
        if not line:
            continue
        tag = line[0]
        if tag == "+":
            if cur_start is None:
                cur_start = new_ln
            new_ln += 1
        elif tag == " ":
            if cur_start is not None:
                ranges.append((cur_start, new_ln - 1))
                cur_start = None
            new_ln += 1
        elif tag == "-":
            # 删除行不占改后行号；标记其锚点附近
            if cur_start is None:
                ranges.append((max(new_ln, 1), max(new_ln, 1)))
        # 其它（\ No newline 等）忽略
    if cur_start is not None:
        ranges.append((cur_start, new_ln - 1))
    return ranges


def enclosing_blocks(file_text: str, line_ranges: list[tuple[int, int]]) -> list[str]:
    """给定完整文件文本 + 改后行区间，返回包含这些行的顶层 def/class 完整体文本。

    确定性词法切分：按缩进判断块边界（Python）。找不到则返回空列表（调用方回退文件级）。
    """
    if not file_text or not line_ranges:
        return []
    lines = file_text.splitlines()
    n = len(lines)
    # 记录每个 def/class 起始行号(1-based)与缩进
    defs = []
    for i, ln in enumerate(lines):
        m = _DEF_RE.match(ln)
        if m:
            defs.append((i + 1, len(m.group(1)), m.group(3)))

    def block_span(start_idx0: int, indent: int) -> tuple[int, int]:
        """从 def 起始行(0-based)按缩进找块结束行(0-based, 含)。"""
        end = start_idx0
        for j in range(start_idx0 + 1, n):
            s = lines[j]
            if not s.strip():
                end = j
                continue
            cur_indent = len(s) - len(s.lstrip())
            if cur_indent <= indent:
                break
            end = j
        return start_idx0, end

    touched: set[tuple[int, int]] = set()
    for (lo, hi) in line_ranges:
        # 找到包含 [lo,hi] 的最内层 def/class
        best = None
        for (dstart, dindent, _name) in defs:
            s0, e0 = block_span(dstart - 1, dindent)
            if s0 + 1 <= lo and hi <= e0 + 1:
                if best is None or (s0, e0) > best:  # 取更内层（起始更靠后）
                    best = (s0, e0)
        if best is not None:
            touched.add(best)
    blocks = []
    for (s0, e0) in sorted(touched):
        blocks.append("\n".join(lines[s0:e0 + 1]))
    return blocks


# --------------------------------------------------------------------------- #
# L3：PR body 里的 issue 引用解析
# --------------------------------------------------------------------------- #
_ISSUE_REF_RE = re.compile(
    r"(?:(?:close|closes|closed|fix|fixes|fixed|resolve|resolves|resolved)\s+)?#(\d+)",
    re.IGNORECASE)


def issue_refs(repo: str, number: int, max_refs: int = 3) -> list[int]:
    """从 PR body 抽取被引用的 issue 号（去重、排除自身、限量）。"""
    try:
        meta = pr_meta(repo, number)
    except KeyError:
        return []
    body = meta.get("body") or ""
    refs = []
    for m in _ISSUE_REF_RE.finditer(body):
        num = int(m.group(1))
        if num != number and num not in refs:
            refs.append(num)
        if len(refs) >= max_refs:
            break
    return refs


# --------------------------------------------------------------------------- #
# L3：历史审查意见（reviews + review_comments），排除作为生成真值的那条
# --------------------------------------------------------------------------- #
def history_reviews(repo: str, number: int, exclude_comment_id: int | None = None) -> list[dict]:
    """返回该 PR 的历史审查意见 [{kind, reviewer, body, path}]，按时间升序。

    kind ∈ {"review","comment"}。排除 exclude_comment_id（防生成真值泄漏，R3/L3）。
    """
    out: list[dict] = []
    # reviews 表（PR 级审查总结）
    rv = load_reviews()
    sub = rv[(rv["repo"] == repo) & (rv["pr_number"] == number)]
    if "submitted_at" in sub.columns:
        sub = sub.sort_values("submitted_at")
    for r in sub.itertuples():
        body = getattr(r, "body", None)
        if isinstance(body, str) and body.strip():
            out.append({"kind": "review", "reviewer": getattr(r, "reviewer", ""),
                        "body": body.strip(), "path": ""})
    # review_comments 表（inline）——排除生成真值那条
    rc = exp4_data.load_review_comments()
    csub = rc[(rc["repo"] == repo) & (rc["pr_number"] == number)]
    if "created_at" in csub.columns:
        csub = csub.sort_values("created_at")
    for r in csub.itertuples():
        cid = getattr(r, "comment_id", None)
        if exclude_comment_id is not None and cid == exclude_comment_id:
            continue
        body = getattr(r, "body", None)
        if isinstance(body, str) and body.strip():
            out.append({"kind": "comment", "reviewer": getattr(r, "reviewer", ""),
                        "body": body.strip(), "path": getattr(r, "path", "") or ""})
    return out


if __name__ == "__main__":
    import json
    sample = json.load(open(config.EXP5_SAMPLES_DIR / "generate_ai.json"))[0]
    repo, num = sample["repo"], sample["number"]
    print("repo", repo, "num", num)
    print("issue_refs:", issue_refs(repo, num))
    hist = history_reviews(repo, num, sample.get("target_comment_id"))
    print("history_reviews:", len(hist))
    patches = pr_python_patches(repo, num)
    if patches:
        rng = changed_line_ranges(patches[0][1])
        print("changed ranges (first file):", rng[:5])
