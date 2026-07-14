"""接缝①：上下文阶梯 L0–L4（与实验四 build_context 同签名，新增 L2–L4 分支）。

单调递增：L0 diff → L4 仓库级。每级独立 token 预算 + 确定性截断（R6，不依赖随机）。
高级别是低级别的超集（除各自块内的预算截断）。L2–L4 触网结果全部走 github_fetch 落盘缓存。

  L0 = 实验四 C1（仅 diff）
  L1 = 实验四 C4（+PR 描述 +全部 commit message）
  L2 = L1 + 改前/改后完整函数（超预算回退文件级截断）
  L3 = L2 + Issue 正文 + 历史审查意见（排除生成真值那条）
  L4 = L3 + 仓库级轻量词法检索（符号出现处 + 同目录文件头 + 贡献规范摘要）
"""
from __future__ import annotations

import importlib

from . import config, data, github_fetch

# 复用实验四 C1/C4 的 diff/desc/commit 块构建（口径逐字一致）
exp4_ctx = importlib.import_module("exp4src.context_builder")


def _truncate(text: str, budget: int) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    return text if len(text) <= budget else text[:budget] + "\n... [已截断]"


# --------------------------------------------------------------------------- #
# L0 / L1：直接复用实验四 C1 / C4
# --------------------------------------------------------------------------- #
def _block_L0(repo: str, number: int) -> str:
    return exp4_ctx.build_context(repo, number, "C1")


def _block_L1_extra(repo: str, number: int) -> str:
    """L1 相对 L0 新增的部分：PR 描述 + Commit（复用实验四 C4 的对应块）。"""
    desc = "## PR 描述\n" + exp4_ctx._build_desc_block(repo, number)
    commit = "## Commit 信息\n" + exp4_ctx._build_commit_block(repo, number)
    return desc + "\n\n" + commit


# --------------------------------------------------------------------------- #
# L2：改前/改后完整函数（超预算回退文件级）
# --------------------------------------------------------------------------- #
def _block_L2(repo: str, number: int) -> str:
    try:
        meta = data.pr_meta(repo, number)
    except KeyError:
        return "(无法定位 PR，跳过完整函数上下文)"
    base_sha = meta.get("base_sha")
    head_sha = meta.get("head_sha")
    files = data.load_files()
    sub = files[(files["repo"] == repo) & (files["pr_number"] == number)]
    py = sub[sub["filename"].str.endswith(".py")] if not sub.empty else sub

    parts: list[str] = []
    used = 0
    for r in py.itertuples():
        if used >= config.L2_TOTAL_BUDGET:
            parts.append("... [剩余文件完整函数已因预算截断]")
            break
        fn = r.filename
        prev_fn = getattr(r, "previous_filename", None)
        before_path = prev_fn if isinstance(prev_fn, str) and prev_fn else fn
        patch = getattr(r, "patch", None)

        after_text = github_fetch.get_file_at_sha(repo, fn, head_sha) if head_sha else None
        before_text = (github_fetch.get_file_at_sha(repo, before_path, base_sha)
                       if base_sha else None)

        ranges = data.changed_line_ranges(patch) if isinstance(patch, str) else []
        after_blocks = data.enclosing_blocks(after_text or "", ranges)
        # 改前块：用改后同名函数不易对齐，简单取改前文件里同区间 enclosing（近似）
        before_blocks = data.enclosing_blocks(before_text or "", ranges)

        seg = [f"### 文件: {fn}"]
        if after_blocks:
            body = _truncate("\n\n".join(after_blocks), config.FULLFILE_CHAR_BUDGET)
            seg.append("改动后完整函数/类:\n```python\n" + body + "\n```")
        elif after_text:  # 回退文件级
            seg.append("改动后文件(截断):\n```python\n"
                       + _truncate(after_text, config.FULLFILE_CHAR_BUDGET) + "\n```")
        if before_blocks:
            body = _truncate("\n\n".join(before_blocks), config.FULLFILE_CHAR_BUDGET)
            seg.append("改动前完整函数/类:\n```python\n" + body + "\n```")
        elif before_text and not after_blocks:
            seg.append("改动前文件(截断):\n```python\n"
                       + _truncate(before_text, config.FULLFILE_CHAR_BUDGET) + "\n```")
        if len(seg) == 1:
            continue  # 该文件既无块也未拉到内容
        block = "\n".join(seg)
        parts.append(block)
        used += len(block)

    if not parts:
        return "(未能获取改前/改后完整函数：可能非 Python 或文件已删除)"
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# L3：Issue 正文 + 历史审查意见
# --------------------------------------------------------------------------- #
def _block_L3(repo: str, number: int, exclude_comment_id: int | None) -> str:
    parts: list[str] = []
    # Issue 正文
    refs = data.issue_refs(repo, number)
    issue_parts = []
    for iref in refs:
        issue = github_fetch.get_issue(repo, iref)
        if issue and (issue.get("title") or issue.get("body")):
            txt = f"[Issue #{iref}] {issue.get('title', '')}\n" \
                  + _truncate(issue.get("body", ""), config.ISSUE_CHAR_BUDGET)
            issue_parts.append(txt)
    if issue_parts:
        parts.append("### 关联 Issue\n" + "\n\n".join(issue_parts))

    # 历史审查意见（排除生成真值那条）
    hist = data.history_reviews(repo, number, exclude_comment_id)
    if hist:
        lines, used = [], 0
        for h in hist:
            prefix = "[审查总结]" if h["kind"] == "review" else f"[行内@{h['path']}]"
            entry = f"- {prefix} {h['reviewer']}: {h['body'].splitlines()[0][:300]}"
            if used + len(entry) > config.HISTORY_REVIEW_CHAR_BUDGET:
                lines.append("... [剩余历史审查意见已截断]")
                break
            lines.append(entry)
            used += len(entry)
        parts.append("### 历史审查意见\n" + "\n".join(lines))

    if not parts:
        return "(无关联 Issue 或历史审查意见)"
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# L4：仓库级轻量词法检索
# --------------------------------------------------------------------------- #
def _changed_symbols(repo: str, number: int) -> list[str]:
    """从改动文件的 patch 抽取被改的顶层符号名（def/class）。"""
    patches = data.pr_python_patches(repo, number)
    syms: list[str] = []
    for _fn, patch in patches:
        for line in (patch or "").splitlines():
            if not line or line[0] not in "+ ":
                continue
            m = data._DEF_RE.match(line[1:])
            if m:
                name = m.group(3)
                if name not in syms:
                    syms.append(name)
            if len(syms) >= config.RETRIEVAL_MAX_SYMBOLS:
                return syms
    return syms


def _block_L4(repo: str, number: int) -> str:
    parts: list[str] = []
    used = 0
    try:
        meta = data.pr_meta(repo, number)
        head_sha = meta.get("head_sha")
    except KeyError:
        head_sha = None

    # ① 被改符号在仓库内其它出现处
    syms = _changed_symbols(repo, number)
    sym_parts = []
    for sym in syms:
        hits = github_fetch.code_search(repo, sym, config.RETRIEVAL_TOPK_FILES)
        for h in hits:
            if used >= config.L4_TOTAL_BUDGET:
                break
            frag = _truncate(h.get("fragment", ""), config.RETRIEVAL_SNIPPET_CHARS)
            if frag:
                entry = f"[检索:{sym} @ {h.get('path','')}]\n{frag}"
                sym_parts.append(entry)
                used += len(entry)
    if sym_parts:
        parts.append("### 符号在仓库内其它出现处（词法检索）\n" + "\n\n".join(sym_parts))

    # ② 同目录同语言文件头部
    if head_sha and used < config.L4_TOTAL_BUDGET:
        patches = data.pr_python_patches(repo, number)
        dirs_seen = set()
        sib_parts = []
        for fn, _ in patches:
            d = "/".join(fn.split("/")[:-1])
            if d in dirs_seen:
                continue
            dirs_seen.add(d)
            siblings = github_fetch.list_dir(repo, d, head_sha)
            for sp in siblings[:2]:
                if not sp.endswith(".py") or sp == fn:
                    continue
                txt = github_fetch.get_file_at_sha(repo, sp, head_sha)
                if txt:
                    head = _truncate(txt, config.SIBLING_FILE_HEAD_CHARS)
                    entry = f"[同目录 {sp} 头部]\n{head}"
                    sib_parts.append(entry)
                    used += len(entry)
                if used >= config.L4_TOTAL_BUDGET:
                    break
            if used >= config.L4_TOTAL_BUDGET:
                break
        if sib_parts:
            parts.append("### 同目录相关文件头部\n" + "\n\n".join(sib_parts))

    # ③ 贡献规范摘要（静态、确定性；不触网亦可）
    parts.append("### 仓库约定摘要\n"
                 "- 提交需遵循该仓库 CONTRIBUTING 约定（测试、代码风格、类型注解）。\n"
                 "- 测试目录通常为 tests/；改动应附带对应测试。")

    if not parts:
        return "(仓库级检索无结果)"
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# 主接口（与实验四同签名，第三参用 level）
# --------------------------------------------------------------------------- #
def build_context(repo: str, number: int, level: str,
                  exclude_comment_id: int | None = None) -> str:
    """返回指定级别的纯文本上下文（不含任务指令）。

    level ∈ {L0,L1,L2,L3,L4}，单调递增。exclude_comment_id 用于 L3 排除生成真值那条。
    """
    if level not in config.LEVELS:
        raise ValueError(f"未知 level: {level}")

    sections: list[str] = [_block_L0(repo, number)]
    if level == "L0":
        return sections[0]

    sections.append(_block_L1_extra(repo, number))
    if level == "L1":
        return "\n\n".join(sections)

    sections.append("## 改前/改后完整函数 (L2)\n" + _block_L2(repo, number))
    if level == "L2":
        return "\n\n".join(sections)

    sections.append("## 关联 Issue 与历史审查 (L3)\n"
                    + _block_L3(repo, number, exclude_comment_id))
    if level == "L3":
        return "\n\n".join(sections)

    sections.append("## 仓库级检索上下文 (L4)\n" + _block_L4(repo, number))
    return "\n\n".join(sections)


if __name__ == "__main__":
    import json
    sample = json.load(open(config.EXP5_SAMPLES_DIR / "classify_ai.json"))[0]
    repo, num = sample["repo"], sample["number"]
    for lv in config.LEVELS:
        ctx = build_context(repo, num, lv)
        print(f"===== {lv} ({config.LEVEL_LABELS[lv]}) len={len(ctx)} =====")
        print(ctx[:300])
        print()
