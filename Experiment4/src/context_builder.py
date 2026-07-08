"""切片 3a：4 种上下文拼装 + token 预算截断。

C1 仅 Diff：拼接该 PR 所有 Python 文件 patch，超预算截断。
C2 Diff + PR 描述：C1 + title + body。
C3 Diff + Commit：C1 + 所有 commit message。
C4 Diff + 全部：C1 + PR 描述 + Commit（最丰富）。

截断策略：diff 优先保证，按字符预算硬截断；body/commit 各有独立预算。
"""
from __future__ import annotations

from . import config, data

CONTEXT_SPEC = {
    "C1": {"diff": True, "desc": False, "commit": False},
    "C2": {"diff": True, "desc": True, "commit": False},
    "C3": {"diff": True, "desc": False, "commit": True},
    "C4": {"diff": True, "desc": True, "commit": True},
}


def _truncate(text: str, budget: int) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    return text if len(text) <= budget else text[:budget] + "\n... [已截断]"


def _build_diff_block(repo: str, number: int) -> str:
    patches = data.pr_python_patches(repo, number)
    if not patches:
        return "(该 PR 无可用的 Python 文件 diff)"
    parts, used = [], 0
    for fn, patch in patches:
        header = f"### 文件: {fn}\n"
        remaining = config.DIFF_CHAR_BUDGET - used
        if remaining <= len(header):
            parts.append("... [剩余文件 diff 已因预算截断]")
            break
        body = _truncate(patch, remaining - len(header))
        block = header + "```diff\n" + body + "\n```"
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)


def _build_desc_block(repo: str, number: int) -> str:
    meta = data.pr_meta(repo, number)
    title = meta.get("title") or ""
    body = _truncate(meta.get("body") or "", config.BODY_CHAR_BUDGET)
    out = f"标题: {title}"
    if body.strip():
        out += f"\n\n描述:\n{body}"
    return out


def _build_commit_block(repo: str, number: int) -> str:
    msgs = data.pr_commit_messages(repo, number)
    if not msgs:
        return "(无 commit message)"
    joined = "\n".join(f"- {m.strip().splitlines()[0]}" if m.strip() else "-"
                       for m in msgs)
    return _truncate(joined, config.COMMIT_CHAR_BUDGET)


def build_context(repo: str, number: int, kind: str) -> str:
    """返回指定上下文类型的纯文本块（不含任务指令）。"""
    spec = CONTEXT_SPEC[kind]
    sections = []
    if spec["diff"]:
        sections.append("## 代码改动 (Diff)\n" + _build_diff_block(repo, number))
    if spec["desc"]:
        sections.append("## PR 描述\n" + _build_desc_block(repo, number))
    if spec["commit"]:
        sections.append("## Commit 信息\n" + _build_commit_block(repo, number))
    return "\n\n".join(sections)


if __name__ == "__main__":
    import json
    sample = json.load(open(config.SAMPLES_DIR / "classify_50.json"))[0]
    for k in config.CONTEXTS:
        ctx = build_context(sample["repo"], sample["number"], k)
        print(f"===== {k} ({config.CONTEXT_LABELS[k]}) len={len(ctx)} =====")
        print(ctx[:400])
        print()
