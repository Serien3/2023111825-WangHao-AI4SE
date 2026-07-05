"""AI 标签判定：启发式识别 AI 生成代码 / AI 审查者，记录命中的具体信号。

判定规则（config 中定义）：
  - is_ai_code: commit/PR body 含 Co-authored-by AI、作者是代码生成 bot、AI label
  - has_ai_reviewer: reviewer/评论者 login 属于已知 AI 审查 bot 集合

所有判定都记录具体命中的信号（便于人工复核、分析弱信号/强信号）。
"""
from __future__ import annotations

from typing import Any

from . import config


def check_ai_code(pr_data: dict) -> tuple[bool, list[str]]:
    """判定 PR 是否为 AI 生成代码。返回 (is_ai, signals)。

    signals 为命中的具体信号列表，便于复核。
    """
    signals: list[str] = []
    detail: dict[str, Any] = pr_data["detail"]

    # 1. 作者是代码生成 bot → 强信号
    author = detail["user"]["login"].lower()
    if author in config.AI_CODE_AUTHOR_BOTS:
        signals.append(f"author_bot:{author}")

    # 2. Co-authored-by AI（commit message 或 PR body）→ 强信号
    body = (detail.get("body") or "").lower()
    for commit in pr_data.get("commits", []):
        msg = commit.get("commit", {}).get("message", "").lower()
        for pat in config.AI_CODE_COAUTHOR_PATTERNS:
            if pat in msg:
                signals.append(f"coauthor_commit:{pat}")
                break
    for pat in config.AI_CODE_COAUTHOR_PATTERNS:
        if pat in body:
            signals.append(f"coauthor_body:{pat}")

    # 3. PR body/title 提及 AI 生成工具 → 弱信号（单独标注）
    title = detail.get("title", "").lower()
    for hint in config.AI_CODE_TEXT_HINTS:
        if hint in body or hint in title:
            signals.append(f"text_hint:{hint}")

    # 4. label 含 AI 相关标签 → 弱信号
    labels = [lbl["name"].lower() for lbl in detail.get("labels", [])]
    ai_label_keywords = {"ai", "copilot", "generated", "bot", "automated"}
    for lbl in labels:
        if any(kw in lbl for kw in ai_label_keywords):
            signals.append(f"label:{lbl}")

    is_ai = len(signals) > 0
    return is_ai, signals


def check_ai_reviewer(pr_data: dict) -> tuple[bool, list[str]]:
    """判定是否存在 AI reviewer。返回 (has_ai, signals)。

    信号：reviewer / issue_comment / review_comment 作者属于已知 AI 审查 bot。
    """
    signals: list[str] = []
    found_reviewers: set[str] = set()

    # 1. reviews 列表里的 reviewer
    for review in pr_data.get("reviews", []):
        user = review.get("user")
        if user:
            login = user["login"].lower()
            if login in config.AI_REVIEWER_BOTS and login not in found_reviewers:
                signals.append(f"reviewer:{login}")
                found_reviewers.add(login)

    # 2. review_comments（inline）作者
    for comment in pr_data.get("review_comments", []):
        user = comment.get("user")
        if user:
            login = user["login"].lower()
            if login in config.AI_REVIEWER_BOTS and login not in found_reviewers:
                signals.append(f"review_comment_author:{login}")
                found_reviewers.add(login)

    # 3. issue_comments（PR 对话区）作者
    for comment in pr_data.get("issue_comments", []):
        user = comment.get("user")
        if user:
            login = user["login"].lower()
            if login in config.AI_REVIEWER_BOTS and login not in found_reviewers:
                signals.append(f"issue_comment_author:{login}")
                found_reviewers.add(login)

    has_ai = len(signals) > 0
    return has_ai, signals


def label_pr(pr_data: dict) -> dict:
    """对单个 PR 进行 AI 标签判定，返回标签字典。

    返回 {"is_ai_code": bool, "ai_code_signals": [str], ...}
    """
    is_ai_code, ai_code_signals = check_ai_code(pr_data)
    has_ai_reviewer, ai_reviewer_signals = check_ai_reviewer(pr_data)
    return {
        "is_ai_code": is_ai_code,
        "ai_code_signals": ai_code_signals,
        "has_ai_reviewer": has_ai_reviewer,
        "ai_reviewer_signals": ai_reviewer_signals,
    }
