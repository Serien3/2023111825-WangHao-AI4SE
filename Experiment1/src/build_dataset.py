"""构建数据集：raw json → 5 张规范化表（parquet + csv）。

输出（results/processed/）：
  prs.*            每 PR 一行：元数据 + 统计 + 标签(is_merged) + AI 标签
  files.*          每 PR 每文件一行：filename/status/additions/deletions/patch/language
  reviews.*        每条 review：reviewer/state/body/submitted_at
  review_comments.* 每条 inline 评论：path/diff_hunk/body（实验三生成任务用）
  commits.*        每个 commit：sha/message/author

清洗规则：
  - 剔除无文件变更的 PR（changed_files==0）
  - 标注超大 diff（GitHub 不返回 patch 的文件）
  - 语言按扩展名识别
  - 去重（同 repo+number 只保留一份）

用法：
    uv run python -m src.build_dataset
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from . import config
from .ai_labeling import label_pr

# 扩展名 → 语言（覆盖数据集主要语言；其余归 other）
EXT_LANG = {
    ".py": "python", ".pyx": "cython", ".pyi": "python",
    ".js": "javascript", ".ts": "typescript", ".tsx": "typescript",
    ".go": "go", ".java": "java", ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cc": "cpp", ".hpp": "cpp", ".rs": "rust",
    ".rb": "ruby", ".md": "markdown", ".rst": "rst", ".txt": "text",
    ".json": "json", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
    ".html": "html", ".css": "css", ".sh": "shell", ".sql": "sql",
}


def detect_language(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    return EXT_LANG.get(suffix, "other")


def iter_raw_pr_files() -> list[Path]:
    return sorted(config.RAW_DIR.glob("*/pr_*.json"))


def _review_decision(reviews: list[dict]) -> str:
    """综合 review 状态得出决策：CHANGES_REQUESTED > APPROVED > COMMENTED > NONE。"""
    states = {r.get("state") for r in reviews}
    if "CHANGES_REQUESTED" in states:
        return "CHANGES_REQUESTED"
    if "APPROVED" in states:
        return "APPROVED"
    if "COMMENTED" in states:
        return "COMMENTED"
    return "NONE"


def build() -> None:
    pr_rows: list[dict] = []
    file_rows: list[dict] = []
    review_rows: list[dict] = []
    review_comment_rows: list[dict] = []
    commit_rows: list[dict] = []

    seen: set[tuple[str, int]] = set()
    skipped_no_files = 0

    for path in tqdm(iter_raw_pr_files(), desc="构建数据集", unit="pr"):
        data = json.loads(path.read_text(encoding="utf-8"))
        repo = data["repo"]
        number = data["number"]
        if (repo, number) in seen:  # 去重
            continue
        seen.add((repo, number))

        detail = data["detail"]
        files = data.get("files", [])
        reviews = data.get("reviews", [])
        review_comments = data.get("review_comments", [])
        issue_comments = data.get("issue_comments", [])
        commits = data.get("commits", [])

        # 清洗：无文件变更的 PR 丢弃
        if detail.get("changed_files", 0) == 0 or len(files) == 0:
            skipped_no_files += 1
            continue

        labels = [lbl["name"] for lbl in detail.get("labels", [])]
        reviewers = sorted(
            {r["user"]["login"] for r in reviews if r.get("user")}
        )
        ai = label_pr(data)

        # 是否有文件缺 patch（超大 diff）
        n_missing_patch = sum(1 for f in files if "patch" not in f)

        pr_rows.append({
            "repo": repo,
            "number": number,
            "title": detail.get("title", ""),
            "body": detail.get("body") or "",
            "author": detail["user"]["login"],
            "author_association": detail.get("author_association"),
            "state": detail.get("state"),
            "is_merged": bool(detail.get("merged")),
            "created_at": detail.get("created_at"),
            "closed_at": detail.get("closed_at"),
            "merged_at": detail.get("merged_at"),
            "additions": detail.get("additions", 0),
            "deletions": detail.get("deletions", 0),
            "changed_files": detail.get("changed_files", 0),
            "num_commits": len(commits),
            "num_reviews": len(reviews),
            "num_reviewers": len(reviewers),
            "num_review_comments": len(review_comments),
            "num_issue_comments": len(issue_comments),
            "num_labels": len(labels),
            "labels": "|".join(labels),
            "review_decision": _review_decision(reviews),
            "base_sha": detail.get("base", {}).get("sha"),
            "head_sha": detail.get("head", {}).get("sha"),
            "merge_commit_sha": detail.get("merge_commit_sha"),
            "n_files_missing_patch": n_missing_patch,
            "is_ai_code": ai["is_ai_code"],
            "ai_code_signals": "|".join(ai["ai_code_signals"]),
            "has_ai_reviewer": ai["has_ai_reviewer"],
            "ai_reviewer_signals": "|".join(ai["ai_reviewer_signals"]),
            "oversampled": bool(data.get("_oversampled", False)),
            "pr_url": detail.get("html_url"),
        })

        for f in files:
            file_rows.append({
                "repo": repo, "pr_number": number,
                "filename": f["filename"],
                "status": f.get("status"),
                "additions": f.get("additions", 0),
                "deletions": f.get("deletions", 0),
                "changes": f.get("changes", 0),
                "language": detect_language(f["filename"]),
                "has_patch": "patch" in f,
                "patch": f.get("patch", ""),
                "previous_filename": f.get("previous_filename"),
            })

        for r in reviews:
            review_rows.append({
                "repo": repo, "pr_number": number,
                "review_id": r["id"],
                "reviewer": r["user"]["login"] if r.get("user") else None,
                "state": r.get("state"),
                "submitted_at": r.get("submitted_at"),
                "body": r.get("body") or "",
            })

        for c in review_comments:
            review_comment_rows.append({
                "repo": repo, "pr_number": number,
                "comment_id": c["id"],
                "reviewer": c["user"]["login"] if c.get("user") else None,
                "path": c.get("path"),
                "diff_hunk": c.get("diff_hunk", ""),
                "body": c.get("body") or "",
                "position": c.get("position"),
                "original_position": c.get("original_position"),
                "in_reply_to_id": c.get("in_reply_to_id"),
                "created_at": c.get("created_at"),
            })

        for c in commits:
            commit_rows.append({
                "repo": repo, "pr_number": number,
                "sha": c.get("sha"),
                "message": c.get("commit", {}).get("message", ""),
                "author": (c.get("commit", {}).get("author", {}) or {}).get("name"),
                "author_login": c["author"]["login"] if c.get("author") else None,
            })

    # 落盘
    tables = {
        "prs": pr_rows,
        "files": file_rows,
        "reviews": review_rows,
        "review_comments": review_comment_rows,
        "commits": commit_rows,
    }
    for name, rows in tables.items():
        df = pd.DataFrame(rows)
        df.to_parquet(config.PROCESSED_DIR / f"{name}.parquet", index=False)
        df.to_csv(config.PROCESSED_DIR / f"{name}.csv", index=False)
        print(f"  {name}: {len(df)} 行 -> {name}.parquet / .csv")

    print(f"\n清洗：跳过无文件变更 PR {skipped_no_files} 个；有效 PR {len(pr_rows)} 个。")


if __name__ == "__main__":
    build()
