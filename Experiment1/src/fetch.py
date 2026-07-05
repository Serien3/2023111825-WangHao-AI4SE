"""抓取脚本：对每个仓库抓取 PR 及其全部子资源，缓存到 results/raw/。

缓存策略（断点续传）：每个 PR 存为 results/raw/<owner>__<repo>/pr_<number>.json。
再次运行时若文件已存在则跳过，可安全中断/续跑。

每个 PR 抓取以下内容（对应实验一步骤一~步骤三）：
  - detail        : PR 基本信息 + additions/deletions/changed_files/merged 等
  - files         : 修改文件 + patch(diff)         → 实验二/三/四
  - reviews       : Review（reviewer/state/body）  → Merge 决策
  - review_comments: inline 评论 + diff_hunk        → 实验三 生成任务的命门
  - issue_comments: PR 对话区评论                   → 用于 AI reviewer 判定/上下文
  - commits       : commit message + author         → 实验二文本特征 / AI 代码判定

用法：
    uv run python -m src.fetch                 # 全量（5 仓库）
    uv run python -m src.fetch --repo django/django --limit 5
    uv run python -m src.fetch --no-ai         # 跳过 AI 定向补采
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from . import config
from .github_client import GitHubClient


def repo_raw_dir(repo: str) -> Path:
    d = config.RAW_DIR / repo.replace("/", "__")
    d.mkdir(parents=True, exist_ok=True)
    return d


def pr_cache_path(repo: str, number: int) -> Path:
    return repo_raw_dir(repo) / f"pr_{number}.json"


def fetch_one_pr(client: GitHubClient, repo: str, number: int) -> dict:
    """抓取单个 PR 的全部子资源，组装成一个 dict。"""
    base = f"/repos/{repo}/pulls/{number}"
    detail = client.get_json(base)
    if detail is None:
        raise RuntimeError(f"PR 不存在: {repo}#{number}")

    files = list(client.paginate(f"{base}/files"))
    reviews = list(client.paginate(f"{base}/reviews"))
    review_comments = list(client.paginate(f"{base}/comments"))
    commits = list(client.paginate(f"{base}/commits"))
    # issue_comments 走 issues 端点（PR 也是 issue）
    issue_comments = list(
        client.paginate(f"/repos/{repo}/issues/{number}/comments")
    )

    return {
        "repo": repo,
        "number": number,
        "detail": detail,
        "files": files,
        "reviews": reviews,
        "review_comments": review_comments,
        "issue_comments": issue_comments,
        "commits": commits,
    }


def save_pr(repo: str, number: int, data: dict) -> None:
    path = pr_cache_path(repo, number)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def list_closed_pr_numbers(
    client: GitHubClient, repo: str, limit: int
) -> list[int]:
    """列出最近 `limit` 个已关闭 PR 的编号（merged 与 unmerged 都要）。"""
    numbers: list[int] = []
    params = {"state": "closed", "sort": "created", "direction": "desc"}
    for pr in client.paginate(f"/repos/{repo}/pulls", params=params, max_items=limit):
        numbers.append(pr["number"])
    return numbers


def ai_oversample_numbers(
    client: GitHubClient, repo: str, limit: int
) -> list[int]:
    """通过 search API 定向搜集带 AI 信号的已关闭 PR 编号。"""
    found: set[int] = set()
    for q in config.AI_SEARCH_QUERIES:
        if len(found) >= limit:
            break
        query = f"repo:{repo} is:pr is:closed {q}"
        try:
            data = client.get_json(
                "/search/issues",
                params={"q": query, "per_page": 30},
            )
        except Exception as exc:  # search 限额较紧，失败就跳过该查询
            print(f"  [ai-search] 查询失败({q}): {exc}")
            continue
        if not data:
            continue
        for item in data.get("items", []):
            found.add(item["number"])
            if len(found) >= limit:
                break
    return sorted(found)


def fetch_repo(
    client: GitHubClient,
    repo: str,
    base_limit: int,
    ai_limit: int,
    include_ai: bool,
) -> None:
    print(f"\n=== {repo} ===")
    base_numbers = list_closed_pr_numbers(client, repo, base_limit)
    print(f"  基础已关闭 PR: {len(base_numbers)} 个")

    ai_numbers: list[int] = []
    if include_ai and ai_limit > 0:
        ai_numbers = ai_oversample_numbers(client, repo, ai_limit)
        print(f"  AI 定向补采候选: {len(ai_numbers)} 个")

    # 合并去重，保留标注来源
    all_numbers = list(dict.fromkeys(base_numbers + ai_numbers))
    oversampled = set(ai_numbers) - set(base_numbers)

    for number in tqdm(all_numbers, desc=f"  抓取 {repo}", unit="pr"):
        path = pr_cache_path(repo, number)
        if path.exists():
            continue  # 断点续传：已缓存则跳过
        try:
            data = fetch_one_pr(client, repo, number)
        except Exception as exc:
            print(f"  [warn] 抓取 {repo}#{number} 失败: {exc}")
            continue
        data["_oversampled"] = number in oversampled
        save_pr(repo, number, data)


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", help="只抓单个仓库，如 django/django")
    ap.add_argument(
        "--limit", type=int, default=config.PRS_PER_REPO, help="每仓库基础 PR 数"
    )
    ap.add_argument(
        "--ai-limit", type=int, default=config.AI_OVERSAMPLE_PER_REPO
    )
    ap.add_argument("--no-ai", action="store_true", help="跳过 AI 定向补采")
    args = ap.parse_args()

    client = GitHubClient()
    repos = [args.repo] if args.repo else config.REPOS
    for repo in repos:
        fetch_repo(
            client,
            repo,
            base_limit=args.limit,
            ai_limit=args.ai_limit,
            include_ai=not args.no_ai,
        )

    rl = client.rate_limit()["resources"]["core"]
    print(f"\n完成。核心限额剩余 {rl['remaining']}/{rl['limit']}")


if __name__ == "__main__":
    main()
