"""API 拉取层（L2–L4 唯一触网模块）：文件内容@sha / issue / code search，全部落盘缓存。

复用实验一 GitHubClient 的限流/退避/分页能力。所有结果按确定性 key 落盘
(config.FETCH_CACHE_DIR)，重跑命中不重复触网（R7：防限流、可复现）。
拉取失败/404 一律返回空并缓存空结果，保证上下文构建对网络异常鲁棒（截断/缺失可复现）。

不做 RAG / 不向量化 / 不 clone 整仓库（R5，Out of Scope）：L4 仅轻量词法检索。
"""
from __future__ import annotations

import base64
import hashlib
import json
import re

from . import config

# 复用实验一 GitHub 客户端（限流/退避/分页）
import importlib.util as _ilu

_exp1_src = config.REPO_ROOT / "Experiment1" / "src"
_spec = _ilu.spec_from_file_location(
    "exp1src", _exp1_src / "__init__.py",
    submodule_search_locations=[str(_exp1_src)])
import sys as _sys

if "exp1src" not in _sys.modules:
    _mod = _ilu.module_from_spec(_spec)
    _sys.modules["exp1src"] = _mod
    _spec.loader.exec_module(_mod)

_client = None


def _get_client():
    """惰性构造 GitHubClient；token 从 .env 读。"""
    global _client
    if _client is None:
        from dotenv import load_dotenv
        load_dotenv(config.ENV_FILE)
        import os
        from exp1src.github_client import GitHubClient
        tok = os.getenv(config.GITHUB_TOKEN_ENV)
        if not tok:
            raise RuntimeError(f"缺少 {config.GITHUB_TOKEN_ENV}（{config.ENV_FILE}）。")
        _client = GitHubClient(token=tok)
    return _client


# --------------------------------------------------------------------------- #
# 落盘缓存（确定性 key）
# --------------------------------------------------------------------------- #
def _cache_path(namespace: str, key: str):
    safe = re.sub(r"[^0-9A-Za-z_.-]", "_", key)
    if len(safe) > 120:  # 过长 key 用哈希收尾，避免文件名超限
        safe = safe[:100] + "_" + hashlib.sha1(key.encode()).hexdigest()[:12]
    return config.FETCH_CACHE_DIR / f"{namespace}__{safe}.json"


def _read_cache(namespace: str, key: str):
    p = _cache_path(namespace, key)
    if p.exists():
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _write_cache(namespace: str, key: str, payload) -> None:
    with open(_cache_path(namespace, key), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _cached(namespace: str, key: str, producer):
    """通用缓存包装：命中返回缓存值，未命中调 producer() 并落盘。

    落盘统一为 {"value": ...} 或 {"__error__": msg}。异常时缓存错误标记并返回 None，
    避免反复触网重试同一失败资源（可复现）。
    """
    hit = _read_cache(namespace, key)
    if isinstance(hit, dict):
        if "__error__" in hit:
            return None
        if "value" in hit:
            return hit["value"]
    try:
        value = producer()
        _write_cache(namespace, key, {"value": value})
        return value
    except Exception as e:  # noqa: BLE001 — 拉取失败不应中断上下文构建
        _write_cache(namespace, key, {"__error__": f"{type(e).__name__}: {e}"})
        return None


# --------------------------------------------------------------------------- #
# 文件内容 @ sha（L2）
# --------------------------------------------------------------------------- #
def get_file_at_sha(repo: str, path: str, sha: str) -> str | None:
    """拉取某 commit sha 下某文件的完整文本；不存在/二进制返回 None。"""
    if not (repo and path and sha):
        return None
    key = f"{repo}__{path}__{sha}"

    def _producer():
        c = _get_client()
        data = c.get_json(f"/repos/{repo}/contents/{path}", params={"ref": sha})
        if not isinstance(data, dict):
            return None
        if data.get("encoding") == "base64" and data.get("content"):
            try:
                raw = base64.b64decode(data["content"])
                return raw.decode("utf-8", errors="replace")
            except Exception:
                return None
        return None

    return _cached("file", key, _producer)


# --------------------------------------------------------------------------- #
# Issue 正文（L3）
# --------------------------------------------------------------------------- #
def get_issue(repo: str, number: int) -> dict | None:
    """拉取 issue/PR 的 title+body（GitHub issues 端点对 PR 也返回）。"""
    key = f"{repo}__{number}"

    def _producer():
        c = _get_client()
        data = c.get_json(f"/repos/{repo}/issues/{number}")
        if not isinstance(data, dict):
            return None
        return {"number": number, "title": data.get("title") or "",
                "body": data.get("body") or ""}

    return _cached("issue", key, _producer)


# --------------------------------------------------------------------------- #
# 代码检索（L4）：仓库内符号其它出现处
# --------------------------------------------------------------------------- #
def code_search(repo: str, symbol: str, top_k: int) -> list[dict]:
    """GitHub code search：仓库内 symbol 出现的文件路径 + 片段。

    返回 [{path, fragment}]。检索失败/无结果返回 []。轻量词法，不做向量化。
    """
    if not symbol or len(symbol) < 3:
        return []
    key = f"{repo}__{symbol}__k{top_k}"

    def _producer():
        c = _get_client()
        # code search 端点：q=symbol repo:owner/name；仅取路径与文本匹配片段
        q = f"{symbol} repo:{repo} in:file language:Python"
        data = c.get_json("/search/code", params={"q": q, "per_page": top_k})
        if not isinstance(data, dict):
            return []
        out = []
        for item in (data.get("items") or [])[:top_k]:
            frag_parts = []
            for tm in (item.get("text_matches") or []):
                if tm.get("fragment"):
                    frag_parts.append(tm["fragment"])
            out.append({"path": item.get("path", ""),
                        "fragment": "\n".join(frag_parts)[:config.RETRIEVAL_SNIPPET_CHARS]})
        return out

    result = _cached("codesearch", key, _producer)
    return result if isinstance(result, list) else []


def list_dir(repo: str, dirpath: str, sha: str) -> list[str]:
    """列出某目录下的文件名（L4 同目录相关文件用）。失败返回 []。"""
    key = f"{repo}__{dirpath or '.'}__{sha}"

    def _producer():
        c = _get_client()
        data = c.get_json(f"/repos/{repo}/contents/{dirpath}", params={"ref": sha})
        if not isinstance(data, list):
            return []
        return [it.get("path", "") for it in data if it.get("type") == "file"]

    result = _cached("listdir", key, _producer)
    return result if isinstance(result, list) else []


if __name__ == "__main__":
    # 冒烟：对已知 PR 拉一个文件（需要 GITHUB_TOKEN 且网络可达）
    import pandas as pd
    prs = pd.read_parquet(config.EXP1_DATA_DIR / "prs.parquet")
    row = prs[prs["repo"] == "home-assistant/core"].iloc[0]
    print("repo", row["repo"], "head_sha", row["head_sha"])
    issue = get_issue(row["repo"], int(row["number"]))
    print("issue title:", (issue or {}).get("title"))
