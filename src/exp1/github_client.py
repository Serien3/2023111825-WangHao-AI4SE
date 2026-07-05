"""GitHub REST API 客户端：认证、限流处理、指数退避重试、自动分页。

只依赖 requests，便于精细控制 rate-limit 与缓存。所有 GET 都经过 `get_json` /
`paginate`，统一处理：
  - 认证头
  - 核心限额耗尽 → 等到 reset 再继续
  - 二级限流(secondary rate limit) / 5xx → 指数退避重试
  - 分页（Link header）
"""
from __future__ import annotations

import time
from typing import Any, Iterator

import requests

from . import config


class GitHubClient:
    def __init__(self, token: str | None = None):
        self.session = requests.Session()
        tok = token or config.get_token()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {tok}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "ai-software-engineer-exp1",
            }
        )

    # ------------------------------------------------------------------ #
    # 底层请求
    # ------------------------------------------------------------------ #
    def _request(self, url: str, params: dict | None = None) -> requests.Response:
        """带重试与限流处理的单次 GET。返回 Response（调用方负责解析）。"""
        last_exc: Exception | None = None
        for attempt in range(config.MAX_RETRIES):
            try:
                resp = self.session.get(
                    url, params=params, timeout=config.REQUEST_TIMEOUT
                )
            except requests.RequestException as exc:  # 网络抖动
                last_exc = exc
                self._backoff_sleep(attempt)
                continue

            # 核心限额耗尽（403/429 且 remaining==0）→ 等到 reset
            if resp.status_code in (403, 429):
                if self._handle_rate_limit(resp, attempt):
                    continue  # 已休眠，重试

            if resp.status_code >= 500:  # 服务端错误 → 退避重试
                self._backoff_sleep(attempt)
                continue

            if resp.status_code == 404:
                # 资源不存在（例如某些子资源）——交给调用方判断，不重试
                return resp

            if resp.ok:
                return resp

            # 其它 4xx：不可重试，直接抛出
            resp.raise_for_status()

        if last_exc:
            raise last_exc
        raise RuntimeError(f"请求失败，已重试 {config.MAX_RETRIES} 次: {url}")

    def _handle_rate_limit(self, resp: requests.Response, attempt: int) -> bool:
        """处理限流。返回 True 表示已休眠、应重试；False 表示非限流的 403。"""
        remaining = resp.headers.get("X-RateLimit-Remaining")
        # 主限额耗尽：等到 reset 时间
        if remaining == "0":
            reset = int(resp.headers.get("X-RateLimit-Reset", "0"))
            wait = max(reset - int(time.time()), 0) + config.RATE_LIMIT_SLEEP_BUFFER
            print(f"  [rate-limit] 核心限额耗尽，休眠 {wait}s 后继续 …")
            time.sleep(wait)
            return True
        # 二级限流：有 Retry-After 或退避
        retry_after = resp.headers.get("Retry-After")
        if retry_after is not None:
            wait = int(retry_after) + config.RATE_LIMIT_SLEEP_BUFFER
            print(f"  [secondary-limit] 休眠 {wait}s …")
            time.sleep(wait)
            return True
        # body 含 secondary rate limit 提示
        if "secondary rate limit" in resp.text.lower():
            self._backoff_sleep(attempt)
            return True
        return False

    @staticmethod
    def _backoff_sleep(attempt: int) -> None:
        wait = config.RETRY_BACKOFF_BASE ** attempt
        time.sleep(wait)

    # ------------------------------------------------------------------ #
    # 高层接口
    # ------------------------------------------------------------------ #
    def get_json(self, path: str, params: dict | None = None) -> Any:
        """GET 单个资源，返回解析后的 JSON。path 可为相对(/repos/..)或完整 URL。"""
        url = path if path.startswith("http") else f"{config.GITHUB_API}{path}"
        resp = self._request(url, params=params)
        if resp.status_code == 404:
            return None
        return resp.json()

    def paginate(
        self, path: str, params: dict | None = None, max_items: int | None = None
    ) -> Iterator[dict]:
        """自动翻页，逐条 yield。max_items 为 None 表示取全部。"""
        url = path if path.startswith("http") else f"{config.GITHUB_API}{path}"
        params = dict(params or {})
        params.setdefault("per_page", 100)
        count = 0
        while url:
            resp = self._request(url, params=params)
            if resp.status_code == 404:
                return
            items = resp.json()
            if not isinstance(items, list):
                return
            for item in items:
                yield item
                count += 1
                if max_items is not None and count >= max_items:
                    return
            url = resp.links.get("next", {}).get("url")
            params = None  # 后续页 URL 已含参数

    def rate_limit(self) -> dict:
        return self.get_json("/rate_limit")


if __name__ == "__main__":  # 手动 smoke test
    from dotenv import load_dotenv

    load_dotenv()
    c = GitHubClient()
    rl = c.rate_limit()["resources"]["core"]
    print("core remaining:", rl["remaining"], "/", rl["limit"])
