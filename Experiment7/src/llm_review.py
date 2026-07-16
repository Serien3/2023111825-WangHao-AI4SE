from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Protocol

from openai import OpenAI

from .config import Settings
from .context_builder import ReviewBatch
from .schemas import BatchReview


class ReviewModel(Protocol):
    model_id: str

    def review(
        self, batch: ReviewBatch, profile: str, *, allow_repair: bool = True
    ) -> tuple[BatchReview, dict]: ...

    def summarize(
        self, reviews: list[BatchReview], coverage: dict, profile: str
    ) -> tuple[BatchReview, dict]: ...


class ModelNotConfigured(RuntimeError):
    pass


class DeepSeekReviewModel:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_id = settings.model_id

    def _messages(self, batch: ReviewBatch, profile: str, repair: str | None = None) -> list[dict]:
        role = (
            "你是极其挑剔的资深维护者。先主动寻找会阻断提交或创建 PR 的缺陷，再反推合并就绪度。"
            if profile == "strict" else "你是高效的资深代码审查者，快速判断本地修改的合并就绪度。"
        )
        schema = (
            '{"decision":"MERGE|REJECT","risk_level":"low|medium|high","summary":"...",'
            '"reasoning":"...","blocking_defects":[],"findings":[{"path":"...","line":1,'
            '"severity":"info|warning|blocker","category":"correctness|security|compatibility|performance|maintainability|testing|style|documentation",'
            '"message":"...","suggestion":"..."}]}'
        )
        if repair is not None:
            return [
                {"role": "system", "content": "修复 JSON 格式。只输出符合指定 schema 的 JSON 对象，不增加事实。"},
                {"role": "user", "content": f"Schema: {schema}\n待修复内容:\n{repair[:12000]}"},
            ]
        return [
            {"role": "system", "content": role},
            {"role": "user", "content": (
                "这是本地 Git 工作树相对 HEAD 的修改，不是 GitHub PR。判断它是否具备进入提交/PR 的条件。"
                "最多返回 5 条意见；意见路径必须属于输入，行号只能是修改后新增行。只输出 JSON。\n"
                f"Schema: {schema}\n\n{batch.context}"
            )},
        ]

    def _call(self, messages: list[dict], cache_key: str) -> dict:
        cache_path = self.settings.cache_dir / f"{cache_key}.json"
        if cache_path.is_file():
            value = json.loads(cache_path.read_text(encoding="utf-8"))
            value["cached"] = True
            return value
        key = os.getenv("DEEPSEEK_API_KEY")
        if not key:
            raise ModelNotConfigured("DEEPSEEK_API_KEY is not configured")
        client = OpenAI(api_key=key, base_url=self.settings.api_base_url, timeout=120)
        last_error: Exception | None = None
        for attempt in range(3):
            started = time.perf_counter()
            try:
                response = client.chat.completions.create(
                    model=self.model_id, messages=messages, temperature=0,
                    max_tokens=4096, response_format={"type": "json_object"},
                )
                usage = response.usage
                value = {
                    "text": response.choices[0].message.content or "",
                    "latency": time.perf_counter() - started,
                    "usage": {
                        "prompt_tokens": getattr(usage, "prompt_tokens", None),
                        "completion_tokens": getattr(usage, "completion_tokens", None),
                        "total_tokens": getattr(usage, "total_tokens", None),
                    },
                    "cached": False,
                }
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
                return value
            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(2 ** attempt)
        raise RuntimeError("LLM request failed after retries") from last_error

    def review(
        self, batch: ReviewBatch, profile: str, *, allow_repair: bool = True
    ) -> tuple[BatchReview, dict]:
        messages = self._messages(batch, profile)
        digest = hashlib.sha256(json.dumps([self.model_id, messages], ensure_ascii=False).encode()).hexdigest()
        first = self._call(messages, digest)
        requests = 0 if first["cached"] else 1
        try:
            parsed = BatchReview.model_validate_json(first["text"])
            return parsed, {**first, "requests": requests, "parse_repaired": False}
        except Exception:
            if not allow_repair:
                raise ValueError("Model output is not valid review JSON")
            repair_messages = self._messages(batch, profile, first["text"])
            repair_digest = hashlib.sha256(json.dumps([self.model_id, repair_messages], ensure_ascii=False).encode()).hexdigest()
            repaired = self._call(repair_messages, repair_digest)
            parsed = BatchReview.model_validate_json(repaired["text"])
            return parsed, {
                "usage": repaired["usage"], "latency": first["latency"] + repaired["latency"],
                "cached": first["cached"] and repaired["cached"],
                "requests": requests + (0 if repaired["cached"] else 1), "parse_repaired": True,
            }

    def summarize(
        self, reviews: list[BatchReview], coverage: dict, profile: str
    ) -> tuple[BatchReview, dict]:
        payload = {
            "batch_results": [review.model_dump() for review in reviews],
            "coverage": coverage,
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "汇总多批本地代码审查结果。只压缩摘要和理由，不得新增 finding、路径或事实；"
                    "任一批 REJECT 或有 blocker 时必须 REJECT。只输出与批审查相同 schema 的 JSON。"
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        digest = hashlib.sha256(
            json.dumps([self.model_id, profile, messages], ensure_ascii=False).encode()
        ).hexdigest()
        response = self._call(messages, f"summary-{digest}")
        parsed = BatchReview.model_validate_json(response["text"])
        return parsed, {
            **response,
            "requests": 0 if response["cached"] else 1,
            "parse_repaired": False,
        }