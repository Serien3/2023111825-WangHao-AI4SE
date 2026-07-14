"""切片 2：DeepSeek 调用层（唯一触网模块，仿实验一 github_client.py）。

- OpenAI 兼容 SDK 指向 DeepSeek base_url。
- chat(messages, temperature, max_tokens) 统一接口，返回文本 + latency。
- 指数退避重试（最多 MAX_RETRIES 次）。
- 磁盘缓存：key = sha1(model, messages, temperature, max_tokens)，命中直接返回，
  实现断点续传 + 省钱。缓存也记录原始 usage。
- 高层 chat_cached(cache_key_parts, ...) 供主循环用 (task,ctx,prompt,pr) 语义 key。
"""
from __future__ import annotations

import hashlib
import json
import re
import time

from dotenv import load_dotenv
from openai import OpenAI

from . import config

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        load_dotenv(config.ENV_FILE)
        import os
        key = os.getenv(config.API_KEY_ENV)
        if not key:
            raise RuntimeError(
                f"缺少 {config.API_KEY_ENV}，请在 {config.ENV_FILE} 中填写 DeepSeek API key。")
        _client = OpenAI(api_key=key, base_url=config.DEEPSEEK_BASE_URL,
                         timeout=config.REQUEST_TIMEOUT)
    return _client


# --------------------------------------------------------------------------- #
# 缓存
# --------------------------------------------------------------------------- #
def _hash_key(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _cache_path(cache_key: str):
    # cache_key 可能含 repo 名里的 '/'（如 django/django），会被误当子目录；
    # 统一把文件系统非法字符替换为 '_'，保证是扁平单文件。
    safe = re.sub(r"[^0-9A-Za-z_.#-]", "_", cache_key)
    return config.CACHE_DIR / f"{safe}.json"


def _read_cache(cache_key: str) -> dict | None:
    path = _cache_path(cache_key)
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _write_cache(cache_key: str, payload: dict) -> None:
    with open(_cache_path(cache_key), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------- #
# 调用
# --------------------------------------------------------------------------- #
def _call_api(messages: list[dict], temperature: float, max_tokens: int,
              json_mode: bool = False, model: str | None = None) -> dict:
    """单次带重试的原始调用，返回 {text, latency, usage, model}。

    model 为 None 时回退 config.MODEL_ID（保持实验四/五既有行为不变）；
    实验六裁判调用传 model="deepseek-v4-pro"，与执行调用天然不同 key。
    """
    client = _get_client()
    model_id = model or config.MODEL_ID
    last_err: Exception | None = None
    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    for attempt in range(config.MAX_RETRIES):
        t0 = time.perf_counter()
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            latency = time.perf_counter() - t0
            usage = resp.usage
            return {
                "text": resp.choices[0].message.content or "",
                "latency": latency,
                "usage": {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                },
                "finish_reason": resp.choices[0].finish_reason,
                "model": resp.model,
            }
        except Exception as e:  # 限流/网络/超时统一退避重试
            last_err = e
            wait = config.RETRY_BACKOFF_BASE ** attempt
            print(f"    [重试 {attempt + 1}/{config.MAX_RETRIES}] {type(e).__name__}: {e} "
                  f"→ {wait:.1f}s 后重试")
            time.sleep(wait)
    raise RuntimeError(f"调用失败（已重试 {config.MAX_RETRIES} 次）: {last_err}")


def chat(messages: list[dict], temperature: float, max_tokens: int,
         cache_key: str | None = None, json_mode: bool = False,
         model: str | None = None) -> dict:
    """带缓存的调用。cache_key 为 None 时不缓存（用于临时冒烟）。

    model 穿透到 _call_api（None 回退 config.MODEL_ID）。缓存 key 的内容哈希由
    上层 chat_semantic 负责纳入 model 维度，这里只透传。
    """
    if cache_key is not None:
        cached = _read_cache(cache_key)
        if cached is not None:
            cached["cached"] = True
            return cached

    result = _call_api(messages, temperature, max_tokens, json_mode=json_mode, model=model)
    result["cached"] = False
    if cache_key is not None:
        _write_cache(cache_key, result)
    return result


def chat_semantic(task: str, context: str, prompt: str, pr_key: str,
                  messages: list[dict], temperature: float, max_tokens: int,
                  json_mode: bool = False, model: str | None = None) -> dict:
    """主循环入口：以 (task,context,prompt,pr_key) + 内容哈希为缓存 key。

    内容哈希纳入 model/messages/温度/max_tokens/json_mode，
    保证 prompt、调用参数或模型变更后不会误命中旧缓存。
    model=None 时 hash 用 config.MODEL_ID —— 与实验四/五既有行为逐字一致（回归）。
    """
    model_id = model or config.MODEL_ID
    content_hash = _hash_key(model_id, json.dumps(messages, ensure_ascii=False),
                             temperature, max_tokens, json_mode)
    cache_key = f"{task}__{context}__{prompt}__{pr_key}__{content_hash[:12]}"
    return chat(messages, temperature, max_tokens, cache_key=cache_key,
                json_mode=json_mode, model=model)


def _smoke() -> None:
    msgs = [{"role": "user", "content": "回复恰好一行：DECISION: MERGE"}]
    print("首次调用（触网）...")
    r1 = chat(msgs, temperature=0.0, max_tokens=32, cache_key="__smoke__")
    print(f"  text={r1['text']!r} latency={r1['latency']:.2f}s "
          f"usage={r1['usage']} cached={r1['cached']}")
    print("二次调用（应命中缓存）...")
    r2 = chat(msgs, temperature=0.0, max_tokens=32, cache_key="__smoke__")
    print(f"  text={r2['text']!r} cached={r2['cached']}")
    (config.CACHE_DIR / "__smoke__.json").unlink(missing_ok=True)
    print("冒烟完成，已清理缓存。")


if __name__ == "__main__":
    _smoke()
