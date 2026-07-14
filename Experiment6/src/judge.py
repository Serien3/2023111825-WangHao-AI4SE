"""接缝③（唯一新接缝）：LLM-judge 多维打分，修复生成任务 BLEU/ROUGE 失明（R16）。

裁判模型 = deepseek-v4-pro（R14，仅评生成任务）；执行仍用 flash。裁判调用经 llm_client.chat
传 model="deepseek-v4-pro"，content_hash 纳入 model 维度 → 与执行调用天然不同 cache key，
互不覆盖（接缝②）。

score(generated, reference, diff) -> {relevance, actionable, correct, hallucination, rationale}
  relevance    : 是否命中真值指出的问题（1–5）
  actionable   : 是否具体可操作（1–5）
  correct      : 技术上是否正确（1–5）
  hallucination: 是否臆测/幻觉（1–5，越高越严重）
裁判自偏声明：pro 评 flash 规避"自评"，但同厂模型，报告需列为 threat to validity。
"""
from __future__ import annotations

import importlib
import json
import re

from . import config

exp4_llm = importlib.import_module("exp4src.llm_client")

JUDGE_SYSTEM = (
    "你是一位公正、严格的代码审查质量评估专家。你的任务是给'一条针对某代码改动的审查意见'"
    "打分，衡量它相对'人类参考审查意见'的质量。只依据事实，不偏袒，不受意见长度影响。"
)

_JUDGE_INSTR = (
    "请评估下面这条'待评审查意见'的质量。给出 4 个维度的 1–5 分整数评分与简短理由。\n\n"
    "评分维度：\n"
    "- relevance（相关性，1–5）：待评意见是否命中了'人类参考意见'所指出的核心问题？"
    "5=完全命中同一问题，1=完全无关。\n"
    "- actionable（可操作性，1–5）：意见是否具体、给出了明确可执行的改进方向？"
    "5=非常具体可操作，1=空泛无法执行。\n"
    "- correct（技术正确性，1–5）：意见在技术上是否成立、针对该改动是否正确？"
    "5=完全正确，1=明显错误。\n"
    "- hallucination（幻觉程度，1–5，越高越糟）：意见是否包含代码中并不存在的臆测/虚构？"
    "5=严重幻觉，1=无幻觉。\n\n"
    "输出要求：只返回一个 JSON 对象，字段为 "
    '{"relevance": int, "actionable": int, "correct": int, "hallucination": int, '
    '"rationale": "一句话综合理由"}。所有分值为 1–5 的整数。'
)

_DIMS = ("relevance", "actionable", "correct", "hallucination")


def _build_messages(generated: str, reference: str, diff: str) -> list[dict]:
    diff_snip = diff if len(diff) <= 4000 else diff[:4000] + "\n... [已截断]"
    user = "\n\n".join([
        _JUDGE_INSTR,
        "## 代码改动 (Diff)\n```diff\n" + diff_snip + "\n```",
        "## 人类参考审查意见（真值）\n" + (reference or "(无)"),
        "## 待评审查意见\n" + (generated or "(空)"),
    ])
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]


def _parse(text: str) -> dict | None:
    if not text or not text.strip():
        return None
    obj = None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    if not isinstance(obj, dict):
        return None
    out = {}
    for d in _DIMS:
        v = obj.get(d)
        try:
            iv = int(round(float(v)))
        except (TypeError, ValueError):
            return None
        out[d] = max(1, min(5, iv))  # 夹到 [1,5]
    out["rationale"] = str(obj.get("rationale", "")).strip()
    return out


def score(generated: str, reference: str, diff: str,
          pr_key: str = "adhoc", cache_ctx: str = "judge") -> dict:
    """对一条生成意见打分。返回 {relevance, actionable, correct, hallucination,
    rationale, latency, cached, parse_error}。

    走 llm_client.chat_semantic，model=JUDGE_MODEL；缓存 key 天然含 model 维度。
    pr_key / cache_ctx 用于构造语义 cache key（同一 PR + 同一执行条件唯一）。
    """
    messages = _build_messages(generated, reference, diff)
    result = exp4_llm.chat_semantic(
        task="judge", context=cache_ctx, prompt="pro", pr_key=pr_key,
        messages=messages, temperature=config.JUDGE_TEMPERATURE,
        max_tokens=config.JUDGE_MAX_TOKENS, json_mode=True,
        model=config.JUDGE_MODEL)
    parsed = _parse(result["text"])
    row = {"latency": result["latency"], "cached": result["cached"]}
    if parsed is None:
        row.update({d: None for d in _DIMS})
        row.update({"rationale": "", "parse_error": True})
    else:
        row.update(parsed)
        row["parse_error"] = False
    return row


if __name__ == "__main__":
    # 冒烟：命中真值的意见 vs 明显幻觉/无关的意见，应给出可区分分档（不断言精确分值）。
    diff = ("### 文件: util.py\n```diff\n+def div(a, b):\n+    return a / b\n```")
    ref = "这里没有处理 b == 0 的除零情况，建议加校验。"
    good = "建议在 div 中校验 b 是否为 0，否则会抛 ZeroDivisionError。"
    bad = "这个函数用了不存在的 numpy 依赖，性能很差，建议改写整个模块。"
    print("命中真值:", {k: score(good, ref, diff, pr_key="smoke_good")[k] for k in _DIMS})
    print("无关幻觉:", {k: score(bad, ref, diff, pr_key="smoke_bad")[k] for k in _DIMS})
