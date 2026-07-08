"""切片 4：主循环。遍历 task × context × prompt × sample → 调 LLM → 解析 → 落盘。

用法：
  uv run python -m src.run_experiments --task classify --limit 3          # 冒烟
  uv run python -m src.run_experiments --task classify --only C1 P1       # 单条件
  uv run python -m src.run_experiments --task classify                    # 全量分类
  uv run python -m src.run_experiments --task generate                    # 全量生成
  uv run python -m src.run_experiments --task all                         # 两任务全量

产物：results/predictions/{task}_predictions.parquet（每行一次调用）。
缓存幂等，重跑跳过已完成调用。
"""
from __future__ import annotations

import argparse
import json
import re

import pandas as pd
from tqdm import tqdm

from . import config, context_builder, llm_client, prompts

_DECISION_RE = re.compile(config.DECISION_PATTERN, re.IGNORECASE)


# --------------------------------------------------------------------------- #
# 分类输出解析（决策 D）
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# 结构化输出解析（JSON 模式）
# --------------------------------------------------------------------------- #
def _load_json(text: str) -> dict | None:
    """稳健解析 JSON：直接 loads，失败则抓第一个 {...} 块再试。"""
    if not text or not text.strip():
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def parse_decision(text: str) -> bool | None:
    """从 JSON 输出取 decision → True(MERGE)/False(REJECT)/None(弃权)。

    兼容旧的正则回退：JSON 缺失时仍尝试抓 DECISION: 标签。
    """
    obj = _load_json(text)
    if obj is not None and "decision" in obj:
        val = str(obj["decision"]).strip().upper()
        if val == "MERGE":
            return True
        if val == "REJECT":
            return False
    matches = _DECISION_RE.findall(text or "")  # 正则回退
    if matches:
        return matches[-1].upper() == "MERGE"
    return None


def _extract_comment(text: str) -> str:
    """从 JSON 输出取 comment 字段；缺失时回退到 COMMENT: 前缀或全文。"""
    obj = _load_json(text)
    if obj is not None and "comment" in obj:
        return str(obj["comment"]).strip()
    if not text:
        return ""
    m = re.search(r"COMMENT:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


# --------------------------------------------------------------------------- #
# 样本加载
# --------------------------------------------------------------------------- #
def load_samples(task: str) -> list[dict]:
    name = "classify_50.json" if task == "classify" else "generate_50.json"
    with open(config.SAMPLES_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def _pr_key(rec: dict) -> str:
    return f"{rec['repo']}#{rec['number']}"


# --------------------------------------------------------------------------- #
# 单条件运行
# --------------------------------------------------------------------------- #
def run_condition(task: str, ctx: str, prm: str, samples: list[dict],
                  retry_parse: bool = True) -> list[dict]:
    temperature = config.TEMPERATURE[task]
    max_tokens = config.MAX_TOKENS[task]
    json_mode = config.JSON_MODE[task]
    rows = []
    for rec in samples:
        repo, number = rec["repo"], rec["number"]
        context_text = context_builder.build_context(repo, number, ctx)
        messages = prompts.build_prompt(task, prm, context_text)
        result = llm_client.chat_semantic(
            task, ctx, prm, _pr_key(rec), messages, temperature, max_tokens,
            json_mode=json_mode)
        text = result["text"]

        row = {
            "task": task, "context": ctx, "prompt": prm,
            "repo": repo, "number": number,
            "latency": result["latency"], "cached": result["cached"],
            "raw_text": text,
            "finish_reason": result.get("finish_reason"),
            "prompt_tokens": (result.get("usage") or {}).get("prompt_tokens"),
            "completion_tokens": (result.get("usage") or {}).get("completion_tokens"),
        }

        if task == "classify":
            pred = parse_decision(text)
            if pred is None and retry_parse:
                # 解析失败重试 1 次（不缓存这次强制重试，避免污染语义缓存）
                retry = llm_client.chat(messages, temperature, max_tokens,
                                        cache_key=None, json_mode=json_mode)
                text2 = retry["text"]
                pred = parse_decision(text2)
                row["raw_text"] = text2
                row["latency"] += retry["latency"]
                row["reparsed"] = True
            row["pred_merge"] = pred
            row["parse_error"] = pred is None
            row["true_merge"] = bool(rec["is_merged"])
        else:
            row["pred_comment"] = _extract_comment(text)
            row["target_comment"] = rec["target_comment"]
            row["target_path"] = rec.get("target_path")

        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# 全矩阵
# --------------------------------------------------------------------------- #
def run_task(task: str, only: tuple[str, str] | None, limit: int | None) -> pd.DataFrame:
    samples = load_samples(task)
    if limit is not None:
        samples = samples[:limit]

    contexts = [only[0]] if only else config.CONTEXTS
    prompts_list = [only[1]] if only else config.PROMPTS
    conditions = [(c, p) for c in contexts for p in prompts_list]

    print(f"\n===== 任务 {task}: {len(conditions)} 条件 × {len(samples)} 样本 "
          f"= {len(conditions) * len(samples)} 次调用 =====")

    all_rows = []
    for ctx, prm in conditions:
        desc = f"{task} {ctx}/{prm}"
        rows = []
        for rec in tqdm(samples, desc=desc, ncols=80):
            rows.extend(run_condition(task, ctx, prm, [rec]))
        all_rows.extend(rows)
        if task == "classify":
            errs = sum(r["parse_error"] for r in rows)
            print(f"  {desc}: parse_error={errs}/{len(rows)}")

    df = pd.DataFrame(all_rows)
    out = config.PREDICTIONS_DIR / f"{task}_predictions.parquet"
    # 增量合并：保留其它条件已有结果
    if out.exists() and (only or limit):
        prev = pd.read_parquet(out)
        keep = prev[~prev.set_index(["context", "prompt", "repo", "number"]).index.isin(
            df.set_index(["context", "prompt", "repo", "number"]).index)]
        df = pd.concat([keep, df], ignore_index=True)
    df.to_parquet(out, index=False)
    print(f"  → 落盘 {out.relative_to(config.PROJECT_ROOT)} ({len(df)} 行)")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["classify", "generate", "all"], default="all")
    ap.add_argument("--only", nargs=2, metavar=("CONTEXT", "PROMPT"),
                    help="只跑单条件，如 --only C1 P1")
    ap.add_argument("--limit", type=int, default=None, help="每条件只取前 N 样本（冒烟）")
    args = ap.parse_args()

    only = tuple(args.only) if args.only else None
    tasks = config.TASKS if args.task == "all" else [args.task]
    for task in tasks:
        run_task(task, only, args.limit)


if __name__ == "__main__":
    main()
