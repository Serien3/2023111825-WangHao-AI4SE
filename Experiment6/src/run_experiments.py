"""主循环（R12/R13）：遍历实验矩阵 → 构建上下文 + Prompt → 调执行模型 → 解析 → 落盘。

条件来自 config.matrix(task)：消融①(上下文阶梯) + 消融②(Prompt) + 归因格 + 人类对照。
- P1–P5：单轮 chat_semantic（复用实验四解析：分类 decision、生成 comment）。
- P6：多轮，把上一轮 assistant 回复回填 messages，每轮独立缓存 key（R10，仅 L4）。
- 生成任务额外调 judge.score（裁判 pro，R16），把 4 维分值并入行。

样本：R1 直接读实验五 AI 池（classify_ai=50 / generate_ai=72）；R2 人类对照读实验五
matched_human_control（classify_control / generate_control），仅在 human_control 格用。

用法：
  uv run python -m src.run_experiments --task classify --limit 1            # 冒烟
  uv run python -m src.run_experiments --task generate --only L0 P3         # 单条件
  uv run python -m src.run_experiments --task all                          # 全量
  uv run python -m src.run_experiments --task generate --no-judge          # 跳过裁判
产物：results/predictions/{task}_predictions.parquet（每行一次执行调用）。
缓存幂等，重跑跳过已完成调用。
"""
from __future__ import annotations

import argparse
import importlib
import json

import pandas as pd
from tqdm import tqdm

from . import config, context_builder, judge, prompts

exp4_run = importlib.import_module("exp4src.run_experiments")
exp4_llm = importlib.import_module("exp4src.llm_client")


# --------------------------------------------------------------------------- #
# 样本加载（R1/R2：直接读实验五 AI 池 + 匹配对照）
# --------------------------------------------------------------------------- #
def load_samples(task: str, group: str) -> list[dict]:
    name = f"{task}_{group}.json"
    with open(config.EXP5_SAMPLES_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def _pr_key(rec: dict) -> str:
    return f"{rec['repo']}#{rec['number']}"


# --------------------------------------------------------------------------- #
# 单样本执行：分类 / 生成，P1–P5 单轮 + P6 多轮
# --------------------------------------------------------------------------- #
def _run_single_turn(task: str, level: str, prm: str, rec: dict) -> dict:
    repo, number = rec["repo"], rec["number"]
    exclude_id = rec.get("target_comment_id")  # 生成真值，L3 排除防泄漏
    context_text = context_builder.build_context(repo, number, level,
                                                  exclude_comment_id=exclude_id)
    messages = prompts.build_prompt(task, prm, context_text)
    result = exp4_llm.chat_semantic(
        task, level, prm, _pr_key(rec), messages,
        config.TEMPERATURE[task], config.MAX_TOKENS[task],
        json_mode=config.JSON_MODE[task], model=config.EXEC_MODEL)
    return _finalize(task, rec, result, context_text)


def _run_multi_turn(task: str, level: str, rec: dict) -> dict:
    """P6：多轮。首轮 + 若干 followup，每轮独立缓存 key；用最后一轮解析结论。"""
    repo, number = rec["repo"], rec["number"]
    exclude_id = rec.get("target_comment_id")
    context_text = context_builder.build_context(repo, number, level,
                                                  exclude_comment_id=exclude_id)
    messages = prompts.build_p6_first(task, context_text)
    total_latency = 0.0
    all_cached = True
    last_result = None
    turns = prompts.p6_followups(task)
    # 首轮（自由文本，非 JSON）
    r0 = exp4_llm.chat_semantic(
        task, level, "P6#0", _pr_key(rec), messages,
        config.TEMPERATURE[task], config.MAX_TOKENS[task],
        json_mode=False, model=config.EXEC_MODEL)
    total_latency += r0["latency"]
    all_cached = all_cached and r0["cached"]
    messages = messages + [{"role": "assistant", "content": r0["text"]}]

    for i, (tname, followup) in enumerate(turns, 1):
        messages = messages + [{"role": "user", "content": followup}]
        is_last = (i == len(turns))
        r = exp4_llm.chat_semantic(
            task, level, f"P6#{i}", _pr_key(rec), messages,
            config.TEMPERATURE[task], config.MAX_TOKENS[task],
            json_mode=is_last and config.JSON_MODE[task], model=config.EXEC_MODEL)
        total_latency += r["latency"]
        all_cached = all_cached and r["cached"]
        messages = messages + [{"role": "assistant", "content": r["text"]}]
        last_result = r

    result = dict(last_result)
    result["latency"] = total_latency
    result["cached"] = all_cached
    return _finalize(task, rec, result, context_text)


def _finalize(task: str, rec: dict, result: dict, context_text: str) -> dict:
    repo, number = rec["repo"], rec["number"]
    text = result["text"]
    row = {
        "task": task, "repo": repo, "number": number,
        "latency": result["latency"], "cached": result["cached"],
        "raw_text": text,
        "finish_reason": result.get("finish_reason"),
        "prompt_tokens": (result.get("usage") or {}).get("prompt_tokens"),
        "completion_tokens": (result.get("usage") or {}).get("completion_tokens"),
        "context_len": len(context_text),
    }
    if task == "classify":
        pred = exp4_run.parse_decision(text)
        row["pred_merge"] = pred
        row["parse_error"] = pred is None
        row["true_merge"] = bool(rec["is_merged"])
    else:
        row["pred_comment"] = exp4_run._extract_comment(text)
        row["target_comment"] = rec.get("target_comment")
        row["target_path"] = rec.get("target_path")
        row["_context_text"] = context_text  # 供 judge 取 diff，落盘前删除
    return row


# --------------------------------------------------------------------------- #
# 条件运行
# --------------------------------------------------------------------------- #
def run_condition(task: str, cond: dict, samples: list[dict], run_judge: bool) -> list[dict]:
    level, prm, group = cond["level"], cond["prompt"], cond["group"]
    rows = []
    desc = f"{task} {level}/{prm}[{group}]"
    for rec in tqdm(samples, desc=desc, ncols=90):
        if prm == "P6":
            row = _run_multi_turn(task, level, rec)
        else:
            row = _run_single_turn(task, level, prm, rec)
        row.update({"level": level, "prompt": prm, "group": group, "cell": cond["cell"]})

        if task == "generate" and run_judge:
            jrow = judge.score(
                row.get("pred_comment", ""), row.get("target_comment", "") or "",
                row.pop("_context_text", ""), pr_key=_pr_key(rec),
                cache_ctx=f"{level}__{prm}")
            row["judge_relevance"] = jrow["relevance"]
            row["judge_actionable"] = jrow["actionable"]
            row["judge_correct"] = jrow["correct"]
            row["judge_hallucination"] = jrow["hallucination"]
            row["judge_rationale"] = jrow["rationale"]
            row["judge_latency"] = jrow["latency"]
            row["judge_cached"] = jrow["cached"]
            row["judge_parse_error"] = jrow["parse_error"]
        else:
            row.pop("_context_text", None)
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# 全矩阵
# --------------------------------------------------------------------------- #
def run_task(task: str, only: tuple[str, str] | None, limit: int | None,
             run_judge: bool) -> pd.DataFrame:
    conds = config.matrix(task)
    if only:
        conds = [c for c in conds if c["level"] == only[0] and c["prompt"] == only[1]]
        if not conds:
            print(f"[{task}] --only {only} 不在矩阵内，跳过。")
            return pd.DataFrame()

    # 按 group 缓存样本
    sample_cache: dict[str, list[dict]] = {}

    def get_samples(group: str) -> list[dict]:
        if group not in sample_cache:
            s = load_samples(task, group)
            sample_cache[group] = s[:limit] if limit is not None else s
        return sample_cache[group]

    n_calls = sum(len(get_samples(c["group"])) for c in conds)
    print(f"\n===== 任务 {task}: {len(conds)} 条件 ≈ {n_calls} 次执行调用 "
          f"（judge={'on' if (run_judge and task == 'generate') else 'off'}）=====")

    all_rows = []
    for cond in conds:
        samples = get_samples(cond["group"])
        rows = run_condition(task, cond, samples, run_judge)
        all_rows.extend(rows)
        if task == "classify":
            errs = sum(r["parse_error"] for r in rows)
            cached = sum(r["cached"] for r in rows)
            print(f"  {cond['level']}/{cond['prompt']}[{cond['group']}]: "
                  f"parse_error={errs}/{len(rows)} cached={cached}/{len(rows)}")
        else:
            cached = sum(r["cached"] for r in rows)
            print(f"  {cond['level']}/{cond['prompt']}[{cond['group']}]: "
                  f"cached={cached}/{len(rows)}")

    df = pd.DataFrame(all_rows)
    out = config.PREDICTIONS_DIR / f"{task}_predictions.parquet"
    key_cols = ["group", "level", "prompt", "repo", "number"]
    if out.exists() and (only or limit):
        prev = pd.read_parquet(out)
        prev_idx = prev.set_index(key_cols).index
        new_idx = df.set_index(key_cols).index
        keep = prev[~prev_idx.isin(new_idx)]
        df = pd.concat([keep, df], ignore_index=True)
    df.to_parquet(out, index=False)
    print(f"\n  → 落盘 {out.relative_to(config.PROJECT_ROOT)}（{len(df)} 行）")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="实验六主循环（上下文阶梯 + Prompt 消融）")
    ap.add_argument("--task", choices=["classify", "generate", "all"], default="all")
    ap.add_argument("--only", nargs=2, metavar=("LEVEL", "PROMPT"),
                    help="只跑单条件，如 --only L0 P1")
    ap.add_argument("--limit", type=int, default=None, help="每条件只取前 N 样本（冒烟）")
    ap.add_argument("--no-judge", action="store_true", help="生成任务跳过 LLM-judge 打分")
    args = ap.parse_args()

    only = tuple(args.only) if args.only else None
    tasks = config.TASKS if args.task == "all" else [args.task]
    for task in tasks:
        run_task(task, only, args.limit, run_judge=not args.no_judge)


if __name__ == "__main__":
    main()
