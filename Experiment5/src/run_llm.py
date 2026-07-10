"""LLM 编排：复用实验四 pipeline，跑 16 条件 × {AI, 对照} × {classify, generate}。

零改动复用实验四：
- context_builder.build_context / prompts.build_prompt：上下文与 Prompt 逐字一致。
- llm_client.chat_semantic：缓存目录 = 实验四 cache（共享），语义 key =
  (task, ctx, prompt, pr_key, content_hash)。生成对照组复用实验四样本可直接命中；
  分类对照组改为无泄露 test 匹配样本，未命中时会正常新增调用。
- run_experiments.run_condition：单条件跑一条样本 + 解析（分类决策 / 生成 comment）。

产物：results/predictions/{task}_predictions.parquet（每行一次调用，含 group 列）。
缓存幂等，重跑跳过已完成调用。

用法：
  uv run python -m src.run_llm --task classify --limit 3          # 冒烟
  uv run python -m src.run_llm --task classify --only C1 P1       # 单条件
  uv run python -m src.run_llm --task all                         # 两任务全量
  uv run python -m src.run_llm --task all --group ai              # 只跑 AI 侧
"""
from __future__ import annotations

import argparse
import importlib
import json

import pandas as pd
from tqdm import tqdm

from . import config

exp4_run = importlib.import_module("exp4src.run_experiments")


# --------------------------------------------------------------------------- #
# 样本加载（实验五 samples 目录）
# --------------------------------------------------------------------------- #
def load_samples(task: str, group: str) -> list[dict]:
    name = f"{task}_{group}.json"
    with open(config.SAMPLES_DIR / name, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# 单任务：遍历 group × 16 条件
# --------------------------------------------------------------------------- #
def run_task(task: str, only: tuple[str, str] | None, limit: int | None,
             groups: list[str]) -> pd.DataFrame:
    contexts = [only[0]] if only else config.CONTEXTS
    prompts_list = [only[1]] if only else config.PROMPTS
    conditions = [(c, p) for c in contexts for p in prompts_list]

    all_rows = []
    for group in groups:
        samples = load_samples(task, group)
        if limit is not None:
            samples = samples[:limit]
        print(f"\n===== 任务 {task} · 组 {group}: {len(conditions)} 条件 × "
              f"{len(samples)} 样本 = {len(conditions) * len(samples)} 次调用 =====")
        for ctx, prm in conditions:
            desc = f"{task}/{group} {ctx}/{prm}"
            rows = []
            for rec in tqdm(samples, desc=desc, ncols=88):
                # 复用实验四单条件单样本逻辑（含缓存 + 解析 + 解析失败重试）
                out = exp4_run.run_condition(task, ctx, prm, [rec])
                for r in out:
                    r["group"] = group
                rows.extend(out)
            all_rows.extend(rows)
            if task == "classify":
                errs = sum(r["parse_error"] for r in rows)
                cached = sum(r["cached"] for r in rows)
                print(f"  {desc}: parse_error={errs}/{len(rows)} cached={cached}/{len(rows)}")
            else:
                cached = sum(r["cached"] for r in rows)
                print(f"  {desc}: cached={cached}/{len(rows)}")

    df = pd.DataFrame(all_rows)
    out = config.PREDICTIONS_DIR / f"{task}_predictions.parquet"
    # 增量合并：保留其它 (group, context, prompt, pr) 已有结果
    key_cols = ["group", "context", "prompt", "repo", "number"]
    if out.exists() and (only or limit or set(groups) != set(config.GROUPS)):
        prev = pd.read_parquet(out)
        prev_idx = prev.set_index(key_cols).index
        new_idx = df.set_index(key_cols).index
        keep = prev[~prev_idx.isin(new_idx)]
        df = pd.concat([keep, df], ignore_index=True)
    df.to_parquet(out, index=False)
    print(f"\n  → 落盘 {out.relative_to(config.PROJECT_ROOT)}（{len(df)} 行）")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="实验五 LLM 编排（AI + 匹配对照）")
    ap.add_argument("--task", choices=["classify", "generate", "all"], default="all")
    ap.add_argument("--only", nargs=2, metavar=("CONTEXT", "PROMPT"),
                    help="只跑单条件，如 --only C1 P1")
    ap.add_argument("--limit", type=int, default=None, help="每条件只取前 N 样本（冒烟）")
    ap.add_argument("--group", choices=config.GROUPS, default=None,
                    help="只跑某一组（ai / control）；默认两组都跑")
    args = ap.parse_args()

    only = tuple(args.only) if args.only else None
    groups = [args.group] if args.group else config.GROUPS
    tasks = config.TASKS if args.task == "all" else [args.task]
    for task in tasks:
        run_task(task, only, args.limit, groups)


if __name__ == "__main__":
    main()
