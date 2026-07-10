"""抽样：LLM 分类/生成的 AI 侧样本 + 匹配人类对照，固定种子落盘。

设计 §2：
- 分类 AI 侧：从 AI 分类池（253）按 repo × is_merged 分层抽 50，与实验四人类侧对齐。
- 生成 AI 侧：用全部 72 条 AI 生成池（含首条顶层 comment 作 target），不抽样。
- 匹配人类对照：
  - classify_control：按 AI 分类抽样的 repo × is_merged 分布，只从实验二 held-out test 集抽同量级，避免训练/调参泄露。
  - generate_control：直接复用实验四 generate_50.json（人类生成样本），作为生成任务对照锚点。
    （实验四人类生成样本已抽自人类 test，含 target_comment，可直接跑 16 条件并大量命中缓存。）

产物：results/samples/{classify_ai,generate_ai,classify_control,generate_control}.json
"""
from __future__ import annotations

import json

import pandas as pd

from . import config, data


def _dump(obj, name: str) -> None:
    path = config.SAMPLES_DIR / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"  → 落盘 {path.relative_to(config.PROJECT_ROOT)}（{len(obj)} 条）")


# --------------------------------------------------------------------------- #
# 分类
# --------------------------------------------------------------------------- #
def sample_classify_ai() -> list[dict]:
    ai = data.ai_classify_pool()
    sampled = data.stratified_sample(
        ai, ["repo", "is_merged"], config.N_CLASSIFY_SAMPLE, config.SEED)
    records = [
        {"repo": r.repo, "number": int(r.number), "is_merged": bool(r.is_merged)}
        for r in sampled.itertuples()
    ]
    print(f"[classify-ai] 抽样 {len(records)} 条，分层分布：")
    print(sampled.groupby(["repo", "is_merged"]).size().to_string())
    return records


def sample_classify_control(ai_records: list[dict]) -> list[dict]:
    """按 AI 分类抽样（ai_records）的 repo × is_merged 分布，从人类池抽匹配对照。"""
    ai_df = pd.DataFrame(ai_records)
    ctrl = data.matched_human_control(ai_df, len(ai_records), config.SEED)
    records = [
        {"repo": r.repo, "number": int(r.number), "is_merged": bool(r.is_merged)}
        for r in ctrl.itertuples()
    ]
    print(f"[classify-control] 匹配对照 {len(records)} 条，分层分布：")
    print(ctrl.groupby(["repo", "is_merged"]).size().to_string())
    return records


# --------------------------------------------------------------------------- #
# 生成
# --------------------------------------------------------------------------- #
def sample_generate_ai() -> list[dict]:
    pool = data.ai_generate_pool()
    records = []
    for r in pool.itertuples():
        records.append({
            "repo": r.repo,
            "number": int(r.number),
            "is_merged": bool(r.is_merged),
            "target_comment": r.target_comment,
            "target_path": r.target_path,
            "target_comment_id": int(r.target_comment_id),
        })
    print(f"[generate-ai] 全量 {len(records)} 条，按仓库：")
    print(pool.groupby("repo").size().to_string())
    return records


def sample_generate_control() -> list[dict]:
    """复用实验四人类生成样本作为生成任务对照锚点（含 target_comment）。"""
    src = config.EXP4_SAMPLES_DIR / "generate_50.json"
    with open(src, encoding="utf-8") as f:
        records = json.load(f)
    print(f"[generate-control] 复用实验四人类生成样本 {len(records)} 条")
    return records


def main() -> None:
    print("=" * 60)
    clf_ai = sample_classify_ai()
    _dump(clf_ai, "classify_ai.json")
    print("=" * 60)
    clf_ctrl = sample_classify_control(clf_ai)
    _dump(clf_ctrl, "classify_control.json")
    print("=" * 60)
    gen_ai = sample_generate_ai()
    _dump(gen_ai, "generate_ai.json")
    print("=" * 60)
    gen_ctrl = sample_generate_control()
    _dump(gen_ctrl, "generate_control.json")
    print("=" * 60)
    print("抽样完成。")


if __name__ == "__main__":
    main()
