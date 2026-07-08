"""切片 5-7：评估。分类指标 + 生成 BLEU/ROUGE，逐条件落盘。

分类（对 50 条真值 is_merged）：Accuracy/Precision/Recall/F1（正类=merged）、
  parse_error_rate（弃权比例）、平均 latency。弃权样本不计入 P/R/F1 混淆矩阵。
生成（对 50 条人类审查意见）：BLEU（sacrebleu 语料级 BLEU-4）、
  ROUGE-1/2/L F-measure（rouge-score）、平均 latency。

用法：
  uv run python -m src.evaluate                 # 两任务都评（有 predictions 才评）
  uv run python -m src.evaluate --task classify
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from . import config


# --------------------------------------------------------------------------- #
# 分类指标
# --------------------------------------------------------------------------- #
def _classification_metrics(sub: pd.DataFrame) -> dict:
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score)

    n_total = len(sub)
    valid = sub[~sub["parse_error"]]
    n_valid = len(valid)
    parse_error_rate = 1.0 - n_valid / n_total if n_total else 0.0

    metrics = {
        "n_total": n_total,
        "n_valid": n_valid,
        "parse_error_rate": parse_error_rate,
        "avg_latency": float(sub["latency"].mean()) if n_total else 0.0,
    }
    if n_valid == 0:
        metrics.update({"accuracy": None, "precision": None,
                        "recall": None, "f1": None})
        return metrics

    y_true = valid["true_merge"].astype(int)
    y_pred = valid["pred_merge"].astype(int)
    metrics.update({
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    })
    return metrics


def evaluate_classify() -> dict:
    path = config.PREDICTIONS_DIR / "classify_predictions.parquet"
    if not path.exists():
        print(f"[classify] 无预测文件 {path}，跳过。")
        return {}
    df = pd.read_parquet(path)
    results = {}
    for ctx in config.CONTEXTS:
        for prm in config.PROMPTS:
            sub = df[(df["context"] == ctx) & (df["prompt"] == prm)]
            if sub.empty:
                continue
            results[f"{ctx}_{prm}"] = _classification_metrics(sub)
    out = config.METRICS_DIR / "classify_metrics.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[classify] {len(results)} 条件指标 → {out.relative_to(config.PROJECT_ROOT)}")
    _print_classify_table(results)
    return results


def _print_classify_table(results: dict) -> None:
    print(f"  {'条件':<10}{'Acc':>7}{'Prec':>7}{'Rec':>7}{'F1':>7}"
          f"{'ParseErr':>10}{'Latency':>9}")
    for k, m in results.items():
        acc = f"{m['accuracy']:.3f}" if m["accuracy"] is not None else "  -  "
        pr = f"{m['precision']:.3f}" if m["precision"] is not None else "  -  "
        rc = f"{m['recall']:.3f}" if m["recall"] is not None else "  -  "
        f1 = f"{m['f1']:.3f}" if m["f1"] is not None else "  -  "
        print(f"  {k:<10}{acc:>7}{pr:>7}{rc:>7}{f1:>7}"
              f"{m['parse_error_rate']:>10.2f}{m['avg_latency']:>9.2f}")


# --------------------------------------------------------------------------- #
# 生成指标
# --------------------------------------------------------------------------- #
def _generation_metrics(sub: pd.DataFrame) -> dict:
    import sacrebleu
    from rouge_score import rouge_scorer

    preds = sub["pred_comment"].fillna("").tolist()
    refs = sub["target_comment"].fillna("").tolist()

    # 语料级 BLEU-4（sacrebleu 要求 refs 为 [[ref1, ref2, ...]]）
    bleu = sacrebleu.corpus_bleu(preds, [refs])

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1 = r2 = rl = 0.0
    for p, r in zip(preds, refs):
        s = scorer.score(r, p)  # (target, prediction)
        r1 += s["rouge1"].fmeasure
        r2 += s["rouge2"].fmeasure
        rl += s["rougeL"].fmeasure
    n = len(preds)
    return {
        "n_total": n,
        "bleu": float(bleu.score),
        "rouge1": r1 / n if n else 0.0,
        "rouge2": r2 / n if n else 0.0,
        "rougeL": rl / n if n else 0.0,
        "avg_latency": float(sub["latency"].mean()) if n else 0.0,
        "avg_pred_len": float(sub["pred_comment"].fillna("").str.len().mean()) if n else 0.0,
    }


def evaluate_generate() -> dict:
    path = config.PREDICTIONS_DIR / "generate_predictions.parquet"
    if not path.exists():
        print(f"[generate] 无预测文件 {path}，跳过。")
        return {}
    df = pd.read_parquet(path)
    results = {}
    for ctx in config.CONTEXTS:
        for prm in config.PROMPTS:
            sub = df[(df["context"] == ctx) & (df["prompt"] == prm)]
            if sub.empty:
                continue
            results[f"{ctx}_{prm}"] = _generation_metrics(sub)
    out = config.METRICS_DIR / "generate_metrics.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[generate] {len(results)} 条件指标 → {out.relative_to(config.PROJECT_ROOT)}")
    _print_generate_table(results)
    return results


def _print_generate_table(results: dict) -> None:
    print(f"  {'条件':<10}{'BLEU':>8}{'ROUGE-1':>9}{'ROUGE-2':>9}"
          f"{'ROUGE-L':>9}{'Latency':>9}")
    for k, m in results.items():
        print(f"  {k:<10}{m['bleu']:>8.2f}{m['rouge1']:>9.3f}{m['rouge2']:>9.3f}"
              f"{m['rougeL']:>9.3f}{m['avg_latency']:>9.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["classify", "generate", "all"], default="all")
    ap.add_argument("--no-figures", action="store_true", help="只算指标，不出图")
    args = ap.parse_args()
    tasks = config.TASKS if args.task == "all" else [args.task]
    if "classify" in tasks:
        evaluate_classify()
        from . import ml_baseline
        try:
            ml_baseline.compute_ml_baseline()
        except Exception as e:  # 实验二产物缺失不应阻断 LLM 指标
            print(f"[ml-baseline] 跳过（{type(e).__name__}: {e}）")
    if "generate" in tasks:
        evaluate_generate()
    if not args.no_figures:
        from . import visualization
        visualization.main()


if __name__ == "__main__":
    main()
