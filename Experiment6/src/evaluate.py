"""评估（R15/R16/R18）：修复两个失明指标 + 与实验五基线对比。

分类（R15，测量仪器非重采样）：
  per-class Precision/Recall（merge 与 non-merge 各一份）+ 混淆矩阵 + balanced accuracy（头条）。
  accuracy/merge-F1 仍列出但不作主结论。B(prompt 框定)生效判据 = non-merge recall 上升。
生成（R16）：LLM-judge 4 维均值（relevance/actionable/correct/hallucination）作主指标；
  保留 BLEU/ROUGE 仅为与实验五表格可比，并在报告点明失明。
对比（R18）：与实验五 AI 基线（llm_*_metrics.json 的 "ai" 组）算改进 Δ。

产物：
  results/metrics/classify_metrics.json   （逐条件 per-class + 混淆矩阵 + balanced acc）
  results/metrics/generate_metrics.json   （逐条件 judge 4 维 + BLEU/ROUGE）
  results/metrics/exp5_baseline_delta.json（与实验五 AI 基线对比 Δ）
"""
from __future__ import annotations

import argparse
import importlib
import json

import pandas as pd

from . import config

exp4_eval = importlib.import_module("exp4src.evaluate")


def _cond_key(level: str, prm: str, group: str) -> str:
    return f"{level}_{prm}_{group}"


# --------------------------------------------------------------------------- #
# 分类：per-class P/R + 混淆矩阵 + balanced accuracy
# --------------------------------------------------------------------------- #
def _classify_metrics(sub: pd.DataFrame) -> dict:
    from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                                 precision_recall_fscore_support)

    n_total = len(sub)
    valid = sub[~sub["parse_error"]]
    n_valid = len(valid)
    m = {
        "n_total": n_total,
        "n_valid": n_valid,
        "parse_error_rate": 1.0 - n_valid / n_total if n_total else 0.0,
        "avg_latency": float(sub["latency"].mean()) if n_total else 0.0,
    }
    if n_valid == 0:
        m.update({"balanced_accuracy": None, "accuracy": None, "merge_f1": None,
                  "per_class": None, "confusion_matrix": None})
        return m

    y_true = valid["true_merge"].astype(int).to_numpy()  # 1=merge, 0=non-merge
    y_pred = valid["pred_merge"].astype(int).to_numpy()

    # 混淆矩阵：labels=[0(non-merge),1(merge)] → [[tn,fp],[fn,tp]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    # per-class（0=non-merge, 1=merge）
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0)
    per_class = {
        "non_merge": {"precision": float(prec[0]), "recall": float(rec[0]),
                      "f1": float(f1[0]), "support": int(support[0])},
        "merge": {"precision": float(prec[1]), "recall": float(rec[1]),
                  "f1": float(f1[1]), "support": int(support[1])},
    }

    m.update({
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),  # 头条
        "accuracy": float((tp + tn) / n_valid),
        "merge_f1": float(f1[1]),                # 旧头条，降级为参考
        "non_merge_recall": float(rec[0]),       # B 生效判据
        "per_class": per_class,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp,
                             "labels": ["non_merge", "merge"]},
    })
    return m


def evaluate_classify() -> dict:
    path = config.PREDICTIONS_DIR / "classify_predictions.parquet"
    if not path.exists():
        print(f"[classify] 无预测文件 {path}，跳过。")
        return {}
    df = pd.read_parquet(path)
    results = {}
    for cond in config.matrix("classify"):
        lv, pm, gp = cond["level"], cond["prompt"], cond["group"]
        sub = df[(df["level"] == lv) & (df["prompt"] == pm) & (df["group"] == gp)]
        if sub.empty:
            continue
        results[_cond_key(lv, pm, gp)] = {**_classify_metrics(sub), "cell": cond["cell"]}
    out = config.METRICS_DIR / "classify_metrics.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[classify] {len(results)} 条件 → {out.relative_to(config.PROJECT_ROOT)}")
    _print_classify(results)
    return results


def _print_classify(results: dict) -> None:
    print(f"  {'条件':<16}{'BalAcc':>8}{'Acc':>7}{'nmR':>7}{'mR':>7}"
          f"{'mF1':>7}{'ParseErr':>9}")
    for k, m in results.items():
        if m.get("balanced_accuracy") is None:
            print(f"  {k:<16}   -")
            continue
        pc = m["per_class"]
        print(f"  {k:<16}{m['balanced_accuracy']:>8.3f}{m['accuracy']:>7.3f}"
              f"{pc['non_merge']['recall']:>7.3f}{pc['merge']['recall']:>7.3f}"
              f"{m['merge_f1']:>7.3f}{m['parse_error_rate']:>9.2f}")


# --------------------------------------------------------------------------- #
# 生成：judge 4 维 + 保留 BLEU/ROUGE
# --------------------------------------------------------------------------- #
def _generate_metrics(sub: pd.DataFrame) -> dict:
    # 保留旧口径 BLEU/ROUGE（复用实验四）
    legacy = exp4_eval._generation_metrics(sub)

    out = dict(legacy)
    # judge 4 维聚合（若有 judge 列）
    dims = ["relevance", "actionable", "correct", "hallucination"]
    judge_cols = [f"judge_{d}" for d in dims]
    if all(c in sub.columns for c in judge_cols):
        valid = sub
        if "judge_parse_error" in sub.columns:
            valid = sub[~sub["judge_parse_error"].fillna(True).astype(bool)]
        n_j = len(valid)
        out["judge_n"] = n_j
        for d in dims:
            col = valid[f"judge_{d}"].dropna()
            out[f"judge_{d}"] = float(col.mean()) if len(col) else None
        if "judge_latency" in sub.columns:
            out["judge_avg_latency"] = float(sub["judge_latency"].dropna().mean()) \
                if sub["judge_latency"].notna().any() else None
    else:
        out["judge_n"] = 0
        for d in dims:
            out[f"judge_{d}"] = None
    return out


def evaluate_generate() -> dict:
    path = config.PREDICTIONS_DIR / "generate_predictions.parquet"
    if not path.exists():
        print(f"[generate] 无预测文件 {path}，跳过。")
        return {}
    df = pd.read_parquet(path)
    results = {}
    for cond in config.matrix("generate"):
        lv, pm, gp = cond["level"], cond["prompt"], cond["group"]
        sub = df[(df["level"] == lv) & (df["prompt"] == pm) & (df["group"] == gp)]
        if sub.empty:
            continue
        results[_cond_key(lv, pm, gp)] = {**_generate_metrics(sub), "cell": cond["cell"]}
    out = config.METRICS_DIR / "generate_metrics.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[generate] {len(results)} 条件 → {out.relative_to(config.PROJECT_ROOT)}")
    _print_generate(results)
    return results


def _print_generate(results: dict) -> None:
    print(f"  {'条件':<16}{'Rel':>6}{'Act':>6}{'Cor':>6}{'Hal':>6}"
          f"{'BLEU':>7}{'RougeL':>8}")
    for k, m in results.items():
        def _f(x, w=6, p=2): return (f"{x:>{w}.{p}f}" if isinstance(x, (int, float))
                                     else f"{'-':>{w}}")
        print(f"  {k:<16}{_f(m.get('judge_relevance'))}{_f(m.get('judge_actionable'))}"
              f"{_f(m.get('judge_correct'))}{_f(m.get('judge_hallucination'))}"
              f"{_f(m.get('bleu'), 7)}{_f(m.get('rougeL'), 8, 3)}")


# --------------------------------------------------------------------------- #
# 与实验五 AI 基线对比 Δ（R18）
# --------------------------------------------------------------------------- #
def _load_json(path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_baseline_delta(clf: dict, gen: dict) -> dict:
    """与实验五 AI 基线对比。实验五键为 Cx_Px（4 上下文×4 prompt），
    实验六 L0=C1、L1=C4；取对应基线格算 Δ。生成基线只有 BLEU/ROUGE（judge 是本实验新增）。"""
    exp5_clf = _load_json(config.EXP5_METRICS_DIR / "llm_classify_metrics.json").get("ai", {})
    exp5_gen = _load_json(config.EXP5_METRICS_DIR / "llm_generate_metrics.json").get("ai", {})

    # 上下文映射到实验五基线格：L0→C1，L1→C4（其余无直接对应，记 None）
    level_to_exp5 = {"L0": "C1", "L1": "C4"}

    out = {"classify": {}, "generate": {},
           "note": ("实验五无 per-class/balanced_acc（仅 pos-class），也无 judge；"
                    "分类 Δ 用 accuracy/merge-F1（可比口径），生成 Δ 用 BLEU/ROUGE。"
                    "judge 4 维与 balanced_acc/non_merge_recall 是本实验新增测量仪器，无基线。")}

    for k, m in clf.items():
        lv, pm, gp = k.split("_")
        if gp != "ai":
            continue
        c5 = level_to_exp5.get(lv)
        base = exp5_clf.get(f"{c5}_{pm}") if c5 else None
        out["classify"][k] = {
            "exp6": {"accuracy": m.get("accuracy"), "merge_f1": m.get("merge_f1"),
                     "balanced_accuracy": m.get("balanced_accuracy"),
                     "non_merge_recall": m.get("non_merge_recall")},
            "exp5_baseline_cell": f"{c5}_{pm}" if c5 else None,
            "delta": ({"accuracy": (m.get("accuracy") - base["accuracy"])
                       if base and base.get("accuracy") is not None
                       and m.get("accuracy") is not None else None,
                       "merge_f1": (m.get("merge_f1") - base["f1"])
                       if base and base.get("f1") is not None
                       and m.get("merge_f1") is not None else None}
                      if base else None),
        }

    for k, m in gen.items():
        lv, pm, gp = k.split("_")
        if gp != "ai":
            continue
        c5 = level_to_exp5.get(lv)
        base = exp5_gen.get(f"{c5}_{pm}") if c5 else None
        out["generate"][k] = {
            "exp6": {"bleu": m.get("bleu"), "rougeL": m.get("rougeL"),
                     "judge_relevance": m.get("judge_relevance"),
                     "judge_hallucination": m.get("judge_hallucination")},
            "exp5_baseline_cell": f"{c5}_{pm}" if c5 else None,
            "delta": ({"bleu": (m.get("bleu") - base["bleu"])
                       if base and base.get("bleu") is not None else None,
                       "rougeL": (m.get("rougeL") - base["rougeL"])
                       if base and base.get("rougeL") is not None else None}
                      if base else None),
        }

    path = config.METRICS_DIR / "exp5_baseline_delta.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[baseline-delta] → {path.relative_to(config.PROJECT_ROOT)}")
    return out


def run() -> None:
    clf = evaluate_classify()
    gen = evaluate_generate()
    if clf or gen:
        build_baseline_delta(clf, gen)


def main() -> None:
    ap = argparse.ArgumentParser(description="实验六评估（修复失明指标 + 基线对比）")
    ap.add_argument("--task", choices=["classify", "generate", "all"], default="all")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    tasks = config.TASKS if args.task == "all" else [args.task]
    clf = evaluate_classify() if "classify" in tasks else {}
    gen = evaluate_generate() if "generate" in tasks else {}
    if clf or gen:
        build_baseline_delta(clf, gen)
    if not args.no_figures:
        from . import nature_viz
        try:
            nature_viz.main()
        except Exception as e:
            print(f"[visualization] 跳过（{type(e).__name__}: {e}）")


if __name__ == "__main__":
    main()
