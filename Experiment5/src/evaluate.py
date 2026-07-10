"""评估：复用实验四指标逻辑，算 LLM (逐组×16 条件) + 汇总 ML，产出 human_vs_ai 对比。

复用实验四 evaluate 的 _classification_metrics / _generation_metrics（BLEU/ROUGE），
对每个 group（ai / control）× 每个条件算指标。人类锚点直接读实验四旧结果：
- LLM 分类人类锚点：Experiment4/results/metrics/classify_metrics.json
- LLM 生成人类锚点：Experiment4/results/metrics/generate_metrics.json
- ML 人类锚点：实验二 test 旧指标（test_metrics_pre/full.json）+ 本实验 control 重预测

产物：
  results/metrics/llm_classify_metrics.json   （group×条件）
  results/metrics/llm_generate_metrics.json   （group×条件）
  results/metrics/human_vs_ai.json            （三方对比：human / ai / control）
"""
from __future__ import annotations

import argparse
import importlib
import json

import pandas as pd

from . import config

exp4_eval = importlib.import_module("exp4src.evaluate")


# --------------------------------------------------------------------------- #
# LLM 指标（逐 group × 条件）
# --------------------------------------------------------------------------- #
def _eval_llm_task(task: str) -> dict:
    path = config.PREDICTIONS_DIR / f"{task}_predictions.parquet"
    if not path.exists():
        print(f"[{task}] 无预测文件，跳过。")
        return {}
    df = pd.read_parquet(path)
    metric_fn = (exp4_eval._classification_metrics if task == "classify"
                 else exp4_eval._generation_metrics)
    results: dict = {}
    for group in config.GROUPS:
        gdf = df[df["group"] == group] if "group" in df.columns else df
        if gdf.empty:
            continue
        results[group] = {}
        for ctx in config.CONTEXTS:
            for prm in config.PROMPTS:
                sub = gdf[(gdf["context"] == ctx) & (gdf["prompt"] == prm)]
                if sub.empty:
                    continue
                results[group][f"{ctx}_{prm}"] = metric_fn(sub)
    out = config.METRICS_DIR / f"llm_{task}_metrics.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[{task}] LLM 指标（{len(results)} 组）→ {out.relative_to(config.PROJECT_ROOT)}")
    return results


# --------------------------------------------------------------------------- #
# 人类锚点（实验四旧结果）
# --------------------------------------------------------------------------- #
def _load_json(path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _human_llm_classify() -> dict:
    return _load_json(config.EXP4_METRICS_DIR / "classify_metrics.json")


def _human_llm_generate() -> dict:
    return _load_json(config.EXP4_METRICS_DIR / "generate_metrics.json")


def _human_ml_test() -> dict:
    """实验二 held-out test 旧指标（真正的人类锚点，无 train 泄漏）。"""
    return {
        "pre": _load_json(config.EXP2_METRICS_DIR / "test_metrics_pre.json"),
        "full": _load_json(config.EXP2_METRICS_DIR / "test_metrics_full.json"),
    }


# --------------------------------------------------------------------------- #
# human_vs_ai 汇总
# --------------------------------------------------------------------------- #
def _condition_gap(human: dict, ai: dict, keys: list[str]) -> dict:
    """对每个条件算 ai - human 的逐指标差值（缺失锚点则记 None）。"""
    gaps = {}
    for cond, ai_m in ai.items():
        h = human.get(cond, {})
        gaps[cond] = {k: (ai_m.get(k) - h[k]) if (h.get(k) is not None
                      and ai_m.get(k) is not None) else None for k in keys}
    return gaps


def build_human_vs_ai(llm_clf: dict, llm_gen: dict, ml_metrics: dict) -> dict:
    hv_clf = _human_llm_classify()
    hv_gen = _human_llm_generate()
    ml_test = _human_ml_test()

    out: dict = {"ml": {}, "llm_classify": {}, "llm_generate": {}}

    # ---- ML：human test（实验二）vs AI vs 匹配 control ----
    for fs in config.FEATURE_SETS:
        out["ml"][fs] = {}
        for model in config.ML_MODELS:
            human = ml_test.get(fs, {}).get(model, {})
            ai = ml_metrics.get("ai", {}).get(fs, {}).get(model, {})
            ctrl = ml_metrics.get("control", {}).get(fs, {}).get(model, {})
            out["ml"][fs][model] = {
                "human_test": {k: human.get(k) for k in ("accuracy", "f1", "precision", "recall")},
                "ai": {k: ai.get(k) for k in ("accuracy", "f1", "precision", "recall")},
                "control": {k: ctrl.get(k) for k in ("accuracy", "f1", "precision", "recall")},
                "gap_ai_vs_human": {
                    k: (ai.get(k) - human[k]) if (human.get(k) is not None
                        and ai.get(k) is not None) else None
                    for k in ("accuracy", "f1")
                },
                "gap_ai_vs_control": {
                    k: (ai.get(k) - ctrl[k]) if (ctrl.get(k) is not None
                        and ai.get(k) is not None) else None
                    for k in ("accuracy", "f1")
                },
            }

    # ---- LLM 分类：human(实验四) vs ai vs control，逐条件 ----
    clf_keys = ["accuracy", "precision", "recall", "f1", "parse_error_rate"]
    out["llm_classify"] = {
        "human": hv_clf,
        "ai": llm_clf.get("ai", {}),
        "control": llm_clf.get("control", {}),
        "gap_ai_vs_human": _condition_gap(hv_clf, llm_clf.get("ai", {}), clf_keys),
        "gap_ai_vs_control": _condition_gap(llm_clf.get("control", {}),
                                            llm_clf.get("ai", {}), clf_keys),
    }

    # ---- LLM 生成：human(实验四) vs ai vs control，逐条件 ----
    gen_keys = ["bleu", "rouge1", "rouge2", "rougeL"]
    out["llm_generate"] = {
        "human": hv_gen,
        "ai": llm_gen.get("ai", {}),
        "control": llm_gen.get("control", {}),
        "gap_ai_vs_human": _condition_gap(hv_gen, llm_gen.get("ai", {}), gen_keys),
        "gap_ai_vs_control": _condition_gap(llm_gen.get("control", {}),
                                            llm_gen.get("ai", {}), gen_keys),
    }

    path = config.METRICS_DIR / "human_vs_ai.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[human_vs_ai] → {path.relative_to(config.PROJECT_ROOT)}")
    return out


# --------------------------------------------------------------------------- #
# 简报打印
# --------------------------------------------------------------------------- #
def _print_summary(hv: dict) -> None:
    print("\n===== 概要：ML pre-review（human test → AI）=====")
    for model in config.ML_MODELS:
        m = hv["ml"].get("pre", {}).get(model, {})
        h = m.get("human_test", {}).get("f1")
        a = m.get("ai", {}).get("f1")
        c = m.get("control", {}).get("f1")
        def _f(x): return f"{x:.3f}" if isinstance(x, (int, float)) else " - "
        print(f"  {model:<10} F1  human_test={_f(h)}  AI={_f(a)}  control={_f(c)}")


def run() -> dict:
    llm_clf = _eval_llm_task("classify")
    llm_gen = _eval_llm_task("generate")
    ml_metrics = _load_json(config.METRICS_DIR / "ml_metrics.json")
    hv = build_human_vs_ai(llm_clf, llm_gen, ml_metrics)
    _print_summary(hv)
    return hv


def main() -> None:
    ap = argparse.ArgumentParser(description="实验五评估 + human_vs_ai 对比")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    run()
    if not args.no_figures:
        from . import visualization
        try:
            visualization.main()
        except Exception as e:
            print(f"[visualization] 跳过（{type(e).__name__}: {e}）")


if __name__ == "__main__":
    main()
