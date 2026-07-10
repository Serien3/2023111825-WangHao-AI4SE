"""实验二 saved 模型对 AI + 匹配对照做 Merge Prediction 预测并算指标。

复用纪律（设计 §0/§3）：模型权重、scaler 一律只 transform 不 fit。
对每个特征集（pre / full）× 每个模型（svm/rf/xgboost/lightgbm）：
  1. 载实验五特征矩阵（ml_features 产出）。
  2. 按该特征集的 scaler.feature_cols 对齐列序 → scaler.transform。
  3. 载模型 predict / predict_proba。
  4. 算 Accuracy/Precision/Recall/F1（正类=merged）。

产物：
  results/predictions/ml_{group}_{feature_set}.parquet   （逐 PR 预测）
  results/metrics/ml_metrics.json                        （group×fs×model 指标）
"""
from __future__ import annotations

import argparse
import json
import pickle
import warnings

import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)

from . import config, data, ml_features

warnings.filterwarnings("ignore", category=UserWarning)  # sklearn feature-name 警告无害


# --------------------------------------------------------------------------- #
# 载入 scaler / 模型
# --------------------------------------------------------------------------- #
def _load_scaler(feature_set: str):
    with open(config.scaler_path(feature_set), "rb") as f:
        obj = pickle.load(f)
    return obj["scaler"], obj["feature_cols"]


def _load_model(name: str, feature_set: str):
    with open(config.model_path(name, feature_set), "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------------- #
# 特征矩阵获取（优先读缓存，缺失则现算）
# --------------------------------------------------------------------------- #
def _group_pr_df(group: str) -> pd.DataFrame:
    if group == "ai":
        return data.ai_classify_pool()
    if group == "control":
        ai = data.ai_classify_pool()
        return data.matched_human_control(ai, config.N_CLASSIFY_SAMPLE, config.SEED)
    raise ValueError(group)


def _features_for(group: str, limit: int | None) -> pd.DataFrame:
    path = config.FEATURES_DIR / f"{group}_features.parquet"
    pr_df = _group_pr_df(group)
    if limit:
        pr_df = pr_df.head(limit)
    if path.exists():
        cached = pd.read_parquet(path)
        keys = set(zip(pr_df["repo"], pr_df["number"]))
        cached = cached[cached.apply(lambda r: (r["repo"], r["number"]) in keys, axis=1)]
        if len(cached) == len(pr_df):
            return cached.reset_index(drop=True)
    # 缓存缺失/不全 → 现算并落盘
    return ml_features.build_and_save(group, pr_df)


# --------------------------------------------------------------------------- #
# 预测 + 指标
# --------------------------------------------------------------------------- #
def _metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "n": int(len(y_true)),
        "positive_rate": float(pd.Series(y_true).mean()),
    }


def predict_group_featureset(group: str, feature_set: str,
                             feat_df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    scaler, feature_cols = _load_scaler(feature_set)
    X = scaler.transform(ml_features.align_to_feature_cols(feat_df, feature_cols))
    y_true = feat_df["is_merged"].astype(int).values

    metrics, pred_rows = {}, None
    base = feat_df[["repo", "number", "is_merged"]].reset_index(drop=True).copy()
    preds_wide = base.copy()
    for name in config.ML_MODELS:
        model = _load_model(name, feature_set)
        y_pred = model.predict(X)
        metrics[name] = _metrics(y_true, y_pred)
        preds_wide[f"pred_{name}"] = y_pred.astype(int)
        # 概率（若可用）供分析用
        if hasattr(model, "predict_proba"):
            try:
                preds_wide[f"proba_{name}"] = model.predict_proba(X)[:, 1]
            except Exception:
                pass
    pred_rows = preds_wide
    return metrics, pred_rows


def run(limit: int | None = None) -> dict:
    all_metrics: dict = {g: {fs: {} for fs in config.FEATURE_SETS} for g in config.GROUPS}
    for group in config.GROUPS:
        feat_df = _features_for(group, limit)
        print(f"\n[{group}] 特征矩阵 {feat_df.shape}，标签正类率 "
              f"{feat_df['is_merged'].mean():.3f}")
        for fs in config.FEATURE_SETS:
            metrics, preds = predict_group_featureset(group, fs, feat_df)
            all_metrics[group][fs] = metrics
            out = config.PREDICTIONS_DIR / f"ml_{group}_{fs}.parquet"
            preds.to_parquet(out, index=False)
            print(f"  [{fs}] 预测 → {out.relative_to(config.PROJECT_ROOT)}")
            for name, m in metrics.items():
                print(f"    {name:<10} acc={m['accuracy']:.3f} f1={m['f1']:.3f} "
                      f"prec={m['precision']:.3f} rec={m['recall']:.3f}")

    out = config.METRICS_DIR / "ml_metrics.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    print(f"\n[ml] 指标 → {out.relative_to(config.PROJECT_ROOT)}")
    return all_metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="实验五 ML 预测（AI + 匹配对照）")
    ap.add_argument("--limit", type=int, default=None, help="每组只取前 N 个 PR（冒烟）")
    args = ap.parse_args()
    run(limit=args.limit)


if __name__ == "__main__":
    main()
