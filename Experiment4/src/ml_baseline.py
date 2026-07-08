"""切片 8a：实验二传统 ML 模型在同一 classify_50 子集上重新预测，得严格同集基线。

设计文档 §6.3：LLM 跑 50 子集、ML 原本在 208 全集，报告需注明；此处用实验二
pre-review 模型（SVM/RF/XGB/LGBM）在这 50 条上重 predict，得到可严格对比的 ML 基线。

产物：results/metrics/ml_baseline_50.json
"""
from __future__ import annotations

import json
import pickle
import warnings

import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)

from . import config, data

warnings.filterwarnings("ignore", category=UserWarning)  # feature-name 警告无害


def _load_scaler():
    with open(config.EXP2_SCALER_PKL, "rb") as f:
        obj = pickle.load(f)
    return obj["scaler"], obj["feature_cols"]


def _classify_50_features() -> tuple[pd.DataFrame, pd.Series]:
    with open(config.SAMPLES_DIR / "classify_50.json", encoding="utf-8") as f:
        samples = json.load(f)
    feat = data.load_features()
    keys = {(r["repo"], r["number"]): bool(r["is_merged"]) for r in samples}
    mask = feat.apply(lambda r: (r["repo"], r["number"]) in keys, axis=1)
    sub = feat[mask].copy()
    y = sub.apply(lambda r: keys[(r["repo"], r["number"])], axis=1).astype(int)
    return sub, y


def compute_ml_baseline() -> dict:
    scaler, feature_cols = _load_scaler()
    sub, y_true = _classify_50_features()
    X = scaler.transform(sub[feature_cols])

    results = {}
    for name, path in config.EXP2_MODEL_FILES.items():
        with open(path, "rb") as f:
            model = pickle.load(f)
        y_pred = model.predict(X)
        results[name] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        }
    out = config.METRICS_DIR / "ml_baseline_50.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[ml-baseline] 同 50 子集重评 4 模型 → {out.relative_to(config.PROJECT_ROOT)}")
    for name, m in results.items():
        print(f"  {name:<10} acc={m['accuracy']:.3f} f1={m['f1']:.3f}")
    return results


if __name__ == "__main__":
    compute_ml_baseline()
