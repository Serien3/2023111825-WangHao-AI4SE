"""实验二 步骤五：模型评估 + 可视化 + 消融分析。

输出：
- 测试集指标（Acc/P/R/F1/AUC）JSON
- 跨仓库性能分解 JSON
- 标签泄漏消融分析 JSON
- 6 张图表（特征重要性、ROC、混淆矩阵、跨仓库对比、泄漏消融、训练时间）

用法：
    uv run python -m src.evaluate --feature-set full
    uv run python -m src.evaluate --feature-set pre
    uv run python -m src.evaluate --ablation  # 对比 pre vs full
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

from . import config

sns.set_theme(style="whitegrid", palette="muted")
# 与实验一一致：英文标签 + sans-serif，避免 matplotlib 中文字体缺失导致乱码
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "axes.unicode_minus": False,
})


# --------------------------------------------------------------------------- #
# 指标计算
# --------------------------------------------------------------------------- #
def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    """计算 Acc/P/R/F1/AUC。"""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        metrics["auc"] = float(auc(fpr, tpr))
    return metrics


def evaluate_model(model, X_test, y_test) -> dict:
    """在测试集上评估单个模型。"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    return compute_metrics(y_test, y_pred, y_proba)


# --------------------------------------------------------------------------- #
# 跨仓库分解
# --------------------------------------------------------------------------- #
def per_repo_analysis(models: dict, X_test, y_test, repos_test) -> dict:
    """按仓库分别计算 F1，返回 {repo: {model: f1}}。"""
    results = {}
    for repo in repos_test.unique():
        mask = repos_test == repo
        X_r = X_test[mask]
        y_r = y_test[mask]
        repo_res = {}
        for name, model in models.items():
            y_pred = model.predict(X_r)
            repo_res[name] = float(f1_score(y_r, y_pred, zero_division=0))
        results[repo] = repo_res
    return results


# --------------------------------------------------------------------------- #
# 可视化
# --------------------------------------------------------------------------- #
def plot_feature_importance(model, feature_names: list[str], top_k: int = 15):
    """RF/XGBoost/LightGBM 特征重要性（横向柱状）。"""
    if not hasattr(model, "feature_importances_"):
        return
    imp = model.feature_importances_
    indices = np.argsort(imp)[-top_k:]
    plt.figure(figsize=(8, 6))
    plt.barh(range(top_k), imp[indices], color="steelblue")
    plt.yticks(range(top_k), [feature_names[i] for i in indices])
    plt.xlabel("Feature Importance")
    plt.title(f"Top {top_k} Features (Random Forest)")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "feature_importance_rf.png", dpi=150)
    plt.close()
    print(f"已保存: feature_importance_rf.png")


def plot_roc_curves(models: dict, X_test, y_test):
    """四个模型的 ROC 曲线。"""
    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        if not hasattr(model, "predict_proba"):
            continue
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name.upper()} (AUC={roc_auc:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (Test Set)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "roc_curves.png", dpi=150)
    plt.close()
    print("已保存: roc_curves.png")


def plot_confusion_matrices(models: dict, X_test, y_test):
    """2×2 子图：四个模型的混淆矩阵。"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    for i, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Non-merge", "Merge"], yticklabels=["Non-merge", "Merge"],
            ax=axes[i]
        )
        axes[i].set_title(name.upper())
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "confusion_matrices.png", dpi=150)
    plt.close()
    print("已保存: confusion_matrices.png")


def plot_cross_repo_performance(per_repo_res: dict):
    """跨仓库 F1 对比（分组柱状）。"""
    df = pd.DataFrame(per_repo_res).T  # rows=repos, cols=models
    df.plot(kind="bar", figsize=(10, 5), rot=15)
    plt.ylabel("F1 Score")
    plt.title("Cross-Repo Performance (F1)")
    plt.legend(title="Model", loc="lower right")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "cross_repo_performance.png", dpi=150)
    plt.close()
    print("已保存: cross_repo_performance.png")


def plot_ablation_comparison(pre_metrics: dict, full_metrics: dict):
    """标签泄漏消融对比（pre-review vs full features）。"""
    models = list(pre_metrics.keys())
    pre_f1 = [pre_metrics[m]["f1"] for m in models]
    full_f1 = [full_metrics[m]["f1"] for m in models]

    x = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, pre_f1, width, label="Pre-review (no leakage)", color="steelblue")
    ax.bar(x + width / 2, full_f1, width, label="Full (with review features)", color="coral")
    ax.set_ylabel("F1 Score")
    ax.set_title("Label-Leakage Ablation: Pre-review vs Full Features")
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models])
    ax.legend()
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "label_leakage_ablation.png", dpi=150)
    plt.close()
    print("已保存: label_leakage_ablation.png")


def plot_training_time(times: dict):
    """四个模型的训练时间对比。"""
    models = list(times.keys())
    vals = [times[m] for m in models]
    plt.figure(figsize=(8, 4))
    plt.bar(models, vals, color="teal")
    plt.ylabel("Training Time (s)")
    plt.title("Model Training Time Comparison")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "training_time.png", dpi=150)
    plt.close()
    print("已保存: training_time.png")


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def evaluate_single_feature_set(feature_set: str) -> dict:
    """评估某个 feature_set（'pre' / 'full'）的全部模型，返回指标字典。"""
    print(f"\n{'='*60}")
    print(f"评估特征集: {feature_set}")
    print(f"{'='*60}")

    # 加载数据划分
    split_path = config.MODELS_DIR / f"split_{feature_set}.pkl"
    if not split_path.exists():
        raise FileNotFoundError(f"{split_path} 不存在。请先运行 train.py")
    with open(split_path, "rb") as f:
        split = pickle.load(f)
    X_test, y_test = split["X_test"], split["y_test"]
    print(f"测试集规模: {len(X_test)} PR")

    # 加载全部模型
    model_names = ["svm", "rf", "xgboost", "lightgbm"]
    models = {}
    for name in model_names:
        model_path = config.MODELS_DIR / f"{name}_{feature_set}.pkl"
        if not model_path.exists():
            print(f"[警告] {model_path} 不存在，跳过")
            continue
        with open(model_path, "rb") as f:
            models[name] = pickle.load(f)

    if not models:
        raise RuntimeError("没有找到任何训练好的模型")

    # 评估
    metrics = {}
    for name, model in models.items():
        print(f"\n[{name.upper()}] 评估中...")
        m = evaluate_model(model, X_test, y_test)
        metrics[name] = m
        print(f"  Acc={m['accuracy']:.4f}, P={m['precision']:.4f}, "
              f"R={m['recall']:.4f}, F1={m['f1']:.4f}, AUC={m.get('auc', 0):.4f}")

    # 保存指标
    metrics_path = config.METRICS_DIR / f"test_metrics_{feature_set}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n已保存测试集指标: {metrics_path}")

    # 跨仓库分解（需要原始 df 的 repo 列）
    feat_df = pd.read_parquet(config.FEATURES_DIR / "features.parquet")
    # 根据索引找对应的 repo
    repos_test = feat_df.loc[X_test.index, "repo"]
    per_repo_res = per_repo_analysis(models, X_test, y_test, repos_test)
    per_repo_path = config.METRICS_DIR / f"per_repo_metrics_{feature_set}.json"
    with open(per_repo_path, "w") as f:
        json.dump(per_repo_res, f, indent=2)
    print(f"已保存跨仓库指标: {per_repo_path}")

    # 可视化
    print("\n生成可视化...")
    if "rf" in models:
        scaler_path = config.MODELS_DIR / f"scaler_{feature_set}.pkl"
        with open(scaler_path, "rb") as f:
            feature_cols = pickle.load(f)["feature_cols"]
        plot_feature_importance(models["rf"], feature_cols, top_k=15)

    plot_roc_curves(models, X_test, y_test)
    plot_confusion_matrices(models, X_test, y_test)
    plot_cross_repo_performance(per_repo_res)

    # 训练时间图（若 train.py 已记录 training_time.json）
    tt_path = config.METRICS_DIR / "training_time.json"
    if tt_path.exists():
        with open(tt_path) as f:
            plot_training_time(json.load(f))

    return metrics


def ablation_analysis():
    """标签泄漏消融：对比 pre-review vs full features。"""
    print("\n" + "="*60)
    print("标签泄漏消融分析 (Pre-review vs Full)")
    print("="*60)

    pre_path = config.METRICS_DIR / "test_metrics_pre.json"
    full_path = config.METRICS_DIR / "test_metrics_full.json"

    if not pre_path.exists() or not full_path.exists():
        print("[警告] 需要先分别训练 pre 和 full 特征集的模型")
        return

    with open(pre_path) as f:
        pre_metrics = json.load(f)
    with open(full_path) as f:
        full_metrics = json.load(f)

    print("\nF1 对比:")
    print(f"{'Model':<12} {'Pre-review':>12} {'Full':>12} {'Delta':>10}")
    print("-" * 50)
    for model in pre_metrics:
        pre_f1 = pre_metrics[model]["f1"]
        full_f1 = full_metrics[model]["f1"]
        delta = full_f1 - pre_f1
        print(f"{model.upper():<12} {pre_f1:>12.4f} {full_f1:>12.4f} {delta:>+10.4f}")

    plot_ablation_comparison(pre_metrics, full_metrics)

    ablation_path = config.METRICS_DIR / "ablation_results.json"
    with open(ablation_path, "w") as f:
        json.dump({"pre_review": pre_metrics, "full": full_metrics}, f, indent=2)
    print(f"\n已保存消融结果: {ablation_path}")


def main():
    ap = argparse.ArgumentParser(description="实验二模型评估")
    ap.add_argument(
        "--feature-set", choices=["pre", "full"], default="full",
        help="评估哪个特征集的模型"
    )
    ap.add_argument(
        "--ablation", action="store_true",
        help="运行标签泄漏消融分析（需要先训练 pre 和 full 两组模型）"
    )
    args = ap.parse_args()

    if args.ablation:
        ablation_analysis()
    else:
        evaluate_single_feature_set(args.feature_set)

    print("\n评估完成！")


if __name__ == "__main__":
    main()
