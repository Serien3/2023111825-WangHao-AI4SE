"""Publication-style visualizations for Experiment 2 evaluation results.

The figures are designed around the report questions in the experiment guide:
model performance, deployable-vs-upper-bound leakage, feature importance,
cross-repository shift, confusion patterns, and training cost.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import auc, confusion_matrix, roc_curve

from . import config


MODEL_ORDER = ["svm", "rf", "xgboost", "lightgbm"]
MODEL_LABELS = {
    "svm": "SVM",
    "rf": "RF",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}

MODEL_COLORS = {
    "svm": "#4B5563",
    "rf": "#0F4D92",
    "xgboost": "#42949E",
    "lightgbm": "#9A4D8E",
}

PALETTE = {
    "blue": "#0F4D92",
    "blue_mid": "#3775BA",
    "teal": "#42949E",
    "violet": "#9A4D8E",
    "red": "#B64342",
    "gold": "#C9A227",
    "green": "#2E9E44",
    "neutral_light": "#D8D8D8",
    "neutral_mid": "#8F8F8F",
    "neutral_dark": "#4D4D4D",
    "neutral_black": "#272727",
}

FEATURE_FAMILY_COLORS = {
    "AST": "#3775BA",
    "CFG": "#42949E",
    "Change statistics": "#C9A227",
    "Text": "#9A4D8E",
    "Review process": "#B64342",
    "Other": "#8F8F8F",
}

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "auc": "ROC-AUC",
}


def apply_publication_style() -> None:
    """Apply compact Nature-style matplotlib settings."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "axes.unicode_minus": False,
    })
    sns.set_theme(style="white", rc=mpl.rcParams)


def save_figure(fig: mpl.figure.Figure, stem: str, legacy_png: str | None = None) -> None:
    """Save editable vector outputs plus high-resolution raster copies."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    base = config.FIGURES_DIR / stem
    for suffix in ("svg", "pdf", "png", "tiff"):
        fig.savefig(base.with_suffix(f".{suffix}"), bbox_inches="tight")
    if legacy_png:
        fig.savefig(config.FIGURES_DIR / legacy_png, bbox_inches="tight")
    plt.close(fig)


def add_panel_label(ax: mpl.axes.Axes, label: str, x: float = -0.12, y: float = 1.05) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color=PALETTE["neutral_black"],
    )


def ordered_models(models: Mapping[str, object] | Sequence[str]) -> list[str]:
    names = list(models.keys()) if isinstance(models, Mapping) else list(models)
    return [m for m in MODEL_ORDER if m in names] + [m for m in names if m not in MODEL_ORDER]


def feature_family(feature_name: str) -> str:
    if feature_name in config.REVIEW_PROCESS_FEATURES:
        return "Review process"
    if feature_name.startswith("ast_"):
        return "AST"
    if feature_name.startswith("cfg_"):
        return "CFG"
    if feature_name.startswith("tfidf_") or feature_name in {
        "title_len",
        "body_len",
        "avg_commit_msg_len",
    } or feature_name.startswith("has_keyword_"):
        return "Text"
    if feature_name in {
        "additions",
        "deletions",
        "changed_files",
        "num_commits",
        "files_per_commit",
        "churn_ratio",
    }:
        return "Change statistics"
    return "Other"


def pretty_feature_name(feature_name: str) -> str:
    if feature_name.startswith("tfidf_"):
        return "tf-idf: " + feature_name.removeprefix("tfidf_")
    return feature_name.replace("has_keyword_", "keyword: ").replace("_", " ")


def performance_dataframe(metrics: Mapping[str, Mapping[str, float]]) -> pd.DataFrame:
    models = ordered_models(metrics)
    rows = [metric for metric in METRIC_LABELS if any(metric in metrics[m] for m in models)]
    return pd.DataFrame(
        [[metrics[m].get(metric, np.nan) for m in models] for metric in rows],
        index=[METRIC_LABELS[m] for m in rows],
        columns=[MODEL_LABELS.get(m, m.upper()) for m in models],
    )


def plot_feature_importance(
    model,
    feature_names: Sequence[str],
    feature_set: str,
    top_k: int = 18,
) -> None:
    """Random Forest feature importance as a top-feature lollipop plus family summary."""
    if not hasattr(model, "feature_importances_"):
        return

    apply_publication_style()
    imp = np.asarray(model.feature_importances_, dtype=float)
    if len(imp) != len(feature_names):
        raise ValueError("feature_names length does not match feature_importances_")

    df = pd.DataFrame({"feature": list(feature_names), "importance": imp})
    df["family"] = df["feature"].map(feature_family)
    top = df.nlargest(top_k, "importance").sort_values("importance")
    family = (
        df.groupby("family", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=True)
    )

    fig = plt.figure(figsize=(7.2, 3.8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.65, 1.0], wspace=0.42)
    ax = fig.add_subplot(gs[0, 0])
    ax_fam = fig.add_subplot(gs[0, 1])

    y = np.arange(len(top))
    colors = [FEATURE_FAMILY_COLORS[f] for f in top["family"]]
    ax.hlines(y, 0, top["importance"], color=colors, lw=2.0, alpha=0.75)
    ax.scatter(top["importance"], y, s=28, color=colors, edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([pretty_feature_name(f) for f in top["feature"]])
    ax.set_xlabel("Gini importance")
    ax.set_title(f"Top predictive signals ({feature_set})", loc="left", pad=6)
    ax.grid(axis="x", color="#E5E7EB", lw=0.7)
    ax.set_axisbelow(True)
    ax.margins(x=0.08)
    add_panel_label(ax, "a")

    fam_colors = [FEATURE_FAMILY_COLORS[f] for f in family["family"]]
    ax_fam.barh(np.arange(len(family)), family["importance"], color=fam_colors, edgecolor="white", lw=0.5)
    ax_fam.set_yticks(np.arange(len(family)))
    ax_fam.set_yticklabels([f.replace("Change statistics", "Change\nstatistics") for f in family["family"]])
    ax_fam.set_xlabel("Total importance")
    ax_fam.set_title("Feature families", loc="left", pad=6)
    ax_fam.grid(axis="x", color="#E5E7EB", lw=0.7)
    ax_fam.set_axisbelow(True)
    add_panel_label(ax_fam, "b", x=-0.22)

    legacy = "feature_importance_rf.png" if feature_set == "full" else None
    save_figure(fig, f"feature_importance_rf_{feature_set}", legacy_png=legacy)
    print(f"Saved: feature_importance_rf_{feature_set}.svg/.pdf/.png/.tiff")


def plot_roc_curves(models: Mapping[str, object], X_test, y_test, feature_set: str) -> None:
    """ROC curves with direct AUC labels."""
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(4.2, 3.5))

    for name in ordered_models(models):
        model = models[name]
        if not hasattr(model, "predict_proba"):
            continue
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        color = MODEL_COLORS.get(name, PALETTE["neutral_mid"])
        ax.plot(fpr, tpr, lw=1.8, color=color, label=f"{MODEL_LABELS.get(name, name)} ({roc_auc:.3f})")
        ax.text(
            fpr[min(len(fpr) - 1, max(1, int(len(fpr) * 0.62)))],
            tpr[min(len(tpr) - 1, max(1, int(len(tpr) * 0.62)))],
            f"{MODEL_LABELS.get(name, name)}",
            color=color,
            fontsize=6,
        )

    ax.plot([0, 1], [0, 1], ls="--", lw=0.9, color=PALETTE["neutral_mid"], label="Random")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"Discrimination on the {feature_set} test set", loc="left", pad=6)
    ax.legend(loc="lower right", title="ROC-AUC", title_fontsize=6.5)
    ax.grid(color="#E5E7EB", lw=0.7)
    ax.set_axisbelow(True)

    legacy = "roc_curves.png" if feature_set == "full" else None
    save_figure(fig, f"roc_curves_{feature_set}", legacy_png=legacy)
    print(f"Saved: roc_curves_{feature_set}.svg/.pdf/.png/.tiff")


def plot_confusion_matrices(models: Mapping[str, object], X_test, y_test, feature_set: str) -> None:
    """Compact 2x2 confusion matrices with counts and row-normalized percentages."""
    apply_publication_style()
    names = ordered_models(models)
    n_cols = 2
    n_rows = math.ceil(len(names) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.2, 2.65 * n_rows), squeeze=False)
    cmap = LinearSegmentedColormap.from_list("exp2_blues", ["#F3F6FA", "#0F4D92"])

    for ax, name in zip(axes.ravel(), names):
        y_pred = models[name].predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        row_sums = cm.sum(axis=1, keepdims=True)
        rates = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
        labels = np.array([
            [f"{cm[i, j]}\n{rates[i, j]:.0%}" for j in range(cm.shape[1])]
            for i in range(cm.shape[0])
        ])
        sns.heatmap(
            rates,
            annot=labels,
            fmt="",
            cmap=cmap,
            cbar=False,
            vmin=0,
            vmax=1,
            square=True,
            linewidths=0.8,
            linecolor="white",
            xticklabels=["Non-merge", "Merge"],
            yticklabels=["Non-merge", "Merge"],
            ax=ax,
        )
        ax.set_title(MODEL_LABELS.get(name, name), loc="left", pad=5)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    for ax in axes.ravel()[len(names):]:
        ax.set_axis_off()

    fig.suptitle(f"Error structure ({feature_set} features)", x=0.02, y=1.0, ha="left", fontsize=9)
    fig.tight_layout()
    legacy = "confusion_matrices.png" if feature_set == "full" else None
    save_figure(fig, f"confusion_matrices_{feature_set}", legacy_png=legacy)
    print(f"Saved: confusion_matrices_{feature_set}.svg/.pdf/.png/.tiff")


def plot_cross_repo_performance(
    per_repo_res: Mapping[str, Mapping[str, float]],
    feature_set: str,
    merge_rates: Mapping[str, float] | None = None,
) -> None:
    """Cross-repository F1 heatmap paired with each repository's base merge rate."""
    apply_publication_style()
    df = pd.DataFrame(per_repo_res).T
    df = df[[m for m in MODEL_ORDER if m in df.columns]]
    if merge_rates:
        order = sorted(df.index, key=lambda r: merge_rates.get(r, np.nan), reverse=True)
        df = df.loc[order]
    else:
        df = df.sort_index()

    fig = plt.figure(figsize=(7.2, 3.55))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.05, 2.85], wspace=0.08)
    ax_rate = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[0, 1])

    repos = list(df.index)
    y = np.arange(len(repos))
    if merge_rates:
        rates = np.array([merge_rates.get(repo, np.nan) for repo in repos])
        ax_rate.hlines(y, 0, rates, color=PALETTE["neutral_light"], lw=5)
        ax_rate.scatter(rates, y, s=30, color=PALETTE["neutral_black"], zorder=3)
        for yi, rate in zip(y, rates):
            ax_rate.text(min(rate + 0.025, 1.0), yi, f"{rate:.0%}", va="center", fontsize=6)
    ax_rate.set_xlim(0, 1.05)
    ax_rate.set_ylim(-0.5, len(repos) - 0.5)
    ax_rate.invert_yaxis()
    ax_rate.set_yticks(y)
    ax_rate.set_yticklabels([repo.replace("/", "/\n") for repo in repos])
    ax_rate.set_xlabel("Merge rate")
    ax_rate.set_title("Dataset prior", loc="left", pad=6)
    ax_rate.grid(axis="x", color="#E5E7EB", lw=0.7)
    add_panel_label(ax_rate, "a", x=-0.30)

    cmap = LinearSegmentedColormap.from_list("repo_f1", ["#F7F7F7", "#B4C0E4", "#0F4D92"])
    sns.heatmap(
        df.rename(columns=MODEL_LABELS),
        ax=ax_heat,
        cmap=cmap,
        annot=True,
        fmt=".2f",
        vmin=0.45,
        vmax=1.0,
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": "F1"},
        yticklabels=False,
    )
    ax_heat.set_xlabel("")
    ax_heat.set_ylabel("")
    ax_heat.set_title(f"Repository-specific F1 ({feature_set})", loc="left", pad=6)
    ax_heat.tick_params(axis="x", rotation=25)
    add_panel_label(ax_heat, "b", x=-0.08)

    legacy = "cross_repo_performance.png" if feature_set == "full" else None
    save_figure(fig, f"cross_repo_performance_{feature_set}", legacy_png=legacy)
    print(f"Saved: cross_repo_performance_{feature_set}.svg/.pdf/.png/.tiff")


def plot_ablation_comparison(
    pre_metrics: Mapping[str, Mapping[str, float]],
    full_metrics: Mapping[str, Mapping[str, float]],
) -> None:
    """Slope chart quantifying temporal leakage from review-process features."""
    apply_publication_style()
    names = ordered_models(pre_metrics)
    fig, ax = plt.subplots(figsize=(5.6, 3.35))
    x = np.array([0, 1])

    for name in names:
        pre_f1 = pre_metrics[name]["f1"]
        full_f1 = full_metrics[name]["f1"]
        color = MODEL_COLORS.get(name, PALETTE["neutral_mid"])
        ax.plot(x, [pre_f1, full_f1], color=color, lw=2.0, marker="o", ms=4.5)

    def spread_positions(values: Mapping[str, float], min_gap: float = 0.012) -> dict[str, float]:
        ordered = sorted(values, key=values.get, reverse=True)
        positions: dict[str, float] = {}
        prev = None
        for name in ordered:
            y = values[name]
            if prev is not None and y > prev - min_gap:
                y = prev - min_gap
            positions[name] = y
            prev = y
        return positions

    full_pos = spread_positions({name: full_metrics[name]["f1"] for name in names})

    for name in names:
        color = MODEL_COLORS.get(name, PALETTE["neutral_mid"])
        pre_f1 = pre_metrics[name]["f1"]
        full_f1 = full_metrics[name]["f1"]
        ax.annotate(
            f"{MODEL_LABELS.get(name, name)} +{full_f1 - pre_f1:.3f}",
            xy=(1, full_f1),
            xytext=(1.06, full_pos[name]),
            ha="left",
            va="center",
            color=color,
            arrowprops={"arrowstyle": "-", "lw": 0.7, "color": color, "shrinkA": 0, "shrinkB": 3},
        )

    ax.set_xlim(-0.16, 1.82)
    ax.set_xticks(x)
    ax.set_xticklabels(["Pre-review\n(deployable)", "Full\n(upper bound)"])
    ax.set_ylabel("F1 score")
    ax.set_title("Review-process features lift F1 but leak future information", loc="left", pad=7)
    ax.grid(axis="y", color="#E5E7EB", lw=0.7)
    ax.set_axisbelow(True)
    label_values = list(full_pos.values())
    ymin = min(label_values + [pre_metrics[m]["f1"] for m in names]) - 0.02
    ymax = max(label_values + [full_metrics[m]["f1"] for m in names]) + 0.025
    ax.set_ylim(ymin, ymax)
    save_figure(fig, "label_leakage_ablation", legacy_png="label_leakage_ablation.png")
    print("Saved: label_leakage_ablation.svg/.pdf/.png/.tiff")


def plot_training_time(times: Mapping[str, float]) -> None:
    """Training cost as a log-scale dot plot."""
    apply_publication_style()
    names = ordered_models(times)
    vals = np.array([times[m] for m in names], dtype=float)

    fig, ax = plt.subplots(figsize=(4.4, 2.45))
    y = np.arange(len(names))
    colors = [MODEL_COLORS.get(m, PALETTE["neutral_mid"]) for m in names]
    ax.hlines(y, max(vals.min() * 0.5, 0.001), vals, color=PALETTE["neutral_light"], lw=3)
    ax.scatter(vals, y, s=42, color=colors, edgecolor="white", linewidth=0.6, zorder=3)
    for yi, val in zip(y, vals):
        ax.text(val * 1.08, yi, f"{val:.3f}s", va="center", fontsize=6.5)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_LABELS.get(m, m) for m in names])
    ax.invert_yaxis()
    ax.set_xlabel("Training time (seconds, log scale)")
    ax.set_title("Training cost remains lightweight for the ML baseline", loc="left", pad=6)
    ax.grid(axis="x", color="#E5E7EB", lw=0.7, which="both")
    ax.set_axisbelow(True)
    save_figure(fig, "training_time", legacy_png="training_time.png")
    print("Saved: training_time.svg/.pdf/.png/.tiff")


def plot_model_performance_matrix(
    pre_metrics: Mapping[str, Mapping[str, float]],
    full_metrics: Mapping[str, Mapping[str, float]],
) -> None:
    """Two-panel metric heatmap for report-level model comparison."""
    apply_publication_style()
    pre_df = performance_dataframe(pre_metrics)
    full_df = performance_dataframe(full_metrics)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.65), sharey=True)
    cmap = LinearSegmentedColormap.from_list("perf", ["#F7F7F7", "#B4C0E4", "#0F4D92"])
    for ax, df, title, label in [
        (axes[0], pre_df, "Pre-review: deployable baseline", "a"),
        (axes[1], full_df, "Full: post-review upper bound", "b"),
    ]:
        sns.heatmap(
            df,
            ax=ax,
            cmap=cmap,
            annot=True,
            fmt=".3f",
            vmin=0.70,
            vmax=0.93,
            linewidths=0.8,
            linecolor="white",
            cbar=ax is axes[1],
            cbar_kws={"label": "Score"} if ax is axes[1] else None,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(title, loc="left", pad=6)
        ax.tick_params(axis="x", rotation=25)
        add_panel_label(ax, label, x=-0.15)

    fig.tight_layout()
    save_figure(fig, "model_performance_matrix")
    print("Saved: model_performance_matrix.svg/.pdf/.png/.tiff")
