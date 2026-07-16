"""实验六出版级结果图：上下文、Prompt、错误转移、人类对照与成本权衡。

图件以 balanced accuracy / per-class recall 和 LLM-judge 为主指标，
BLEU/ROUGE 仅保留在结果表中用于前序实验可比。全部图由逐样本预测重算，
以固定随机种子的 bootstrap 给出 95% CI，并导出 SVG/PDF/PNG 与源数据 CSV。
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402

from . import config  # noqa: E402

C_BASE = "#0F4D92"
C_CONTEXT = "#42949E"
C_REFLECT = "#B64342"
C_MULTI = "#9A4D8E"
C_CONTROL = "#4E9A6B"
C_INK = "#272727"
C_MUTE = "#767676"
C_GRID = "#D8D8D8"
C_LIGHT = "#F3F4F5"
C_BAD = "#D24B40"
C_GOOD = "#2E7D4F"
COLORS = {"P1": C_BASE, "P3": C_BASE, "P5": C_REFLECT, "P6": C_MULTI}
RNG_SEED = 42
N_BOOT = 2000


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Noto Sans CJK SC", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "axes.linewidth": 0.8,
        "axes.edgecolor": C_INK,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.color": C_INK,
        "ytick.color": C_INK,
        "text.color": C_INK,
        "axes.labelcolor": C_INK,
        "legend.frameon": False,
        "legend.fontsize": 7,
        "figure.dpi": 120,
    })


def _panel(ax, label: str, x: float = -0.15, y: float = 1.04) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="bottom", ha="left")


def _save(fig, base: str, source: pd.DataFrame | None = None) -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (("svg", {}), ("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(config.FIGURES_DIR / f"{base}.{suffix}", bbox_inches="tight", **kwargs)
    if source is not None:
        source.to_csv(config.FIGURES_DIR / f"{base}_source.csv", index=False)
    plt.close(fig)
    print(f"  -> {base}.{{svg,pdf,png}}")


def _metric(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> float:
    if name == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, y_pred))
    target = 0 if name == "non_merge_recall" else 1
    mask = y_true == target
    return float(np.mean(y_pred[mask] == target)) if mask.any() else np.nan


def _bootstrap_classification(sub: pd.DataFrame, metric: str) -> tuple[float, float, float]:
    valid = sub[~sub["parse_error"]]
    y_true = valid["true_merge"].astype(int).to_numpy()
    y_pred = valid["pred_merge"].astype(int).to_numpy()
    estimate = _metric(y_true, y_pred, metric)
    rng = np.random.default_rng(RNG_SEED)
    values = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(valid), len(valid))
        if len(np.unique(y_true[idx])) < 2 and metric == "balanced_accuracy":
            continue
        values.append(_metric(y_true[idx], y_pred[idx], metric))
    low, high = np.quantile(values, [0.025, 0.975])
    return estimate, float(low), float(high)


def _bootstrap_mean(values: pd.Series) -> tuple[float, float, float]:
    data = values.dropna().to_numpy(dtype=float)
    estimate = float(np.mean(data))
    rng = np.random.default_rng(RNG_SEED)
    draws = np.mean(data[rng.integers(0, len(data), (N_BOOT, len(data)))], axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return estimate, float(low), float(high)


def _condition(df: pd.DataFrame, level: str, prompt: str, group: str = "ai") -> pd.DataFrame:
    return df[(df["level"] == level) & (df["prompt"] == prompt) & (df["group"] == group)]


def _point(ax, x, est, low, high, color, marker="o", label=None, zorder=3):
    ax.errorbar(x, est, yerr=[[est - low], [high - est]], color=color, marker=marker,
                ms=5, lw=1.4, capsize=2.5, label=label, zorder=zorder)


def fig1_overview(clf: pd.DataFrame, gen: pd.DataFrame) -> None:
    cp, gp = config.PSTAR["classify"], config.PSTAR["generate"]
    conditions = [(cp, "旧最优"), ("P5", "Self-Reflection"), ("P6", "多轮")]
    source = []
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05))
    ax = axes[0]
    x = np.arange(3)
    width = 0.34
    for offset, metric, hatch in [(-width / 2, "non_merge_recall", ""),
                                   (width / 2, "merge_recall", "///")]:
        for pos, (prompt, _) in enumerate(conditions):
            est, low, high = _bootstrap_classification(_condition(clf, "L4", prompt), metric)
            ax.bar(pos + offset, est, width, color=COLORS[prompt], edgecolor="white",
                   linewidth=0.6, hatch=hatch, zorder=2)
            ax.errorbar(pos + offset, est, yerr=[[est-low], [high-est]], fmt="none",
                        ecolor=C_INK, elinewidth=0.8, capsize=2, zorder=4)
            source.append({"panel": "classification", "condition": f"L4_{prompt}",
                           "metric": metric, "estimate": est, "ci_low": low, "ci_high": high})
    ax.set_xticks(x); ax.set_xticklabels([label for _, label in conditions])
    ax.set_ylim(0, 1.05); ax.set_ylabel("Recall")
    ax.grid(axis="y", color=C_GRID, lw=0.6, zorder=0)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor="#B8B8B8", label="Non-merge recall"),
                       plt.Rectangle((0, 0), 1, 1, facecolor="#B8B8B8", hatch="///", label="Merge recall")],
              loc="lower left")
    ax.set_title("Merge prediction: reflection restores rejection")
    _panel(ax, "a")

    ax = axes[1]
    for pos, (prompt, _) in enumerate([(gp, "旧最优"), ("P5", "Self-Reflection"), ("P6", "多轮")]):
        sub = _condition(gen, "L4", prompt)
        good = (sub["judge_relevance"] + sub["judge_correct"] +
                sub["judge_actionable"] + (6 - sub["judge_hallucination"])) / 4
        est, low, high = _bootstrap_mean(good)
        _point(ax, pos, est, low, high, COLORS[prompt])
        source.append({"panel": "generation", "condition": f"L4_{prompt}",
                       "metric": "composite_quality", "estimate": est,
                       "ci_low": low, "ci_high": high})
    ax.set_xticks(x); ax.set_xticklabels(["旧最优", "Self-Reflection", "多轮"])
    ax.set_ylim(1, 5); ax.set_ylabel("Composite judge score (1-5)")
    ax.grid(axis="y", color=C_GRID, lw=0.6)
    ax.text(0.02, 0.03, "mean of 4 judge dimensions; hallucination reversed\n95% bootstrap CI",
            transform=ax.transAxes, fontsize=6.3, color=C_MUTE, va="bottom")
    ax.set_title("Review comments: gains are modest and uneven")
    _panel(ax, "b")
    fig.suptitle("Input engineering changes the error profile more than overall performance",
                 fontsize=9.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "fig1_result_overview", pd.DataFrame(source))


def fig2_context_ladder(clf: pd.DataFrame, gen: pd.DataFrame) -> None:
    cp, gp = config.PSTAR["classify"], config.PSTAR["generate"]
    levels = config.LEVELS
    source = []
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.1), sharex="col")
    for ax, metric, title, color in [
        (axes[0, 0], "balanced_accuracy", "Balanced accuracy", C_BASE),
        (axes[1, 0], "non_merge_recall", "Non-merge recall", C_REFLECT),
    ]:
        vals = []
        for pos, level in enumerate(levels):
            est, low, high = _bootstrap_classification(_condition(clf, level, cp), metric)
            vals.append(est); _point(ax, pos, est, low, high, color)
            source.append({"panel": "classification", "level": level, "prompt": cp,
                           "metric": metric, "estimate": est, "ci_low": low, "ci_high": high})
        ax.plot(range(5), vals, color=color, lw=1.2, zorder=2)
        ax.axhline(0.5, color=C_MUTE, ls="--", lw=0.8)
        ax.set_ylim(0, 1); ax.set_ylabel(title); ax.grid(axis="y", color=C_GRID, lw=0.6)
    axes[0, 0].set_title(f"Classification x {cp}"); _panel(axes[0, 0], "a")
    for ax, metric, title, color in [
        (axes[0, 1], "judge_relevance", "Relevance", C_CONTEXT),
        (axes[1, 1], "judge_hallucination", "Hallucination (lower is better)", C_BAD),
    ]:
        vals = []
        for pos, level in enumerate(levels):
            valid = _condition(gen, level, gp).query("not judge_parse_error")
            est, low, high = _bootstrap_mean(valid[metric])
            vals.append(est); _point(ax, pos, est, low, high, color)
            source.append({"panel": "generation", "level": level, "prompt": gp,
                           "metric": metric, "estimate": est, "ci_low": low, "ci_high": high})
        ax.plot(range(5), vals, color=color, lw=1.2, zorder=2)
        ax.set_ylim(1, 2.25 if metric == "judge_relevance" else 2.1)
        ax.set_ylabel(title); ax.grid(axis="y", color=C_GRID, lw=0.6)
    axes[0, 1].set_title(f"Comment generation x {gp}"); _panel(axes[0, 1], "b")
    labels = ["L0\nDiff", "L1\n+ PR", "L2\n+ code", "L3\n+ issue/review", "L4\n+ repository"]
    for ax in axes[1]:
        ax.set_xticks(range(5)); ax.set_xticklabels(labels, fontsize=6.8)
    fig.suptitle("More context helps selectively; repository context is not a monotonic gain",
                 fontsize=9.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "fig2_context_ladder", pd.DataFrame(source))


def fig3_prompt_tradeoff(clf: pd.DataFrame, gen: pd.DataFrame) -> None:
    cp, gp = config.PSTAR["classify"], config.PSTAR["generate"]
    labels = ["旧最优", "Self-Reflection", "多轮"]
    source = []
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25))
    ax = axes[0]
    for prompt, label in zip([cp, "P5", "P6"], labels):
        sub = _condition(clf, "L4", prompt)
        nmr, nlo, nhi = _bootstrap_classification(sub, "non_merge_recall")
        mr, mlo, mhi = _bootstrap_classification(sub, "merge_recall")
        ax.errorbar(nmr, mr, xerr=[[nmr-nlo], [nhi-nmr]], yerr=[[mr-mlo], [mhi-mr]],
                    fmt="o", ms=6, color=COLORS[prompt], capsize=2, label=label, zorder=3)
        source.append({"panel": "classification", "prompt": prompt,
                       "non_merge_recall": nmr, "merge_recall": mr,
                       "nmr_ci_low": nlo, "nmr_ci_high": nhi,
                       "mr_ci_low": mlo, "mr_ci_high": mhi})
    ax.plot([0, 1], [1, 0], color=C_GRID, ls="--", lw=0.8)
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Non-merge recall"); ax.set_ylabel("Merge recall")
    ax.grid(color=C_GRID, lw=0.5); ax.legend(loc="lower left")
    ax.set_title("Classification operating point"); _panel(ax, "a")

    ax = axes[1]
    dimensions = ["Relevance", "Actionable", "Correct", "Non-hallucination"]
    x = np.arange(4)
    for prompt, label in zip([gp, "P5", "P6"], labels):
        valid = _condition(gen, "L4", prompt).query("not judge_parse_error")
        means, lows, highs = [], [], []
        columns = ["judge_relevance", "judge_actionable", "judge_correct", "judge_hallucination"]
        for column in columns:
            values = 6 - valid[column] if column == "judge_hallucination" else valid[column]
            est, low, high = _bootstrap_mean(values)
            means.append(est); lows.append(low); highs.append(high)
            source.append({"panel": "generation", "prompt": prompt, "metric": column,
                           "estimate": est, "ci_low": low, "ci_high": high,
                           "direction": "reversed" if column == "judge_hallucination" else "higher_better"})
        ax.errorbar(x, means, yerr=[np.array(means)-lows, np.array(highs)-means], marker="o",
                    ms=4.5, lw=1.3, capsize=2, color=COLORS[prompt], label=label)
    ax.set_xticks(x); ax.set_xticklabels(dimensions, rotation=18, ha="right")
    ax.set_ylim(1, 5); ax.set_ylabel("Judge score (higher is better)")
    ax.grid(axis="y", color=C_GRID, lw=0.6); ax.legend(loc="lower right")
    ax.set_title("Generation quality profile"); _panel(ax, "b")
    fig.suptitle("Prompt choice moves the operating point; no prompt dominates every objective",
                 fontsize=9.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "fig3_prompt_tradeoff", pd.DataFrame(source))


def fig4_error_transitions(clf: pd.DataFrame) -> None:
    cp = config.PSTAR["classify"]
    base = _condition(clf, "L4", cp).query("not parse_error")
    reflect = _condition(clf, "L4", "P5").query("not parse_error")
    paired = base.merge(reflect, on=["repo", "number", "true_merge"], suffixes=("_base", "_p5"))
    paired["base_correct"] = paired["pred_merge_base"].astype(bool) == paired["true_merge"]
    paired["p5_correct"] = paired["pred_merge_p5"].astype(bool) == paired["true_merge"]
    paired["transition"] = np.select(
        [paired["base_correct"] & paired["p5_correct"],
         ~paired["base_correct"] & paired["p5_correct"],
         paired["base_correct"] & ~paired["p5_correct"]],
        ["correct in both", "fixed by P5", "broken by P5"], default="wrong in both")
    order = ["correct in both", "fixed by P5", "broken by P5", "wrong in both"]
    colors = [C_GOOD, C_CONTEXT, C_BAD, C_MUTE]
    labels = ["Both correct", "Fixed by P5", "Broken by P5", "Both wrong"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.0, 1.15]})
    ax = axes[0]
    bottom = np.zeros(2); rows = []
    for transition, color, label in zip(order, colors, labels):
        values = []
        for true_value in [False, True]:
            count = int(((paired["true_merge"] == true_value) &
                         (paired["transition"] == transition)).sum())
            values.append(count)
            rows.append({"true_class": "merge" if true_value else "non_merge",
                         "transition": transition, "count": count})
        ax.bar([0, 1], values, bottom=bottom, color=color, width=0.62,
               edgecolor="white", linewidth=0.6, label=label)
        bottom += values
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Non-merge\n(n=13)", "Merge\n(n=37)"])
    ax.set_ylabel("PR count"); ax.legend(loc="upper left", labelspacing=0.25)
    ax.set_title("P5 fixes rejection errors but creates merge errors"); _panel(ax, "a")

    ax = axes[1]
    for idx, (prompt, label) in enumerate([(cp, "旧最优"), ("P5", "Self-Reflection")]):
        sub = _condition(clf, "L4", prompt).query("not parse_error")
        counts = pd.crosstab(sub["true_merge"], sub["pred_merge"]).reindex(
            index=[False, True], columns=[False, True], fill_value=0).to_numpy()
        x0 = idx * 2.7
        for row in range(2):
            for col in range(2):
                face = C_LIGHT if row == col else "#F3D4D1"
                ax.add_patch(plt.Rectangle((x0 + col, 1-row), 0.92, 0.92,
                                           facecolor=face, edgecolor="white"))
                ax.text(x0 + col + 0.46, 1-row + 0.46, str(counts[row, col]),
                        ha="center", va="center", fontsize=11, fontweight="bold")
        ax.text(x0 + 0.96, 2.18, label, ha="center", fontsize=8, fontweight="bold")
    ax.set_xlim(-0.05, 4.65); ax.set_ylim(-0.08, 2.45)
    ax.set_xticks([0.46, 1.46, 3.16, 4.16]); ax.set_xticklabels(["pred non", "pred merge"] * 2, fontsize=6.5)
    ax.set_yticks([0.46, 1.46]); ax.set_yticklabels(["true merge", "true non"])
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.tick_params(length=0); ax.set_title("Confusion structure at L4"); _panel(ax, "b", x=-0.12)
    fig.suptitle("Self-reflection corrects class bias, not every individual decision",
                 fontsize=9.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "fig4_error_transitions", pd.DataFrame(rows))


def fig5_human_control(clf: pd.DataFrame, gen: pd.DataFrame) -> None:
    cp, gp = config.PSTAR["classify"], config.PSTAR["generate"]
    source = []
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax = axes[0]; x = np.arange(2)
    for offset, group, color, label in [(-0.08, "ai", C_REFLECT, "AI code"),
                                         (0.08, "control", C_CONTROL, "Human control")]:
        vals = []
        for pos, prompt in enumerate([cp, "P5"]):
            est, low, high = _bootstrap_classification(_condition(clf, "L4", prompt, group), "balanced_accuracy")
            _point(ax, pos + offset, est, low, high, color, label=label if pos == 0 else None)
            vals.append(est)
            source.append({"panel": "classification", "group": group, "prompt": prompt,
                           "metric": "balanced_accuracy", "estimate": est,
                           "ci_low": low, "ci_high": high})
        ax.plot(x + offset, vals, color=color, lw=1.2)
    ax.axhline(0.5, color=C_MUTE, ls="--", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(["旧最优", "Self-Reflection"])
    ax.set_ylim(0.3, 0.75); ax.set_ylabel("Balanced accuracy")
    ax.grid(axis="y", color=C_GRID, lw=0.6); ax.legend(loc="upper right")
    ax.set_title("Merge prediction"); _panel(ax, "a")

    ax = axes[1]
    for offset, group, color, label in [(-0.08, "ai", C_REFLECT, "AI code"),
                                         (0.08, "control", C_CONTROL, "Human control")]:
        vals = []
        for pos, prompt in enumerate([gp, "P5"]):
            valid = _condition(gen, "L4", prompt, group).query("not judge_parse_error")
            est, low, high = _bootstrap_mean(valid["judge_relevance"])
            _point(ax, pos + offset, est, low, high, color, label=label if pos == 0 else None)
            vals.append(est)
            source.append({"panel": "generation", "group": group, "prompt": prompt,
                           "metric": "judge_relevance", "estimate": est,
                           "ci_low": low, "ci_high": high})
        ax.plot(x + offset, vals, color=color, lw=1.2)
    ax.set_xticks(x); ax.set_xticklabels(["旧最优", "Self-Reflection"])
    ax.set_ylim(1.1, 2.25); ax.set_ylabel("Judge relevance (1-5)")
    ax.grid(axis="y", color=C_GRID, lw=0.6); ax.legend(loc="upper left")
    ax.set_title("Review comment generation"); _panel(ax, "b")
    fig.suptitle("The improved input does not establish a universal AI-human gap",
                 fontsize=9.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "fig5_human_control", pd.DataFrame(source))


def fig6_quality_cost(clf: pd.DataFrame, gen: pd.DataFrame) -> None:
    cp, gp = config.PSTAR["classify"], config.PSTAR["generate"]
    source = []
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    specs = [(axes[0], clf, [cp, "P5", "P6"], "classification"),
             (axes[1], gen, [gp, "P5", "P6"], "generation")]
    for ax, df, prompts, task in specs:
        for prompt, label in zip(prompts, ["旧最优", "Self-Reflection", "多轮"]):
            sub = _condition(df, "L4", prompt)
            latency = float(sub["latency"].mean())
            if task == "classification":
                quality = _bootstrap_classification(sub, "balanced_accuracy")[0]
                ylabel = "Balanced accuracy"
            else:
                valid = sub[~sub["judge_parse_error"]]
                quality = float(((valid["judge_relevance"] + valid["judge_correct"] +
                                  valid["judge_actionable"] + (6-valid["judge_hallucination"])) / 4).mean())
                ylabel = "Composite judge score"
            ax.scatter(latency, quality, s=55, color=COLORS[prompt], zorder=3)
            ax.annotate(label, (latency, quality), xytext=(4, 4), textcoords="offset points", fontsize=7)
            source.append({"task": task, "prompt": prompt, "avg_latency_s": latency, "quality": quality})
        ax.set_xlabel("Mean execution latency (s, cached and uncached)")
        ax.set_ylabel(ylabel); ax.grid(color=C_GRID, lw=0.5)
    axes[0].set_title("Merge prediction"); axes[1].set_title("Review comment generation")
    _panel(axes[0], "a"); _panel(axes[1], "b")
    fig.suptitle("Multi-turn prompting adds substantial latency without clear quality gain",
                 fontsize=9.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "fig6_quality_cost", pd.DataFrame(source))


def main() -> None:
    apply_style()
    clf_path = config.PREDICTIONS_DIR / "classify_predictions.parquet"
    gen_path = config.PREDICTIONS_DIR / "generate_predictions.parquet"
    if not clf_path.exists() or not gen_path.exists():
        print("[nature-viz] 缺少逐样本预测文件，请先运行实验与评估。")
        return
    clf = pd.read_parquet(clf_path)
    gen = pd.read_parquet(gen_path)
    print("生成出版级图表 -> results/figures/")
    fig1_overview(clf, gen)
    fig2_context_ladder(clf, gen)
    fig3_prompt_tradeoff(clf, gen)
    fig4_error_transitions(clf)
    fig5_human_control(clf, gen)
    fig6_quality_cost(clf, gen)


if __name__ == "__main__":
    main()