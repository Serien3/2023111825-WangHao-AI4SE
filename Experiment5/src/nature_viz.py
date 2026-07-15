"""Experiment 5 — publication-grade result figures (nature-figure style).

Core conclusion the figure set defends
--------------------------------------
Existing human-code review models (Exp2 ML / Exp4 LLM) lose *discriminative
power* when transferred to AI-generated code. The F1 scores that look flat or
even higher are class-imbalance artifacts of the 74% merge rate; the true
bottleneck is the input (context / prompt), not the model — which motivates
Experiment 6.

Evidence chain (one link per figure)
------------------------------------
fig1  ML merge prediction: tree-model F1 declines on AI vs BOTH the human
      held-out test AND the leakage-free matched control -> a real
      generalization drop, not a distribution artifact.
fig2  LLM 16-condition landscape (4 context x 4 prompt): classification F1
      stays high while recall saturates to ~1.0 -> sets up the artifact.
fig3  Context sensitivity C1->C4: gain ~ 0 for classification, +0.005 for
      generation -> "AI needs more context" holds only weakly, only for
      generation.
fig4  Stratified-control ablation + precision-recall plane: (a) AI vs a
      leakage-free, distribution-aware control shows that tree-model drops
      persist after partial control of repository and label composition;
      (b) the P-R plane exposes the majority-predictor artifact shared by SVM
      and every LLM condition.
fig5  Process-feature gain Delta(full - pre): review-phase features help
      humans but help AI less / inconsistently.
fig6  Error-case attribution: all five selected misclassifications are false
      positives on high-surface-quality AI PRs.

Backend: Python / matplotlib only. Outputs SVG (editable text) + PDF + PNG@600
into results/figures/ under the SAME base names the report indexes (fig1..6).

Run:  cd Experiment5 && uv run python -m src.nature_viz
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# --------------------------------------------------------------------------- #
# Paths & constants (self-contained; no heavy cross-experiment imports)
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]            # Experiment5/
METRICS = ROOT / "results" / "metrics"
CASES = ROOT / "results" / "cases"
FIGDIR = ROOT / "results" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

MODELS = ["svm", "rf", "xgboost", "lightgbm"]
MODEL_LABELS = ["SVM", "RF", "XGBoost", "LightGBM"]
CONTEXTS = ["C1", "C2", "C3", "C4"]
CONTEXT_LABELS = ["C1\ndiff only", "C2\n+PR desc.", "C3\n+commit", "C4\n+all"]
PROMPTS = ["P1", "P2", "P3", "P4"]
PROMPT_LABELS = ["P1\nzero", "P2\nfew", "P3\nCoT", "P4\nrole"]

AI_POS_RATE = 0.7391304347826086   # 74% merge rate in the AI pool

# ---- unified palette --------------------------------------------------------
C_HUMAN = "#0F4D92"   # deep blue  — human anchor
C_AI = "#C0433F"      # brick red  — AI (the subject under test)
C_CTRL = "#4E9A6B"    # green      — leakage-free matched control
C_INK = "#272727"
C_GRID = "#D9D9D9"
C_MUTE = "#8F8F8F"
DELTA_UP = "#2E9E44"
DELTA_DOWN = "#D24B40"
COHORT = {"human": C_HUMAN, "ai": C_AI, "control": C_CTRL}
COHORT_LABEL = {"human": "Human test", "ai": "AI code", "control": "Matched control"}


# --------------------------------------------------------------------------- #
# Publication style (editable SVG text is mandatory)
# --------------------------------------------------------------------------- #
def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"],
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
        "legend.fontsize": 7.5,
        "figure.dpi": 120,
    })


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _panel_tag(ax, tag, x=-0.16, y=1.06):
    ax.text(x, y, tag, transform=ax.transAxes, fontsize=11, fontweight="bold",
            va="bottom", ha="left", color=C_INK)


def _save(fig, base: str):
    fig.savefig(FIGDIR / f"{base}.svg", bbox_inches="tight")
    fig.savefig(FIGDIR / f"{base}.pdf", bbox_inches="tight")
    fig.savefig(FIGDIR / f"{base}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {base}.{{svg,pdf,png}}")


# --------------------------------------------------------------------------- #
# Fig 1 — ML merge prediction: the generalization drop  (HERO of Exp A)
# --------------------------------------------------------------------------- #
def fig1_ml_generalization(hv: dict) -> None:
    ml = hv.get("ml", {})
    if not ml:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    x = np.arange(len(MODELS))
    w = 0.26
    order = [("human_test", "human"), ("ai", "ai"), ("control", "control")]
    for ax, fs in zip(axes, ["pre", "full"]):
        for k, (gk, ck) in enumerate(order):
            vals = [ml[fs][m][gk]["f1"] for m in MODELS]
            off = (k - 1) * w
            bars = ax.bar(x + off, vals, w, color=COHORT[ck],
                          edgecolor="white", linewidth=0.6,
                          label=COHORT_LABEL[ck], zorder=3)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=5.6, color=C_INK)
        # drop annotation on the worst tree model (AI vs human) for pre panel
        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_LABELS, fontsize=7.5)
        ax.set_ylim(0, 1.0)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.axhspan(0, 0.5, color="#F4F4F4", zorder=0)  # subtle "weak" band
        ax.grid(axis="y", color=C_GRID, lw=0.6, zorder=1)
        ax.set_title(f"{'Pre-review' if fs == 'pre' else 'Full'} features", pad=6)
    axes[0].set_ylabel("Merge-prediction F1")
    _panel_tag(axes[0], "a")
    _panel_tag(axes[1], "b")
    # annotate the collapse: RF pre AI vs human
    ax = axes[0]
    ai_rf = ml["pre"]["rf"]["ai"]["f1"]
    hu_rf = ml["pre"]["rf"]["human_test"]["f1"]
    xa = 1 - w / 2          # between the human and AI bars of RF
    ax.annotate("", xy=(xa, ai_rf), xytext=(xa, hu_rf),
                arrowprops=dict(arrowstyle="-|>", color=C_AI, lw=1.1))
    ax.text(xa - 0.06, (ai_rf + hu_rf) / 2, f"−{hu_rf - ai_rf:.2f}",
            color=C_AI, fontsize=7, fontweight="bold", va="center", ha="right")
    axes[1].legend(loc="lower right", ncol=1, handlelength=1.1,
                   borderpad=0.3, labelspacing=0.3)
    fig.suptitle("Most human-trained models lose merge-prediction power on AI code",
                 fontsize=9.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig1_ml_performance")


# --------------------------------------------------------------------------- #
# Fig 2 — LLM 16-condition landscape (heatmaps)
# --------------------------------------------------------------------------- #
def _grid(metrics: dict, key: str) -> np.ndarray:
    g = np.full((len(CONTEXTS), len(PROMPTS)), np.nan)
    for i, c in enumerate(CONTEXTS):
        for j, p in enumerate(PROMPTS):
            v = metrics.get(f"{c}_{p}", {}).get(key)
            if isinstance(v, (int, float)):
                g[i, j] = v
    return g


def _heat(ax, grid, cmap, vmin, vmax, title, fmt="{:.2f}", txt_thresh=0.5):
    im = ax.imshow(grid, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    norm = plt.Normalize(vmin, vmax)
    cm = plt.get_cmap(cmap)
    for (i, j), v in np.ndenumerate(grid):
        if np.isnan(v):
            continue
        r, g, b, _ = cm(norm(v))
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=6.6,
                color="white" if lum < txt_thresh else C_INK)
    ax.set_xticks(range(len(PROMPTS)))
    ax.set_xticklabels(PROMPT_LABELS, fontsize=6.6)
    ax.set_yticks(range(len(CONTEXTS)))
    ax.set_yticklabels(CONTEXT_LABELS, fontsize=6.6)
    ax.set_title(title, pad=6, fontsize=8.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def fig2_llm_landscape(clf: dict, gen: dict) -> None:
    ai_clf = clf.get("ai", {})
    ai_gen = gen.get("ai", {})
    if not ai_clf and not ai_gen:
        return
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.7))
    specs = [
        (ai_clf, "f1", "Blues", 0.80, 0.87, "Classification F1", "{:.2f}", 0.55),
        (ai_clf, "recall", "Reds", 0.90, 1.0, "Classification recall", "{:.2f}", 0.55),
        (ai_gen, "rougeL", "Greens", 0.035, 0.06, "Generation ROUGE-L", "{:.3f}", 0.55),
    ]
    for ax, (src, k, cmap, vmin, vmax, ttl, fmt, th) in zip(axes, specs):
        grid = _grid(src, k)
        im = _heat(ax, grid, cmap, vmin, vmax, ttl, fmt, th)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=6, length=0)
        cb.outline.set_visible(False)
    for ax, t in zip(axes, "abc"):
        _panel_tag(ax, t, x=-0.28, y=1.10)
    axes[0].set_ylabel("Context granularity")
    fig.suptitle("AI-side landscape: F1 looks strong, but recall is saturated "
                 "(the artifact)", fontsize=9.5, fontweight="bold", y=1.04)
    fig.tight_layout()
    _save(fig, "fig2_ai_heatmap")


# --------------------------------------------------------------------------- #
# Fig 3 — Context sensitivity C1 -> C4
# --------------------------------------------------------------------------- #
def fig3_context(ctx: dict) -> None:
    specs = [("classify_f1", "Classification F1", "Merge-prediction F1"),
             ("generate_rougeL", "Generation ROUGE-L", "Review-comment ROUGE-L")]
    have = [(k, t, yl) for k, t, yl in specs if ctx.get(k)]
    if not have:
        return
    fig, axes = plt.subplots(1, len(have), figsize=(6.6, 3.0))
    if len(have) == 1:
        axes = [axes]
    for ax, (key, ttl, ylab) in zip(axes, have):
        for grp in ["human", "ai", "control"]:
            by = ctx[key].get(grp, {}).get("by_context", {})
            ys = [by.get(c, np.nan) for c in CONTEXTS]
            if all(np.isnan(y) for y in ys):
                continue
            ax.plot(range(len(CONTEXTS)), ys, marker="o", ms=4.5, lw=1.6,
                    color=COHORT[grp], label=COHORT_LABEL[grp], zorder=3,
                    clip_on=False)
        # annotate C4-C1 gain for AI and human
        gain_ai = ctx[key].get("ai", {}).get("gain_C4_minus_C1")
        gain_hu = ctx[key].get("human", {}).get("gain_C4_minus_C1")
        note = []
        if gain_ai is not None:
            note.append(f"AI  Δ(C4−C1) = {gain_ai:+.3f}")
        if gain_hu is not None:
            note.append(f"Human Δ = {gain_hu:+.3f}")
        ax.text(0.03, 0.03, "\n".join(note), transform=ax.transAxes,
                fontsize=6.8, va="bottom", ha="left", color=C_MUTE)
        if key == "generate_rougeL":
            ax.text(0.97, 0.97, "control = human\n(reused reference)",
                    transform=ax.transAxes, fontsize=6.2, va="top", ha="right",
                    color=C_MUTE, style="italic")
        ax.set_xticks(range(len(CONTEXTS)))
        ax.set_xticklabels(CONTEXT_LABELS, fontsize=6.8)
        ax.set_ylabel(ylab)
        ax.grid(axis="y", color=C_GRID, lw=0.6)
        ax.margins(x=0.04)
    _panel_tag(axes[0], "a")
    if len(axes) > 1:
        _panel_tag(axes[1], "b")
    axes[0].legend(loc="upper left", handlelength=1.4, bbox_to_anchor=(0.0, 1.0))
    fig.suptitle("More context barely helps classification; it helps AI "
                 "generation only weakly", fontsize=9.3, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig3_context_sensitivity")


# --------------------------------------------------------------------------- #
# Fig 4 — Matched-control ablation  +  precision-recall artifact plane
# --------------------------------------------------------------------------- #
def fig4_ablation_artifact(hv: dict, clf: dict) -> None:
    ml = hv.get("ml", {})
    if not ml:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4),
                             gridspec_kw={"width_ratios": [1, 1.15]})

    # ---- panel a: AI vs matched control, per model (pre) — dumbbell -------- #
    ax = axes[0]
    y = np.arange(len(MODELS))[::-1]
    for yi, m in zip(y, MODELS):
        ai = ml["pre"][m]["ai"]["f1"]
        ct = ml["pre"][m]["control"]["f1"]
        ax.plot([ai, ct], [yi, yi], color=C_MUTE, lw=1.4, zorder=1)
        ax.scatter([ai], [yi], color=C_AI, s=42, zorder=3)
        ax.scatter([ct], [yi], color=C_CTRL, s=42, zorder=3)
        d = ai - ct
        ax.text(min(ai, ct) - 0.02, yi, f"{d:+.2f}", ha="right", va="center",
                fontsize=6.6, color=DELTA_DOWN if d < 0 else DELTA_UP)
    ax.set_yticks(y)
    ax.set_yticklabels(MODEL_LABELS, fontsize=7.5)
    ax.set_xlim(0.45, 0.95)
    ax.set_xlabel("Merge-prediction F1 (pre features)")
    ax.grid(axis="x", color=C_GRID, lw=0.6)
    ax.set_title("Leakage-free stratified control:\nAI trails on all tree models",
                 fontsize=8, pad=6)
    ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=C_AI,
                              markersize=7, label="AI code"),
                       Line2D([0], [0], marker="o", color="w", markerfacecolor=C_CTRL,
                              markersize=7, label="Matched control")],
              loc="lower right", handletextpad=0.2)
    _panel_tag(ax, "a", x=-0.30)

    # ---- panel b: precision-recall plane (the artifact) ------------------- #
    ax = axes[1]
    # majority-predictor guide: always-positive => recall=1, precision=pos_rate
    ax.axhline(1.0, color=C_GRID, lw=0.8, ls="--", zorder=1)
    ax.axvline(AI_POS_RATE, color=C_GRID, lw=0.8, ls="--", zorder=1)
    ax.scatter([AI_POS_RATE], [1.0], marker="*", s=170, color=C_INK, zorder=6)
    ax.annotate('"always merge"\nmajority predictor',
                xy=(AI_POS_RATE, 1.0), xytext=(0.60, 0.63), fontsize=6.6,
                color=C_INK, ha="center",
                arrowprops=dict(arrowstyle="-|>", color=C_INK, lw=0.9))
    # ML models on AI code (per-model label offsets to avoid collisions)
    lab_off = {  # dx, dy, ha, va
        "svm": (-0.012, 0.0, "right", "center"),
        "rf": (-0.012, 0.0, "right", "center"),
        "xgboost": (0.012, 0.0, "left", "center"),
        "lightgbm": (0.0, 0.030, "center", "bottom"),
    }
    for m, lab in zip(MODELS, MODEL_LABELS):
        d = ml["pre"][m]["ai"]
        mk = "^" if m == "svm" else "o"
        ax.scatter(d["precision"], d["recall"], s=55, marker=mk,
                   color=C_AI, edgecolor="white", linewidth=0.6, zorder=5)
        dx, dy, ha, va = lab_off[m]
        ax.text(d["precision"] + dx, d["recall"] + dy, lab, fontsize=6,
                ha=ha, va=va, color=C_AI)
    # LLM classify conditions on AI code (cloud)
    ai_clf = clf.get("ai", {})
    if ai_clf:
        ps = [v["precision"] for v in ai_clf.values()]
        rs = [v["recall"] for v in ai_clf.values()]
        ax.scatter(ps, rs, s=26, marker="s", facecolor="none",
                   edgecolor="#7C6CCF", linewidth=1.0, zorder=4)
    ax.set_xlim(0.6, 0.92)
    ax.set_ylim(0.42, 1.06)
    ax.set_xlabel("Precision (AI code)")
    ax.set_ylabel("Recall (AI code)")
    ax.grid(color=C_GRID, lw=0.5, zorder=0)
    ax.set_title("Recall collapse (trees) vs recall saturation (SVM / LLM)",
                 fontsize=8, pad=6)
    ax.legend(handles=[
        Line2D([0], [0], marker="^", color="w", markerfacecolor=C_AI, markersize=8,
               label="SVM (AI)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_AI, markersize=8,
               label="Tree models (AI)"),
        Line2D([0], [0], marker="s", color="w", markeredgecolor="#7C6CCF",
               markerfacecolor="none", markersize=8, label="LLM conditions (AI)"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor=C_INK, markersize=12,
               label="Majority predictor"),
    ], loc="lower left", handletextpad=0.2, labelspacing=0.3)
    _panel_tag(ax, "b", x=-0.20)

    fig.suptitle("Tree-model drops persist (a); high F1 can reflect class "
                 "imbalance (b)", fontsize=9.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "fig4_control_ablation")


# --------------------------------------------------------------------------- #
# Fig 5 — Process-feature gain  Delta(full - pre) F1
# --------------------------------------------------------------------------- #
def fig5_leakage_gain(hv: dict) -> None:
    ml = hv.get("ml", {})
    if not (ml.get("pre") and ml.get("full")):
        return
    fig, ax = plt.subplots(figsize=(5.4, 3.1))
    x = np.arange(len(MODELS))
    w = 0.26
    order = [("human_test", "human"), ("ai", "ai"), ("control", "control")]
    for k, (gk, ck) in enumerate(order):
        deltas = [ml["full"][m][gk]["f1"] - ml["pre"][m][gk]["f1"] for m in MODELS]
        off = (k - 1) * w
        ax.bar(x + off, deltas, w, color=COHORT[ck], edgecolor="white",
               linewidth=0.6, label=COHORT_LABEL[ck], zorder=3)
    ax.axhline(0, color=C_INK, lw=0.9, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_LABELS, fontsize=7.5)
    ax.set_ylabel("Δ F1  (full − pre)")
    ax.grid(axis="y", color=C_GRID, lw=0.6, zorder=1)
    ax.legend(loc="upper left", ncol=1, handlelength=1.1)
    ax.text(0.98, 0.03, "adding review-phase features", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.6, color=C_MUTE)
    fig.suptitle("Post-review features raise AI tree-model F1; SVM is the "
                 "exception", fontsize=8.6, fontweight="bold", y=1.0)
    fig.tight_layout()
    _save(fig, "fig5_leakage_gain")


# --------------------------------------------------------------------------- #
# Fig 6 — Error-case attribution
# --------------------------------------------------------------------------- #
def fig6_error_cases(cases: dict) -> None:
    cs = cases.get("cases", [])
    if not cs:
        return
    n = len(cs)
    fp = sum(1 for c in cs if c.get("pred_merge") and not c.get("true_merge"))
    fn = sum(1 for c in cs if not c.get("pred_merge") and c.get("true_merge"))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9),
                             gridspec_kw={"width_ratios": [0.85, 1.3]})

    # ---- panel a: error direction (all are false positives) --------------- #
    ax = axes[0]
    kinds = ["False positive\n(pred merge,\nactually not)",
             "False negative\n(pred not,\nactually merge)"]
    vals = [fp, fn]
    bars = ax.bar([0, 1], vals, width=0.62, color=[C_AI, C_MUTE],
                  edgecolor="white", linewidth=0.8, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.06, str(v),
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(kinds, fontsize=6.8)
    ax.set_ylabel(f"Mis-classified AI PRs (n = {n})")
    ax.set_ylim(0, max(vals) + 1)
    ax.grid(axis="y", color=C_GRID, lw=0.6, zorder=1)
    ax.set_title("All five selected errors are over-optimistic", fontsize=8, pad=6)
    _panel_tag(ax, "a", x=-0.26)

    # ---- panel b: the concrete cases (only structured fields) ------------- #
    ax = axes[1]
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, n - 0.3)
    ax.axis("off")
    # column x anchors
    x_id, x_pred, x_true, x_ctx = 0.02, 0.44, 0.66, 0.90
    # header
    hy = n - 0.45
    for xx, htxt, ha in [(x_id, "AI pull request", "left"), (x_pred, "predicted", "center"),
                         (x_true, "actual", "center"), (x_ctx, "context", "center")]:
        ax.text(xx, hy, htxt, fontsize=6.4, color=C_MUTE, fontweight="bold",
                va="bottom", ha=ha)
    ax.plot([0, 1], [hy - 0.12, hy - 0.12], color=C_GRID, lw=0.8)
    for i, c in enumerate(cs):
        y = n - 1 - i
        repo = c["repo"].split("/")[-1]
        ax.text(x_id, y, f"{repo} #{c['number']}", fontsize=7, va="center",
                ha="left", color=C_INK)
        # predicted = merge (red pill)
        ax.text(x_pred, y, "MERGE", fontsize=6.4, va="center", ha="center",
                color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc=C_AI, ec="none"))
        # actual = not merged (grey pill)
        ax.text(x_true, y, "not merged", fontsize=6.4, va="center", ha="center",
                color=C_INK,
                bbox=dict(boxstyle="round,pad=0.25", fc="#ECECEC", ec=C_MUTE, lw=0.5))
        # context condition
        ax.text(x_ctx, y, c["condition"].split("_")[0], fontsize=6.6, va="center",
                ha="center", color=C_MUTE)
    _panel_tag(ax, "b", x=-0.02, y=1.02)

    fig.suptitle("Five selected diff-only errors are false positives on "
                 "surface-complete AI PRs",
                 fontsize=9.0, fontweight="bold", y=1.04)
    fig.tight_layout()
    _save(fig, "fig6_error_cases")


# --------------------------------------------------------------------------- #
def main() -> None:
    apply_style()
    hv = _load(METRICS / "human_vs_ai.json")
    clf = _load(METRICS / "llm_classify_metrics.json")
    gen = _load(METRICS / "llm_generate_metrics.json")
    ctx = _load(METRICS / "context_sensitivity.json")
    cases = _load(CASES / "error_cases.json")

    print("[nature_viz] rendering publication figures ...")
    fig1_ml_generalization(hv)
    fig2_llm_landscape(clf, gen)
    fig3_context(ctx)
    fig4_ablation_artifact(hv, clf)
    fig5_leakage_gain(hv)
    fig6_error_cases(cases)
    print(f"done -> {FIGDIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
