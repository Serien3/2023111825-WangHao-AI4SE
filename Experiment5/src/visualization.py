"""可视化：设计 §7 的 6 张图。全部数据驱动，缺数据的图自动跳过。

图 1 人类 vs AI 分类性能对比（ML 四模型 pre/full）
图 2 AI 侧 16 条件热力图（分类 F1 / 生成 ROUGE-L）
图 3 上下文敏感度对比（C1→C4，human/ai/control 叠加）
图 4 匹配对照消融（AI vs 匹配人类对照）
图 5 ML 泄漏增益变化（pre vs full 的 ΔF1，human/ai/control）
图 6 错误案例类型分布

产物：results/figures/*.png
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import config  # noqa: E402

plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "Noto Sans CJK SC",
                                   "DejaVu Sans", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def _load(path):
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save(fig, name):
    out = config.FIGURES_DIR / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out.relative_to(config.PROJECT_ROOT)}")


# --------------------------------------------------------------------------- #
# 图 1：ML 分类性能对比（human_test / ai / control × pre/full）
# --------------------------------------------------------------------------- #
def fig_ml_performance(hv: dict) -> None:
    ml = hv.get("ml", {})
    if not ml:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, fs in zip(axes, config.FEATURE_SETS):
        models = config.ML_MODELS
        human = [ml.get(fs, {}).get(m, {}).get("human_test", {}).get("f1") or 0 for m in models]
        ai = [ml.get(fs, {}).get(m, {}).get("ai", {}).get("f1") or 0 for m in models]
        ctrl = [ml.get(fs, {}).get(m, {}).get("control", {}).get("f1") or 0 for m in models]
        x = np.arange(len(models))
        w = 0.26
        ax.bar(x - w, human, w, label="人类 test", color="#4C72B0")
        ax.bar(x, ai, w, label="AI", color="#C44E52")
        ax.bar(x + w, ctrl, w, label="匹配对照", color="#55A868")
        ax.set_title(f"{fs} 特征集")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("F1")
    axes[0].legend()
    fig.suptitle("图1 ML Merge Prediction 性能对比（F1）：人类 test vs AI vs 匹配对照")
    _save(fig, "fig1_ml_performance.png")


# --------------------------------------------------------------------------- #
# 图 2：AI 侧 16 条件热力图
# --------------------------------------------------------------------------- #
def _heatmap(ax, metrics: dict, metric: str, title: str):
    grid = np.full((len(config.CONTEXTS), len(config.PROMPTS)), np.nan)
    for i, ctx in enumerate(config.CONTEXTS):
        for j, prm in enumerate(config.PROMPTS):
            v = metrics.get(f"{ctx}_{prm}", {}).get(metric)
            if isinstance(v, (int, float)):
                grid[i, j] = v
    im = ax.imshow(grid, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(config.PROMPTS)))
    ax.set_xticklabels([config.PROMPT_LABELS[p] for p in config.PROMPTS], rotation=20)
    ax.set_yticks(range(len(config.CONTEXTS)))
    ax.set_yticklabels([config.CONTEXT_LABELS[c] for c in config.CONTEXTS])
    for i in range(len(config.CONTEXTS)):
        for j in range(len(config.PROMPTS)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                        color="w", fontsize=9)
    ax.set_title(title)
    return im


def fig_ai_heatmap(llm_clf: dict, llm_gen: dict) -> None:
    ai_clf = llm_clf.get("ai", {})
    ai_gen = llm_gen.get("ai", {})
    panels = []
    if ai_clf:
        panels.append(("分类 F1", ai_clf, "f1"))
    if ai_gen:
        panels.append(("生成 ROUGE-L", ai_gen, "rougeL"))
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5))
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, metrics, metric) in zip(axes, panels):
        im = _heatmap(ax, metrics, metric, f"AI 侧 {title}")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("图2 AI 生成代码上的 16 条件（4 上下文 × 4 Prompt）")
    _save(fig, "fig2_ai_heatmap.png")


# --------------------------------------------------------------------------- #
# 图 3：上下文敏感度 C1→C4
# --------------------------------------------------------------------------- #
def fig_context_sensitivity(ctx_sens: dict) -> None:
    panels = [("classify_f1", "分类 F1"), ("generate_rougeL", "生成 ROUGE-L")]
    have = [(k, t) for k, t in panels if ctx_sens.get(k)]
    if not have:
        return
    fig, axes = plt.subplots(1, len(have), figsize=(6.5 * len(have), 5))
    if len(have) == 1:
        axes = [axes]
    colors = {"human": "#4C72B0", "ai": "#C44E52", "control": "#55A868"}
    for ax, (key, title) in zip(axes, have):
        for group in ["human", "ai", "control"]:
            by_ctx = ctx_sens[key].get(group, {}).get("by_context", {})
            ys = [by_ctx.get(c) for c in config.CONTEXTS]
            if all(y is None for y in ys):
                continue
            ax.plot(config.CONTEXTS, [y if y is not None else np.nan for y in ys],
                    marker="o", label=group, color=colors[group])
        ax.set_title(title)
        ax.set_xlabel("上下文（C1 仅diff → C4 全部）")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("图3 上下文敏感度：C1→C4 性能变化（人类 vs AI vs 对照）")
    _save(fig, "fig3_context_sensitivity.png")


# --------------------------------------------------------------------------- #
# 图 4：匹配对照消融（ML，AI vs control）
# --------------------------------------------------------------------------- #
def fig_control_ablation(hv: dict) -> None:
    ml = hv.get("ml", {})
    if not ml:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    models = config.ML_MODELS
    fs = "pre"
    ai = [ml.get(fs, {}).get(m, {}).get("ai", {}).get("f1") or 0 for m in models]
    ctrl = [ml.get(fs, {}).get(m, {}).get("control", {}).get("f1") or 0 for m in models]
    x = np.arange(len(models))
    w = 0.35
    ax.bar(x - w / 2, ai, w, label="AI", color="#C44E52")
    ax.bar(x + w / 2, ctrl, w, label="匹配人类对照", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.set_ylabel("F1 (pre)")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    ax.set_title("图4 匹配对照消融：同分布下 AI vs 人类（证明差异非分布假象）")
    _save(fig, "fig4_control_ablation.png")


# --------------------------------------------------------------------------- #
# 图 5：ML 泄漏增益变化（full - pre 的 ΔF1）
# --------------------------------------------------------------------------- #
def fig_leakage_gain(hv: dict) -> None:
    ml = hv.get("ml", {})
    if not ml.get("pre") or not ml.get("full"):
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    models = config.ML_MODELS
    groups = [("human_test", "人类 test", "#4C72B0"),
              ("ai", "AI", "#C44E52"), ("control", "匹配对照", "#55A868")]
    x = np.arange(len(models))
    w = 0.26
    for i, (gk, label, color) in enumerate(groups):
        deltas = []
        for m in models:
            pre = ml["pre"].get(m, {}).get(gk, {}).get("f1")
            full = ml["full"].get(m, {}).get(gk, {}).get("f1")
            deltas.append((full - pre) if (pre is not None and full is not None) else 0)
        ax.bar(x + (i - 1) * w, deltas, w, label=label, color=color)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.set_ylabel("ΔF1 (full − pre)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    ax.set_title("图5 审查过程特征的泄漏增益：full−pre（人类 vs AI vs 对照）")
    _save(fig, "fig5_leakage_gain.png")


# --------------------------------------------------------------------------- #
# 图 6：错误案例类型分布
# --------------------------------------------------------------------------- #
def fig_error_cases(cases: dict) -> None:
    counts = cases.get("type_counts", {})
    if not counts:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = list(counts.keys())
    vals = [counts[k] for k in labels]
    ax.bar(labels, vals, color="#8172B3")
    for i, v in enumerate(vals):
        ax.text(i, v, str(v), ha="center", va="bottom")
    ax.set_ylabel("案例数")
    ax.set_title("图6 错误案例类型分布（AI 侧）")
    _save(fig, "fig6_error_cases.png")


def main() -> None:
    hv = _load(config.METRICS_DIR / "human_vs_ai.json")
    llm_clf = _load(config.METRICS_DIR / "llm_classify_metrics.json")
    llm_gen = _load(config.METRICS_DIR / "llm_generate_metrics.json")
    ctx_sens = _load(config.METRICS_DIR / "context_sensitivity.json")
    cases = _load(config.CASES_DIR / "error_cases.json")

    print("[visualization] 出图 ...")
    fig_ml_performance(hv)
    fig_ai_heatmap(llm_clf, llm_gen)
    fig_context_sensitivity(ctx_sens)
    fig_control_ablation(hv)
    fig_leakage_gain(hv)
    fig_error_cases(cases)
    print("出图完成。")


if __name__ == "__main__":
    main()
