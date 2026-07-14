"""可视化：实验六核心结论叙事的图（对齐 6.9 五问）。全部数据驱动，缺数据自动跳过。

图 1 上下文阶梯增益曲线（L0→L4 × P*，balanced_acc / non_merge_recall；judge relevance）—— 答 Q1
图 2 Prompt 消融（L4 × {P*,P5,P6}，non_merge_recall 与 judge 4 维）—— 答 Q2
图 3 归因对比（L0×P5 vs L4×P* vs L4×P5）—— 答 Q3（上下文 vs prompt 功劳）
图 4 混淆矩阵网格（关键条件）—— 修复分类失明
图 5 AI vs 人类对照（L4×P*、L4×P5）—— 答"改进后是否追平人类"

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


def _k(level, prm, group="ai"):
    return f"{level}_{prm}_{group}"


# --------------------------------------------------------------------------- #
# 图 1：上下文阶梯增益曲线
# --------------------------------------------------------------------------- #
def fig_context_ladder(clf: dict, gen: dict) -> None:
    ps_c, ps_g = config.PSTAR["classify"], config.PSTAR["generate"]
    levels = config.LEVELS
    bal = [clf.get(_k(lv, ps_c), {}).get("balanced_accuracy") for lv in levels]
    nmr = [clf.get(_k(lv, ps_c), {}).get("non_merge_recall") for lv in levels]
    rel = [gen.get(_k(lv, ps_g), {}).get("judge_relevance") for lv in levels]
    if not any(v is not None for v in bal + rel):
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(levels))
    ax = axes[0]
    ax.plot(x, [b or np.nan for b in bal], "o-", color="#4C72B0", label="balanced acc")
    ax.plot(x, [n or np.nan for n in nmr], "s--", color="#C44E52", label="non-merge recall")
    ax.set_xticks(x); ax.set_xticklabels(levels)
    ax.set_title(f"分类：上下文阶梯 × {ps_c}"); ax.set_ylim(0, 1)
    ax.legend(); ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(x, [r or np.nan for r in rel], "o-", color="#55A868", label="judge relevance")
    ax.set_xticks(x); ax.set_xticklabels(levels)
    ax.set_title(f"生成：上下文阶梯 × {ps_g}"); ax.set_ylim(1, 5)
    ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle("图1 上下文阶梯增益曲线（答 Q1：L4 是否弥合差距）")
    _save(fig, "fig1_context_ladder.png")


# --------------------------------------------------------------------------- #
# 图 2：Prompt 消融（L4 × {P*,P5,P6}）
# --------------------------------------------------------------------------- #
def fig_prompt_ablation(clf: dict, gen: dict) -> None:
    ps_c, ps_g = config.PSTAR["classify"], config.PSTAR["generate"]
    clf_prompts = [ps_c, "P5", "P6"]
    gen_prompts = [ps_g, "P5", "P6"]
    nmr = [clf.get(_k("L4", p), {}).get("non_merge_recall") for p in clf_prompts]
    if not any(v is not None for v in nmr):
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    x = np.arange(len(clf_prompts))
    ax.bar(x, [n or 0 for n in nmr], color=["#4C72B0", "#C44E52", "#8172B3"])
    ax.set_xticks(x); ax.set_xticklabels([config.PROMPT_LABELS.get(p, p) for p in clf_prompts])
    ax.set_title(f"分类 L4：non-merge recall（B 生效判据）"); ax.set_ylim(0, 1)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    dims = ["relevance", "actionable", "correct", "hallucination"]
    x = np.arange(len(dims)); w = 0.25
    for i, p in enumerate(gen_prompts):
        vals = [gen.get(_k("L4", p), {}).get(f"judge_{d}") or 0 for d in dims]
        ax.bar(x + (i - 1) * w, vals, w, label=config.PROMPT_LABELS.get(p, p))
    ax.set_xticks(x); ax.set_xticklabels(dims, rotation=15)
    ax.set_title("生成 L4：judge 4 维"); ax.set_ylim(0, 5)
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.suptitle("图2 Prompt 消融（答 Q2：哪种 Prompt 最适合）")
    _save(fig, "fig2_prompt_ablation.png")


# --------------------------------------------------------------------------- #
# 图 3：归因（L0×P5 vs L4×P* vs L4×P5）
# --------------------------------------------------------------------------- #
def fig_attribution(clf: dict, gen: dict) -> None:
    ps_c = config.PSTAR["classify"]
    cells = [("L4", ps_c, "L4×P*"), ("L0", "P5", "L0×P5"), ("L4", "P5", "L4×P5")]
    nmr = [clf.get(_k(lv, pm), {}).get("non_merge_recall") for lv, pm, _ in cells]
    bal = [clf.get(_k(lv, pm), {}).get("balanced_accuracy") for lv, pm, _ in cells]
    if not any(v is not None for v in nmr):
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(cells)); w = 0.38
    ax.bar(x - w / 2, [n or 0 for n in nmr], w, label="non-merge recall", color="#C44E52")
    ax.bar(x + w / 2, [b or 0 for b in bal], w, label="balanced acc", color="#4C72B0")
    ax.set_xticks(x); ax.set_xticklabels([c[2] for c in cells])
    ax.set_ylim(0, 1); ax.legend(); ax.grid(alpha=0.3, axis="y")
    ax.set_title("图3 归因：上下文(L4) vs prompt(P5) 各自的功劳（答 Q3）")
    _save(fig, "fig3_attribution.png")


# --------------------------------------------------------------------------- #
# 图 4：混淆矩阵网格
# --------------------------------------------------------------------------- #
def fig_confusion(clf: dict) -> None:
    ps_c = config.PSTAR["classify"]
    picks = [("L0", ps_c), ("L4", ps_c), ("L4", "P5"), ("L0", "P5")]
    picks = [(lv, pm) for lv, pm in picks if clf.get(_k(lv, pm), {}).get("confusion_matrix")]
    if not picks:
        return
    fig, axes = plt.subplots(1, len(picks), figsize=(4 * len(picks), 4))
    if len(picks) == 1:
        axes = [axes]
    for ax, (lv, pm) in zip(axes, picks):
        cm = clf[_k(lv, pm)]["confusion_matrix"]
        mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        ax.imshow(mat, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, mat[i, j], ha="center", va="center",
                        color="black", fontsize=13)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["pred non", "pred merge"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["true non", "true merge"])
        ax.set_title(f"{lv}×{pm}")
    fig.suptitle("图4 混淆矩阵（修复分类失明：看清 non-merge 判别力）")
    _save(fig, "fig4_confusion.png")


# --------------------------------------------------------------------------- #
# 图 5：AI vs 人类对照
# --------------------------------------------------------------------------- #
def fig_human_control(clf: dict, gen: dict) -> None:
    ps_c, ps_g = config.PSTAR["classify"], config.PSTAR["generate"]
    conds = [("L4", ps_c), ("L4", "P5")]
    labels = [f"L4×{ps_c}", "L4×P5"]
    ai_bal = [clf.get(_k(lv, pm, "ai"), {}).get("balanced_accuracy") for lv, pm in conds]
    hu_bal = [clf.get(_k(lv, pm, "control"), {}).get("balanced_accuracy") for lv, pm in conds]
    if not any(v is not None for v in ai_bal + hu_bal):
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    x = np.arange(len(conds)); w = 0.38
    ax.bar(x - w / 2, [a or 0 for a in ai_bal], w, label="AI", color="#C44E52")
    ax.bar(x + w / 2, [h or 0 for h in hu_bal], w, label="人类对照", color="#55A868")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1); ax.legend(); ax.grid(alpha=0.3, axis="y")
    ax.set_title("分类 balanced acc：AI vs 人类对照")

    ax = axes[1]
    ai_rel = [gen.get(_k(lv, pm, "ai"), {}).get("judge_relevance") for lv, pm in conds]
    hu_rel = [gen.get(_k(lv, pm, "control"), {}).get("judge_relevance") for lv, pm in conds]
    ax.bar(x - w / 2, [a or 0 for a in ai_rel], w, label="AI", color="#C44E52")
    ax.bar(x + w / 2, [h or 0 for h in hu_rel], w, label="人类对照", color="#55A868")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 5); ax.legend(); ax.grid(alpha=0.3, axis="y")
    ax.set_title("生成 judge relevance：AI vs 人类对照")
    fig.suptitle("图5 改进后 AI 是否追平人类（关键格对照）")
    _save(fig, "fig5_human_control.png")


def main() -> None:
    clf = _load(config.METRICS_DIR / "classify_metrics.json")
    gen = _load(config.METRICS_DIR / "generate_metrics.json")
    if not clf and not gen:
        print("[visualization] 无指标文件，跳过。请先跑 evaluate。")
        return
    print("生成图表 → results/figures/")
    fig_context_ladder(clf, gen)
    fig_prompt_ablation(clf, gen)
    fig_attribution(clf, gen)
    fig_confusion(clf)
    fig_human_control(clf, gen)


if __name__ == "__main__":
    main()
