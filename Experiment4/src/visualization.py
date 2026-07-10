"""切片 8b：结果可视化（Nature-style 出版级组图）。

设计原则（图服务于科学论证，见设计文档 §8）：
不再输出 11 张零散小图，而是构建 **4 张多面板组图**，每张捍卫一个结论，
统一风格，支撑「实验→结果→结论」逻辑链：

  fig1_classify_landscape.{svg,png}   分类：上下文/Prompt 的性能地形
      → 结论 1：上下文比 Prompt 更重要；C2(Diff+PR描述) 最佳；CoT 无增益
  fig2_llm_failure.{svg,png}          分类：LLM 失效模式 + 与传统 ML 同集对比
      → 结论 2：LLM 倾向全判 MERGE（高召回低精确），仅微超基线；训练过的 ML 更均衡且更强
  fig3_generate_landscape.{svg,png}   生成：审查意见质量地形
      → 结论 3：生成任务普遍困难；富上下文无益；输出偏短于人类
  fig4_latency_cost.{svg,png}         成本-收益：CoT 的推理时间税
      → 结论 4：CoT 大幅拉高延迟，却无质量回报

依赖 evaluate 产出的 metrics JSON、ml_baseline_50.json、predictions parquet。
所有文本以可编辑 SVG 保存（svg.fonttype=none），并附 PNG 供报告嵌入。
"""
from __future__ import annotations

import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from . import config

# --------------------------------------------------------------------------- #
# 出版风格：可编辑 SVG 文本 + 中文字体
# --------------------------------------------------------------------------- #
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Noto Sans CJK SC", "Noto Serif CJK SC",
                        "AR PL UMing TW", "DejaVu Sans"],
    "svg.fonttype": "none",       # SVG 文本保持可编辑
    "pdf.fonttype": 42,
    "axes.unicode_minus": False,
    "font.size": 9,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "figure.dpi": 120,
})

# --------------------------------------------------------------------------- #
# 调色板（一个中性族 + 一个信号族 + 一个强调色，全图统一）
# --------------------------------------------------------------------------- #
PALETTE = {
    "blue_main": "#0F4D92",
    "blue_sec": "#3775BA",
    "teal": "#42949E",
    "green": "#4C9A64",
    "red": "#B64342",
    "gold": "#D9A441",
    "violet": "#7C6CCF",
    "n_light": "#CFCECE",
    "n_mid": "#767676",
    "n_dark": "#4D4D4D",
}
# 上下文有序 → 蓝色渐深；Prompt 类别 → 分类色
CTX_COLORS = ["#BAD2EC", "#7FAAD8", "#3775BA", "#0F4D92"]
PROMPT_COLORS = [PALETTE["n_mid"], PALETTE["teal"], PALETTE["violet"], PALETTE["gold"]]
CMAP_SEQ = LinearSegmentedColormap.from_list("blseq", ["#F2F7FB", "#BAD2EC", "#3775BA", "#0F4D92"])
CMAP_WARM = LinearSegmentedColormap.from_list("warm", ["#FBF3EE", "#F0C9A8", "#D9A441", "#B64342"])

CTX_LAB = [config.CONTEXT_LABELS[c] for c in config.CONTEXTS]
PRM_LAB = [config.PROMPT_LABELS[p] for p in config.PROMPTS]
CTX_TICK = [f"{c}\n{config.CONTEXT_LABELS[c]}" for c in config.CONTEXTS]
PRM_TICK = [f"{p}\n{config.PROMPT_LABELS[p]}" for p in config.PROMPTS]


# --------------------------------------------------------------------------- #
# 数据加载
# --------------------------------------------------------------------------- #
def _load(name: str) -> dict:
    path = config.METRICS_DIR / name
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _matrix(metrics: dict, key: str) -> np.ndarray:
    """行=上下文 C1..C4，列=Prompt P1..P4。"""
    m = np.full((len(config.CONTEXTS), len(config.PROMPTS)), np.nan)
    for i, c in enumerate(config.CONTEXTS):
        for j, p in enumerate(config.PROMPTS):
            cell = metrics.get(f"{c}_{p}")
            if cell and cell.get(key) is not None:
                m[i, j] = cell[key]
    return m


import warnings

CTX_SHORT = list(config.CONTEXTS)   # C1..C4
PRM_SHORT = list(config.PROMPTS)    # P1..P4


def _save(fig, stem: str) -> None:
    with warnings.catch_warnings():   # twinx/colorbar 与 tight_layout 的无害告警
        warnings.simplefilter("ignore")
        fig.tight_layout(pad=1.1)
    for ext in ("svg", "png"):
        out = config.FIGURES_DIR / f"{stem}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {stem}.svg / .png")


def _panel_label(ax, s: str) -> None:
    ax.text(-0.14, 1.06, s, transform=ax.transAxes, fontsize=13,
            fontweight="bold", ha="left", va="bottom")


def _heatmap(ax, mat, cmap, fmt="{:.3f}", vmin=None, vmax=None,
             xticklab=PRM_TICK, yticklab=CTX_TICK) -> "matplotlib.image.AxesImage":
    im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(xticklab, fontsize=8)
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(yticklab, fontsize=8)
    lo = np.nanmin(mat) if vmin is None else vmin
    hi = np.nanmax(mat) if vmax is None else vmax
    rng = (hi - lo) or 1.0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                lum = (mat[i, j] - lo) / rng
                ax.text(j, i, fmt.format(mat[i, j]), ha="center", va="center",
                        fontsize=8.5, color="white" if lum > 0.6 else PALETTE["n_dark"])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-.5, mat.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, mat.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    return im


# --------------------------------------------------------------------------- #
# 图 1：分类性能地形（上下文 vs Prompt）
# --------------------------------------------------------------------------- #
def fig1_classify_landscape() -> None:
    cls = _load("classify_metrics.json")
    if not cls:
        print("[fig1] 无分类指标，跳过。")
        return
    f1 = _matrix(cls, "f1")
    lat = _matrix(cls, "avg_latency")

    fig = plt.figure(figsize=(11, 4.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.42)

    # (a) hero: F1 热力图
    ax0 = fig.add_subplot(gs[0])
    im = _heatmap(ax0, f1, CMAP_SEQ, fmt="{:.3f}")
    ax0.set_title("F1（4 上下文 × 4 Prompt）", fontsize=10, pad=8)
    cb = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.03)
    cb.set_label("F1", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    _panel_label(ax0, "a")

    # (b) 边际效应：上下文（对 Prompt 取均值）
    ax1 = fig.add_subplot(gs[1])
    ctx_mean = np.nanmean(f1, axis=1)
    ctx_std = np.nanstd(f1, axis=1)
    ax1.bar(range(4), ctx_mean, yerr=ctx_std, color=CTX_COLORS,
            edgecolor=PALETTE["n_dark"], linewidth=0.8,
            error_kw=dict(elinewidth=1, capsize=3, ecolor=PALETTE["n_mid"]))
    ax1.set_xticks(range(4))
    ax1.set_xticklabels(CTX_SHORT, fontsize=9)
    ax1.set_ylabel("F1（对 Prompt 平均）", fontsize=9)
    ax1.set_ylim(0.70, 0.83)
    ax1.set_title("上下文的边际效应", fontsize=10, pad=8)
    best = int(np.nanargmax(ctx_mean))
    ax1.annotate("最佳", xy=(best, ctx_mean[best]), xytext=(best, 0.822),
                 ha="center", fontsize=8, color=PALETTE["red"],
                 arrowprops=dict(arrowstyle="->", color=PALETTE["red"], lw=1))
    _panel_label(ax1, "b")

    # (c) 边际效应：Prompt（对上下文取均值）+ 延迟叠加
    ax2 = fig.add_subplot(gs[2])
    prm_mean = np.nanmean(f1, axis=0)
    prm_std = np.nanstd(f1, axis=0)
    ax2.bar(range(4), prm_mean, yerr=prm_std, color=PROMPT_COLORS,
            edgecolor=PALETTE["n_dark"], linewidth=0.8,
            error_kw=dict(elinewidth=1, capsize=3, ecolor=PALETTE["n_mid"]))
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(PRM_SHORT, fontsize=9)
    ax2.set_ylabel("F1（对上下文平均）", fontsize=9)
    ax2.set_ylim(0.70, 0.83)
    ax2.set_title("Prompt 的边际效应", fontsize=10, pad=8)
    # 叠加延迟折线（右轴）——凸显 CoT 无增益却更慢
    ax2b = ax2.twinx()
    ax2b.plot(range(4), np.nanmean(lat, axis=0), "o-", color=PALETTE["red"],
              lw=1.5, ms=5, label="平均推理时间")
    ax2b.set_ylabel("推理时间 (s)", fontsize=9, color=PALETTE["red"])
    ax2b.tick_params(axis="y", labelcolor=PALETTE["red"], labelsize=8)
    ax2b.spines["top"].set_visible(False)
    ax2b.spines["right"].set_visible(True)
    ax2b.spines["right"].set_color(PALETTE["red"])
    _panel_label(ax2, "c")

    fig.suptitle("分类性能地形：上下文的作用大于 Prompt，CoT(P3) 更慢却无增益",
                 fontsize=11.5, fontweight="bold", y=1.04)
    _save(fig, "fig1_classify_landscape")


# --------------------------------------------------------------------------- #
# 图 2：LLM 失效模式 + 与传统 ML 同 50 子集对比
# --------------------------------------------------------------------------- #
def _base_rate() -> float:
    with open(config.SAMPLES_DIR / "classify_50.json", encoding="utf-8") as f:
        s = json.load(f)
    return sum(1 for r in s if r["is_merged"]) / len(s)


def fig2_llm_failure() -> None:
    cls = _load("classify_metrics.json")
    ml = _load("ml_baseline_50.json")
    if not cls or not ml:
        print("[fig2] 缺分类或 ML 基线，跳过。")
        return
    br = _base_rate()

    fig = plt.figure(figsize=(11, 4.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1, 1.05], wspace=0.42)

    # (a) 精确率-召回率散点：LLM 16 条件挤在高召回低精确角，ML 更均衡
    ax0 = fig.add_subplot(gs[0])
    prec = [cls[k]["precision"] for k in cls]
    rec = [cls[k]["recall"] for k in cls]
    ax0.scatter(rec, prec, s=42, color=PALETTE["blue_sec"], edgecolor=PALETTE["blue_main"],
                linewidth=0.6, alpha=0.9, label="LLM（16 条件）", zorder=3)
    mlp = [ml[m]["precision"] for m in ml]
    mlr = [ml[m]["recall"] for m in ml]
    ax0.scatter(mlr, mlp, s=90, marker="D", color=PALETTE["red"],
                edgecolor="white", linewidth=0.8, label="传统 ML（同 50）", zorder=4)
    # 各 ML 点标签手动错开，避免 RF/XGBOOST 重叠
    label_off = {"rf": (5, 4, "left"), "xgboost": (-5, 4, "right"),
                 "lightgbm": (6, -2, "left"), "svm": (6, -2, "left")}
    for m in ml:
        dx, dy, ha = label_off.get(m, (4, 4, "left"))
        ax0.annotate(m.upper(), (ml[m]["recall"], ml[m]["precision"]),
                     fontsize=6.5, color=PALETTE["red"], ha=ha,
                     xytext=(dx, dy), textcoords="offset points")
    ax0.set_xlabel("召回率 Recall（正类=merged）", fontsize=9)
    ax0.set_ylabel("精确率 Precision", fontsize=9)
    ax0.set_xlim(0.6, 1.03)
    ax0.set_ylim(0.6, 0.95)
    ax0.legend(fontsize=7.5, loc="lower left")
    ax0.set_title("LLM 挤在「高召回·低精确」角", fontsize=10, pad=8)
    _panel_label(ax0, "a")

    # (b) 准确率对比条：最佳 LLM vs 4 个 ML，基线线
    ax1 = fig.add_subplot(gs[1])
    best_key = max(cls, key=lambda k: cls[k]["accuracy"])
    names = [m.upper() for m in ml] + [f"LLM\n{best_key}"]
    accs = [ml[m]["accuracy"] for m in ml] + [cls[best_key]["accuracy"]]
    colors = [PALETTE["n_mid"]] * len(ml) + [PALETTE["blue_main"]]
    bars = ax1.bar(range(len(names)), accs, color=colors,
                   edgecolor=PALETTE["n_dark"], linewidth=0.8)
    ax1.axhline(br, color=PALETTE["red"], ls="--", lw=1.2)
    ax1.text(-0.4, br - 0.004, f"全判 MERGE 基线 {br:.2f}",
             fontsize=7.5, color=PALETTE["red"], ha="left", va="top")
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, fontsize=7.5, rotation=20, ha="right")
    ax1.set_ylabel("Accuracy", fontsize=9)
    ax1.set_ylim(0.55, 0.85)
    ax1.set_title("训练过的 ML 更强", fontsize=10, pad=8)
    for b, a in zip(bars, accs):
        ax1.text(b.get_x() + b.get_width() / 2, a + 0.006, f"{a:.2f}",
                 ha="center", fontsize=7.5)
    _panel_label(ax1, "b")

    # (c) REJECT 检出能力：每条件真正判 REJECT 且判对的数量 / 18
    ax2 = fig.add_subplot(gs[2])
    df = pd.read_parquet(config.PREDICTIONS_DIR / "classify_predictions.parquet")
    n_true_reject = int((~df.drop_duplicates(["repo", "number"])["true_merge"]).sum())
    conds = [f"{c}_{p}" for c in config.CONTEXTS for p in config.PROMPTS]
    correct_reject = []
    for k in conds:
        c, p = k.split("_")
        g = df[(df["context"] == c) & (df["prompt"] == p)]
        correct_reject.append(int(((g["pred_merge"] == False) & (g["true_merge"] == False)).sum()))
    ax2.bar(range(len(conds)), correct_reject, color=PALETTE["teal"],
            edgecolor=PALETTE["n_dark"], linewidth=0.6)
    ax2.axhline(n_true_reject, color=PALETTE["red"], ls="--", lw=1.2)
    ax2.text(len(conds) - 0.5, n_true_reject - 0.6, f"真实 REJECT 数={n_true_reject}",
             fontsize=7.5, color=PALETTE["red"], ha="right", va="top")
    ax2.set_xticks(range(len(conds)))
    ax2.set_xticklabels(conds, rotation=90, fontsize=6)
    ax2.set_ylabel("正确识别的 REJECT 数", fontsize=9)
    ax2.set_ylim(0, n_true_reject + 1)
    ax2.set_title("几乎抓不到该拒绝的 PR", fontsize=10, pad=8)
    _panel_label(ax2, "c")

    fig.suptitle("LLM 失效模式：倾向一律判 MERGE，仅微超基线；训练过的传统 ML 更均衡且更准",
                 fontsize=11.5, fontweight="bold", y=1.04)
    _save(fig, "fig2_llm_failure")


# --------------------------------------------------------------------------- #
# 图 3：生成质量地形（BLEU / ROUGE-L + 边际 + 长度失配）
# --------------------------------------------------------------------------- #
def fig3_generate_landscape() -> None:
    gen = _load("generate_metrics.json")
    if not gen:
        print("[fig3] 无生成指标，跳过。")
        return
    bleu = _matrix(gen, "bleu")
    rougeL = _matrix(gen, "rougeL")

    fig = plt.figure(figsize=(11, 4.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 1], wspace=0.5)

    # (a) BLEU 热力图
    ax0 = fig.add_subplot(gs[0])
    im0 = _heatmap(ax0, bleu, CMAP_WARM, fmt="{:.2f}")
    ax0.set_title("BLEU（语料级 BLEU-4）", fontsize=10, pad=8)
    cb0 = fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.03)
    cb0.ax.tick_params(labelsize=7)
    _panel_label(ax0, "a")

    # (b) ROUGE-L 热力图
    ax1 = fig.add_subplot(gs[1])
    im1 = _heatmap(ax1, rougeL, CMAP_WARM, fmt="{:.3f}")
    ax1.set_title("ROUGE-L（F-measure）", fontsize=10, pad=8)
    cb1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.03)
    cb1.ax.tick_params(labelsize=7)
    _panel_label(ax1, "b")

    # (c) 长度失配：预测 vs 人类 target 的字符长度分布
    ax2 = fig.add_subplot(gs[2])
    df = pd.read_parquet(config.PREDICTIONS_DIR / "generate_predictions.parquet")
    pred_len = df["pred_comment"].fillna("").str.len()
    tgt_len = df.drop_duplicates(["repo", "number"])["target_comment"].fillna("").str.len()
    bins = np.linspace(0, 500, 26)
    ax2.hist(tgt_len, bins=bins, color=PALETTE["n_mid"], alpha=0.65,
             label=f"人类意见 (中位 {int(tgt_len.median())})")
    ax2.hist(pred_len, bins=bins, color=PALETTE["red"], alpha=0.6,
             label=f"LLM 生成 (中位 {int(pred_len.median())})")
    ax2.axvline(tgt_len.median(), color=PALETTE["n_dark"], ls="--", lw=1)
    ax2.axvline(pred_len.median(), color=PALETTE["red"], ls="--", lw=1)
    ax2.set_xlabel("审查意见长度（字符）", fontsize=9)
    ax2.set_ylabel("频数", fontsize=9)
    ax2.legend(fontsize=7.5)
    ax2.set_title("生成偏短于人类意见", fontsize=10, pad=8)
    _panel_label(ax2, "c")

    fig.suptitle("生成质量地形：BLEU/ROUGE 普遍低（任务本质困难），富上下文无稳定增益",
                 fontsize=11.5, fontweight="bold", y=1.04)
    _save(fig, "fig3_generate_landscape")


# --------------------------------------------------------------------------- #
# 图 4：成本-收益——CoT 的推理时间税
# --------------------------------------------------------------------------- #
def fig4_latency_cost() -> None:
    cls = _load("classify_metrics.json")
    gen = _load("generate_metrics.json")
    if not cls or not gen:
        print("[fig4] 缺指标，跳过。")
        return
    conds = [f"{c}_{p}" for c in config.CONTEXTS for p in config.PROMPTS]

    def prm_of(k):  # 该条件属于哪个 Prompt
        return k.split("_")[1]

    fig = plt.figure(figsize=(11, 4.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1, 1], wspace=0.42)

    # (a) 各条件延迟条形，按 Prompt 着色 —— CoT(P3) 显著高
    ax0 = fig.add_subplot(gs[0])
    cls_lat = [cls[k]["avg_latency"] for k in conds]
    gen_lat = [gen[k]["avg_latency"] for k in conds]
    x = np.arange(len(conds))
    w = 0.4
    bar_colors = [PROMPT_COLORS[config.PROMPTS.index(prm_of(k))] for k in conds]
    ax0.bar(x - w / 2, cls_lat, w, color=bar_colors, edgecolor=PALETTE["n_dark"],
            linewidth=0.5, label="分类")
    ax0.bar(x + w / 2, gen_lat, w, color=bar_colors, alpha=0.5,
            edgecolor=PALETTE["n_dark"], linewidth=0.5, label="生成", hatch="//")
    ax0.set_xticks(x)
    ax0.set_xticklabels(conds, rotation=90, fontsize=6)
    ax0.set_ylabel("平均推理时间 (s)", fontsize=9)
    ax0.set_title("各条件推理时间（P3=CoT 着色最深)", fontsize=10, pad=8)
    ax0.legend(fontsize=7.5, loc="upper left")
    _panel_label(ax0, "a")

    # (b) 分类：延迟 vs F1 —— CoT 点在右侧但不更高
    ax1 = fig.add_subplot(gs[1])
    for p_i, p in enumerate(config.PROMPTS):
        ks = [k for k in conds if prm_of(k) == p]
        ax1.scatter([cls[k]["avg_latency"] for k in ks],
                    [cls[k]["f1"] for k in ks],
                    s=55, color=PROMPT_COLORS[p_i], edgecolor="white",
                    linewidth=0.6, label=config.PROMPT_LABELS[p], zorder=3)
    ax1.set_xlabel("推理时间 (s)", fontsize=9)
    ax1.set_ylabel("F1", fontsize=9)
    ax1.set_title("分类：更慢 ≠ 更准", fontsize=10, pad=8)
    ax1.legend(fontsize=7, loc="lower right")
    _panel_label(ax1, "b")

    # (c) 生成：延迟 vs BLEU
    ax2 = fig.add_subplot(gs[2])
    for p_i, p in enumerate(config.PROMPTS):
        ks = [k for k in conds if prm_of(k) == p]
        ax2.scatter([gen[k]["avg_latency"] for k in ks],
                    [gen[k]["bleu"] for k in ks],
                    s=55, color=PROMPT_COLORS[p_i], edgecolor="white",
                    linewidth=0.6, label=config.PROMPT_LABELS[p], zorder=3)
    ax2.set_xlabel("推理时间 (s)", fontsize=9)
    ax2.set_ylabel("BLEU", fontsize=9)
    ax2.set_title("生成：更慢 ≠ 更好", fontsize=10, pad=8)
    ax2.legend(fontsize=7, loc="upper right")
    _panel_label(ax2, "c")

    fig.suptitle("成本-收益：CoT(P3) 在两任务上都大幅拉高推理时间，却无质量回报",
                 fontsize=11.5, fontweight="bold", y=1.04)
    _save(fig, "fig4_latency_cost")


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def main() -> None:
    print("[figures] 生成出版级组图...")
    fig1_classify_landscape()
    fig2_llm_failure()
    fig3_generate_landscape()
    fig4_latency_cost()
    print("[figures] 完成：4 张组图（每张 .svg + .png）。")


if __name__ == "__main__":
    main()
