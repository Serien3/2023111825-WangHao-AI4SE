"""EDA 分析：统计 + 专业可视化（对应实验一步骤五）。

生成 Nature 期刊风格图表到 data/figures/，并打印统计摘要。
所有图表使用英文标签（避免 matplotlib 中文字体缺失），同时附详细中文注释。

图表清单（兼容旧文件名，同时新增综合概览图）：
  1. merge_vs_nonmerge.png    Merge/Non-merge 分布（柱状 + 饼图 + 跨仓库 merge 率）
  2. review_comment_dist.png  Review comment 分布（零膨胀直方图 + 仓库箱线图）
  3. label_dist.png           Top-15 标签频次（横向柱状，带数值标注）
  4. reviewer_dist.png        Reviewer 数量分布（柱状 + 按 merge 状态分组）
  5. pr_size_dist.png         PR 大小分布（对数直方图 + 百分位线 + 仓库小提琴图）
  6. ai_vs_human.png          AI 参与分析（信号类型饼图 + merge 率对比 + AI reviewer）
  7. cross_repo.png           跨仓库多维对比（merge 率 / 评论 / PR 大小三联图）
  8. eda_overview.png         综合概览（3×3 九宫格，实验报告一图覆全）

用法：
    uv run python -m src.exp1.analyze
"""
from __future__ import annotations

# ── 非交互后端（headless 环境必须放在 pyplot 之前）
import matplotlib
matplotlib.use("Agg")

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns
from collections import Counter

from . import config

# ---------------------------------------------------------------------------
# Nature 风格 rcParams（参照 nature-figure skill）
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":         "sans-serif",
    "font.sans-serif":     ["Arial", "DejaVu Sans", "Liberation Sans"],
    "svg.fonttype":        "none",       # SVG 内保留可编辑文字节点
    "pdf.fonttype":        42,           # PDF 内嵌 TrueType 字体
    "font.size":           9,
    "axes.spines.right":   False,
    "axes.spines.top":     False,
    "axes.linewidth":      1.0,
    "xtick.major.width":   0.8,
    "ytick.major.width":   0.8,
    "xtick.major.size":    3,
    "ytick.major.size":    3,
    "legend.frameon":      False,
    "legend.fontsize":     8,
    "figure.dpi":          150,
})

# ---------------------------------------------------------------------------
# 调色板（与 nature-figure skill 的 PALETTE 保持一致）
# ---------------------------------------------------------------------------
PALETTE = {
    "blue_main":      "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_3":        "#8BCF8B",
    "red_strong":     "#B64342",
    "teal":           "#42949E",
    "violet":         "#9A4D8E",
    "gold":           "#E8A020",
    "neutral_light":  "#CFCECE",
    "neutral_mid":    "#767676",
    "neutral_dark":   "#4D4D4D",
}

# 5 个仓库的专用颜色（跨仓库图表保持一致）
REPO_COLORS = {
    "apache/airflow":          "#0F4D92",
    "home-assistant/core":     "#42949E",
    "pandas-dev/pandas":       "#8BCF8B",
    "django/django":           "#E8A020",
    "huggingface/transformers":"#9A4D8E",
}
REPO_SHORT = {
    "apache/airflow":          "airflow",
    "home-assistant/core":     "home-asst",
    "pandas-dev/pandas":       "pandas",
    "django/django":           "django",
    "huggingface/transformers":"HF-transformers",
}

FIG = config.FIGURES_DIR


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _load() -> pd.DataFrame:
    """读取主表 prs.parquet，返回 DataFrame。"""
    df = pd.read_parquet(config.PROCESSED_DIR / "prs.parquet")
    df["pr_size"] = df["additions"] + df["deletions"]
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["closed_at"]  = pd.to_datetime(df["closed_at"])
    df["lifetime_h"] = (df["closed_at"] - df["created_at"]).dt.total_seconds() / 3600
    df["repo_short"] = df["repo"].map(REPO_SHORT)
    return df


def _panel_label(ax: plt.Axes, label: str, x: float = -0.10, y: float = 1.03) -> None:
    """在 Axes 左上角添加 Nature 风格 (a)(b)(c) 分图标签。"""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom", ha="left")


def _annotate_bar(ax: plt.Axes, bars, fmt: str = "{:.0f}",
                  va: str = "bottom", offset: float = 0, fontsize: int = 8) -> None:
    """在每根柱子顶部/末尾添加数值标注。"""
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + offset,
                fmt.format(h),
                ha="center", va=va, fontsize=fontsize, color=PALETTE["neutral_dark"],
            )


def _save(fig: plt.Figure, name: str, dpi: int = 300) -> None:
    """保存为 PNG（300 dpi）和 SVG，然后关闭 figure。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # 忽略含饼图时 tight_layout 的兼容告警
        fig.tight_layout()
    fig.savefig(FIG / f"{name}.png",  dpi=dpi, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.svg",  bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 图 1：Merge 分布 —— 数量柱状 + 比例饼图 + 跨仓库 merge 率
# ---------------------------------------------------------------------------
def fig_merge(prs: pd.DataFrame) -> None:
    """Merge Prediction 的标签分布，是全课程的核心因变量 y。

    三联图揭示：(a) 整体正负样本量；(b) 类别不平衡程度；
    (c) 各仓库 merge 率差异 —— 呼应指导书思考题 5（类别不平衡）与拓展目标（社区差异）。
    """
    merged = int(prs["is_merged"].sum())
    non    = len(prs) - merged

    fig = plt.figure(figsize=(12, 4.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.35], wspace=0.35)
    ax1, ax2, ax3 = (fig.add_subplot(gs[0, i]) for i in range(3))

    # (a) 数量柱状
    bars = ax1.bar(["Merged", "Non-merged"], [merged, non],
                   color=[PALETTE["green_3"], PALETTE["red_strong"]],
                   edgecolor="black", linewidth=0.8, width=0.6)
    _annotate_bar(ax1, bars, offset=merged * 0.01)
    ax1.set_ylabel("PR count")
    ax1.set_title("Merge outcome (count)", fontsize=10)
    ax1.set_ylim(0, merged * 1.15)
    _panel_label(ax1, "a")

    # (b) 比例饼图（甜甜圈）
    wedges, _, autotexts = ax2.pie(
        [merged, non], labels=["Merged", "Non-merged"],
        autopct=lambda p: f"{p:.1f}%\n({int(round(p*len(prs)/100))})",
        colors=[PALETTE["green_3"], PALETTE["red_strong"]],
        startangle=90, counterclock=False,
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=1.5),
        textprops=dict(fontsize=8),
    )
    for t in autotexts:
        t.set_fontsize(8); t.set_color("white"); t.set_fontweight("bold")
    ax2.text(0, 0, f"{merged/len(prs):.0%}\nmerged", ha="center", va="center",
             fontsize=11, fontweight="bold", color=PALETTE["neutral_dark"])
    ax2.set_title("Class balance (ratio ≈ 2:1)", fontsize=10)
    _panel_label(ax2, "b", x=-0.05)

    # (c) 各仓库 merge 率（排序 + 全局均值参考线）
    rate = prs.groupby("repo")["is_merged"].mean().sort_values()
    ncnt = prs.groupby("repo").size()
    colors = [REPO_COLORS[r] for r in rate.index]
    ypos = np.arange(len(rate))
    bars = ax3.barh(ypos, rate.values, color=colors, edgecolor="black",
                    linewidth=0.8, height=0.62)
    ax3.axvline(prs["is_merged"].mean(), color=PALETTE["neutral_dark"],
                linestyle="--", linewidth=1.1)
    ax3.text(prs["is_merged"].mean(), len(rate) - 0.3,
             f" overall {prs['is_merged'].mean():.0%}", fontsize=7.5,
             color=PALETTE["neutral_dark"], va="center")
    for i, (r, v) in enumerate(rate.items()):
        ax3.text(v + 0.01, i, f"{v:.0%}  (n={ncnt[r]})", va="center",
                 fontsize=7.5, color=PALETTE["neutral_dark"])
    ax3.set_yticks(ypos)
    ax3.set_yticklabels([REPO_SHORT[r] for r in rate.index], fontsize=8)
    ax3.set_xlim(0, 1.0)
    ax3.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax3.set_xlabel("Merge rate")
    ax3.set_title("Merge rate varies 48%–90% across repos", fontsize=10)
    _panel_label(ax3, "c", x=-0.22)

    fig.suptitle("Merge Prediction target: label distribution and cross-repo imbalance",
                 fontsize=11, fontweight="bold", y=1.02)
    _save(fig, "merge_vs_nonmerge")


# ---------------------------------------------------------------------------
# 图 2：Review comment 分布 —— 零膨胀直方图 + 仓库箱线图
# ---------------------------------------------------------------------------
def fig_review_comments(prs: pd.DataFrame) -> None:
    """inline review comment 数量分布。

    (a) 整体直方图（截尾 p98）凸显严重零膨胀 —— 实验三只能用有评论的 PR；
    (b) 各仓库有评论 PR 占比 —— 揭示不同社区的审查密度差异。
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2),
                                   gridspec_kw={"width_ratios": [1.1, 1]})

    # (a) 零膨胀直方图
    x = prs["num_review_comments"]
    p98 = int(x.quantile(0.98))
    data = x.clip(upper=p98)
    ax1.hist(data, bins=range(0, p98 + 2), color=PALETTE["blue_secondary"],
             edgecolor="white", linewidth=0.6, align="left")
    zero_frac = (x == 0).mean()
    ax1.axvline(x.median(), color=PALETTE["red_strong"], linestyle="--", linewidth=1.1)
    ax1.text(0.55, 0.92,
             f"{zero_frac:.0%} of PRs have\nZERO inline comments\n"
             f"median={int(x.median())},  p90={int(x.quantile(.9))},  max={int(x.max())}",
             transform=ax1.transAxes, fontsize=8, va="top",
             bbox=dict(boxstyle="round", fc="#FdF3E7", ec=PALETTE["gold"], lw=0.8))
    ax1.set_xlabel(f"# inline review comments per PR (clipped at p98={p98})")
    ax1.set_ylabel("PR count")
    ax1.set_title("Inline comments are strongly zero-inflated", fontsize=10)
    _panel_label(ax1, "a")

    # (b) 各仓库“有评论 PR”占比 + 均值
    g = prs.groupby("repo")
    frac_nonzero = (g["num_review_comments"].apply(lambda s: (s > 0).mean())
                    .sort_values(ascending=False))
    mean_c = g["num_review_comments"].mean()
    colors = [REPO_COLORS[r] for r in frac_nonzero.index]
    ypos = np.arange(len(frac_nonzero))
    bars = ax2.barh(ypos, frac_nonzero.values, color=colors, edgecolor="black",
                    linewidth=0.8, height=0.62)
    for i, r in enumerate(frac_nonzero.index):
        ax2.text(frac_nonzero[r] + 0.008, i,
                 f"{frac_nonzero[r]:.0%}  (μ={mean_c[r]:.1f})", va="center",
                 fontsize=7.5, color=PALETTE["neutral_dark"])
    ax2.set_yticks(ypos)
    ax2.set_yticklabels([REPO_SHORT[r] for r in frac_nonzero.index], fontsize=8)
    ax2.set_xlim(0, max(frac_nonzero.values) * 1.35)
    ax2.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax2.set_xlabel("Share of PRs with ≥1 inline comment")
    ax2.set_title("Review density differs by community", fontsize=10)
    _panel_label(ax2, "b", x=-0.20)

    fig.suptitle("Review Comment Generation source: sparsity of inline comments",
                 fontsize=11, fontweight="bold", y=1.02)
    _save(fig, "review_comment_dist")


# ---------------------------------------------------------------------------
# 图 3：Top-15 标签频次
# ---------------------------------------------------------------------------
def fig_labels(prs: pd.DataFrame) -> None:
    """PR 标签（label）频次分布，反映各仓库的分类/流程标记体系。"""
    all_labels: list[str] = []
    for s in prs["labels"]:
        if s:
            all_labels.extend(s.split("|"))
    if not all_labels:
        return
    top = pd.Series(all_labels).value_counts().head(15)

    fig, ax = plt.subplots(figsize=(8.5, 6))
    ypos = np.arange(len(top))[::-1]
    # 颜色随频次深浅渐变（单一蓝色族）
    norm = plt.Normalize(top.values.min(), top.values.max())
    cmap = sns.light_palette(PALETTE["blue_main"], as_cmap=True)
    colors = [cmap(norm(v)) for v in top.values]
    bars = ax.barh(ypos, top.values, color=colors, edgecolor="black",
                   linewidth=0.6, height=0.72)
    for y, v in zip(ypos, top.values):
        ax.text(v + top.values.max() * 0.01, y, str(v), va="center",
                fontsize=8, color=PALETTE["neutral_dark"])
    ax.set_yticks(ypos)
    ax.set_yticklabels(top.index, fontsize=8.5)
    ax.set_xlim(0, top.values.max() * 1.12)
    ax.set_xlabel("Frequency (number of PRs)")
    ax.set_title(f"Top 15 PR labels  (from {len(set(all_labels))} distinct labels, "
                 f"{len(all_labels)} label uses)", fontsize=10, fontweight="bold")
    _save(fig, "label_dist")


# ---------------------------------------------------------------------------
# 图 4：Reviewer 数量分布
# ---------------------------------------------------------------------------
def fig_reviewers(prs: pd.DataFrame) -> None:
    """参与审查的不同 reviewer 数量分布，并按 merge 状态分组。

    观察：reviewer 越多的 PR 是否越容易 merge？（审查投入 vs 结果）
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # (a) 总体分布
    vc = prs["num_reviewers"].value_counts().sort_index()
    bars = ax1.bar(vc.index.astype(int).astype(str), vc.values,
                   color=PALETTE["gold"], edgecolor="black", linewidth=0.8, width=0.7)
    _annotate_bar(ax1, bars, offset=vc.values.max() * 0.01)
    ax1.set_xlabel("# distinct reviewers per PR")
    ax1.set_ylabel("PR count")
    ax1.set_title(f"Reviewer count (mean={prs['num_reviewers'].mean():.2f})", fontsize=10)
    _panel_label(ax1, "a")

    # (b) 按 reviewer 数分组的 merge 率
    tmp = prs.copy()
    tmp["rev_bucket"] = tmp["num_reviewers"].clip(upper=4).astype(int)
    grp = tmp.groupby("rev_bucket")["is_merged"].agg(["mean", "size"])
    labels = [f"{i}" if i < 4 else "4+" for i in grp.index]
    bars = ax2.bar(labels, grp["mean"], color=PALETTE["blue_secondary"],
                   edgecolor="black", linewidth=0.8, width=0.7)
    for i, (m, n) in enumerate(zip(grp["mean"], grp["size"])):
        ax2.text(i, m + 0.02, f"{m:.0%}\n(n={n})", ha="center", va="bottom", fontsize=7.5,
                 color=PALETTE["neutral_dark"])
    ax2.axhline(prs["is_merged"].mean(), color=PALETTE["red_strong"],
                linestyle="--", linewidth=1.1, label=f"overall {prs['is_merged'].mean():.0%}")
    ax2.set_ylim(0, 1.05)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax2.set_xlabel("# distinct reviewers per PR")
    ax2.set_ylabel("Merge rate")
    ax2.set_title("More reviewers → higher merge rate", fontsize=10)
    ax2.legend(loc="lower right")
    _panel_label(ax2, "b")

    _save(fig, "reviewer_dist")


# ---------------------------------------------------------------------------
# 图 5：PR 大小分布 —— 对数直方图 + 百分位 + 仓库小提琴图
# ---------------------------------------------------------------------------
def fig_pr_size(prs: pd.DataFrame) -> None:
    """PR 改动规模（additions+deletions）分布。

    (a) 对数直方图 + 关键百分位线，凸显极端长尾（中位数 32 行，最大近百万行）；
    (b) 各仓库小提琴图 —— pandas 因附带大量测试/文档改动而整体偏大。
    """
    size = prs["pr_size"].clip(lower=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4),
                                   gridspec_kw={"width_ratios": [1, 1.15]})

    # (a) 对数直方图
    bins = np.logspace(0, np.log10(size.max()), 45)
    ax1.hist(size, bins=bins, color=PALETTE["blue_main"],
             edgecolor="white", linewidth=0.4)
    ax1.set_xscale("log")
    for q, ls, c in [(0.5, "-", PALETTE["red_strong"]),
                     (0.9, "--", PALETTE["gold"]),
                     (0.99, ":", PALETTE["neutral_dark"])]:
        val = size.quantile(q)
        ax1.axvline(val, color=c, linestyle=ls, linewidth=1.3)
        ax1.text(val, ax1.get_ylim()[1] * (0.95 - 0.10 * (q > 0.5) - 0.10 * (q > 0.9)),
                 f" p{int(q*100)}={int(val)}", color=c, fontsize=7.5, rotation=0)
    ax1.set_xlabel("Changed lines = additions + deletions (log scale)")
    ax1.set_ylabel("PR count")
    ax1.set_title("Heavy-tailed PR size (spans 6 orders of magnitude)", fontsize=10)
    _panel_label(ax1, "a")

    # (b) 各仓库小提琴图（log 尺度）
    order = (prs.groupby("repo")["pr_size"].median().sort_values().index.tolist())
    data = [np.log10(prs.loc[prs["repo"] == r, "pr_size"].clip(lower=1)) for r in order]
    parts = ax2.violinplot(data, showmedians=True, showextrema=False,
                           orientation="horizontal", widths=0.85)
    for pc, r in zip(parts["bodies"], order):
        pc.set_facecolor(REPO_COLORS[r]); pc.set_alpha(0.75); pc.set_edgecolor("black")
        pc.set_linewidth(0.6)
    parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(1.2)
    for i, r in enumerate(order):
        med = prs.loc[prs["repo"] == r, "pr_size"].median()
        ax2.text(np.log10(med), i + 1.32, f"med={int(med)}", ha="center",
                 fontsize=7, color=PALETTE["neutral_dark"])
    ax2.set_yticks(range(1, len(order) + 1))
    ax2.set_yticklabels([REPO_SHORT[r] for r in order], fontsize=8)
    ax2.set_xlabel("PR size (log10 changed lines)")
    ax2.set_title("Size distribution by repo (pandas skews largest)", fontsize=10)
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"$10^{{{int(v)}}}$"))
    _panel_label(ax2, "b", x=-0.16)

    fig.suptitle("PR size: a heavy-tailed feature — log scale is mandatory",
                 fontsize=11, fontweight="bold", y=1.02)
    _save(fig, "pr_size_dist")


# ---------------------------------------------------------------------------
# 图 6：AI 参与分析
# ---------------------------------------------------------------------------
def fig_ai_vs_human(prs: pd.DataFrame) -> None:
    """AI 生成代码 / AI reviewer 的分布与影响（实验五的数据基础）。

    (a) AI 代码信号类型构成（强/弱信号）；
    (b) AI 代码 vs 人类代码 的 merge 率对比；
    (c) 有/无 AI reviewer 的 merge 率对比（AI reviewer 与高 merge 率强相关）。
    """
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    # (a) AI 代码信号类型（强弱分色）
    sig = Counter()
    for s in prs["ai_code_signals"].dropna():
        for tok in str(s).split("|"):
            if tok:
                sig[tok.split(":")[0]] += 1
    strong = {"coauthor_commit", "coauthor_body", "author_bot"}
    order = sorted(sig, key=lambda k: -sig[k])
    vals = [sig[k] for k in order]
    colors = [PALETTE["blue_main"] if k in strong else PALETTE["neutral_light"]
              for k in order]
    ax = axes[0]
    wedges, _, autotexts = ax.pie(
        vals, labels=[k.replace("_", "\n") for k in order],
        autopct=lambda p: f"{int(round(p*sum(vals)/100))}",
        colors=colors, startangle=90, counterclock=False,
        wedgeprops=dict(edgecolor="white", linewidth=1.2),
        textprops=dict(fontsize=7.5),
    )
    for t in autotexts:
        t.set_fontsize(8); t.set_fontweight("bold")
    ax.set_title("is_ai_code signal mix\n(blue = strong signals)", fontsize=9.5)
    _panel_label(ax, "a", x=-0.08)

    # (b) AI vs Human merge 率
    ax = axes[1]
    grp = prs.groupby("is_ai_code")["is_merged"].agg(["mean", "size"])
    cats = ["Human code", "AI code"]
    means = [grp.loc[False, "mean"], grp.loc[True, "mean"]]
    sizes = [grp.loc[False, "size"], grp.loc[True, "size"]]
    bars = ax.bar(cats, means, color=[PALETTE["neutral_mid"], PALETTE["red_strong"]],
                  edgecolor="black", linewidth=0.8, width=0.6)
    for i, (m, n) in enumerate(zip(means, sizes)):
        ax.text(i, m + 0.02, f"{m:.1%}\n(n={n})", ha="center", va="bottom",
                fontsize=8, color=PALETTE["neutral_dark"])
    ax.axhline(prs["is_merged"].mean(), color=PALETTE["neutral_dark"],
               linestyle="--", linewidth=1.0)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_ylabel("Merge rate")
    ax.set_title("AI-code PRs merge at similar rate", fontsize=9.5)
    _panel_label(ax, "b")

    # (c) AI reviewer merge 率
    ax = axes[2]
    grp = prs.groupby("has_ai_reviewer")["is_merged"].agg(["mean", "size"])
    cats = ["No AI reviewer", "Has AI reviewer"]
    means = [grp.loc[False, "mean"], grp.loc[True, "mean"]]
    sizes = [grp.loc[False, "size"], grp.loc[True, "size"]]
    bars = ax.bar(cats, means, color=[PALETTE["neutral_mid"], PALETTE["green_3"]],
                  edgecolor="black", linewidth=0.8, width=0.6)
    for i, (m, n) in enumerate(zip(means, sizes)):
        ax.text(i, m + 0.02, f"{m:.1%}\n(n={n})", ha="center", va="bottom",
                fontsize=8, color=PALETTE["neutral_dark"])
    ax.axhline(prs["is_merged"].mean(), color=PALETTE["neutral_dark"],
               linestyle="--", linewidth=1.0)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_ylabel("Merge rate")
    ax.set_title("AI-reviewed PRs merge far more often", fontsize=9.5)
    _panel_label(ax, "c")

    fig.suptitle("AI participation: signal composition and association with merge outcome",
                 fontsize=11, fontweight="bold", y=1.02)
    _save(fig, "ai_vs_human")


# ---------------------------------------------------------------------------
# 图 7：跨仓库多维对比
# ---------------------------------------------------------------------------
def fig_cross_repo(prs: pd.DataFrame) -> pd.DataFrame:
    """跨仓库多维对比（呼应实验一拓展目标：分析社区间审查流程差异）。"""
    g = prs.groupby("repo")
    stats = pd.DataFrame({
        "merge_rate":          g["is_merged"].mean(),
        "avg_review_comments": g["num_review_comments"].mean(),
        "median_pr_size":      g["pr_size"].median(),
        "avg_reviewers":       g["num_reviewers"].mean(),
        "ai_code_share":       g["is_ai_code"].mean(),
        "ai_rev_share":        g["has_ai_reviewer"].mean(),
        "pr_count":            g.size(),
    }).sort_values("merge_rate")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    order = stats.index.tolist()
    colors = [REPO_COLORS[r] for r in order]
    short = [REPO_SHORT[r] for r in order]

    def _barh(ax, series, title, fmt, xlabel, pct=False):
        y = np.arange(len(series))
        bars = ax.barh(y, series.values, color=colors, edgecolor="black",
                       linewidth=0.8, height=0.66)
        for i, v in enumerate(series.values):
            ax.text(v + series.values.max() * 0.02, i, fmt.format(v), va="center",
                    fontsize=8, color=PALETTE["neutral_dark"])
        ax.set_yticks(y); ax.set_yticklabels(short, fontsize=8.5)
        ax.set_xlim(0, series.values.max() * 1.25)
        if pct:
            ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.set_xlabel(xlabel); ax.set_title(title, fontsize=10)

    _barh(axes[0], stats["merge_rate"], "Merge rate", "{:.0%}", "merge rate", pct=True)
    axes[0].axvline(prs["is_merged"].mean(), color=PALETTE["neutral_dark"],
                    linestyle="--", linewidth=1.0)
    _panel_label(axes[0], "a", x=-0.28)
    _barh(axes[1], stats["avg_review_comments"], "Avg inline comments / PR",
          "{:.2f}", "avg review comments")
    _panel_label(axes[1], "b", x=-0.20)
    _barh(axes[2], stats["median_pr_size"], "Median PR size (lines)",
          "{:.0f}", "median changed lines")
    _panel_label(axes[2], "c", x=-0.20)

    fig.suptitle("Cross-repository comparison: review culture differs by community",
                 fontsize=11, fontweight="bold", y=1.02)
    _save(fig, "cross_repo")
    return stats


# ---------------------------------------------------------------------------
# 图 8：综合概览九宫格（实验报告一图总览）
# ---------------------------------------------------------------------------
def fig_overview(prs: pd.DataFrame) -> None:
    """3×3 综合概览：一张图覆盖指导书步骤五要求的全部统计维度。"""
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))

    # (1) Merge 饼图
    ax = axes[0, 0]
    m, n = int(prs["is_merged"].sum()), int((~prs["is_merged"]).sum())
    ax.pie([m, n], labels=["Merged", "Non-merged"], autopct="%1.1f%%",
           colors=[PALETTE["green_3"], PALETTE["red_strong"]], startangle=90,
           wedgeprops=dict(width=0.42, edgecolor="white"), textprops=dict(fontsize=8))
    ax.set_title("Merge outcome", fontsize=9.5, fontweight="bold")

    # (2) Review comment 直方图
    ax = axes[0, 1]
    x = prs["num_review_comments"]; p98 = int(x.quantile(0.98))
    ax.hist(x.clip(upper=p98), bins=range(0, p98 + 2), color=PALETTE["blue_secondary"],
            edgecolor="white", linewidth=0.4, align="left")
    ax.set_title("Inline comments/PR", fontsize=9.5, fontweight="bold")
    ax.set_xlabel(f"count (clip p98={p98})"); ax.set_ylabel("PRs")

    # (3) PR 大小对数直方图
    ax = axes[0, 2]
    size = prs["pr_size"].clip(lower=1)
    ax.hist(size, bins=np.logspace(0, np.log10(size.max()), 35),
            color=PALETTE["blue_main"], edgecolor="white", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_title("PR size (log)", fontsize=9.5, fontweight="bold")
    ax.set_xlabel("changed lines"); ax.set_ylabel("PRs")

    # (4) Reviewer 分布
    ax = axes[1, 0]
    vc = prs["num_reviewers"].value_counts().sort_index()
    ax.bar(vc.index.astype(int).astype(str), vc.values, color=PALETTE["gold"],
           edgecolor="black", linewidth=0.6, width=0.7)
    ax.set_title("Reviewers/PR", fontsize=9.5, fontweight="bold")
    ax.set_xlabel("# reviewers"); ax.set_ylabel("PRs")

    # (5) review_decision 分布
    ax = axes[1, 1]
    dc = prs["review_decision"].value_counts()
    ax.bar(range(len(dc)), dc.values, color=PALETTE["teal"], edgecolor="black",
           linewidth=0.6, width=0.7)
    ax.set_xticks(range(len(dc)))
    ax.set_xticklabels(dc.index, rotation=25, ha="right", fontsize=7.5)
    ax.set_title("Review decision", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("PRs")

    # (6) author_association 分布
    ax = axes[1, 2]
    ac = prs["author_association"].value_counts()
    ax.barh(range(len(ac))[::-1], ac.values, color=PALETTE["violet"],
            edgecolor="black", linewidth=0.6, height=0.7)
    ax.set_yticks(range(len(ac))[::-1]); ax.set_yticklabels(ac.index, fontsize=7.5)
    ax.set_title("Author association", fontsize=9.5, fontweight="bold")
    ax.set_xlabel("PRs")

    # (7) 各仓库 PR 数
    ax = axes[2, 0]
    rc = prs["repo"].value_counts()
    ax.bar(range(len(rc)), rc.values, color=[REPO_COLORS[r] for r in rc.index],
           edgecolor="black", linewidth=0.6, width=0.7)
    ax.set_xticks(range(len(rc)))
    ax.set_xticklabels([REPO_SHORT[r] for r in rc.index], rotation=25, ha="right", fontsize=7)
    ax.set_title("PRs per repo", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("PRs")

    # (8) AI 标签
    ax = axes[2, 1]
    ai = int(prs["is_ai_code"].sum()); air = int(prs["has_ai_reviewer"].sum())
    xs = np.arange(2)
    ax.bar(xs - 0.2, [len(prs) - ai, len(prs) - air], width=0.4,
           color=PALETTE["neutral_light"], edgecolor="black", linewidth=0.6, label="No")
    ax.bar(xs + 0.2, [ai, air], width=0.4, color=PALETTE["red_strong"],
           edgecolor="black", linewidth=0.6, label="Yes")
    ax.set_xticks(xs); ax.set_xticklabels(["is_ai_code", "has_ai_reviewer"], fontsize=8)
    ax.set_title("AI participation", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("PRs"); ax.legend(fontsize=7)

    # (9) PR 生命周期（关闭耗时，对数）
    ax = axes[2, 2]
    lt = prs["lifetime_h"].clip(lower=0.05)
    ax.hist(lt, bins=np.logspace(np.log10(lt.min()), np.log10(lt.max()), 35),
            color=PALETTE["blue_secondary"], edgecolor="white", linewidth=0.3)
    ax.set_xscale("log")
    ax.axvline(lt.median(), color=PALETTE["red_strong"], linestyle="--", linewidth=1.1)
    ax.text(lt.median(), ax.get_ylim()[1] * 0.9, f" median={lt.median():.1f}h",
            fontsize=7, color=PALETTE["red_strong"])
    ax.set_title("PR lifetime (open→close)", fontsize=9.5, fontweight="bold")
    ax.set_xlabel("hours (log)"); ax.set_ylabel("PRs")

    for i, ax in enumerate(axes.flat):
        _panel_label(ax, chr(ord("a") + i), x=-0.12, y=1.05)

    fig.suptitle("Exp-1 dataset overview: 1,706 PRs from 5 repositories",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(FIG / "eda_overview.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "eda_overview.svg", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 统计摘要（控制台打印）
# ---------------------------------------------------------------------------
def print_summary(prs: pd.DataFrame) -> None:
    print("\n" + "=" * 62)
    print(f"数据集规模: {len(prs)} 个有效 PR，来自 {prs['repo'].nunique()} 个仓库")
    print("=" * 62)
    print(f"Merge 分布: merged={int(prs['is_merged'].sum())}, "
          f"non-merged={int((~prs['is_merged']).sum())}, "
          f"merge_rate={prs['is_merged'].mean():.1%}")
    print(f"AI 生成代码 PR: {int(prs['is_ai_code'].sum())} ({prs['is_ai_code'].mean():.1%})")
    print(f"含 AI reviewer PR: {int(prs['has_ai_reviewer'].sum())} "
          f"({prs['has_ai_reviewer'].mean():.1%})")
    print(f"定向补采 PR: {int(prs['oversampled'].sum())}")
    print(f"平均 review comments: {prs['num_review_comments'].mean():.2f} "
          f"(有评论 PR 占比 {(prs['num_review_comments'] > 0).mean():.1%})")
    print(f"平均 reviewers: {prs['num_reviewers'].mean():.2f}")
    print(f"PR 大小(增+删): 中位数 {prs['pr_size'].median():.0f} 行, "
          f"p90={prs['pr_size'].quantile(.9):.0f}, max={prs['pr_size'].max():.0f}")
    print(f"PR 生命周期: 中位数 {prs['lifetime_h'].median():.1f} 小时")
    print("\n各仓库 PR 数与 merge 率:")
    for repo, sub in prs.groupby("repo"):
        print(f"  {repo:32s} n={len(sub):4d}  merge_rate={sub['is_merged'].mean():.1%}"
              f"  ai_code={int(sub['is_ai_code'].sum())}"
              f"  ai_rev={int(sub['has_ai_reviewer'].sum())}")


def main() -> None:
    sns.set_style("ticks")
    prs = _load()
    fig_merge(prs)
    fig_review_comments(prs)
    fig_labels(prs)
    fig_reviewers(prs)
    fig_pr_size(prs)
    fig_ai_vs_human(prs)
    fig_cross_repo(prs)
    fig_overview(prs)
    print_summary(prs)
    print(f"\n图表已保存到 {FIG}")
    print("生成: merge_vs_nonmerge, review_comment_dist, label_dist, reviewer_dist,")
    print("      pr_size_dist, ai_vs_human, cross_repo, eda_overview  (各含 .png + .svg)")


if __name__ == "__main__":
    main()
