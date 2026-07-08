"""切片 8b：图表。设计文档 §8 图表清单。

依赖 evaluate 产出的 metrics JSON 与 ml_baseline_50.json。
产物落 results/figures/：
  classify_f1_heatmap.png        分类 F1 热力图 (4 上下文 × 4 Prompt)
  generate_bleu_heatmap.png      生成 BLEU 热力图
  generate_rougeL_heatmap.png    生成 ROUGE-L 热力图
  classify_context_bars.png      固定 Prompt，4 上下文对比
  classify_prompt_bars.png       固定上下文，4 Prompt 对比
  generate_context_bars.png / generate_prompt_bars.png
  llm_vs_ml.png                  最佳 LLM 条件 vs 实验二四模型（同 50 子集）
  latency_compare.png            各条件平均推理时间
"""
from __future__ import annotations

import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Serif CJK SC",
                                   "WenQuanYi Zen Hei", "AR PL UMing TW",
                                   "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _load(name: str) -> dict:
    path = config.METRICS_DIR / name
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _matrix(metrics: dict, key: str) -> np.ndarray:
    m = np.full((len(config.CONTEXTS), len(config.PROMPTS)), np.nan)
    for i, c in enumerate(config.CONTEXTS):
        for j, p in enumerate(config.PROMPTS):
            cell = metrics.get(f"{c}_{p}")
            if cell and cell.get(key) is not None:
                m[i, j] = cell[key]
    return m


def _heatmap(mat: np.ndarray, title: str, fname: str, fmt: str = "{:.3f}") -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    im = ax.imshow(mat, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(config.PROMPTS)))
    ax.set_xticklabels([f"{p}\n{config.PROMPT_LABELS[p]}" for p in config.PROMPTS])
    ax.set_yticks(range(len(config.CONTEXTS)))
    ax.set_yticklabels([f"{c}\n{config.CONTEXT_LABELS[c]}" for c in config.CONTEXTS])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, fmt.format(mat[i, j]), ha="center", va="center",
                        color="black", fontsize=9)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    out = config.FIGURES_DIR / fname
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  → {out.relative_to(config.PROJECT_ROOT)}")


def _grouped_bars(mat: np.ndarray, group_by: str, metric_name: str, fname: str) -> None:
    """group_by='context'：每组一个上下文，柱为 4 Prompt；'prompt' 反之。"""
    if group_by == "context":
        groups, series = config.CONTEXTS, config.PROMPTS
        glabels = [config.CONTEXT_LABELS[c] for c in groups]
        slabels = [config.PROMPT_LABELS[p] for p in series]
        data = mat  # rows=context, cols=prompt
    else:
        groups, series = config.PROMPTS, config.CONTEXTS
        glabels = [config.PROMPT_LABELS[p] for p in groups]
        slabels = [config.CONTEXT_LABELS[c] for c in series]
        data = mat.T
    x = np.arange(len(groups))
    width = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(8, 5))
    for k in range(len(series)):
        ax.bar(x + k * width, data[:, k], width, label=slabels[k])
    ax.set_xticks(x + width * (len(series) - 1) / 2)
    ax.set_xticklabels(glabels)
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name}：按{'上下文' if group_by=='context' else 'Prompt'}分组")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = config.FIGURES_DIR / fname
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  → {out.relative_to(config.PROJECT_ROOT)}")


def figures_classify() -> None:
    metrics = _load("classify_metrics.json")
    if not metrics:
        print("[figures] 无分类指标，跳过分类图。")
        return
    f1 = _matrix(metrics, "f1")
    _heatmap(f1, "分类 F1（4 上下文 × 4 Prompt）", "classify_f1_heatmap.png")
    acc = _matrix(metrics, "accuracy")
    _heatmap(acc, "分类 Accuracy（4 上下文 × 4 Prompt）", "classify_acc_heatmap.png")
    _grouped_bars(f1, "context", "F1", "classify_context_bars.png")
    _grouped_bars(f1, "prompt", "F1", "classify_prompt_bars.png")
    lat = _matrix(metrics, "avg_latency")
    _heatmap(lat, "分类平均推理时间 (s)", "classify_latency_heatmap.png", fmt="{:.1f}")


def figures_generate() -> None:
    metrics = _load("generate_metrics.json")
    if not metrics:
        print("[figures] 无生成指标，跳过生成图。")
        return
    bleu = _matrix(metrics, "bleu")
    _heatmap(bleu, "生成 BLEU（4 上下文 × 4 Prompt）", "generate_bleu_heatmap.png",
             fmt="{:.1f}")
    rl = _matrix(metrics, "rougeL")
    _heatmap(rl, "生成 ROUGE-L（4 上下文 × 4 Prompt）", "generate_rougeL_heatmap.png")
    _grouped_bars(bleu, "context", "BLEU", "generate_context_bars.png")
    _grouped_bars(bleu, "prompt", "BLEU", "generate_prompt_bars.png")


def figure_llm_vs_ml() -> None:
    cls = _load("classify_metrics.json")
    ml = _load("ml_baseline_50.json")
    if not cls or not ml:
        print("[figures] 缺分类或 ML 基线，跳过 LLM vs ML。")
        return
    best_key = max(cls, key=lambda k: (cls[k].get("f1") or -1))
    best = cls[best_key]
    names = list(ml.keys()) + [f"LLM\n({best_key})"]
    f1s = [ml[n]["f1"] for n in ml] + [best["f1"]]
    accs = [ml[n]["accuracy"] for n in ml] + [best["accuracy"]]

    x = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, accs, width, label="Accuracy")
    ax.bar(x + width / 2, f1s, width, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("分数")
    ax.set_title("分类：最佳 LLM 条件 vs 实验二传统 ML（同 50 子集）")
    ax.legend()
    for i, (a, f) in enumerate(zip(accs, f1s)):
        ax.text(i - width / 2, a + 0.01, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(i + width / 2, f + 0.01, f"{f:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    out = config.FIGURES_DIR / "llm_vs_ml.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  → {out.relative_to(config.PROJECT_ROOT)}")


def figure_latency() -> None:
    cls = _load("classify_metrics.json")
    gen = _load("generate_metrics.json")
    conds = [f"{c}_{p}" for c in config.CONTEXTS for p in config.PROMPTS]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(conds))
    width = 0.4
    if cls:
        ax.bar(x - width / 2, [cls.get(k, {}).get("avg_latency", 0) for k in conds],
               width, label="分类")
    if gen:
        ax.bar(x + width / 2, [gen.get(k, {}).get("avg_latency", 0) for k in conds],
               width, label="生成")
    ax.set_xticks(x)
    ax.set_xticklabels(conds, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("平均推理时间 (s)")
    ax.set_title("各条件平均推理时间")
    ax.legend()
    fig.tight_layout()
    out = config.FIGURES_DIR / "latency_compare.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  → {out.relative_to(config.PROJECT_ROOT)}")


def main() -> None:
    print("[figures] 生成图表...")
    figures_classify()
    figures_generate()
    figure_llm_vs_ml()
    figure_latency()
    print("[figures] 完成。")


if __name__ == "__main__":
    main()
