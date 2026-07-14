"""R11：P* 选取脚本——从实验四/五 AI 预测指标经验选取旧最优 prompt，可复现。

不硬编码：读实验五 AI 组各 prompt 跨上下文均值，分类看 accuracy/f1，生成看 bleu/rougeL，
取最高者作 P*。结果写入 results/metrics/pstar_selection.json，供报告引用避免事后挑选偏差。

用法：uv run python -m src.select_pstar
"""
from __future__ import annotations

import json
from collections import defaultdict

from . import config


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def select() -> dict:
    clf = _load(config.EXP5_METRICS_DIR / "llm_classify_metrics.json")["ai"]
    gen = _load(config.EXP5_METRICS_DIR / "llm_generate_metrics.json")["ai"]

    def avg_by_prompt(metrics: dict, keys: list[str]) -> dict:
        agg = defaultdict(lambda: defaultdict(float))
        cnt = defaultdict(int)
        for cond, m in metrics.items():
            _, prm = cond.split("_")
            cnt[prm] += 1
            for k in keys:
                v = m.get(k)
                if isinstance(v, (int, float)):
                    agg[prm][k] += v
        return {p: {k: agg[p][k] / cnt[p] for k in keys} for p in agg}

    clf_avg = avg_by_prompt(clf, ["accuracy", "f1", "recall"])
    gen_avg = avg_by_prompt(gen, ["bleu", "rougeL"])

    # 分类：主看 accuracy（旧口径，f1 tie-break）；生成：主看 bleu（rougeL tie-break）
    clf_star = max(clf_avg, key=lambda p: (clf_avg[p]["accuracy"], clf_avg[p]["f1"]))
    gen_star = max(gen_avg, key=lambda p: (gen_avg[p]["bleu"], gen_avg[p]["rougeL"]))

    out = {
        "classify": {"selected": clf_star, "by_prompt_avg": clf_avg,
                     "criterion": "max mean accuracy (f1 tie-break) over contexts on AI pool"},
        "generate": {"selected": gen_star, "by_prompt_avg": gen_avg,
                     "criterion": "max mean BLEU (rougeL tie-break) over contexts on AI pool"},
        "note": "BLEU/ROUGE 本身失明，但作为'沿用旧口径经验选取'依据自洽；见报告。",
        "hardcoded_in_config": config.PSTAR,
    }
    path = config.METRICS_DIR / "pstar_selection.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[P* 选取] classify → {clf_star}   generate → {gen_star}")
    print(f"  config.PSTAR = {config.PSTAR}")
    consistent = (config.PSTAR["classify"] == clf_star
                  and config.PSTAR["generate"] == gen_star)
    print(f"  与 config 一致: {consistent}")
    print(f"  → {path.relative_to(config.PROJECT_ROOT)}")
    return out


if __name__ == "__main__":
    select()
