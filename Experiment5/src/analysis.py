"""分析产物：三条证据链（性能差值、上下文敏感度、错误案例归因）。

输出全部数据驱动，供报告写作直接引用：
1. metrics/performance_gap_localization.json：人类 vs AI / 匹配对照逐指标差值。
2. metrics/context_sensitivity.json：C1→C4 增益，AI vs human/control。
3. cases/error_cases.json + cases/error_case_attribution_template.md：3–5 个典型错误案例及归因建议。
"""
from __future__ import annotations

import json
from collections import Counter

import pandas as pd

from . import config, data


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def _load_json(path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _dump_json(obj, path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"  → {path.relative_to(config.PROJECT_ROOT)}")


# --------------------------------------------------------------------------- #
# 证据链 1：性能差值定位
# --------------------------------------------------------------------------- #
def performance_gap_localization(hv: dict) -> dict:
    """汇总 ML + LLM 的 AI 性能差值，定位下降发生在哪类模型/条件。"""
    out = {"ml": [], "llm_classify": [], "llm_generate": []}

    # ML：pre/full × model
    for fs, models in hv.get("ml", {}).items():
        for model, m in models.items():
            row = {
                "feature_set": fs,
                "model": model,
                "human_test_f1": m.get("human_test", {}).get("f1"),
                "ai_f1": m.get("ai", {}).get("f1"),
                "control_f1": m.get("control", {}).get("f1"),
                "gap_ai_vs_human_f1": m.get("gap_ai_vs_human", {}).get("f1"),
                "gap_ai_vs_control_f1": m.get("gap_ai_vs_control", {}).get("f1"),
            }
            out["ml"].append(row)

    # LLM 分类：逐条件 F1
    for cond, m in hv.get("llm_classify", {}).get("ai", {}).items():
        h = hv.get("llm_classify", {}).get("human", {}).get(cond, {})
        c = hv.get("llm_classify", {}).get("control", {}).get(cond, {})
        out["llm_classify"].append({
            "condition": cond,
            "human_f1": h.get("f1"),
            "ai_f1": m.get("f1"),
            "control_f1": c.get("f1"),
            "gap_ai_vs_human_f1": (m.get("f1") - h.get("f1"))
                                    if h.get("f1") is not None and m.get("f1") is not None else None,
            "gap_ai_vs_control_f1": (m.get("f1") - c.get("f1"))
                                      if c.get("f1") is not None and m.get("f1") is not None else None,
            "parse_error_rate": m.get("parse_error_rate"),
        })

    # LLM 生成：逐条件 ROUGE-L / BLEU
    for cond, m in hv.get("llm_generate", {}).get("ai", {}).items():
        h = hv.get("llm_generate", {}).get("human", {}).get(cond, {})
        c = hv.get("llm_generate", {}).get("control", {}).get(cond, {})
        out["llm_generate"].append({
            "condition": cond,
            "human_rougeL": h.get("rougeL"),
            "ai_rougeL": m.get("rougeL"),
            "control_rougeL": c.get("rougeL"),
            "gap_ai_vs_human_rougeL": (m.get("rougeL") - h.get("rougeL"))
                                        if h.get("rougeL") is not None and m.get("rougeL") is not None else None,
            "gap_ai_vs_control_rougeL": (m.get("rougeL") - c.get("rougeL"))
                                          if c.get("rougeL") is not None and m.get("rougeL") is not None else None,
            "ai_bleu": m.get("bleu"),
        })
    return out


# --------------------------------------------------------------------------- #
# 证据链 2：上下文敏感度（C1→C4）
# --------------------------------------------------------------------------- #
def _avg_by_context(metrics: dict, metric: str) -> dict:
    vals = {ctx: [] for ctx in config.CONTEXTS}
    for cond, m in metrics.items():
        ctx = cond.split("_")[0]
        v = m.get(metric)
        if ctx in vals and isinstance(v, (int, float)):
            vals[ctx].append(v)
    return {ctx: (sum(v) / len(v) if v else None) for ctx, v in vals.items()}


def context_sensitivity(hv: dict) -> dict:
    """按 Prompt 对 C1/C2/C3/C4 求均值，报告 C4-C1 增益。"""
    out = {"classify_f1": {}, "generate_rougeL": {}}
    for section, metric, target in [
        ("llm_classify", "f1", "classify_f1"),
        ("llm_generate", "rougeL", "generate_rougeL"),
    ]:
        for group in ["human", "ai", "control"]:
            avg = _avg_by_context(hv.get(section, {}).get(group, {}), metric)
            gain = None
            if avg.get("C1") is not None and avg.get("C4") is not None:
                gain = avg["C4"] - avg["C1"]
            out[target][group] = {"by_context": avg, "gain_C4_minus_C1": gain}
    return out


# --------------------------------------------------------------------------- #
# 证据链 3：错误案例抽取 + 归因模板
# --------------------------------------------------------------------------- #
def _snippet(repo: str, number: int, budget: int = 1200) -> str:
    patches = data.pr_python_patches(repo, number)
    text = "\n".join(f"--- {fn} ---\n{patch}" for fn, patch in patches)
    return text[:budget] + ("\n... [truncated]" if len(text) > budget else "")


def _classify_error_cases(limit: int = 5) -> list[dict]:
    path = config.PREDICTIONS_DIR / "classify_predictions.parquet"
    if not path.exists():
        return []
    df = pd.read_parquet(path)
    rows = []
    # 以 AI 侧为主，优先解析失败，其次误判；按条件去重 PR，最多 limit 个
    ai = df[df.get("group", "ai") == "ai"].copy()
    if ai.empty:
        return []
    candidates = ai[(ai.get("parse_error", False) == True) |
                    (ai["pred_merge"].notna() &
                     (ai["pred_merge"].astype(bool) != ai["true_merge"].astype(bool)))]
    seen = set()
    for r in candidates.itertuples():
        key = (r.repo, int(r.number))
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "case_type": "parse_error" if bool(getattr(r, "parse_error", False)) else "misclassification",
            "repo": r.repo,
            "number": int(r.number),
            "condition": f"{r.context}_{r.prompt}",
            "true_merge": bool(r.true_merge),
            "pred_merge": None if pd.isna(r.pred_merge) else bool(r.pred_merge),
            "suggested_attribution": "局部 diff 信号弱 / AI 代码风格统一导致模型过度乐观或保守（需人工复核）",
            "diff_snippet": _snippet(r.repo, int(r.number)),
        })
        if len(rows) >= limit:
            break
    return rows


def _generate_error_cases(limit: int = 5) -> list[dict]:
    path = config.PREDICTIONS_DIR / "generate_predictions.parquet"
    if not path.exists():
        return []
    df = pd.read_parquet(path)
    ai = df[df.get("group", "ai") == "ai"].copy()
    if ai.empty:
        return []
    # 简单数据驱动启发：空/过短/过长/疑似泛泛而谈的生成，作为“跑偏”候选
    def bad(row):
        text = str(row.get("pred_comment") or "")
        if len(text.strip()) < 8 or len(text) > 600:
            return True
        generic = ["looks good", "LGTM", "建议检查", "需要注意", "可能存在问题"]
        return any(g.lower() in text.lower() for g in generic)
    cand = ai[ai.apply(bad, axis=1)]
    rows, seen = [], set()
    for r in cand.itertuples():
        key = (r.repo, int(r.number))
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "case_type": "generation_drift",
            "repo": r.repo,
            "number": int(r.number),
            "condition": f"{r.context}_{r.prompt}",
            "target_comment": getattr(r, "target_comment", ""),
            "pred_comment": getattr(r, "pred_comment", ""),
            "suggested_attribution": "生成意见泛化/未对齐真实 inline comment，可能缺少跨文件或项目上下文（需人工复核）",
            "diff_snippet": _snippet(r.repo, int(r.number)),
        })
        if len(rows) >= limit:
            break
    return rows


def error_case_analysis() -> dict:
    cases = _classify_error_cases(limit=5)
    if len(cases) < 5:
        cases += _generate_error_cases(limit=5 - len(cases))
    type_counts = dict(Counter(c["case_type"] for c in cases))
    out = {"cases": cases, "type_counts": type_counts}
    return out


def _write_case_template(cases: list[dict]) -> None:
    path = config.CASES_DIR / "error_case_attribution_template.md"
    lines = ["# 实验五错误案例人工归因模板", "", "> 下面案例由数据自动抽取；请人工复核 attribution 字段。", ""]
    for i, c in enumerate(cases, 1):
        lines += [
            f"## Case {i}: {c['case_type']} · {c['repo']}#{c['number']} · {c.get('condition')}",
            "",
            f"- 自动归因建议：{c.get('suggested_attribution')}",
            "- 人工归因（待填）：",
            "- 证据摘录：",
            "",
            "```diff",
            c.get("diff_snippet", "")[:1200],
            "```",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  → {path.relative_to(config.PROJECT_ROOT)}")


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def main() -> None:
    hv = _load_json(config.METRICS_DIR / "human_vs_ai.json")
    if not hv:
        raise FileNotFoundError("缺少 metrics/human_vs_ai.json，请先运行 uv run python -m src.evaluate")

    perf = performance_gap_localization(hv)
    _dump_json(perf, config.METRICS_DIR / "performance_gap_localization.json")

    ctx = context_sensitivity(hv)
    _dump_json(ctx, config.METRICS_DIR / "context_sensitivity.json")

    cases = error_case_analysis()
    _dump_json(cases, config.CASES_DIR / "error_cases.json")
    _write_case_template(cases["cases"])
    print("分析产物完成。")


if __name__ == "__main__":
    main()
