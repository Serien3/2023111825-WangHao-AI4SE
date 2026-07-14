"""实验六测试（验外部行为，不验实现细节）。参照设计 Testing Decisions。

覆盖：
- 接缝①上下文：L0–L4 单调（高 level ⊇ 低 level 除截断）；各 level 含应有块标记。
  L2–L4 触网被 monkeypatch 为确定性桩，离线可跑。
- 接缝②LLM：不同 model → 不同 cache key；model=None 时 key 等于原实验四行为（回归）。
- 接缝③裁判：命中真值 vs 无关幻觉 → 可区分分档（judge 的 LLM 调用被桩替换）。
- 分类指标：per-class P/R + 混淆矩阵 + balanced accuracy 计算正确。
- 解析兼容：P5/P6 输出结构被实验四解析器正确解析（decision / comment）。

用法：uv run python -m src.tests
"""
from __future__ import annotations

import importlib
import json

from . import config, context_builder, data, github_fetch, judge, prompts

exp4_run = importlib.import_module("exp4src.run_experiments")
exp4_llm = importlib.import_module("exp4src.llm_client")

_PASS = 0
_FAIL = 0


def check(cond: bool, msg: str) -> None:
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  ✓ {msg}")
    else:
        _FAIL += 1
        print(f"  ✗ FAIL: {msg}")


# --------------------------------------------------------------------------- #
# 接缝②：LLM cache key 随 model 变化，model=None 回归实验四行为
# --------------------------------------------------------------------------- #
def test_llm_model_seam():
    print("\n[接缝②] LLM model 维度 cache key")
    msgs = [{"role": "user", "content": "hi"}]
    # 复制 chat_semantic 里的 hash 逻辑做外部行为断言：不同 model → 不同 key
    import hashlib

    def key_for(model):
        model_id = model or exp4_llm.config.MODEL_ID
        h = exp4_llm._hash_key(model_id, json.dumps(msgs, ensure_ascii=False), 0.0, 32, True)
        return h[:12]

    k_none = key_for(None)
    k_flash = key_for(config.EXEC_MODEL)     # = MODEL_ID
    k_pro = key_for(config.JUDGE_MODEL)
    check(k_none == k_flash, "model=None 与 flash(=MODEL_ID) 同 key（回归实验四行为）")
    check(k_pro != k_none, "裁判 model=pro 与 None 不同 key（互不覆盖）")


# --------------------------------------------------------------------------- #
# 接缝①：上下文阶梯单调 + 各级含应有块
# --------------------------------------------------------------------------- #
def test_context_ladder():
    print("\n[接缝①] 上下文阶梯 L0–L4")
    # 用离线桩替换 L2–L4 的触网，保证确定性
    orig = {
        "get_file_at_sha": github_fetch.get_file_at_sha,
        "get_issue": github_fetch.get_issue,
        "code_search": github_fetch.code_search,
        "list_dir": github_fetch.list_dir,
    }
    github_fetch.get_file_at_sha = lambda repo, path, sha: (
        "def changed_func(x):\n    return x + 1\n" if path.endswith(".py") else None)
    github_fetch.get_issue = lambda repo, number: {
        "number": number, "title": "ISSUE_TITLE_MARK", "body": "ISSUE_BODY_MARK 修复问题"}
    github_fetch.code_search = lambda repo, symbol, top_k: [
        {"path": "other/x.py", "fragment": "RETRIEVAL_MARK usage of " + symbol}]
    github_fetch.list_dir = lambda repo, d, sha: [f"{d}/sibling.py"] if d else ["sibling.py"]
    # 同时给 context_builder 引用的句柄打桩
    context_builder.github_fetch = github_fetch

    # 找一个有 Python patch 且 body 含 #ref 的样本；否则退而用任意分类样本
    samples = json.load(open(config.EXP5_SAMPLES_DIR / "classify_ai.json"))
    rec = None
    for s in samples:
        if data.pr_python_patches(s["repo"], s["number"]):
            rec = s
            break
    rec = rec or samples[0]
    repo, num = rec["repo"], rec["number"]

    ctxs = {lv: context_builder.build_context(repo, num, lv) for lv in config.LEVELS}
    for lv in config.LEVELS:
        print(f"    {lv}: len={len(ctxs[lv])}")

    # 单调：高 level 长度 >= 低 level（我们的块是纯追加）
    lens = [len(ctxs[lv]) for lv in config.LEVELS]
    check(all(lens[i] <= lens[i + 1] for i in range(len(lens) - 1)),
          "长度单调不减 L0≤L1≤L2≤L3≤L4")

    # L1 含 PR 描述块标记
    check("PR 描述" in ctxs["L1"], "L1 含 PR 描述块")
    # L2 含完整函数块标记
    check("完整函数" in ctxs["L2"], "L2 含改前/改后完整函数块")
    # L3 含 issue 正文串（若该 PR 有 issue ref）
    refs = data.issue_refs(repo, num)
    if refs:
        check("ISSUE_BODY_MARK" in ctxs["L3"], "L3 含 issue 正文串")
    else:
        check("关联 Issue 与历史审查" in ctxs["L3"], "L3 含 L3 块标题（该 PR 无 issue ref）")
    # L4 含检索片段标记
    check("检索" in ctxs["L4"] and "仓库级检索" in ctxs["L4"], "L4 含仓库级检索块")

    # 超集性质：低 level 的核心 L0 块出现在所有更高 level 中
    check(ctxs["L0"] in ctxs["L4"], "L4 是 L0 的超集（L0 文本内嵌于 L4）")

    for k, v in orig.items():
        setattr(github_fetch, k, v)


# --------------------------------------------------------------------------- #
# 接缝③：裁判可区分命中 vs 幻觉（LLM 调用打桩）
# --------------------------------------------------------------------------- #
def test_judge_seam():
    print("\n[接缝③] 裁判多维打分可区分")
    orig_chat = judge.exp4_llm.chat_semantic

    def fake_chat(task, context, prompt, pr_key, messages, temperature,
                  max_tokens, json_mode=False, model=None):
        # 桩：依据"待评意见"文本里的关键词返回不同分档，模拟真裁判的区分能力
        user = messages[-1]["content"]
        if "校验" in user and "ZeroDivision" in user:
            obj = {"relevance": 5, "actionable": 5, "correct": 5,
                   "hallucination": 1, "rationale": "命中除零"}
        else:
            obj = {"relevance": 1, "actionable": 2, "correct": 2,
                   "hallucination": 5, "rationale": "无关且幻觉"}
        return {"text": json.dumps(obj, ensure_ascii=False), "latency": 0.0,
                "cached": False, "usage": {}, "finish_reason": "stop", "model": model}

    judge.exp4_llm.chat_semantic = fake_chat
    try:
        diff = "def div(a,b):\n    return a/b"
        ref = "没有处理 b==0 的除零情况"
        good = judge.score("建议校验 b 是否为 0，否则抛 ZeroDivisionError", ref, diff,
                           pr_key="t_good")
        bad = judge.score("这个函数依赖不存在的库，建议重写整个模块", ref, diff,
                          pr_key="t_bad")
        check(good["relevance"] > bad["relevance"], "命中真值 relevance 明显更高")
        check(good["hallucination"] < bad["hallucination"], "命中真值 hallucination 明显更低")
        check(all(1 <= good[d] <= 5 for d in
                  ("relevance", "actionable", "correct", "hallucination")),
              "分值夹在 [1,5]")
    finally:
        judge.exp4_llm.chat_semantic = orig_chat


# --------------------------------------------------------------------------- #
# 分类指标：per-class + 混淆矩阵 + balanced accuracy
# --------------------------------------------------------------------------- #
def test_classify_metrics():
    print("\n[指标] per-class P/R + 混淆矩阵 + balanced accuracy")
    import pandas as pd
    from . import evaluate
    # 构造：4 true-merge(3 对 1 错), 4 true-nonmerge(2 对 2 错)
    rows = []
    truth = [1, 1, 1, 1, 0, 0, 0, 0]
    pred = [1, 1, 1, 0, 0, 0, 1, 1]
    for t, p in zip(truth, pred):
        rows.append({"true_merge": bool(t), "pred_merge": bool(p),
                     "parse_error": False, "latency": 1.0})
    m = evaluate._classify_metrics(pd.DataFrame(rows))
    cm = m["confusion_matrix"]
    # tp=3, fn=1, tn=2, fp=2
    check(cm == {"tn": 2, "fp": 2, "fn": 1, "tp": 3, "labels": ["non_merge", "merge"]},
          f"混淆矩阵正确 {cm}")
    # non-merge recall = tn/(tn+fp)=2/4=0.5; merge recall=tp/(tp+fn)=3/4=0.75
    check(abs(m["per_class"]["non_merge"]["recall"] - 0.5) < 1e-9, "non-merge recall=0.5")
    check(abs(m["per_class"]["merge"]["recall"] - 0.75) < 1e-9, "merge recall=0.75")
    # balanced acc = (0.5+0.75)/2 = 0.625
    check(abs(m["balanced_accuracy"] - 0.625) < 1e-9, "balanced accuracy=0.625")


# --------------------------------------------------------------------------- #
# P5/P6 输出解析兼容
# --------------------------------------------------------------------------- #
def test_prompt_parse_compat():
    print("\n[解析兼容] P5/P6 结构化输出可被实验四解析器解析")
    p5_clf = json.dumps({"blocking_defects": ["缺测试"], "reasoning": "有缺陷",
                         "decision": "REJECT"}, ensure_ascii=False)
    check(exp4_run.parse_decision(p5_clf) is False, "P5 分类输出解析为 REJECT")
    p5_gen = json.dumps({"blocking_defects": ["空输入"], "comment": "建议校验空输入"},
                        ensure_ascii=False)
    check(exp4_run._extract_comment(p5_gen) == "建议校验空输入", "P5 生成输出解析出 comment")
    # P5 messages 结构
    msgs = prompts.build_prompt("classify", "P5", "CTX")
    check(msgs[0]["role"] == "system" and "maintainer" in msgs[0]["content"],
          "P5 system 设批判性 maintainer 角色")
    check("blocking" in msgs[1]["content"].lower(), "P5 user 强制先列 blocking 缺陷")


def main():
    test_llm_model_seam()
    test_context_ladder()
    test_judge_seam()
    test_classify_metrics()
    test_prompt_parse_compat()
    print(f"\n===== 通过 {_PASS} / 失败 {_FAIL} =====")
    if _FAIL:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
