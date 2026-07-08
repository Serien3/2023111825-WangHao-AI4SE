"""切片 3b：4 种 Prompt 策略 × 2 任务，构建 messages 列表。

策略（指导书 4.7.3）：
  P1 Zero-shot：直接任务指令，无示例。
  P2 Few-shot：注入 train 集示例（防泄漏，见 sampling）。
  P3 Chain-of-Thought：要求先逐步分析，最后一行给结论。
  P4 Role-based：设定资深 reviewer / maintainer 角色。

分类：强制结尾 `DECISION: MERGE` 或 `DECISION: REJECT`（解析见 run_experiments）。
生成：产出一条简洁的行内审查意见。
"""
from __future__ import annotations

import json
from functools import lru_cache

from . import config

# --------------------------------------------------------------------------- #
# few-shot 示例加载
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=2)
def _fewshot(task: str) -> list[dict]:
    path = config.SAMPLES_DIR / f"fewshot_{task}.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# 任务指令片段
# --------------------------------------------------------------------------- #
CLASSIFY_TASK = (
    "任务：判断下面这个 GitHub Pull Request 最终是否会被合并（merged）。\n"
    "综合代码改动质量、完整性、是否符合项目规范等因素做出判断。"
)
# 结构化输出：JSON 模式要求 prompt 含 "json" 字样 + 给出示例（DeepSeek 官方要求）。
# reasoning 字段承载分析过程；decision 字段为干净标签，解析零歧义、绝不因截断丢失。
CLASSIFY_FORMAT = (
    "输出要求：只返回一个 JSON 对象，字段为 "
    '{"reasoning": "一句话简要理由", "decision": "MERGE" 或 "REJECT"}。'
    "decision 必须是 MERGE 或 REJECT 之一。示例：\n"
    '{"reasoning": "改动小且符合规范，测试完整。", "decision": "MERGE"}'
)
CLASSIFY_FORMAT_COT = (
    "输出要求：只返回一个 JSON 对象，字段为 "
    '{"reasoning": "分步骤的详细分析（改动规模、风险、规范符合度等）", '
    '"decision": "MERGE" 或 "REJECT"}。'
    "请在 reasoning 中一步步充分推理，再在 decision 给出最终结论。示例：\n"
    '{"reasoning": "1) 改动规模适中；2) 无破坏性变更；3) 符合项目风格。综合判断可合并。", '
    '"decision": "MERGE"}'
)

GENERATE_TASK = (
    "任务：作为代码审查者，针对下面的代码改动，写出一条具体、可操作的行内审查意见"
    "（review comment），指出潜在问题或改进点。"
)
GENERATE_FORMAT = (
    "输出要求：只返回一个 JSON 对象，字段为 "
    '{"comment": "一条简洁（1-3 句）的审查意见"}。不要复述代码。示例：\n'
    '{"comment": "建议为这个边界条件补充单元测试。"}'
)
GENERATE_FORMAT_COT = (
    "输出要求：只返回一个 JSON 对象，字段为 "
    '{"reasoning": "对改动潜在问题的分析", "comment": "最终的一条审查意见"}。'
    "请先在 reasoning 中分析，再在 comment 给出结论。示例：\n"
    '{"reasoning": "该函数未处理空输入，可能抛异常。", '
    '"comment": "建议在入口处校验空输入并给出明确报错。"}'
)

ROLE_SYSTEM = {
    "classify": "你是一位经验丰富的开源项目维护者（maintainer），负责决定 PR 是否合入主干。",
    "generate": "你是一位资深代码审查专家（senior reviewer），以严谨、具体、建设性著称。",
}


# --------------------------------------------------------------------------- #
# few-shot 文本渲染
# --------------------------------------------------------------------------- #
def _render_fewshot_classify() -> str:
    exs = _fewshot("classify")
    blocks = []
    for i, e in enumerate(exs, 1):
        decision = "MERGE" if e["is_merged"] else "REJECT"
        out = json.dumps({"reasoning": e["reason"], "decision": decision},
                         ensure_ascii=False)
        blocks.append(f"[示例 {i}]\n代码改动摘要:\n{e['diff']}\n输出: {out}")
    return "以下是若干判断示例：\n\n" + "\n\n".join(blocks)


def _render_fewshot_generate() -> str:
    exs = _fewshot("generate")
    blocks = []
    for i, e in enumerate(exs, 1):
        out = json.dumps({"comment": e["comment"]}, ensure_ascii=False)
        blocks.append(f"[示例 {i}]\n代码改动:\n{e['diff_hunk']}\n输出: {out}")
    return "以下是若干审查意见示例：\n\n" + "\n\n".join(blocks)


# --------------------------------------------------------------------------- #
# 主接口
# --------------------------------------------------------------------------- #
def build_prompt(task: str, strategy: str, context_text: str) -> list[dict]:
    """返回 OpenAI chat messages 列表。

    task ∈ {classify, generate}；strategy ∈ {P1,P2,P3,P4}；
    context_text 为 context_builder.build_context 的输出。
    """
    if task == "classify":
        task_instr = CLASSIFY_TASK
        fmt = CLASSIFY_FORMAT
        fmt_cot = CLASSIFY_FORMAT_COT
        render_fewshot = _render_fewshot_classify
    else:
        task_instr = GENERATE_TASK
        fmt = GENERATE_FORMAT
        fmt_cot = GENERATE_FORMAT_COT
        render_fewshot = _render_fewshot_generate

    system = "你是一个严谨的代码审查助手。"
    user_parts = [task_instr]

    if strategy == "P2":  # Few-shot
        user_parts.append(render_fewshot())
    if strategy == "P4":  # Role-based
        system = ROLE_SYSTEM[task]

    # CoT 用不同的格式约束
    if strategy == "P3":
        fmt_final = fmt_cot
        user_parts.append("请一步步思考（Chain-of-Thought）。")
    else:
        fmt_final = fmt

    user_parts.append("=" * 40)
    user_parts.append(context_text)
    user_parts.append("=" * 40)
    user_parts.append(fmt_final)

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


if __name__ == "__main__":
    from . import context_builder
    sample = json.load(open(config.SAMPLES_DIR / "classify_50.json"))[0]
    ctx = context_builder.build_context(sample["repo"], sample["number"], "C2")
    for task in config.TASKS:
        for strat in config.PROMPTS:
            msgs = build_prompt(task, strat, ctx)
            print(f"\n########## task={task} strategy={strat} "
                  f"({config.PROMPT_LABELS[strat]}) ##########")
            print(f"[system] {msgs[0]['content']}")
            print(f"[user] (len={len(msgs[1]['content'])})")
            print(msgs[1]['content'][:500])
