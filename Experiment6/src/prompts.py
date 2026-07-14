"""Prompt 构建：P1–P4 复用实验四（R8），新增 P5 Self-Reflection、P6 多轮（R9/R10）。

P5：单轮内两阶段结构化——① 扮演挑剔 maintainer 主动找 blocking 缺陷（不预设 PR 好），
    ② 基于"是否存在 blocking 缺陷"反推 merge 判断 / 生成审查意见。结论行/意见体与
    实验四 *_FORMAT 保持解析兼容（分类 decision 字段、生成 comment 字段）。
P6：多轮交互式，用多个 messages 回合串起对话（build_p6_turns 返回逐轮 messages 序列）。

P1–P4 直接委托实验四 prompts.build_prompt，保证旧口径逐字一致。
"""
from __future__ import annotations

import importlib

from . import config

exp4_prompts = importlib.import_module("exp4src.prompts")


# --------------------------------------------------------------------------- #
# P5 Self-Reflection：批判性 maintainer + 强制先列 blocking 缺陷清单
# --------------------------------------------------------------------------- #
P5_SYSTEM = {
    "classify": (
        "你是一位极其挑剔、以质量把关著称的开源项目维护者（maintainer）。"
        "你默认 PR 可能存在问题，绝不假设它是好的——你的职责是主动找出所有会阻断合并的"
        "(blocking) 缺陷：正确性 bug、破坏性变更、缺失测试、安全隐患、不符合项目规范等。"
    ),
    "generate": (
        "你是一位极其挑剔的资深代码审查专家（senior reviewer）。你默认代码可能存在问题，"
        "绝不假设它是好的——你主动寻找 blocking 缺陷（正确性、边界、安全、可维护性、测试缺失），"
        "并据此写出最有价值的一条审查意见。"
    ),
}

P5_CLASSIFY_INSTR = (
    "任务：判断下面这个 GitHub Pull Request 是否应当被合并（merged）。\n"
    "请按以下两阶段推理，不要跳过第一阶段：\n"
    "阶段①（主动找茬）：以挑剔 maintainer 视角，列出所有你能发现的 blocking 缺陷"
    "（若确无缺陷则显式写：未发现 blocking 缺陷）。不要预设 PR 是好的。\n"
    "阶段②（反推结论）：仅当不存在 blocking 缺陷时才判 MERGE；只要存在任一 blocking 缺陷"
    "就倾向 REJECT。"
)
P5_CLASSIFY_FORMAT = (
    "输出要求：只返回一个 JSON 对象，字段为 "
    '{"blocking_defects": ["缺陷1", "缺陷2", ...], '
    '"reasoning": "基于缺陷清单的结论理由", "decision": "MERGE" 或 "REJECT"}。'
    "blocking_defects 为空数组表示未发现 blocking 缺陷。decision 必须是 MERGE 或 REJECT。示例：\n"
    '{"blocking_defects": ["新增函数缺少单元测试", "边界条件未处理"], '
    '"reasoning": "存在 2 处 blocking 缺陷，不宜合并。", "decision": "REJECT"}'
)

P5_GENERATE_INSTR = (
    "任务：作为挑剔的代码审查者，针对下面的代码改动写出一条最有价值、具体、可操作的行内审查意见。\n"
    "请按两阶段推理：\n"
    "阶段①（主动找茬）：列出你发现的 blocking 缺陷（正确性/边界/安全/测试缺失等），不要假设代码是好的。\n"
    "阶段②（择要成文）：从缺陷清单中挑出最关键的一点，写成一条简洁的审查意见。"
)
P5_GENERATE_FORMAT = (
    "输出要求：只返回一个 JSON 对象，字段为 "
    '{"blocking_defects": ["缺陷1", ...], "comment": "一条简洁(1-3句)的审查意见"}。'
    "comment 应针对最关键的缺陷、具体可操作、不要复述代码。示例：\n"
    '{"blocking_defects": ["空输入未校验"], '
    '"comment": "建议在入口处校验空输入并给出明确报错，否则会抛未捕获异常。"}'
)


def build_prompt(task: str, strategy: str, context_text: str) -> list[dict]:
    """返回单轮 messages。P1–P4 委托实验四；P5 为本实验新增。

    P6 是多轮，不走这里——用 build_p6_turns。
    """
    if strategy in ("P1", "P2", "P3", "P4"):
        return exp4_prompts.build_prompt(task, strategy, context_text)
    if strategy == "P5":
        return _build_p5(task, context_text)
    raise ValueError(f"build_prompt 不支持 {strategy}（P6 用 build_p6_turns）")


def _build_p5(task: str, context_text: str) -> list[dict]:
    if task == "classify":
        instr, fmt = P5_CLASSIFY_INSTR, P5_CLASSIFY_FORMAT
    else:
        instr, fmt = P5_GENERATE_INSTR, P5_GENERATE_FORMAT
    user = "\n\n".join([instr, "=" * 40, context_text, "=" * 40, fmt])
    return [
        {"role": "system", "content": P5_SYSTEM[task]},
        {"role": "user", "content": user},
    ]


# --------------------------------------------------------------------------- #
# P6 多轮交互式（R10，仅 L4 一格）
# --------------------------------------------------------------------------- #
# 分类：审代码 → 追问风险 → 定论；生成：初稿意见 → 自我批评 → 修订。
# build_p6_turns 返回 [(turn_name, followup_user_message)]；主循环把上一轮 assistant
# 回复回填进 messages 再发下一轮。首轮 messages 由 build_p6_first 给出。
P6_CLASSIFY_FOLLOWUPS = [
    ("probe_risk",
     "现在请专门审视风险面：这个 PR 是否存在被你忽略的正确性 bug、破坏性变更、"
     "安全隐患或缺失测试？逐条列出，若确无则说明理由。"),
    ("verdict",
     "综合前两轮分析给出最终结论。只返回一个 JSON 对象："
     '{"reasoning": "综合结论理由", "decision": "MERGE" 或 "REJECT"}。'
     "只要存在任一 blocking 缺陷就倾向 REJECT。"),
]
P6_GENERATE_FOLLOWUPS = [
    ("self_critique",
     "请自我批评你刚才的审查意见：它是否命中了最关键的问题？是否具体可操作、"
     "是否有臆测/幻觉？指出可改进之处。"),
    ("revise",
     "基于自我批评给出修订后的最终审查意见。只返回一个 JSON 对象："
     '{"comment": "一条简洁(1-3句)、具体可操作的审查意见"}。不要复述代码。'),
]

P6_CLASSIFY_FIRST_INSTR = (
    "任务：审查下面这个 GitHub Pull Request 的代码改动，判断它是否应被合并。\n"
    "这是多轮审查的第一轮：请先客观描述改动做了什么、初步质量印象，暂不下最终结论。"
)
P6_GENERATE_FIRST_INSTR = (
    "任务：作为代码审查者，针对下面的代码改动写出一条初稿审查意见。\n"
    "这是多轮审查的第一轮：给出你的初稿意见（后续轮次会自我批评并修订）。"
)


def build_p6_first(task: str, context_text: str) -> list[dict]:
    """P6 首轮 messages。"""
    if task == "classify":
        instr = P6_CLASSIFY_FIRST_INSTR
        sys_msg = "你是一位经验丰富、注重风险的开源项目维护者，正在进行多轮审查。"
    else:
        instr = P6_GENERATE_FIRST_INSTR
        sys_msg = "你是一位资深代码审查专家，正在进行多轮迭代审查。"
    user = "\n\n".join([instr, "=" * 40, context_text, "=" * 40])
    return [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user},
    ]


def p6_followups(task: str) -> list[tuple[str, str]]:
    return P6_CLASSIFY_FOLLOWUPS if task == "classify" else P6_GENERATE_FOLLOWUPS


if __name__ == "__main__":
    from . import context_builder
    import json
    sample = json.load(open(config.EXP5_SAMPLES_DIR / "classify_ai.json"))[0]
    ctx = context_builder.build_context(sample["repo"], sample["number"], "L0")
    for task in config.TASKS:
        print(f"\n##### P5 {task} #####")
        msgs = _build_p5(task, ctx)
        print("[system]", msgs[0]["content"][:80])
        print("[user]", msgs[1]["content"][:200])
        print(f"\n##### P6 {task} first turn + {len(p6_followups(task))} followups #####")
        first = build_p6_first(task, ctx)
        print("[system]", first[0]["content"][:80])
