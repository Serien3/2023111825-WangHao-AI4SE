"""实验六配置：上下文阶梯 L0–L4、Prompt（P*/P5/P6）、R12 矩阵、模型、token 预算。

设计见 docs/specs/2026-07-10-exp6-ai-code-review-improvement-design.md。实验六在
「只优化输入(上下文+Prompt)、不改模型」前提下改进 AI 代码审查，是实验四/五的下游。

沿用实验五的别名加载：把兄弟实验 src 以 exp2src/exp4src 载入，复用其
context_builder/prompts/llm_client/evaluate。LLM 缓存目录**指向实验四 cache**（共享，
省钱）；新上下文/Prompt 改变 messages → content_hash 变 → 不误命中旧缓存。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# 路径
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]          # Experiment6/
REPO_ROOT = PROJECT_ROOT.parent                             # 仓库根
RESULTS_DIR = PROJECT_ROOT / "results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
# API 拉取（文件/issue/检索）落盘缓存（R7），与 LLM cache 分开
FETCH_CACHE_DIR = RESULTS_DIR / "fetch_cache"

for _d in (PREDICTIONS_DIR, METRICS_DIR, FIGURES_DIR, FETCH_CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# 前序实验目录
EXP1_DATA_DIR = REPO_ROOT / "Experiment1" / "results" / "processed"
EXP2_DIR = REPO_ROOT / "Experiment2"
EXP4_DIR = REPO_ROOT / "Experiment4"
EXP5_DIR = REPO_ROOT / "Experiment5"

# 实验五 AI 池样本（R1：直接读，禁止重新采集）
EXP5_SAMPLES_DIR = EXP5_DIR / "results" / "samples"
EXP5_METRICS_DIR = EXP5_DIR / "results" / "metrics"

ENV_FILE = REPO_ROOT / ".env"
GITHUB_TOKEN_ENV = "GITHUB_TOKEN"
GITHUB_API = "https://api.github.com"


# --------------------------------------------------------------------------- #
# 兄弟实验 src 包按别名载入（零改动复用，与实验五同款）
# --------------------------------------------------------------------------- #
def _load_sibling_src(exp_dir: Path, alias: str):
    if alias in sys.modules:
        return sys.modules[alias]
    src_dir = exp_dir / "src"
    init = src_dir / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        alias, init, submodule_search_locations=[str(src_dir)]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


_load_sibling_src(EXP2_DIR, "exp2src")
_load_sibling_src(EXP4_DIR, "exp4src")

import exp4src as exp4  # noqa: E402

importlib.import_module("exp4src.config")
importlib.import_module("exp4src.data")

# LLM 缓存共享实验四（关键：exp4.llm_client 直接写 exp4.config.CACHE_DIR）
CACHE_DIR = exp4.config.CACHE_DIR


# --------------------------------------------------------------------------- #
# 模型（R14）：执行 = flash，裁判 = pro（仅评生成任务）
# --------------------------------------------------------------------------- #
EXEC_MODEL = exp4.config.MODEL_ID           # "deepseek-v4-flash"
JUDGE_MODEL = "deepseek-v4-pro"

TASKS = ["classify", "generate"]
TEMPERATURE = exp4.config.TEMPERATURE       # {"classify":0.0,"generate":0.3}
MAX_TOKENS = exp4.config.MAX_TOKENS
JSON_MODE = exp4.config.JSON_MODE
JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_TOKENS = 2048


# --------------------------------------------------------------------------- #
# 上下文阶梯（R4）：5 级单调递增，L0=Exp4 C1，L1=Exp4 C4
# --------------------------------------------------------------------------- #
LEVELS = ["L0", "L1", "L2", "L3", "L4"]
LEVEL_LABELS = {
    "L0": "Diff",
    "L1": "Diff+PR描述+Commit",
    "L2": "L1+改前后完整函数/文件",
    "L3": "L2+Issue正文+历史审查意见",
    "L4": "L3+仓库级轻量检索",
}

# --------------------------------------------------------------------------- #
# Prompt：P*（旧最优）+ P5 Self-Reflection + P6 多轮（R8–R11）
# --------------------------------------------------------------------------- #
# R11：P* = 实验四/五 AI 数据上表现最好的旧 prompt，从预测指标经验选取（不硬编码）。
# 选取依据（脚本 select_pstar.py 复现）：
#   分类：按 AI 组各 prompt 跨上下文均值，P1 acc=0.760/f1=0.860（最高）→ P*_classify=P1
#   生成：按 AI 组各 prompt 跨上下文均值，P3 bleu=1.125/rougeL=0.054（最高）→ P*_generate=P3
# 说明：BLEU/ROUGE 本身失明（见报告），但作为"沿用旧口径经验选取"的依据是自洽的；
#       选取过程与依据写入报告，避免事后挑选偏差。
PSTAR = {"classify": "P1", "generate": "P3"}
NEW_PROMPTS = ["P5", "P6"]
PROMPT_LABELS = {
    **exp4.config.PROMPT_LABELS,
    "P5": "Self-Reflection",
    "P6": "多轮交互式",
}


# --------------------------------------------------------------------------- #
# 实验矩阵（R12）：非全网格
# --------------------------------------------------------------------------- #
def matrix(task: str) -> list[dict]:
    """返回该任务要跑的条件列表 [{level, prompt, group, cell}]。

    group ∈ {"ai","control"}；cell 是人类可读标签。仅在关键格跑对照。
    条件类别：
      - 消融①(上下文): {L0..L4} × P*                      —— AI
      - 消融②(Prompt): L4 × {P*, P5, P6}                   —— AI（P6 仅 L4）
      - 归因格: L0 × P5                                     —— AI
      - 人类对照: L4×P*、L4×P5 两格                          —— control
    """
    pstar = PSTAR[task]
    conds: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    def add(level: str, prompt: str, group: str, cell: str):
        key = (level, prompt, group)
        if key not in seen:
            seen.add(key)
            conds.append({"level": level, "prompt": prompt, "group": group, "cell": cell})

    # 消融①：上下文阶梯 × P*
    for lv in LEVELS:
        add(lv, pstar, "ai", "context_ladder")
    # 消融②：L4 × {P*, P5, P6}
    for pm in [pstar, "P5", "P6"]:
        add("L4", pm, "ai", "prompt_ablation")
    # 归因格：L0 × P5
    add("L0", "P5", "ai", "attribution")
    # 人类对照：仅 L4×P*、L4×P5
    add("L4", pstar, "control", "human_control")
    add("L4", "P5", "control", "human_control")
    return conds


# --------------------------------------------------------------------------- #
# Token / 长度预算（R6，每级独立；沿用实验四字符估算）
# --------------------------------------------------------------------------- #
CHARS_PER_TOKEN = exp4.config.CHARS_PER_TOKEN
DIFF_CHAR_BUDGET = exp4.config.DIFF_CHAR_BUDGET          # 24000
BODY_CHAR_BUDGET = exp4.config.BODY_CHAR_BUDGET          # 4000
COMMIT_CHAR_BUDGET = exp4.config.COMMIT_CHAR_BUDGET      # 3000
# L2：改前/改后完整函数或文件
FULLFILE_CHAR_BUDGET = 12000        # 单文件改前+改后合计上限
L2_TOTAL_BUDGET = 24000             # L2 全部文件合计上限
# L3：issue 正文 + 历史审查意见
ISSUE_CHAR_BUDGET = 4000
HISTORY_REVIEW_CHAR_BUDGET = 4000
# L4：仓库级轻量检索
RETRIEVAL_SNIPPET_CHARS = 600       # 单个检索片段截断
RETRIEVAL_TOPK_FILES = 3            # code search 每符号 top-k 文件
RETRIEVAL_MAX_SYMBOLS = 5           # 最多检索几个被改符号
SIBLING_FILE_HEAD_CHARS = 800       # 同目录相关文件头部
L4_TOTAL_BUDGET = 12000             # L4 检索块合计上限

# --------------------------------------------------------------------------- #
# API 限流/并发（R20）
# --------------------------------------------------------------------------- #
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0
REQUEST_TIMEOUT = 60
RATE_LIMIT_SLEEP_BUFFER = 2

SEED = 42


def print_matrix() -> None:
    print("实验六（改进 AI 代码审查）配置")
    print(f"  执行模型: {EXEC_MODEL}   裁判模型: {JUDGE_MODEL}（仅评生成）")
    print(f"  LLM 缓存目录（共享实验四）: {CACHE_DIR}")
    print(f"  上下文阶梯: " + ", ".join(f"{lv}={LEVEL_LABELS[lv]}" for lv in LEVELS))
    print(f"  P*（旧最优）: classify={PSTAR['classify']}, generate={PSTAR['generate']}")
    print(f"  新增 Prompt: P5={PROMPT_LABELS['P5']}, P6={PROMPT_LABELS['P6']}")
    for task in TASKS:
        conds = matrix(task)
        print(f"\n  === 任务 {task}: {len(conds)} 条件 ===")
        for c in conds:
            print(f"    {c['level']:<3} {c['prompt']:<3} [{c['group']:<7}] {c['cell']}")


if __name__ == "__main__":
    print_matrix()
