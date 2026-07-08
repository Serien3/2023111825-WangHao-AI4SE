"""实验四配置：路径、模型 id、实验矩阵、采样参数、token 预算、随机种子。

设计见 docs/design-docs/exp4-llm.md。所有路径以 Experiment4/ 为根，
跨实验消费实验一 processed 表与实验二 split/features/模型。
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# 路径
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]          # Experiment4/
REPO_ROOT = PROJECT_ROOT.parent                              # 仓库根
RESULTS_DIR = PROJECT_ROOT / "results"
SAMPLES_DIR = RESULTS_DIR / "samples"
CACHE_DIR = RESULTS_DIR / "cache"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"

for _d in (SAMPLES_DIR, CACHE_DIR, PREDICTIONS_DIR, METRICS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# 前序实验产物
EXP1_DATA_DIR = REPO_ROOT / "Experiment1" / "results" / "processed"
EXP2_DIR = REPO_ROOT / "Experiment2" / "results"
EXP2_SPLIT_PKL = EXP2_DIR / "models" / "split_pre.pkl"
EXP2_FEATURES = EXP2_DIR / "features" / "features.parquet"
EXP2_MODELS_DIR = EXP2_DIR / "models"
EXP2_SCALER_PKL = EXP2_MODELS_DIR / "scaler_pre.pkl"
EXP2_METRICS = EXP2_DIR / "metrics" / "test_metrics_pre.json"
EXP2_MODEL_FILES = {
    "svm": EXP2_MODELS_DIR / "svm_pre.pkl",
    "rf": EXP2_MODELS_DIR / "rf_pre.pkl",
    "xgboost": EXP2_MODELS_DIR / "xgboost_pre.pkl",
    "lightgbm": EXP2_MODELS_DIR / "lightgbm_pre.pkl",
}

ENV_FILE = REPO_ROOT / ".env"

# --------------------------------------------------------------------------- #
# 模型与调用层
# --------------------------------------------------------------------------- #
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL_ID = "deepseek-v4-flash"             # DeepSeek-V4，快/省一档（flash）
API_KEY_ENV = "DEEPSEEK_API_KEY"

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0                    # 指数退避基数（秒）
REQUEST_TIMEOUT = 120                       # 单次调用超时（秒）

# 任务相关的采样温度（分类求稳，生成留一点多样性）
TEMPERATURE = {"classify": 0.0, "generate": 0.3}
# 放开输出预算：给足空间让模型在 CoT 下充分推理，不因预算截断。
# 取模型可用的大额上限；JSON 模式下也保证结构完整不被截断。
MAX_TOKENS = {"classify": 8192, "generate": 4096}

# 结构化输出：分类与生成都用 DeepSeek JSON 模式，解析零歧义（见 prompts.py 的 schema）。
JSON_MODE = {"classify": True, "generate": True}

# --------------------------------------------------------------------------- #
# 实验矩阵：4 上下文 × 4 Prompt
# --------------------------------------------------------------------------- #
CONTEXTS = ["C1", "C2", "C3", "C4"]         # 见 context_builder.CONTEXT_SPEC
PROMPTS = ["P1", "P2", "P3", "P4"]          # Zero-shot / Few-shot / CoT / Role
TASKS = ["classify", "generate"]

CONTEXT_LABELS = {
    "C1": "Diff",
    "C2": "Diff+PR描述",
    "C3": "Diff+Commit",
    "C4": "Diff+全部",
}
PROMPT_LABELS = {
    "P1": "Zero-shot",
    "P2": "Few-shot",
    "P3": "CoT",
    "P4": "Role",
}

# --------------------------------------------------------------------------- #
# 采样
# --------------------------------------------------------------------------- #
SEED = 42
N_SAMPLE = 50                               # 每任务抽样条数（老师规约）

# --------------------------------------------------------------------------- #
# Token / 长度预算（近似：按字符估算，1 token ≈ 4 字符）
# --------------------------------------------------------------------------- #
CHARS_PER_TOKEN = 4
DIFF_TOKEN_BUDGET = 6000                    # 单 PR 拼接 diff 的 token 上限
DIFF_CHAR_BUDGET = DIFF_TOKEN_BUDGET * CHARS_PER_TOKEN
BODY_CHAR_BUDGET = 4000                     # PR body 截断
COMMIT_CHAR_BUDGET = 3000                   # commit messages 拼接截断
FEWSHOT_DIFF_CHAR_BUDGET = 1500             # few-shot 示例内 diff 摘要截断

# 分类解析
DECISION_PATTERN = r"DECISION:\s*(MERGE|REJECT)"


def print_matrix() -> None:
    print("实验四矩阵定义")
    print(f"  模型: {MODEL_ID} @ {DEEPSEEK_BASE_URL}")
    print(f"  任务: {TASKS}")
    print(f"  上下文 (4): " + ", ".join(f"{c}={CONTEXT_LABELS[c]}" for c in CONTEXTS))
    print(f"  Prompt (4): " + ", ".join(f"{p}={PROMPT_LABELS[p]}" for p in PROMPTS))
    print(f"  抽样 N={N_SAMPLE}, 种子={SEED}")
    print(f"  条件数/任务: {len(CONTEXTS) * len(PROMPTS)} → 调用/任务: "
          f"{len(CONTEXTS) * len(PROMPTS) * N_SAMPLE}")
    print(f"  两任务总调用: {len(TASKS) * len(CONTEXTS) * len(PROMPTS) * N_SAMPLE}")
    print(f"  diff 预算: {DIFF_TOKEN_BUDGET} tokens (~{DIFF_CHAR_BUDGET} 字符)")


if __name__ == "__main__":
    print_matrix()
