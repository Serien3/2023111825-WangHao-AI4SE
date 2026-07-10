"""实验五配置：路径、AI 筛选、抽样参数、种子、矩阵定义。

设计见 docs/specs/2026-07-08-exp5-ai-code-review-design.md。实验五是实验二/四的
纯下游消费者，因此这里通过 importlib 把兄弟实验的 `src` 包以别名载入，供其它模块
`from .config import exp2, exp4` 复用其函数/常量，实现零改动复用。

- exp2 = Experiment2/src（feature_extraction、config、模型/scaler）
- exp4 = Experiment4/src（context_builder、prompts、llm_client、evaluate…）

LLM 缓存复用实验四 cache 目录（exp4.config.CACHE_DIR），匹配对照里凡实验四已跑过的
PR 直接命中，不重复付费。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# 路径
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]          # Experiment5/
REPO_ROOT = PROJECT_ROOT.parent                             # 仓库根
RESULTS_DIR = PROJECT_ROOT / "results"
SAMPLES_DIR = RESULTS_DIR / "samples"
FEATURES_DIR = RESULTS_DIR / "features"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
METRICS_DIR = RESULTS_DIR / "metrics"
CASES_DIR = RESULTS_DIR / "cases"
FIGURES_DIR = RESULTS_DIR / "figures"

for _d in (SAMPLES_DIR, FEATURES_DIR, PREDICTIONS_DIR, METRICS_DIR, CASES_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# 前序实验目录
EXP1_DATA_DIR = REPO_ROOT / "Experiment1" / "results" / "processed"
EXP2_DIR = REPO_ROOT / "Experiment2"
EXP4_DIR = REPO_ROOT / "Experiment4"


# --------------------------------------------------------------------------- #
# 兄弟实验 src 包按别名载入（零改动复用）
# --------------------------------------------------------------------------- #
def _load_sibling_src(exp_dir: Path, alias: str):
    """把 <exp_dir>/src 以别名 alias 注册为顶层包，供子模块相对 import 正常解析。"""
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

# 便捷句柄（子模块按需 importlib.import_module("exp4src.xxx")）
import exp2src as exp2  # noqa: E402
import exp4src as exp4  # noqa: E402

# 预载常用子模块，使 exp2.config / exp4.config 等属性可直接访问
importlib.import_module("exp2src.config")
importlib.import_module("exp4src.config")

# --------------------------------------------------------------------------- #
# AI 筛选 & 抽样参数
# --------------------------------------------------------------------------- #
SEED = 42
N_CLASSIFY_SAMPLE = 50      # LLM 分类抽样条数（与实验四人类侧对齐）
# 生成任务用全部 72 条 AI 真值池（不抽样）

# 匹配人类对照组：按 AI 分类池的 repo × is_merged 分布分层抽样，规模与 AI 侧同量级
CONTROL_MULTIPLIER = 1.0    # 对照组规模 = AI 抽样规模 × 该系数

# --------------------------------------------------------------------------- #
# 实验矩阵（引用实验四定义，保持逐字一致）
# --------------------------------------------------------------------------- #
CONTEXTS = exp4.config.CONTEXTS          # ["C1","C2","C3","C4"]
PROMPTS = exp4.config.PROMPTS            # ["P1","P2","P3","P4"]
TASKS = exp4.config.TASKS                # ["classify","generate"]
CONTEXT_LABELS = exp4.config.CONTEXT_LABELS
PROMPT_LABELS = exp4.config.PROMPT_LABELS

# 两个受试组（AI 生成代码 vs 匹配人类对照）
GROUPS = ["ai", "control"]

# --------------------------------------------------------------------------- #
# ML：特征列定义与模型/scaler（引用实验二）
# --------------------------------------------------------------------------- #
PRE_REVIEW_FEATURES = exp2.config.PRE_REVIEW_FEATURES
REVIEW_PROCESS_FEATURES = exp2.config.REVIEW_PROCESS_FEATURES
FULL_FEATURES = exp2.config.FULL_FEATURES
TFIDF_TOP_K = exp2.config.TFIDF_TOP_K

EXP2_FEATURES_PARQUET = EXP2_DIR / "results" / "features" / "features.parquet"
EXP2_MODELS_DIR = EXP2_DIR / "results" / "models"
EXP2_METRICS_DIR = EXP2_DIR / "results" / "metrics"
EXP4_METRICS_DIR = EXP4_DIR / "results" / "metrics"
EXP4_SAMPLES_DIR = EXP4_DIR / "results" / "samples"

# 每个特征集对应的 4 个模型 + scaler + split
FEATURE_SETS = ["pre", "full"]
ML_MODELS = ["svm", "rf", "xgboost", "lightgbm"]


def model_path(name: str, feature_set: str) -> Path:
    return EXP2_MODELS_DIR / f"{name}_{feature_set}.pkl"


def scaler_path(feature_set: str) -> Path:
    return EXP2_MODELS_DIR / f"scaler_{feature_set}.pkl"


def print_matrix() -> None:
    print("实验五（AI 代码审查 · 泛化测试）配置")
    print(f"  复用 LLM: {exp4.config.MODEL_ID} @ {exp4.config.DEEPSEEK_BASE_URL}")
    print(f"  LLM 缓存目录（共享实验四）: {exp4.config.CACHE_DIR}")
    print(f"  受试组: {GROUPS}")
    print(f"  上下文: {CONTEXTS} / Prompt: {PROMPTS} / 任务: {TASKS}")
    print(f"  ML 特征集: {FEATURE_SETS} × 模型: {ML_MODELS}")
    print(f"  种子={SEED}, LLM 分类抽样 N={N_CLASSIFY_SAMPLE}, 生成用全 72")


if __name__ == "__main__":
    print_matrix()
