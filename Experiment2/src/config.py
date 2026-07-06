"""实验二配置：路径、特征组定义、常量。"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# 路径
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
FEATURES_DIR = RESULTS_DIR / "features"
MODELS_DIR = RESULTS_DIR / "models"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"

for _d in (FEATURES_DIR, MODELS_DIR, FIGURES_DIR, METRICS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# 实验一数据路径
EXP1_DATA_DIR = PROJECT_ROOT.parent / "Experiment1" / "results" / "processed"

# --------------------------------------------------------------------------- #
# 特征组定义（用于消融实验与标签泄漏分析）
# --------------------------------------------------------------------------- #

# Pre-review features: 提交时即可获得，无时间泄漏
PRE_REVIEW_FEATURES = [
    # AST features (will be prefixed with 'ast_' in actual feature names)
    'ast_total_nodes', 'ast_max_depth', 'ast_avg_branching',
    'ast_function_definition', 'ast_class_definition',
    'ast_if_statement', 'ast_for_statement', 'ast_while_statement',
    'ast_try_statement', 'ast_call', 'ast_import_statement',
    'ast_assignment', 'ast_binary_operator', 'ast_comparison_operator',
    'ast_return_statement', 'ast_expression_statement',

    # CFG features
    'cfg_nodes', 'cfg_edges', 'cfg_complexity', 'cfg_max_nesting',

    # Statistical features (from PR metadata at submission time)
    'additions', 'deletions', 'changed_files', 'num_commits',
    'files_per_commit', 'churn_ratio',

    # Text features
    'title_len', 'body_len', 'avg_commit_msg_len',
    'has_keyword_fix', 'has_keyword_bug', 'has_keyword_test',
    'has_keyword_refactor', 'has_keyword_add', 'has_keyword_remove',
]

# Review-process features: 审查过程中产生，含时间泄漏
REVIEW_PROCESS_FEATURES = [
    'num_reviews', 'num_reviewers', 'num_review_comments',
    'num_issue_comments', 'review_density',
]

# Full feature set (for upper-bound analysis)
FULL_FEATURES = PRE_REVIEW_FEATURES + REVIEW_PROCESS_FEATURES

# TF-IDF features (dynamically generated, not listed here)
TFIDF_TOP_K = 50

# --------------------------------------------------------------------------- #
# 模型超参数搜索空间
# --------------------------------------------------------------------------- #

SVM_PARAM_GRID = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'kernel': ['rbf'],
    'class_weight': ['balanced'],
}

RF_PARAM_GRID = {
    'n_estimators': [100, 200, 500],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'class_weight': ['balanced'],
}

XGB_PARAM_GRID = {
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 300, 500],
    'subsample': [0.8, 1.0],
}

LGBM_PARAM_GRID = {
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 300, 500],
    'subsample': [0.8, 1.0],
}

# --------------------------------------------------------------------------- #
# 训练常量
# --------------------------------------------------------------------------- #

RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.2  # of remaining after test split
CV_FOLDS = 5

# 类别不平衡处理
IMBALANCE_THRESHOLD = 3.0  # 正负样本比超过此值时应用 SMOTE
SMOTE_ENABLED = False  # 2:1 不算严重，默认不用 SMOTE，靠 class_weight

# --------------------------------------------------------------------------- #
# 并行 / 资源控制（重要：防止 OOM）
# --------------------------------------------------------------------------- #
# 背景：本机 32 核但内存较小（WSL2 默认 ~宿主 50%）。GridSearchCV 若用
# n_jobs=-1 会一次 fork 与核心数相等的 worker，每个 worker 复制一份数据，
# 且树模型内部还会各自多线程 → 内存/线程双重超订，直接把 WSL2 打爆退出。
#
# 因此显式限制并行度：GridSearch 外层并行 N_JOBS 个，模型内层线程 MODEL_N_THREADS
# 个，二者乘积不超过物理核心，且总内存可控。可按机器实际内存调整。
N_JOBS = 4            # GridSearchCV / cross_val 外层并行 worker 数
MODEL_N_THREADS = 2   # 单个树模型（RF/XGB/LGBM）内部线程数
# 二者乘积 4×2=8，远低于 32 核，内存峰值 ~= N_JOBS 份数据副本，安全。

# --------------------------------------------------------------------------- #
# AST 节点类型（tree-sitter Python 常见节点，用于特征提取）
# --------------------------------------------------------------------------- #

AST_NODE_TYPES = [
    'function_definition',
    'class_definition',
    'if_statement',
    'for_statement',
    'while_statement',
    'try_statement',
    'import_statement',
    'import_from_statement',
    'call',
    'assignment',
    'augmented_assignment',
    'binary_operator',
    'comparison_operator',
    'boolean_operator',
    'return_statement',
    'expression_statement',
    'decorated_definition',
    'with_statement',
    'raise_statement',
    'assert_statement',
]

# --------------------------------------------------------------------------- #
# 文本特征关键词
# --------------------------------------------------------------------------- #

TEXT_KEYWORDS = ['fix', 'bug', 'test', 'refactor', 'add', 'remove']
