"""实验二 步骤四：模型训练（SVM, RF, XGBoost, LightGBM）。

包含标签泄漏分析（pre-review vs full features）、跨仓库分层划分、类别不平衡处理。

用法：
    uv run python -m src.train --model all          # 训练全部四个模型
    uv run python -m src.train --model svm          # 仅训练 SVM
    uv run python -m src.train --feature-set full   # 使用全特征（默认）
    uv run python -m src.train --feature-set pre    # 仅用 pre-review 特征
"""
from __future__ import annotations

import argparse
import os

# 在导入 numpy/sklearn 之前限制底层线程库（BLAS/OpenMP），
# 防止每个 GridSearch worker 内部再各自开满线程导致超订 → OOM。
# 必须在 import numpy 之前设置才生效。
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "2")

import pickle
import time
import json
import warnings
from pathlib import Path

import lightgbm as lgb
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config

warnings.filterwarnings("ignore", category=UserWarning)


# --------------------------------------------------------------------------- #
# 特征选择（根据 feature-set 参数筛选列）
# --------------------------------------------------------------------------- #
def select_features(df: pd.DataFrame, feature_set: str) -> tuple[pd.DataFrame, list[str]]:
    """根据 feature_set ('pre' / 'full') 选择特征列，返回 (X_df, feature_names)。"""
    # 保留 repo/number/is_merged 外的全部特征列
    meta_cols = {"repo", "number", "is_merged"}
    all_features = [c for c in df.columns if c not in meta_cols]

    if feature_set == "pre":
        # Pre-review: 排除审查过程特征
        exclude = set(config.REVIEW_PROCESS_FEATURES)
        selected = [c for c in all_features if c not in exclude]
    elif feature_set == "full":
        selected = all_features
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")

    print(f"特征集 '{feature_set}': {len(selected)} 列")
    return df[selected], selected


# --------------------------------------------------------------------------- #
# 数据划分（跨仓库分层 + 标准化）
# --------------------------------------------------------------------------- #
def split_data(
    df: pd.DataFrame, feature_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, StandardScaler]:
    """分层划分 train/val/test，标准化特征，返回 (X_train, X_val, X_test, y_*, scaler)。"""
    # 按 repo 分层（保证每个仓库在 train/val/test 都有代表）
    X = df[feature_cols].copy()
    y = df["is_merged"].astype(int)
    repos = df["repo"]

    # 先切出 test（20%）
    X_temp, X_test, y_temp, y_test, repo_temp, repo_test = train_test_split(
        X, y, repos, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=repos
    )

    # 再从剩余中切出 val（剩余的 20%，即全体的 16%）
    X_train, X_val, y_train, y_val, _, _ = train_test_split(
        X_temp, y_temp, repo_temp,
        test_size=config.VAL_SIZE, random_state=config.RANDOM_STATE, stratify=repo_temp
    )

    print(f"数据划分: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    print(f"train 正负比: {y_train.sum()} merged / {(~y_train.astype(bool)).sum()} non-merged")

    # 标准化（SVM/XGBoost 需要；RF 不需要但不影响）
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val), columns=X_val.columns, index=X_val.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler


# --------------------------------------------------------------------------- #
# 模型训练（GridSearchCV + 5-fold CV）
# --------------------------------------------------------------------------- #
def train_svm(X_train, y_train) -> SVC:
    """训练 SVM，使用 GridSearchCV 调参。"""
    print("\n[SVM] 开始训练 + 超参数搜索 ...")
    clf = GridSearchCV(
        SVC(random_state=config.RANDOM_STATE, probability=True),
        config.SVM_PARAM_GRID,
        cv=config.CV_FOLDS,
        scoring="f1",
        n_jobs=config.N_JOBS,  # 受控并行，防 OOM（见 config.N_JOBS 说明）
        verbose=1,
    )
    clf.fit(X_train, y_train)
    print(f"[SVM] 最佳参数: {clf.best_params_}")
    print(f"[SVM] 交叉验证 F1: {clf.best_score_:.4f}")
    return clf.best_estimator_


def train_rf(X_train, y_train) -> RandomForestClassifier:
    """训练 Random Forest，使用 GridSearchCV 调参。"""
    print("\n[RF] 开始训练 + 超参数搜索 ...")
    clf = GridSearchCV(
        RandomForestClassifier(random_state=config.RANDOM_STATE, n_jobs=config.MODEL_N_THREADS),
        config.RF_PARAM_GRID,
        cv=config.CV_FOLDS,
        scoring="f1",
        n_jobs=config.N_JOBS,  # 外层并行受控；RF 内部线程 = MODEL_N_THREADS
        verbose=1,
    )
    clf.fit(X_train, y_train)
    print(f"[RF] 最佳参数: {clf.best_params_}")
    print(f"[RF] 交叉验证 F1: {clf.best_score_:.4f}")
    return clf.best_estimator_


def train_xgboost(X_train, y_train, X_val, y_val) -> xgb.XGBClassifier:
    """训练 XGBoost，使用 GridSearchCV + early stopping。"""
    print("\n[XGBoost] 开始训练 + 超参数搜索 ...")
    # 计算类别权重（处理不平衡）
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    clf = GridSearchCV(
        xgb.XGBClassifier(
            random_state=config.RANDOM_STATE,
            scale_pos_weight=scale_pos_weight,
            early_stopping_rounds=10,
            eval_metric="logloss",
            n_jobs=config.MODEL_N_THREADS,  # 内部线程受控
        ),
        config.XGB_PARAM_GRID,
        cv=config.CV_FOLDS,
        scoring="f1",
        n_jobs=config.N_JOBS,
        verbose=1,
    )
    # 注意：GridSearchCV 会用内部 CV，这里 eval_set 只在最终 best_estimator_ refit 时用
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print(f"[XGBoost] 最佳参数: {clf.best_params_}")
    print(f"[XGBoost] 交叉验证 F1: {clf.best_score_:.4f}")
    return clf.best_estimator_


def train_lightgbm(X_train, y_train, X_val, y_val) -> lgb.LGBMClassifier:
    """训练 LightGBM，使用 GridSearchCV + early stopping。"""
    print("\n[LightGBM] 开始训练 + 超参数搜索 ...")
    clf = GridSearchCV(
        lgb.LGBMClassifier(
            random_state=config.RANDOM_STATE,
            is_unbalance=True,
            verbose=-1,
            n_jobs=config.MODEL_N_THREADS,  # 内部线程受控
        ),
        config.LGBM_PARAM_GRID,
        cv=config.CV_FOLDS,
        scoring="f1",
        n_jobs=config.N_JOBS,
        verbose=1,
    )
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(10, verbose=False)],
    )
    print(f"[LightGBM] 最佳参数: {clf.best_params_}")
    print(f"[LightGBM] 交叉验证 F1: {clf.best_score_:.4f}")
    return clf.best_estimator_


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="实验二模型训练")
    ap.add_argument(
        "--model", choices=["svm", "rf", "xgboost", "lightgbm", "all"], default="all"
    )
    ap.add_argument(
        "--feature-set", choices=["pre", "full"], default="full",
        help="'pre'=pre-review 特征（无泄漏），'full'=全特征（含审查过程特征）"
    )
    args = ap.parse_args()

    # 读取特征矩阵
    feat_path = config.FEATURES_DIR / "features.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(
            f"{feat_path} 不存在。请先运行 feature_extraction.py"
        )

    print(f"读取特征矩阵: {feat_path}")
    df = pd.read_parquet(feat_path)
    print(f"数据规模: {df.shape[0]} PR × {df.shape[1]} 列")

    # 特征选择
    X_df, feature_cols = select_features(df, args.feature_set)

    # 数据划分 + 标准化
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = split_data(df, feature_cols)

    # 保存 scaler 与特征列表（evaluate 时需要）
    scaler_path = config.MODELS_DIR / f"scaler_{args.feature_set}.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump({"scaler": scaler, "feature_cols": feature_cols}, f)
    print(f"已保存 scaler: {scaler_path}")

    # 保存划分后的数据（evaluate 时需要）
    split_path = config.MODELS_DIR / f"split_{args.feature_set}.pkl"
    with open(split_path, "wb") as f:
        pickle.dump({
            "X_train": X_train, "X_val": X_val, "X_test": X_test,
            "y_train": y_train, "y_val": y_val, "y_test": y_test,
        }, f)
    print(f"已保存数据划分: {split_path}")

    # 训练模型
    models_to_train = (
        ["svm", "rf", "xgboost", "lightgbm"] if args.model == "all" else [args.model]
    )

    train_times: dict = {}
    for model_name in models_to_train:
        t0 = time.perf_counter()
        if model_name == "svm":
            model = train_svm(X_train, y_train)
        elif model_name == "rf":
            model = train_rf(X_train, y_train)
        elif model_name == "xgboost":
            model = train_xgboost(X_train, y_train, X_val, y_val)
        elif model_name == "lightgbm":
            model = train_lightgbm(X_train, y_train, X_val, y_val)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        train_times[model_name] = round(time.perf_counter() - t0, 3)

        # 保存模型
        model_path = config.MODELS_DIR / f"{model_name}_{args.feature_set}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"已保存模型: {model_path}\n")

    # 记录训练耗时（含网格搜索），供 evaluate.py 画训练时间图
    # 仅在 full 特征集且训练全部模型时写入，避免部分覆盖
    if args.feature_set == "full" and args.model == "all":
        tt_path = config.METRICS_DIR / "training_time.json"
        with open(tt_path, "w") as f:
            json.dump(train_times, f, indent=2)
        print(f"已保存训练耗时: {tt_path}")

    print("\n训练完成！运行 evaluate.py 进行测试集评估。")


if __name__ == "__main__":
    main()
