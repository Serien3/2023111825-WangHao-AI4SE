from __future__ import annotations

import importlib
import importlib.util
import pickle
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tree_sitter import Parser

from .config import REPOSITORY_ROOT, Settings
from .workspace import WorkspaceSnapshot


DISCLAIMER = "传统模型在 GitHub PR 特征上训练；当前本地输入缺少部分 PR 元数据，结果仅作实验外迁移参考，不参与合并就绪度结论。"


def _load_exp2():
    alias = "exp7_exp2src"
    if alias not in sys.modules:
        src = REPOSITORY_ROOT / "Experiment2" / "src"
        spec = importlib.util.spec_from_file_location(alias, src / "__init__.py", submodule_search_locations=[str(src)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[alias] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return importlib.import_module(f"{alias}.feature_extraction")


@lru_cache(maxsize=1)
def _restored_tfidf() -> TfidfVectorizer:
    features_path = REPOSITORY_ROOT / "Experiment2" / "results" / "features" / "features.parquet"
    prs_path = REPOSITORY_ROOT / "Experiment1" / "results" / "processed" / "prs.parquet"
    feature_keys = pd.read_parquet(features_path, columns=["repo", "number"])
    feature_columns = pd.read_parquet(features_path).columns
    vocabulary = [column.removeprefix("tfidf_") for column in feature_columns if column.startswith("tfidf_")]
    prs = pd.read_parquet(prs_path, columns=["repo", "number", "title", "body"])
    corpus = feature_keys.merge(prs, on=["repo", "number"], how="left")
    texts = (corpus["title"].fillna("") + " " + corpus["body"].fillna("")).tolist()
    vectorizer = TfidfVectorizer(vocabulary=vocabulary, stop_words="english")
    vectorizer.fit(texts)
    return vectorizer


class MLReference:
    def __init__(self, settings: Settings):
        self.repo_root = settings.repo_root
        self.model_dir = REPOSITORY_ROOT / "Experiment2" / "results" / "models"

    def predict(self, snapshot: WorkspaceSnapshot) -> dict:
        python_changes = [item for item in snapshot.files if item.reviewable and item.language == "python"]
        if not python_changes:
            return {"status": "not_applicable", "reason": "该语言未经过实验二模型验证", "models": [], "disclaimer": DISCLAIMER}
        missing = ["pr_title", "pr_body", "num_commits"]
        try:
            feature_module = _load_exp2()
            parser = Parser(feature_module.PY_LANGUAGE)
            patches = [item.diff for item in python_changes]
            row = feature_module.extract_code_features_for_pr(patches, parser)
            additions = sum(item.additions for item in snapshot.files)
            deletions = sum(item.deletions for item in snapshot.files)
            row.update(feature_module.extract_statistical_features(pd.Series({
                "additions": additions, "deletions": deletions,
                "changed_files": len(snapshot.files), "num_commits": 0,
            })))
            message = subprocess.run(
                ["git", "log", "-1", "--pretty=%s"], cwd=self.repo_root,
                capture_output=True, text=True,
            ).stdout.strip()
            row.update(feature_module.extract_text_features(pd.Series({"title": "", "body": ""}), [message] if message else []))
            with open(self.model_dir / "scaler_pre.pkl", "rb") as handle:
                scaler_obj = pickle.load(handle)
            feature_cols = scaler_obj["feature_cols"]
            vectorizer = _restored_tfidf()
            local_text = ""
            tfidf_values = vectorizer.transform([local_text]).toarray()[0]
            row.update({
                f"tfidf_{term}": float(value)
                for term, value in zip(vectorizer.get_feature_names_out(), tfidf_values, strict=True)
            })
            values = pd.DataFrame([{column: float(row.get(column, 0)) for column in feature_cols}])
            transformed = pd.DataFrame(
                scaler_obj["scaler"].transform(values[feature_cols]), columns=feature_cols
            )
            results = []
            for name in ("svm", "rf", "xgboost", "lightgbm"):
                with open(self.model_dir / f"{name}_pre.pkl", "rb") as handle:
                    model = pickle.load(handle)
                prediction = int(model.predict(transformed)[0])
                probability = None
                if hasattr(model, "predict_proba"):
                    probability = float(model.predict_proba(transformed)[0][1])
                results.append({"model": name, "decision": "MERGE" if prediction == 1 else "REJECT", "merge_probability": probability})
            return {
                "status": "ok", "models": results, "missing_features": missing,
                "feature_count": len(feature_cols), "tfidf_idf_source": "Experiment 2 human PR corpus",
                "tfidf_local_transform": "empty PR title/body", "disclaimer": DISCLAIMER,
            }
        except Exception as exc:
            return {"status": "error", "models": [], "error": f"ML reference unavailable: {type(exc).__name__}", "missing_features": missing, "disclaimer": DISCLAIMER}