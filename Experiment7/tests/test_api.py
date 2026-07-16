from __future__ import annotations

import json
import subprocess
import threading
from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient

from src.app import create_app
from src.config import Settings
from src.history import HistoryStore
from src.ml_reference import MLReference
from src.schemas import BatchReview


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "sample"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.com")
    (repo / "example.py").write_text("answer = 41\n", encoding="utf-8")
    _git(repo, "add", "example.py")
    _git(repo, "commit", "-m", "initial")
    return repo


class StubReviewModel:
    model_id = "stub-review"

    def __init__(self) -> None:
        self.calls = 0
        self.summary_calls = 0

    def review(self, batch, profile, *, allow_repair=True):
        self.calls += 1
        return BatchReview.model_validate({
            "decision": "MERGE",
            "risk_level": "low",
            "summary": "修改可审查",
            "reasoning": "桩模型未发现其他问题",
            "blocking_defects": ["存在阻断缺陷"],
            "findings": [
                {
                    "path": batch.paths[0], "line": min(batch.valid_lines[batch.paths[0]]),
                    "severity": "blocker", "category": "correctness",
                    "message": "新增逻辑破坏边界条件", "suggestion": "补充边界校验",
                },
                {
                    "path": batch.paths[0], "line": 9999,
                    "severity": "warning", "category": "testing",
                    "message": "不可定位意见", "suggestion": "补充测试",
                },
            ],
        }), {
            "requests": 1, "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "latency": 0.01, "cached": False,
        }

    def summarize(self, reviews, coverage, profile):
        self.summary_calls += 1
        return BatchReview.model_validate({
            "decision": "MERGE", "risk_level": "low",
            "summary": "多批结构化汇总", "reasoning": "汇总不能推翻批次阻断结论",
            "blocking_defects": [], "findings": [],
        }), {
            "requests": 1, "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
            "latency": 0.01, "cached": False,
        }


class StubMLReference:
    def predict(self, snapshot):
        return {"status": "ok", "models": [{"model": name, "decision": "MERGE", "merge_probability": 0.5} for name in ("svm", "rf", "xgboost", "lightgbm")]}


class BlockingReviewModel(StubReviewModel):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def review(self, batch, profile, *, allow_repair=True):
        self.started.set()
        assert self.release.wait(timeout=5)
        return super().review(batch, profile, allow_repair=allow_repair)


def test_health_is_degraded_without_api_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    repo = make_repo(tmp_path)
    settings = Settings.from_repo(repo, history_path=tmp_path / "history.jsonl")

    with TestClient(create_app(settings)) as client:
        response = client.get("/api/v1/health", headers={"X-Request-ID": "test-request"})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "test-request"
    assert response.json()["status"] == "degraded"
    assert response.json()["repository"]["name"] == "sample"
    assert response.json()["ml"]["models"] == ["svm", "rf", "xgboost", "lightgbm"]


def test_workspace_combines_staged_unstaged_and_untracked(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    (repo / "example.py").write_text("answer = 42\n", encoding="utf-8")
    _git(repo, "add", "example.py")
    (repo / "example.py").write_text("answer = 43\nprint(answer)\n", encoding="utf-8")
    (repo / "new.py").write_text("value = 1\n", encoding="utf-8")
    settings = Settings.from_repo(repo, history_path=tmp_path / "history.jsonl")

    with TestClient(create_app(settings)) as client:
        workspace = client.get("/api/v1/workspace").json()
        diff = client.get("/api/v1/diff", params={"path": "example.py"}).json()

    assert [item["path"] for item in workspace["files"]] == ["example.py", "new.py"]
    assert diff["files"][0]["additions"] == 2
    assert diff["files"][0]["deletions"] == 1
    assert "+answer = 43" in diff["files"][0]["diff"]
    assert "+answer = 42" not in diff["files"][0]["diff"]
    assert diff["files"][0]["hunks"][0]["changed_new_lines"] == [1, 2]


def test_preflight_blocks_credentials_and_detects_conflict(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    (repo / "safe.py").write_text("value = 1\n", encoding="utf-8")
    (repo / "secret.py").write_text(
        'api_key = "abcdefghijklmnop123456"\n', encoding="utf-8"
    )
    settings = Settings.from_repo(repo, history_path=tmp_path / "history.jsonl")

    with TestClient(create_app(settings)) as client:
        observed = client.get("/api/v1/workspace").json()["snapshot_hash"]
        payload = {"scope": "workspace", "profile": "strict", "expected_snapshot_hash": observed}
        report = client.post("/api/v1/preflight", json=payload)
        (repo / "safe.py").write_text("value = 2\n", encoding="utf-8")
        conflict = client.post("/api/v1/preflight", json=payload)

    assert report.status_code == 200
    assert report.json()["blocked"] is True
    assert report.json()["excluded_items"] == [
        {"path": "secret.py", "reason": "credential_detected", "rules": ["api_key"]}
    ]
    assert "abcdefghijklmnop123456" not in report.text
    assert conflict.status_code == 409
    assert conflict.json()["code"] == "snapshot_conflict"


def test_review_enforces_blocker_and_discards_invalid_location(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    (repo / "example.py").write_text("answer = 42\n", encoding="utf-8")
    settings = Settings.from_repo(repo, history_path=tmp_path / "history.jsonl")
    model = StubReviewModel()

    with TestClient(create_app(settings, review_model=model, ml_reference=StubMLReference())) as client:
        snapshot_hash = client.get(
            "/api/v1/diff", params={"path": "example.py"}
        ).json()["snapshot_hash"]
        response = client.post("/api/v1/reviews", json={
            "scope": "file", "path": "example.py", "profile": "strict",
            "expected_snapshot_hash": snapshot_hash,
        }, headers={"X-Request-ID": "review-request"})
        progress = client.get("/api/v1/progress/review-request")

    assert response.status_code == 200
    result = response.json()
    assert result["decision"] == "REJECT"
    assert result["risk_level"] == "high"
    assert len(result["findings"]) == 1
    assert result["coverage"]["warnings"] == ["Discarded unlocatable finding for example.py"]
    assert result["coverage"]["planned_batches"] == 1
    assert model.calls == 1
    assert len(result["ml_reference"]["models"]) == 4
    assert progress.json()["status"] == "complete"


def test_history_stale_exports_delete_and_clear(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    (repo / "example.py").write_text("answer = 42\n", encoding="utf-8")
    settings = Settings.from_repo(repo, history_path=tmp_path / "history.jsonl")

    with TestClient(create_app(settings, review_model=StubReviewModel(), ml_reference=StubMLReference())) as client:
        snapshot_hash = client.get("/api/v1/diff", params={"path": "example.py"}).json()["snapshot_hash"]
        created = client.post("/api/v1/reviews", json={
            "scope": "file", "path": "example.py", "profile": "strict",
            "expected_snapshot_hash": snapshot_hash, "include_diff_in_history": False,
        }).json()
        listing = client.get("/api/v1/history")
        exported_md = client.get(f"/api/v1/exports/{created['id']}.md")
        exported_json = client.get(f"/api/v1/exports/{created['id']}.json")
        (repo / "example.py").write_text("answer = 43\n", encoding="utf-8")
        detail = client.get(f"/api/v1/reviews/{created['id']}")
        deleted = client.delete(f"/api/v1/history/{created['id']}")
        cleared = client.delete("/api/v1/history")

    assert listing.json()["total"] == 1
    assert "saved_diff" not in settings.history_path.read_text(encoding="utf-8")
    assert "合并就绪度" in exported_md.text
    assert exported_json.json()["id"] == created["id"]
    assert detail.json()["stale"] is True
    assert deleted.json()["deleted"] is True
    assert cleared.json()["deleted"] == 0


def test_review_caps_four_code_calls_plus_one_summary(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    for index in range(5):
        (repo / f"part_{index}.py").write_text(
            f"value_{index} = {index}\n" + "# context\n" * 8, encoding="utf-8"
        )
    base = Settings.from_repo(repo, history_path=tmp_path / "history.jsonl")
    settings = replace(base, batch_char_budget=170, max_review_batches=4)
    model = StubReviewModel()

    with TestClient(create_app(settings, review_model=model, ml_reference=StubMLReference())) as client:
        snapshot_hash = client.get("/api/v1/workspace").json()["snapshot_hash"]
        response = client.post("/api/v1/reviews", json={
            "scope": "workspace", "profile": "strict",
            "expected_snapshot_hash": snapshot_hash,
        })

    assert response.status_code == 200
    result = response.json()
    assert model.calls == 4
    assert model.summary_calls == 1
    assert result["requests"] == {"code": 4, "summary": 1}
    assert result["status"] == "INCOMPLETE"
    assert result["decision"] == "REJECT"
    assert result["coverage"]["unreviewed_items"]


def test_workspace_rename_delete_and_path_escape(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    (repo / "deleted.py").write_text("old = True\n", encoding="utf-8")
    _git(repo, "add", "deleted.py")
    _git(repo, "commit", "-m", "add deletable")
    _git(repo, "mv", "example.py", "renamed.py")
    (repo / "deleted.py").unlink()
    settings = Settings.from_repo(repo, history_path=tmp_path / "history.jsonl")

    with TestClient(create_app(settings)) as client:
        workspace = client.get("/api/v1/workspace")
        escaped = client.get("/api/v1/diff", params={"path": "../outside.py"})

    files = {item["path"]: item for item in workspace.json()["files"]}
    assert files["renamed.py"]["status"] == "renamed"
    assert files["renamed.py"]["old_path"] == "example.py"
    assert files["deleted.py"]["status"] == "deleted"
    assert escaped.status_code == 400
    assert escaped.json()["code"] == "invalid_path"


def test_history_retains_latest_two_hundred(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path / "reviews.jsonl", limit=200)
    for index in range(205):
        store.append({"id": str(index), "created_at": f"2026-07-16T00:{index // 60:02d}:{index % 60:02d}+00:00"})

    records, warnings = store.all()

    assert len(records) == 200
    assert records[0]["id"] == "204"
    assert records[-1]["id"] == "5"
    assert warnings == []


def test_real_ml_reference_uses_four_pre_review_models(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    (repo / "example.py").write_text("answer = 42\n", encoding="utf-8")
    settings = Settings.from_repo(repo, history_path=tmp_path / "history.jsonl")
    from src.workspace import WorkspaceRepository

    result = MLReference(settings).predict(WorkspaceRepository(settings).snapshot("file", "example.py"))

    assert result["status"] == "ok", result
    assert result["feature_count"] == 93
    assert [item["model"] for item in result["models"]] == ["svm", "rf", "xgboost", "lightgbm"]
    assert result["missing_features"] == ["pr_title", "pr_body", "num_commits"]


def test_equivalent_concurrent_review_is_rejected_and_progress_is_readable(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    (repo / "example.py").write_text("answer = 42\n", encoding="utf-8")
    settings = Settings.from_repo(repo, history_path=tmp_path / "history.jsonl")
    model = BlockingReviewModel()
    app = create_app(settings, review_model=model, ml_reference=StubMLReference())

    with TestClient(app) as client:
        snapshot_hash = client.get("/api/v1/diff", params={"path": "example.py"}).json()["snapshot_hash"]
        payload = {
            "scope": "file", "path": "example.py", "profile": "strict",
            "expected_snapshot_hash": snapshot_hash,
        }
        first_response = {}

        def run_first() -> None:
            first_response["value"] = client.post(
                "/api/v1/reviews", json=payload, headers={"X-Request-ID": "first-review"}
            )

        thread = threading.Thread(target=run_first)
        thread.start()
        assert model.started.wait(timeout=5)
        progress = client.get("/api/v1/progress/first-review")
        duplicate = client.post(
            "/api/v1/reviews", json=payload, headers={"X-Request-ID": "duplicate-review"}
        )
        model.release.set()
        thread.join(timeout=5)

    assert progress.status_code == 200
    assert progress.json()["status"] == "reviewing"
    assert duplicate.status_code == 409
    assert duplicate.json()["code"] == "review_in_progress"
    assert first_response["value"].status_code == 200


def test_binary_and_symlink_are_excluded_without_content_leak(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    binary_secret = b"binary-secret\x00payload"
    (repo / "image.bin").write_bytes(binary_secret)
    outside = tmp_path / "outside.py"
    outside.write_text("outside_secret = 'do-not-read'\n", encoding="utf-8")
    (repo / "linked.py").symlink_to(outside)
    settings = Settings.from_repo(repo, history_path=tmp_path / "history.jsonl")

    with TestClient(create_app(settings)) as client:
        workspace = client.get("/api/v1/workspace")
        binary_diff = client.get("/api/v1/diff", params={"path": "image.bin"})
        linked_diff = client.get("/api/v1/diff", params={"path": "linked.py"})

    files = {item["path"]: item for item in workspace.json()["files"]}
    assert files["image.bin"]["excluded_reason"] == "binary_file"
    assert files["linked.py"]["excluded_reason"] == "unsafe_file"
    assert "binary-secret" not in binary_diff.text
    assert "do-not-read" not in linked_diff.text


def test_explicit_diff_history_is_security_checked_and_budgeted(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    (repo / "example.py").write_text("answer = 42\n" + "print(answer)\n" * 20, encoding="utf-8")
    base = Settings.from_repo(repo, history_path=tmp_path / "history.jsonl")
    settings = replace(base, batch_char_budget=300)

    with TestClient(create_app(settings, review_model=StubReviewModel(), ml_reference=StubMLReference())) as client:
        snapshot_hash = client.get("/api/v1/diff", params={"path": "example.py"}).json()["snapshot_hash"]
        created = client.post("/api/v1/reviews", json={
            "scope": "file", "path": "example.py", "profile": "strict",
            "expected_snapshot_hash": snapshot_hash, "include_diff_in_history": True,
        })

    assert created.status_code == 200
    record = json.loads(settings.history_path.read_text(encoding="utf-8"))
    assert len(record["saved_diff"]) <= settings.batch_char_budget
    assert record["saved_diff_truncated"] is True