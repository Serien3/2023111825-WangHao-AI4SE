from __future__ import annotations

import socket
import subprocess
import threading
from pathlib import Path

import uvicorn
from playwright.sync_api import sync_playwright

from src.app import create_app
from src.config import Settings
from src.schemas import BatchReview


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


def browser_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "browser-repo"
    repo.mkdir()
    git(repo, "init", "-b", "main")
    git(repo, "config", "user.name", "Browser Test")
    git(repo, "config", "user.email", "browser@example.com")
    (repo / "example.py").write_text("answer = 41\n", encoding="utf-8")
    git(repo, "add", "example.py")
    git(repo, "commit", "-m", "initial")
    (repo / "example.py").write_text("answer = 42\nprint(answer)\n", encoding="utf-8")
    return repo


class BrowserReviewModel:
    model_id = "browser-stub"

    def review(self, batch, profile, *, allow_repair=True):
        path = batch.paths[0]
        line = min(batch.valid_lines[path])
        return BatchReview.model_validate({
            "decision": "REJECT", "risk_level": "high",
            "summary": "发现一个可定位的阻断问题", "reasoning": "边界行为需要修复",
            "blocking_defects": ["边界缺陷"],
            "findings": [{
                "path": path, "line": line, "severity": "blocker",
                "category": "correctness", "message": "新增逻辑未保护边界",
                "suggestion": "在执行前增加显式校验",
            }],
        }), {
            "requests": 1, "usage": {"prompt_tokens": 12, "completion_tokens": 6, "total_tokens": 18},
            "latency": 0.01, "cached": False,
        }

    def summarize(self, reviews, coverage, profile):
        return reviews[0], {
            "requests": 1, "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "latency": 0.01, "cached": False,
        }


class BrowserMLReference:
    def predict(self, snapshot):
        return {
            "status": "ok", "disclaimer": "传统模型仅作实验外迁移参考。",
            "models": [{"model": name, "decision": "MERGE", "merge_probability": 0.5} for name in ("svm", "rf", "xgboost", "lightgbm")],
        }


def test_desktop_review_finding_and_mobile_tabs(tmp_path: Path) -> None:
    repo = browser_repo(tmp_path)
    settings = Settings.from_repo(repo, history_path=tmp_path / "history.jsonl")
    app = create_app(settings, review_model=BrowserReviewModel(), ml_reference=BrowserMLReference())
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            page.goto(f"http://127.0.0.1:{port}")
            page.locator(".file-item").first.wait_for()
            assert page.locator(".review-grid .panel:visible").count() == 3
            assert page.evaluate("document.body.scrollWidth === document.body.clientWidth")

            page.get_by_role("button", name="审查当前文件").click()
            page.get_by_text("新增逻辑未保护边界").wait_for()
            assert page.locator("#result-panel strong", has_text="REJECT").is_visible()
            page.locator(".finding").click()
            page.locator(".diff-line.highlight").wait_for()
            screenshot = page.screenshot()
            assert len(screenshot) > 10_000

            page.set_viewport_size({"width": 390, "height": 844})
            page.reload()
            page.locator(".file-item").first.wait_for()
            assert page.locator(".review-grid .panel:visible").count() == 1
            page.get_by_role("button", name="Diff", exact=True).click()
            assert page.locator("#diff-panel").is_visible()
            assert page.evaluate("document.body.scrollWidth === document.body.clientWidth")
            panels_fit = page.evaluate("""
                [...document.querySelectorAll('.review-grid .panel')]
                    .filter((panel) => getComputedStyle(panel).display !== 'none')
                    .every((panel) => {
                    const rect = panel.getBoundingClientRect();
                    return rect.left >= 0 && rect.right <= innerWidth + 1;
                })
            """)
            assert panels_fit
            browser.close()
    finally:
        server.should_exit = True
        thread.join(timeout=5)
