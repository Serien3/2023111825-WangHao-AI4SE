from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .config import Settings
from .context_builder import ContextBuilder
from .llm_review import DeepSeekReviewModel
from .ml_reference import MLReference
from .review_engine import ReviewEngine
from .workspace import WorkspaceRepository


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explicit paid strict-profile DeepSeek smoke test"
    )
    parser.add_argument("--repo", required=True, help="Locked Git worktree root")
    parser.add_argument("--path", required=True, help="Modified repository-relative file")
    parser.add_argument(
        "--confirm-paid",
        action="store_true",
        help="Confirm that this command may make paid DeepSeek requests",
    )
    args = parser.parse_args()
    if not args.confirm_paid:
        raise SystemExit("Refusing to call DeepSeek without --confirm-paid")
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise SystemExit("DEEPSEEK_API_KEY is not configured")

    settings = Settings.from_repo(args.repo)
    snapshot = WorkspaceRepository(settings).snapshot("file", args.path)
    excluded = [item.path for item in snapshot.files if not item.reviewable]
    if excluded:
        raise SystemExit(f"Security preflight blocked the selected file: {excluded}")
    engine = ReviewEngine(
        ContextBuilder(settings), DeepSeekReviewModel(settings), MLReference(settings)
    )
    result = engine.run(snapshot, "strict", "paid-smoke")
    result.update({
        "schema_version": 1,
        "scope": "file",
        "path": args.path,
        "snapshot_hash": snapshot.snapshot_hash,
    })
    output = settings.repo_root / "Experiment7" / "results" / "metrics" / "paid_smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "status": result["status"],
        "decision": result["decision"],
        "usage": result["usage"],
        "latency": result["latency"],
        "cached": result["cached"],
        "coverage": result["coverage"],
        "requests": result["requests"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()