from __future__ import annotations

import argparse
import json
import logging
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .config import ConfigurationError, REPOSITORY_ROOT, Settings
from .context_builder import ContextBuilder
from .export import markdown_review
from .history import HistoryStore
from .llm_review import DeepSeekReviewModel, ModelNotConfigured, ReviewModel
from .ml_reference import MLReference
from .review_engine import ReviewEngine
from .schemas import HealthResponse, PreflightResponse, ReviewRequest
from .security import UnsafePathError
from .workspace import WorkspaceError, WorkspaceRepository


LOGGER = logging.getLogger("experiment7")
STATIC_DIR = Path(__file__).resolve().parent / "static"


def _error(request: Request, status_code: int, code: str, message: str, **details: object) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "code": code,
            "message": message,
            "details": details,
            "request_id": getattr(request.state, "request_id", "unknown"),
        },
    )


def create_app(
    settings: Settings,
    *,
    review_model: ReviewModel | None = None,
    ml_reference: MLReference | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        settings.history_path.parent.mkdir(parents=True, exist_ok=True)
        settings.cache_dir.mkdir(parents=True, exist_ok=True)
        yield

    app = FastAPI(title="Local Code Review Workbench", version="1.0.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.workspace = WorkspaceRepository(settings)
    app.state.review_model = review_model or DeepSeekReviewModel(settings)
    app.state.ml_reference = ml_reference or MLReference(settings)
    app.state.history = HistoryStore(settings.history_path, settings.history_limit)
    app.state.review_engine = ReviewEngine(
        ContextBuilder(settings), app.state.review_model, app.state.ml_reference
    )
    app.state.progress = {}
    app.state.active_reviews = set()
    app.state.active_reviews_lock = threading.Lock()

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request.state.request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [
            {key: value for key, value in error.items() if key not in {"input", "ctx"}}
            for error in exc.errors()
        ]
        return _error(request, 422, "validation_error", "Request validation failed", errors=errors)

    @app.exception_handler(UnsafePathError)
    async def unsafe_path(request: Request, exc: UnsafePathError) -> JSONResponse:
        return _error(request, 400, "invalid_path", str(exc))

    @app.exception_handler(WorkspaceError)
    async def workspace_error(request: Request, exc: WorkspaceError) -> JSONResponse:
        return _error(request, 400, "workspace_error", str(exc))

    @app.exception_handler(ModelNotConfigured)
    async def model_not_configured(request: Request, exc: ModelNotConfigured) -> JSONResponse:
        return _error(request, 503, "llm_not_configured", str(exc))

    @app.exception_handler(Exception)
    async def internal_error(request: Request, exc: Exception) -> JSONResponse:
        LOGGER.exception("Unhandled request error", extra={"request_id": request.state.request_id})
        return _error(request, 500, "internal_error", "The request could not be completed")

    @app.get("/api/v1/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        model_dir = REPOSITORY_ROOT / "Experiment2" / "results" / "models"
        model_names = ("svm", "rf", "xgboost", "lightgbm")
        available_models = [name for name in model_names if (model_dir / f"{name}_pre.pkl").is_file()]
        history_ready = settings.history_path.parent.is_dir()
        llm_ready = settings.llm_configured or review_model is not None
        return HealthResponse(
            status="ok" if llm_ready else "degraded",
            repository={"id": settings.repo_id, "name": settings.repo_root.name, "ready": True},
            llm={
                "ready": llm_ready,
                "model": settings.model_id,
                "reason": None if llm_ready else "DEEPSEEK_API_KEY is not configured",
            },
            ml={"ready": len(available_models) == 4, "models": available_models},
            history={"ready": history_ready},
        )

    @app.get("/api/v1/workspace")
    async def workspace(request: Request) -> dict:
        snapshot = request.app.state.workspace.snapshot()
        value = snapshot.as_dict()
        value["repository"] = {"id": settings.repo_id, "name": settings.repo_root.name}
        return value

    @app.get("/api/v1/diff")
    async def diff(path: str, request: Request) -> dict:
        snapshot = request.app.state.workspace.snapshot("file", path)
        return snapshot.as_dict(include_diff=True)

    def current_snapshot(payload: ReviewRequest, request: Request):
        snapshot = request.app.state.workspace.snapshot(payload.scope, payload.path)
        if snapshot.snapshot_hash != payload.expected_snapshot_hash:
            return snapshot, _error(
                request, 409, "snapshot_conflict", "The workspace changed; refresh preflight",
                current_snapshot_hash=snapshot.snapshot_hash,
            )
        return snapshot, None

    @app.post("/api/v1/preflight", response_model=PreflightResponse)
    async def preflight(payload: ReviewRequest, request: Request):
        snapshot, conflict = current_snapshot(payload, request)
        if conflict:
            return conflict
        included = [item for item in snapshot.files if item.reviewable]
        excluded = [item for item in snapshot.files if not item.reviewable]
        blocked = any(item.security_rules or item.excluded_reason == "sensitive_path" for item in excluded)
        plan = ContextBuilder(settings).build(snapshot)
        return PreflightResponse(
            snapshot_hash=snapshot.snapshot_hash,
            included_items=[{"path": item.path, "characters": len(item.diff)} for item in included],
            excluded_items=[{
                "path": item.path, "reason": item.excluded_reason, "rules": item.security_rules,
            } for item in excluded],
            context_sources=plan.sources,
            char_count=sum(len(batch.context) for batch in plan.batches),
            batch_plan=[{"batch": batch.number, "items": batch.items} for batch in plan.batches],
            unreviewed_items=plan.unreviewed_items, blocked=blocked,
        )

    @app.post("/api/v1/reviews")
    def create_review(payload: ReviewRequest, request: Request):
        snapshot, conflict = current_snapshot(payload, request)
        if conflict:
            return conflict
        if not snapshot.files:
            return _error(request, 400, "no_changes", "The selected scope has no changes")
        if not any(item.reviewable for item in snapshot.files):
            return _error(
                request, 400, "sensitive_content_blocked",
                "No reviewable files remain after security checks",
                excluded=[{"path": item.path, "reason": item.excluded_reason} for item in snapshot.files],
            )
        progress_key = request.state.request_id
        review_key = (
            snapshot.snapshot_hash, payload.profile, payload.scope, payload.path,
            request.app.state.review_model.model_id,
        )
        with request.app.state.active_reviews_lock:
            if review_key in request.app.state.active_reviews:
                return _error(
                    request, 409, "review_in_progress",
                    "An equivalent review is already running",
                )
            request.app.state.active_reviews.add(review_key)
        request.app.state.progress[progress_key] = {
            "status": "reviewing", "current_batch": 0, "request_id": progress_key,
        }
        try:
            def update_progress(current_batch: int, total_batches: int) -> None:
                request.app.state.progress[progress_key] = {
                    "status": "reviewing", "current_batch": current_batch,
                    "total_batches": total_batches, "request_id": progress_key,
                }

            result = request.app.state.review_engine.run(
                snapshot, payload.profile, progress_key, update_progress
            )
            current = request.app.state.workspace.snapshot(payload.scope, payload.path)
            result["stale"] = current.snapshot_hash != snapshot.snapshot_hash
            result.update({
                "schema_version": 1,
                "scope": payload.scope,
                "path": payload.path,
                "files": [item.path for item in snapshot.files],
                "snapshot_hash": snapshot.snapshot_hash,
                "diff_summary": {
                    "sha256": snapshot.snapshot_hash,
                    "additions": sum(item.additions for item in snapshot.files),
                    "deletions": sum(item.deletions for item in snapshot.files),
                },
            })
            history_record = dict(result)
            history_record["stale"] = False
            if payload.include_diff_in_history:
                safe_diff = "\n".join(item.diff for item in snapshot.files if item.reviewable)
                history_record["saved_diff"] = safe_diff[:settings.batch_char_budget]
                history_record["saved_diff_truncated"] = len(safe_diff) > settings.batch_char_budget
            request.app.state.history.append(history_record)
            request.app.state.progress[progress_key] = {
                "status": "complete", "current_batch": result["coverage"]["reviewed_batches"],
                "total_batches": result["coverage"]["planned_batches"], "review_id": result["id"],
            }
            return result
        except ValueError as exc:
            request.app.state.progress[progress_key] = {"status": "error"}
            return _error(request, 400, "context_empty", str(exc))
        finally:
            with request.app.state.active_reviews_lock:
                request.app.state.active_reviews.discard(review_key)

    @app.get("/api/v1/progress/{request_id}")
    async def progress(request_id: str, request: Request):
        state = request.app.state.progress.get(request_id)
        if state is None:
            return _error(request, 404, "progress_not_found", "Review progress was not found")
        return state

    def with_dynamic_stale(record: dict, request: Request) -> dict:
        value = dict(record)
        try:
            current = request.app.state.workspace.snapshot(record["scope"], record.get("path"))
            value["stale"] = current.snapshot_hash != record["snapshot_hash"]
        except (WorkspaceError, UnsafePathError):
            value["stale"] = True
        return value

    @app.get("/api/v1/history/export")
    async def export_history(request: Request) -> Response:
        records, warnings = request.app.state.history.all()
        payload = {"schema_version": 1, "items": records, "storage_warnings": warnings}
        return Response(
            json.dumps(payload, ensure_ascii=False, indent=2),
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=review-history.json"},
        )

    @app.get("/api/v1/history")
    async def history(
        request: Request,
        page: int = Query(1, ge=1),
        page_size: int = Query(20, ge=1, le=100),
        path: str | None = None,
        status: str | None = None,
    ) -> dict:
        return request.app.state.history.query(
            page=page, page_size=page_size, path=path, status=status
        )

    @app.get("/api/v1/reviews/{review_id}")
    async def review_detail(review_id: str, request: Request):
        record = request.app.state.history.get(review_id)
        if record is None:
            return _error(request, 404, "review_not_found", "Review history record was not found")
        return with_dynamic_stale(record, request)

    @app.delete("/api/v1/history/{review_id}")
    async def delete_history(review_id: str, request: Request):
        if not request.app.state.history.delete(review_id):
            return _error(request, 404, "review_not_found", "Review history record was not found")
        return {"deleted": True, "id": review_id}

    @app.delete("/api/v1/history")
    async def clear_history(request: Request) -> dict:
        return {"deleted": request.app.state.history.clear()}

    @app.get("/api/v1/exports/{review_id}.{format}")
    async def export_review(review_id: str, format: str, request: Request):
        record = request.app.state.history.get(review_id)
        if record is None:
            return _error(request, 404, "review_not_found", "Review history record was not found")
        if format == "json":
            content = json.dumps(with_dynamic_stale(record, request), ensure_ascii=False, indent=2)
            media_type = "application/json"
        elif format == "md":
            content = markdown_review(with_dynamic_stale(record, request))
            media_type = "text/markdown; charset=utf-8"
        else:
            return _error(request, 400, "invalid_export_format", "Export format must be json or md")
        return Response(
            content,
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename=review-{review_id}.{format}"},
        )

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment 7 local code review workbench")
    subparsers = parser.add_subparsers(dest="command", required=True)
    serve = subparsers.add_parser("serve", help="serve the locked Git worktree")
    serve.add_argument("--repo", required=True, help="Git worktree root")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        settings = Settings.from_repo(args.repo, host=args.host, port=args.port)
    except ConfigurationError as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc
    if settings.host not in {"127.0.0.1", "::1", "localhost"}:
        LOGGER.warning("Listening beyond loopback exposes access to repository source code")
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
