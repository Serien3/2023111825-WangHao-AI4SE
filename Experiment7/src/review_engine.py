from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime
from typing import Callable

from .context_builder import ContextBuilder
from .llm_review import ReviewModel
from .ml_reference import MLReference
from .schemas import Finding
from .workspace import WorkspaceSnapshot


SEVERITY_ORDER = {"blocker": 0, "warning": 1, "info": 2}


class ReviewEngine:
    def __init__(self, context_builder: ContextBuilder, model: ReviewModel, ml: MLReference):
        self.context_builder = context_builder
        self.model = model
        self.ml = ml

    def run(
        self,
        snapshot: WorkspaceSnapshot,
        profile: str,
        request_id: str,
        progress: Callable[[int, int], None] | None = None,
    ) -> dict:
        started = time.perf_counter()
        plan = self.context_builder.build(snapshot)
        if not plan.batches:
            raise ValueError("No reviewable context")
        batch_results = []
        warnings: list[str] = []
        total_requests = 0
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        total_latency = 0.0
        all_cached = True
        summary_requests = 0
        for batch in plan.batches:
            if progress:
                progress(batch.number, len(plan.batches))
            if total_requests >= 4:
                plan.unreviewed_items.extend(batch.items)
                continue
            try:
                result, metadata = self.model.review(
                    batch, profile, allow_repair=total_requests <= 2
                )
                total_requests += metadata.get("requests", 1)
                if total_requests > 4:
                    plan.unreviewed_items.extend(batch.items)
                    break
                valid_findings = []
                for finding in result.findings:
                    valid_lines = batch.valid_lines.get(finding.path)
                    if valid_lines is None or (finding.line is not None and finding.line not in valid_lines):
                        warnings.append(f"Discarded unlocatable finding for {finding.path}")
                        continue
                    valid_findings.append(finding)
                result.findings = valid_findings
                if result.blocking_defects or any(item.severity == "blocker" for item in result.findings):
                    result.decision = "REJECT"
                    result.risk_level = "high"
                batch_results.append(result)
                for key in usage:
                    usage[key] += metadata.get("usage", {}).get(key) or 0
                total_latency += metadata.get("latency", 0.0)
                all_cached = all_cached and metadata.get("cached", False)
            except Exception as exc:
                warnings.append(f"Batch {batch.number} failed: {type(exc).__name__}")

        valid = len(batch_results)
        incomplete = valid != len(plan.batches) or bool(plan.unreviewed_items)
        findings: list[Finding] = []
        seen = set()
        for result in batch_results:
            for finding in result.findings:
                key = (finding.path, finding.line, finding.category, finding.message.strip().lower())
                if key not in seen:
                    seen.add(key); findings.append(finding)
        findings.sort(key=lambda item: (SEVERITY_ORDER[item.severity], item.path, item.line or 0))
        findings = findings[:5]
        reject = any(result.decision == "REJECT" for result in batch_results) or any(item.severity == "blocker" for item in findings)
        decision = "REJECT" if reject or incomplete else "MERGE"
        risk = "high" if reject else ("medium" if incomplete else max((result.risk_level for result in batch_results), default="low"))
        status = "INCOMPLETE" if incomplete else "COMPLETED"
        summary = batch_results[0].summary if len(batch_results) == 1 else f"已审查 {valid}/{len(plan.batches)} 个批次"
        reasoning = "；".join(result.reasoning for result in batch_results)[:4000] or "没有形成有效批次结论"
        if len(batch_results) > 1:
            try:
                aggregate, metadata = self.model.summarize(
                    batch_results,
                    {
                        "reviewed_batches": valid,
                        "planned_batches": len(plan.batches),
                        "unreviewed_items": plan.unreviewed_items,
                    },
                    profile,
                )
                summary = aggregate.summary
                reasoning = aggregate.reasoning
                summary_requests = metadata.get("requests", 1)
                for key in usage:
                    usage[key] += metadata.get("usage", {}).get(key) or 0
                total_latency += metadata.get("latency", 0.0)
                all_cached = all_cached and metadata.get("cached", False)
            except Exception as exc:
                warnings.append(f"Summary failed: {type(exc).__name__}")
        ml_result = self.ml.predict(snapshot)
        return {
            "id": str(uuid.uuid4()), "request_id": request_id, "status": status,
            "snapshot": snapshot.as_dict(), "profile": profile, "decision": decision,
            "risk_level": risk, "summary": summary, "reasoning": reasoning,
            "findings": [item.model_dump() for item in findings],
            "coverage": {
                "reviewed_batches": valid, "planned_batches": len(plan.batches),
                "reviewed_items": [item for batch in plan.batches[:valid] for item in batch.items],
                "unreviewed_items": plan.unreviewed_items,
                "excluded_items": [{"path": item.path, "reason": item.excluded_reason} for item in snapshot.files if not item.reviewable],
                "context_sources": plan.sources, "warnings": warnings,
            },
            "ml_reference": ml_result, "usage": usage, "latency": {
                "model_seconds": total_latency, "total_seconds": time.perf_counter() - started,
            }, "cached": all_cached, "model": self.model.model_id,
            "requests": {"code": total_requests, "summary": summary_requests},
            "created_at": datetime.now(UTC).isoformat(), "stale": False,
        }