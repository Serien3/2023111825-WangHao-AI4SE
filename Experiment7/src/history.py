from __future__ import annotations

import json
import os
import tempfile
import threading
from copy import deepcopy
from datetime import datetime
from pathlib import Path


class HistoryStore:
    def __init__(self, path: Path, limit: int = 200):
        self.path = path
        self.limit = limit
        self._lock = threading.RLock()

    def _read_unlocked(self) -> tuple[list[dict], list[str]]:
        if not self.path.is_file():
            return [], []
        records: list[dict] = []
        warnings: list[str] = []
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    value = json.loads(line)
                    if isinstance(value, dict) and value.get("id"):
                        records.append(value)
                    else:
                        warnings.append(f"Skipped invalid history record at line {line_number}")
                except json.JSONDecodeError:
                    warnings.append(f"Skipped damaged history record at line {line_number}")
        return records, warnings

    def _write_unlocked(self, records: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                for record in records[-self.limit:]:
                    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, self.path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    def append(self, record: dict) -> None:
        with self._lock:
            records, _ = self._read_unlocked()
            records.append(deepcopy(record))
            self._write_unlocked(records)

    def get(self, review_id: str) -> dict | None:
        with self._lock:
            records, _ = self._read_unlocked()
        return next((deepcopy(item) for item in records if item.get("id") == review_id), None)

    def query(
        self,
        *,
        page: int = 1,
        page_size: int = 20,
        path: str | None = None,
        status: str | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
    ) -> dict:
        with self._lock:
            records, warnings = self._read_unlocked()
        records.reverse()
        if path:
            needle = path.lower()
            records = [item for item in records if any(needle in value.lower() for value in item.get("files", []))]
        if status:
            expected = status.upper()
            records = [item for item in records if item.get("status", "").upper() == expected or item.get("decision", "").upper() == expected]
        if created_after:
            records = [item for item in records if datetime.fromisoformat(item["created_at"]) >= created_after]
        if created_before:
            records = [item for item in records if datetime.fromisoformat(item["created_at"]) <= created_before]
        total = len(records)
        start = (page - 1) * page_size
        summaries = []
        for record in records[start:start + page_size]:
            summaries.append({key: value for key, value in record.items() if key not in {"saved_diff", "reasoning", "snapshot"}})
        return {"items": summaries, "page": page, "page_size": page_size, "total": total, "storage_warnings": warnings}

    def all(self) -> tuple[list[dict], list[str]]:
        with self._lock:
            records, warnings = self._read_unlocked()
        return [deepcopy(item) for item in reversed(records)], warnings

    def delete(self, review_id: str) -> bool:
        with self._lock:
            records, _ = self._read_unlocked()
            remaining = [item for item in records if item.get("id") != review_id]
            if len(remaining) == len(records):
                return False
            self._write_unlocked(remaining)
            return True

    def clear(self) -> int:
        with self._lock:
            records, _ = self._read_unlocked()
            self._write_unlocked([])
            return len(records)
