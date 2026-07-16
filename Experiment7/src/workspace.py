from __future__ import annotations

import difflib
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from .config import Settings
from .security import (
    UnsafePathError,
    credential_rules,
    is_probably_binary,
    resolve_regular_path,
    sensitive_path_reason,
)


HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
LANGUAGES = {
    ".py": "python", ".js": "javascript", ".ts": "typescript", ".tsx": "typescript",
    ".java": "java", ".go": "go", ".rs": "rust", ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cc": "cpp", ".cs": "csharp", ".rb": "ruby", ".php": "php",
    ".html": "html", ".css": "css", ".md": "markdown", ".json": "json",
    ".yaml": "yaml", ".yml": "yaml", ".toml": "toml", ".sh": "shell",
}


class WorkspaceError(RuntimeError):
    pass


@dataclass(slots=True)
class DiffHunk:
    id: str
    header: str
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[str]
    changed_new_lines: list[int]

    def as_dict(self) -> dict:
        return {
            "id": self.id, "header": self.header, "old_start": self.old_start,
            "old_count": self.old_count, "new_start": self.new_start,
            "new_count": self.new_count, "lines": self.lines,
            "changed_new_lines": self.changed_new_lines,
        }


@dataclass(slots=True)
class FileChange:
    path: str
    status: str
    diff: str
    old_path: str | None = None
    language: str = "text"
    additions: int = 0
    deletions: int = 0
    binary: bool = False
    reviewable: bool = True
    excluded_reason: str | None = None
    security_rules: list[str] = field(default_factory=list)
    hunks: list[DiffHunk] = field(default_factory=list)

    def as_dict(self, *, include_diff: bool = False) -> dict:
        value = {
            "path": self.path, "old_path": self.old_path, "status": self.status,
            "language": self.language, "additions": self.additions,
            "deletions": self.deletions, "binary": self.binary,
            "reviewable": self.reviewable, "excluded_reason": self.excluded_reason,
            "security_rules": self.security_rules,
            "hunks": [hunk.as_dict() for hunk in self.hunks] if self.reviewable else [],
        }
        if include_diff and self.reviewable:
            value["diff"] = self.diff
        return value


@dataclass(slots=True)
class WorkspaceSnapshot:
    repo_root_id: str
    branch: str
    head_sha: str
    scope: str
    path: str | None
    files: list[FileChange]
    normalized_diff: str
    snapshot_hash: str
    created_at: str

    def as_dict(self, *, include_diff: bool = False) -> dict:
        return {
            "repo_root_id": self.repo_root_id, "branch": self.branch,
            "head_sha": self.head_sha, "scope": self.scope, "path": self.path,
            "files": [item.as_dict(include_diff=include_diff) for item in self.files],
            "snapshot_hash": self.snapshot_hash, "created_at": self.created_at,
        }


def parse_hunks(diff: str) -> list[DiffHunk]:
    hunks: list[DiffHunk] = []
    current: DiffHunk | None = None
    old_line = new_line = 0
    for line in diff.splitlines():
        match = HUNK_RE.match(line)
        if match:
            old_start, old_count, new_start, new_count = (
                int(match.group(1)), int(match.group(2) or 1),
                int(match.group(3)), int(match.group(4) or 1),
            )
            current = DiffHunk(
                id=f"H{len(hunks) + 1}", header=line, old_start=old_start,
                old_count=old_count, new_start=new_start, new_count=new_count,
                lines=[], changed_new_lines=[],
            )
            hunks.append(current)
            old_line, new_line = old_start, new_start
            continue
        if current is None:
            continue
        current.lines.append(line)
        if line.startswith("+") and not line.startswith("+++"):
            current.changed_new_lines.append(new_line)
            new_line += 1
        elif line.startswith("-") and not line.startswith("---"):
            old_line += 1
        elif not line.startswith("\\"):
            old_line += 1
            new_line += 1
    return hunks


class WorkspaceRepository:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.root = settings.repo_root

    def _git(self, *args: str, text: bool = True) -> str | bytes:
        try:
            result = subprocess.run(
                ["git", "--no-pager", *args], cwd=self.root, check=True,
                capture_output=True, text=text, timeout=20,
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise WorkspaceError("Git workspace query failed") from exc
        return result.stdout

    def _change_specs(self) -> list[tuple[str, str, str | None]]:
        raw = self._git("diff", "--name-status", "-z", "--find-renames", "HEAD", "--")
        fields = raw.split("\0")
        specs: list[tuple[str, str, str | None]] = []
        index = 0
        while index < len(fields) and fields[index]:
            code = fields[index]
            index += 1
            if code.startswith("R"):
                old_path, path = fields[index], fields[index + 1]
                index += 2
                specs.append((path, "renamed", old_path))
            else:
                path = fields[index]
                index += 1
                status = {"A": "added", "D": "deleted", "M": "modified"}.get(code[0], "modified")
                specs.append((path, status, None))
        untracked = self._git("ls-files", "--others", "--exclude-standard", "-z")
        specs.extend((path, "added", None) for path in untracked.split("\0") if path)
        return sorted(specs, key=lambda item: item[0])

    def _untracked_diff(self, path: str, data: bytes) -> str:
        text = data.decode("utf-8")
        lines = text.splitlines(keepends=True)
        return "".join(difflib.unified_diff([], lines, fromfile="/dev/null", tofile=f"b/{path}", n=3))

    def _read_change(self, path: str, status: str, old_path: str | None) -> FileChange:
        reason = sensitive_path_reason(path)
        data = b""
        binary = False
        security_rules: list[str] = []
        if status != "deleted":
            try:
                resolved = resolve_regular_path(self.root, path)
                if resolved.stat().st_size > self.settings.max_file_bytes:
                    reason = reason or "file_too_large"
                else:
                    data = resolved.read_bytes()
                    binary = is_probably_binary(data)
                    if not binary:
                        security_rules = credential_rules(data.decode("utf-8"))
            except (OSError, UnsafePathError):
                reason = reason or "unsafe_file"

        tracked = bool(self._git("ls-files", "--error-unmatch", "--", path).strip()) if status != "added" else False
        if status == "added" and not tracked:
            diff = "" if binary or reason == "unsafe_file" else self._untracked_diff(path, data)
        else:
            diff = self._git("diff", "--no-ext-diff", "--unified=3", "--find-renames", "HEAD", "--", old_path or path, path)
        if any(
            line.startswith("Binary files ") or line == "GIT binary patch"
            for line in diff.splitlines()
        ):
            binary = True
        hunks = parse_hunks(diff)
        additions = sum(len(hunk.changed_new_lines) for hunk in hunks)
        deletions = sum(1 for hunk in hunks for line in hunk.lines if line.startswith("-") and not line.startswith("---"))
        excluded = reason or ("binary_file" if binary else None) or ("credential_detected" if security_rules else None)
        return FileChange(
            path=path, old_path=old_path, status=status, diff=diff.replace("\r\n", "\n"),
            language=LANGUAGES.get(Path(path).suffix.lower(), "text"), additions=additions,
            deletions=deletions, binary=binary, reviewable=excluded is None,
            excluded_reason=excluded, security_rules=security_rules, hunks=hunks,
        )

    def snapshot(self, scope: str = "workspace", path: str | None = None) -> WorkspaceSnapshot:
        if scope not in {"workspace", "file"}:
            raise WorkspaceError("Unknown review scope")
        changes = [self._read_change(*spec) for spec in self._change_specs()]
        if scope == "file":
            if not path:
                raise WorkspaceError("File scope requires a path")
            if sensitive_path_reason(path) == "path_escape":
                raise UnsafePathError("Path must be repository-relative")
            changes = [change for change in changes if change.path == path]
            if not changes:
                raise WorkspaceError("The requested path is not modified")
        normalized_parts = []
        for change in changes:
            if change.reviewable:
                normalized_parts.append(change.diff.rstrip("\n"))
            elif change.diff:
                normalized_parts.append(
                    f"EXCLUDED {change.path} {hashlib.sha256(change.diff.encode('utf-8')).hexdigest()}"
                )
        normalized = "\n".join(normalized_parts) + ("\n" if normalized_parts else "")
        head_sha = str(self._git("rev-parse", "HEAD"))
        branch = str(self._git("branch", "--show-current")) or "DETACHED"
        payload = json.dumps(
            {"schema": 1, "head": head_sha, "scope": scope, "path": path, "diff": normalized},
            ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return WorkspaceSnapshot(
            repo_root_id=self.settings.repo_id, branch=branch, head_sha=head_sha,
            scope=scope, path=path, files=changes, normalized_diff=normalized,
            snapshot_hash=f"sha256:{digest}", created_at=datetime.now(UTC).isoformat(),
        )
