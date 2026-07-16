from __future__ import annotations

import ast
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import Settings
from .security import is_probably_binary, resolve_regular_path, sensitive_path_reason
from .workspace import FileChange, WorkspaceSnapshot


@dataclass(slots=True)
class ReviewBatch:
    number: int
    items: list[str]
    paths: list[str]
    context: str
    valid_lines: dict[str, set[int]]


@dataclass(slots=True)
class ContextPlan:
    batches: list[ReviewBatch]
    sources: list[dict]
    unreviewed_items: list[str]
    truncated: bool


class ContextBuilder:
    def __init__(self, settings: Settings):
        self.settings = settings

    def _file_context(self, change: FileChange) -> tuple[str, list[dict]]:
        blocks = [f"FILE:{change.path} STATUS:{change.status}", change.diff]
        sources = [{"type": "diff", "path": change.path, "truncated": False}]
        if change.status == "deleted":
            return "\n".join(blocks), sources
        try:
            path = resolve_regular_path(self.settings.repo_root, change.path)
            data = path.read_bytes()
            if len(data) <= 24_000 and not is_probably_binary(data):
                text = data.decode("utf-8")
                content = text
                if change.language == "python":
                    try:
                        ast.parse(text)
                    except SyntaxError:
                        content = text[:12_000]
                else:
                    content = text[:12_000]
                blocks.append(f"POST_CHANGE_FILE:{change.path}\n{content}")
                sources.append({"type": "post_change_file", "path": change.path, "truncated": len(content) < len(text)})
        except (OSError, ValueError):
            pass
        return "\n\n".join(blocks), sources

    def _conventions(self) -> tuple[str, list[dict]]:
        for name in ("AGENTS.md", "CLAUDE.md", "CONTRIBUTING.md", "README.md"):
            path = self.settings.repo_root / name
            if path.is_file() and not path.is_symlink() and not sensitive_path_reason(name):
                try:
                    text = path.read_text(encoding="utf-8")[:4_000]
                except (OSError, UnicodeDecodeError):
                    continue
                return f"PROJECT_CONVENTIONS:{name}\n{text}", [{"type": "conventions", "path": name, "truncated": len(text) == 4_000}]
        return "", []

    def _changed_symbols(self, change: FileChange) -> list[str]:
        if change.language != "python" or change.status == "deleted":
            return []
        try:
            source = resolve_regular_path(self.settings.repo_root, change.path).read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (OSError, UnicodeDecodeError, SyntaxError, ValueError):
            return []
        changed_lines = {line for hunk in change.hunks for line in hunk.changed_new_lines}
        symbols = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                end = getattr(node, "end_lineno", node.lineno)
                if any(node.lineno <= line <= end for line in changed_lines):
                    symbols.append(node.name)
        return symbols[:5]

    def _repository_context(self, change: FileChange) -> tuple[str, list[dict]]:
        try:
            output = subprocess.run(
                ["git", "ls-files", "-z"], cwd=self.settings.repo_root, check=True,
                capture_output=True, timeout=10,
            ).stdout.decode("utf-8")
        except (OSError, UnicodeDecodeError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return "", []
        paths = [path for path in output.split("\0") if path and path != change.path]
        stem = Path(change.path).stem.removeprefix("test_")
        tests = sorted(
            path for path in paths
            if ("test" in Path(path).parts or Path(path).name.startswith("test_"))
            and stem in Path(path).stem
        )[:2]
        symbols = self._changed_symbols(change)
        candidates = [
            path for path in paths
            if Path(path).suffix.lower() in {".py", ".js", ".ts", ".tsx", ".java", ".go", ".rs"}
            and not sensitive_path_reason(path)
        ][:500]
        parts: list[str] = []
        sources: list[dict] = []
        used = 0
        for path in tests:
            try:
                text = resolve_regular_path(self.settings.repo_root, path).read_text(encoding="utf-8")[:4_000]
            except (OSError, UnicodeDecodeError, ValueError):
                continue
            parts.append(f"RELATED_TEST:{path}\n{text}")
            sources.append({"type": "related_test", "path": path, "truncated": len(text) == 4_000})
            used += len(text)
        for symbol in symbols:
            pattern = re.compile(rf"\b{re.escape(symbol)}\b")
            for path in candidates:
                if used >= 8_000 or any(source.get("path") == path and source["type"] == "symbol_reference" for source in sources):
                    continue
                try:
                    text = resolve_regular_path(self.settings.repo_root, path).read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError, ValueError):
                    continue
                match = pattern.search(text)
                if not match:
                    continue
                start = max(0, match.start() - 500)
                snippet = text[start:match.end() + 800]
                parts.append(f"SYMBOL_REFERENCE:{symbol}@{path}\n{snippet}")
                sources.append({"type": "symbol_reference", "path": path, "symbol": symbol, "truncated": True})
                used += len(snippet)
                if sum(source.get("symbol") == symbol for source in sources) >= 2:
                    break
        return "\n\n".join(parts), sources

    def build(self, snapshot: WorkspaceSnapshot) -> ContextPlan:
        reviewable = sorted(
            (item for item in snapshot.files if item.reviewable),
            key=lambda item: (item.language != "python", item.path),
        )
        units: list[tuple[list[str], list[str], str, dict[str, set[int]], list[dict]]] = []
        for change in reviewable:
            context, sources = self._file_context(change)
            repository_context, repository_sources = self._repository_context(change)
            if repository_context and len(context) + len(repository_context) <= self.settings.batch_char_budget:
                context += "\n\n" + repository_context
                sources.extend(repository_sources)
            valid = {change.path: {line for hunk in change.hunks for line in hunk.changed_new_lines}}
            if len(context) <= self.settings.batch_char_budget:
                units.append(([change.path], [change.path], context, valid, sources))
                continue
            for hunk in change.hunks:
                hunk_text = "\n".join([hunk.header, *hunk.lines])
                units.append((
                    [f"{change.path}:{hunk.id}"], [change.path],
                    f"FILE:{change.path} HUNK:{hunk.id}\n{hunk_text}",
                    {change.path: set(hunk.changed_new_lines)},
                    [{"type": "diff_hunk", "path": change.path, "hunk": hunk.id, "truncated": True}],
                ))

        conventions, convention_sources = self._conventions()
        batches: list[ReviewBatch] = []
        sources: list[dict] = []
        unreviewed: list[str] = []
        current_items: list[str] = []
        current_paths: list[str] = []
        current_contexts: list[str] = []
        current_valid: dict[str, set[int]] = {}

        def flush() -> None:
            if not current_items:
                return
            context = "\n\n".join(current_contexts)
            if conventions and len(context) + len(conventions) <= self.settings.batch_char_budget:
                context += "\n\n" + conventions
            batches.append(ReviewBatch(len(batches) + 1, current_items.copy(), current_paths.copy(), context, {k: v.copy() for k, v in current_valid.items()}))
            current_items.clear(); current_paths.clear(); current_contexts.clear(); current_valid.clear()

        for items, paths, context, valid, unit_sources in units:
            if current_contexts and sum(map(len, current_contexts)) + len(context) > self.settings.batch_char_budget:
                flush()
            if len(batches) >= self.settings.max_review_batches:
                unreviewed.extend(items)
                continue
            current_items.extend(items); current_paths.extend(path for path in paths if path not in current_paths)
            current_contexts.append(context[:self.settings.batch_char_budget])
            for path, lines in valid.items():
                current_valid.setdefault(path, set()).update(lines)
            sources.extend(unit_sources)
        flush()
        if conventions and batches:
            sources.extend(convention_sources)
        return ContextPlan(batches[:self.settings.max_review_batches], sources, unreviewed, bool(unreviewed))