from __future__ import annotations

import fnmatch
import re
from pathlib import Path, PurePosixPath


SENSITIVE_PARTS = {".git", ".ssh", ".aws", "node_modules", ".venv", "dist", "build"}
SENSITIVE_NAMES = (".env", ".env.*", "*.pem", "*.key", "*.p12", "id_rsa*")
CREDENTIAL_PATTERNS = (
    ("private_key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("github_token", re.compile(r"\b(?:gh[opusr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b")),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("bearer_token", re.compile(r"(?i)\bauthorization\s*[:=]\s*bearer\s+[A-Za-z0-9._~+/=-]{16,}")),
    ("api_key", re.compile(r"(?i)\b(?:api[_-]?key|secret[_-]?key)\s*[:=]\s*['\"]?[A-Za-z0-9._~+/=-]{16,}")),
    ("password_assignment", re.compile(r"(?i)\bpassword\s*[:=]\s*['\"][^'\"\s]{8,}['\"]")),
)


class UnsafePathError(ValueError):
    pass


def sensitive_path_reason(path: str) -> str | None:
    normalized = path.replace("\\", "/")
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or ".." in pure.parts:
        return "path_escape"
    if any(part in SENSITIVE_PARTS for part in pure.parts):
        return "sensitive_directory"
    if any(fnmatch.fnmatch(pure.name, pattern) for pattern in SENSITIVE_NAMES):
        return "sensitive_path"
    return None


def resolve_regular_path(root: Path, relative_path: str, *, allow_missing: bool = False) -> Path:
    if sensitive_path_reason(relative_path) == "path_escape":
        raise UnsafePathError("Path must be a repository-relative path")
    candidate = root / relative_path
    if candidate.is_symlink():
        raise UnsafePathError("Symbolic links are not readable")
    resolved = candidate.resolve(strict=not allow_missing)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise UnsafePathError("Path escapes the locked repository") from exc
    if not allow_missing and not resolved.is_file():
        raise UnsafePathError("Path is not a regular file")
    return resolved


def credential_rules(text: str) -> list[str]:
    return [name for name, pattern in CREDENTIAL_PATTERNS if pattern.search(text)]


def is_probably_binary(data: bytes) -> bool:
    if b"\x00" in data[:8192]:
        return True
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return True
    return False
