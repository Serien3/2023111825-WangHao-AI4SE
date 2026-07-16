from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent


class ConfigurationError(ValueError):
    """Raised when the service cannot safely lock the requested repository."""


@dataclass(frozen=True, slots=True)
class Settings:
    repo_root: Path
    repo_id: str
    host: str = "127.0.0.1"
    port: int = 8765
    model_id: str = "deepseek-v4-flash"
    api_base_url: str = "https://api.deepseek.com"
    history_path: Path = PROJECT_ROOT / "results" / "history" / "reviews.jsonl"
    cache_dir: Path = PROJECT_ROOT / "results" / "cache"
    max_file_bytes: int = 512_000
    batch_char_budget: int = 48_000
    max_review_batches: int = 4
    history_limit: int = 200

    @property
    def llm_configured(self) -> bool:
        return bool(os.getenv("DEEPSEEK_API_KEY"))

    @classmethod
    def from_repo(
        cls,
        repo: str | Path,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        history_path: Path | None = None,
    ) -> "Settings":
        root = Path(repo).expanduser().resolve()
        if not root.is_dir():
            raise ConfigurationError(f"Repository does not exist or is not a directory: {root}")
        if not os.access(root, os.R_OK):
            raise ConfigurationError(f"Repository is not readable: {root}")

        def git(*args: str) -> str:
            try:
                result = subprocess.run(
                    ["git", *args],
                    cwd=root,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
            except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                raise ConfigurationError(f"Invalid Git worktree: {root}") from exc
            return result.stdout.strip()

        if git("rev-parse", "--is-inside-work-tree") != "true":
            raise ConfigurationError(f"Not a Git worktree: {root}")
        git("rev-parse", "--verify", "HEAD^{commit}")
        top_level = Path(git("rev-parse", "--show-toplevel")).resolve()
        if top_level != root:
            raise ConfigurationError(f"--repo must be the Git worktree root: {top_level}")

        repo_hash = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:12]
        values: dict[str, object] = {
            "repo_root": root,
            "repo_id": f"{root.name}:{repo_hash}",
            "host": host,
            "port": port,
        }
        if history_path is not None:
            values["history_path"] = history_path
        return cls(**values)
