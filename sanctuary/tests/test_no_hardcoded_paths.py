"""Repo-hygiene test: no hardcoded user-specific or OS-specific paths.

This test exists to catch a class of bug that broke this codebase before:
a developer's local path (e.g. ``C:/Users/Hasha Smokes/...``) embedded as
a default in source. Such a default works on the original developer's
machine and silently fails everywhere else — Linux, Docker, any other
checkout, CI. Because the failure mode is "works for me," it doesn't
get noticed during ordinary development.

The remediation pattern: read paths from environment variables,
``Path(__file__)``-relative discovery, or explicit configuration. Never
embed a literal user home or developer-specific path in source.

If this test fails, fix the offending file rather than disabling the
test. If a path *must* appear in source for a documented reason (e.g.
a comment, a docstring example, an error message), prefix it with the
escape sequence ``# pragma: allow-hardcoded-path`` on the same line.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Paths that should never appear as literals in source. Each pattern is a
# regex that matches the *user-specific* portion — we don't ban the
# string "C:" globally, only "C:/Users/<name>/..."-style fragments.
FORBIDDEN_PATTERNS: list[tuple[str, str]] = [
    (
        r"[Cc]:[\\/]Users[\\/][A-Za-z0-9 _.\-]+",
        "Windows user home path (use env var, e.g. $LUTHI_PATH, or Path-relative discovery)",
    ),
    (
        r"/home/[A-Za-z0-9_.\-]+",
        "Linux user home path (use $HOME, env vars, or Path-relative discovery)",
    ),
    (
        r"/Users/[A-Za-z0-9 _.\-]+",
        "macOS user home path (use $HOME, env vars, or Path-relative discovery)",
    ),
]

# Allow specific files/dirs that legitimately contain example paths in
# documentation strings (CLAUDE.md, AGENTS.md, READMEs, .docs).
ALLOWLIST_SUFFIXES = (".md", ".txt", ".rst")
ALLOWLIST_DIR_NAMES = {".docs", "docs", "reference_material", ".claude", ".memories"}

# Honor the per-line escape pragma.
ESCAPE_PRAGMA = "pragma: allow-hardcoded-path"

SCAN_EXTENSIONS = {".py", ".pyi", ".toml", ".yaml", ".yml", ".json", ".cfg", ".ini"}

SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    "node_modules",
    ".pytest_cache",
    ".hypothesis",
    "build",
    "dist",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    "uv.lock",  # generated lockfile contains paths in some envs
}


def _iter_source_files(root: Path):
    """Yield files to scan. Prefers tracked files via ``git ls-files`` so
    untracked work-in-progress doesn't fail the test until it's committed
    (at which point the author has to clean it up). Falls back to a
    direct walk if git isn't available.
    """
    import subprocess

    tracked = None
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        tracked = {
            (root / line).resolve()
            for line in result.stdout.splitlines()
            if line.strip()
        }
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        tracked = None  # git not available; walk everything

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SCAN_EXTENSIONS:
            continue
        if any(part in SKIP_DIR_NAMES for part in path.parts):
            continue
        if any(part in ALLOWLIST_DIR_NAMES for part in path.parts):
            continue
        if path.name == "uv.lock":
            continue
        # Skip this test file itself — it contains the patterns by definition.
        if path.name == "test_no_hardcoded_paths.py":
            continue
        # If git is available, only scan tracked files (skip WIP / untracked).
        if tracked is not None and path.resolve() not in tracked:
            continue
        yield path


def test_no_hardcoded_user_paths_in_source():
    """Fail loudly if any tracked source file embeds a user-specific path."""
    findings: list[str] = []
    compiled = [(re.compile(pat), label) for pat, label in FORBIDDEN_PATTERNS]

    for path in _iter_source_files(REPO_ROOT):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Binary or non-utf8 file — skip.
            continue

        for line_num, line in enumerate(text.splitlines(), start=1):
            if ESCAPE_PRAGMA in line:
                continue
            for pattern, label in compiled:
                match = pattern.search(line)
                if match:
                    rel = path.relative_to(REPO_ROOT)
                    findings.append(
                        f"  {rel}:{line_num}: {match.group(0)!r}  ({label})"
                    )

    if findings:
        msg = (
            "Hardcoded user-specific paths found in source. These break "
            "anyone whose checkout lives at a different path (Linux, "
            "Docker, CI, another developer's machine). Fix by using "
            "env vars, Path(__file__)-relative discovery, or explicit "
            "configuration. If the literal must remain (e.g. in an "
            "error message or comment), append "
            f"`# {ESCAPE_PRAGMA}` to that line.\n\n"
            + "\n".join(findings)
        )
        pytest.fail(msg)
