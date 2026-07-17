"""Security invariant: keep Sanctuary outside the preconditions of the
ChromaDB pre-auth RCE (CVE-2026-45829 / GHSA-f4j7-r4q5-qw2c).

The advisory: in chromadb >= 1.0.0 (through 1.5.9, the latest release —
**no patched version exists**), an unauthenticated attacker can achieve
remote code execution by POSTing a malicious model repository with
``trust_remote_code`` set to true to the Chroma *server's*
``/api/v2/.../collections`` endpoint. The dangerous model code is then
executed during embedding.

Sanctuary is not exposed, because it meets none of the three preconditions:

1. **No Chroma server.** The live memory path uses embedded
   ``chromadb.PersistentClient`` (in-process, local disk). Nothing launches
   a Chroma HTTP server; ``infrastructure/remote_memory.py`` is only a
   *client* and is not wired into the runner.
2. **No ``trust_remote_code``.** The embedding trigger is never enabled.
3. **No attacker-supplied model.** The embedding model is a hardcoded
   sentence-transformers checkpoint, not a caller-controlled repo.

Because there is no upstream fix to upgrade to, the mitigation is to *keep*
this true. This test makes precondition (2) — the decisive, unambiguous
trigger — a CI-enforced invariant: if anyone ever introduces
``trust_remote_code=True`` (in an embedding config, a transformers/
sentence-transformers load, or anywhere else), the RCE precondition is back
and this test fails loudly, pointing at the CVE.

If a future change legitimately requires ``trust_remote_code`` for a
*trusted, vendored, offline* model, append ``# pragma: allow-trust-remote-code``
to that line AND record the justification in
``docs/audits/CVE-2026-45829_chromadb.md``. Do not silence this test by
editing it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Matches trust_remote_code being set truthy in any of the forms it appears
# across Python source and config: ``trust_remote_code=True``,
# ``trust_remote_code = True``, ``"trust_remote_code": True``,
# ``trust_remote_code: true`` (yaml/json).
TRUST_REMOTE_CODE_TRUE = re.compile(
    r"""trust_remote_code["']?\s*[:=]\s*(?:True|true|1)\b"""
)

ESCAPE_PRAGMA = "pragma: allow-trust-remote-code"

SCAN_EXTENSIONS = {".py", ".pyi", ".toml", ".yaml", ".yml", ".json", ".cfg", ".ini"}

SKIP_DIR_NAMES = {
    ".git", ".venv", "venv", "env", "__pycache__", "node_modules",
    ".pytest_cache", ".hypothesis", "build", "dist", ".mypy_cache",
    ".ruff_cache", ".tox",
}


def _iter_source_files(root: Path):
    """Yield tracked source files to scan (``git ls-files``), falling back to
    a full walk if git is unavailable. Mirrors ``test_no_hardcoded_paths``."""
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
        tracked = None

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SCAN_EXTENSIONS:
            continue
        if any(part in SKIP_DIR_NAMES for part in path.parts):
            continue
        if path.name == "uv.lock":
            continue
        # Skip this test file itself — it names the pattern by definition.
        if path.name == "test_chromadb_rce_guard.py":
            continue
        if tracked is not None and path.resolve() not in tracked:
            continue
        yield path


def test_trust_remote_code_never_enabled():
    """Fail loudly if any tracked source enables ``trust_remote_code``.

    This is the decisive precondition of CVE-2026-45829 that Sanctuary
    controls. Keeping it false keeps the RCE unreachable even though no
    upstream chromadb patch exists.
    """
    findings: list[str] = []

    for path in _iter_source_files(REPO_ROOT):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        for line_num, line in enumerate(text.splitlines(), start=1):
            if ESCAPE_PRAGMA in line:
                continue
            if TRUST_REMOTE_CODE_TRUE.search(line):
                rel = path.relative_to(REPO_ROOT)
                findings.append(f"  {rel}:{line_num}: {line.strip()!r}")

    if findings:
        msg = (
            "`trust_remote_code` is enabled somewhere in source. This is the "
            "code-execution trigger of CVE-2026-45829 (ChromaDB pre-auth RCE, "
            "no upstream fix). Enabling it — especially with a caller- or "
            "network-supplied model — reintroduces the RCE precondition.\n\n"
            "Use a trusted, vendored, hardcoded model without trust_remote_code. "
            "If it is genuinely required for a trusted offline model, append "
            f"`# {ESCAPE_PRAGMA}` to the line and document the justification in "
            "docs/audits/CVE-2026-45829_chromadb.md.\n\n"
            + "\n".join(findings)
        )
        pytest.fail(msg)
