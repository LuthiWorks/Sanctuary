"""Regression guards: the file tools are confined to configured sandbox roots.

read_file / write_file / list_directory used to accept any absolute path with
no confinement -- there was no jail to escape. These tests pin the jail down:

  - FAIL CLOSED: with no roots configured, every filesystem call is denied.
  - Paths inside an allowed root work (read, write, nested dirs, list).
  - `..` traversal that resolves outside a root is denied.
  - An absolute path outside every root is denied.
  - A symlink INSIDE a root that points OUTSIDE it does not grant escape
    (resolution follows the link before the root check).
  - A write is confined too -- it cannot create files outside the root.

They fail if a future edit drops the resolve-then-check, compares the raw
(unresolved) path, or reintroduces an unconfined default.

Authored by Fable 5 (adversarial seat), 2026-07-02.
"""
from __future__ import annotations

import pytest

from sanctuary.tools import builtin
from sanctuary.tools.builtin import _read_file, _write_file, _list_directory


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A configured sandbox root with one file already inside it."""
    (tmp_path / "inside.txt").write_text("secret-but-allowed", encoding="utf-8")
    monkeypatch.setattr(builtin._config, "filesystem_roots", (str(tmp_path),))
    return tmp_path


# ---------------------------------------------------------------------------
# Fail closed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_roots_denies_read(monkeypatch, tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("x", encoding="utf-8")
    monkeypatch.setattr(builtin._config, "filesystem_roots", ())
    result = await _read_file({"path": str(f)})
    assert not result.success
    assert "not configured" in result.error


@pytest.mark.asyncio
async def test_no_roots_denies_write(monkeypatch, tmp_path):
    monkeypatch.setattr(builtin._config, "filesystem_roots", ())
    result = await _write_file({"path": str(tmp_path / "f.txt"), "content": "x"})
    assert not result.success
    assert not (tmp_path / "f.txt").exists()


# ---------------------------------------------------------------------------
# Inside the root: allowed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_inside_root(root):
    result = await _read_file({"path": str(root / "inside.txt")})
    assert result.success
    assert result.output == "secret-but-allowed"


@pytest.mark.asyncio
async def test_write_inside_root_including_nested(root):
    target = root / "sub" / "dir" / "new.txt"
    result = await _write_file({"path": str(target), "content": "ok"})
    assert result.success
    assert target.read_text(encoding="utf-8") == "ok"


@pytest.mark.asyncio
async def test_list_inside_root(root):
    result = await _list_directory({"path": str(root)})
    assert result.success
    assert "inside.txt" in [e["name"] for e in result.output]


# ---------------------------------------------------------------------------
# Escapes: denied
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_traversal_escape_denied(root):
    # ..-traversal that resolves outside the root.
    escape = str(root / ".." / ".." / "etc" / "passwd")
    result = await _read_file({"path": escape})
    assert not result.success
    assert "outside the allowed" in result.error


@pytest.mark.asyncio
async def test_absolute_outside_denied(root):
    # A sibling of the root (its parent) is outside the jail.
    outside = root.parent / "sibling.txt"
    outside.write_text("nope", encoding="utf-8")
    result = await _read_file({"path": str(outside)})
    assert not result.success
    assert "outside the allowed" in result.error


@pytest.mark.asyncio
async def test_write_cannot_escape_root(root):
    outside = root.parent / "escaped.txt"
    result = await _write_file({"path": str(outside), "content": "pwned"})
    assert not result.success
    assert not outside.exists()


@pytest.mark.asyncio
async def test_symlink_inside_root_pointing_outside_denied(root, tmp_path):
    # A symlink that lives inside the root but targets a file outside it must
    # not grant access -- resolution follows the link before the root check.
    outside_dir = tmp_path.parent / "outside_target_dir"
    outside_dir.mkdir(exist_ok=True)
    secret = outside_dir / "secret.txt"
    secret.write_text("exfiltrate me", encoding="utf-8")

    link = root / "backdoor"
    try:
        link.symlink_to(outside_dir, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted on this platform/session")

    result = await _read_file({"path": str(link / "secret.txt")})
    assert not result.success
    assert "outside the allowed" in result.error
