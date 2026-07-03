"""Regression guards: the shell tool is safe-by-default (no shell injection).

The shell tool used to run every command with ``shell=True``, so a single
command string could chain / pipe / substitute (``foo; rm -rf ~``). It is now
safe by default: the command is tokenized into argv and run WITHOUT a shell,
so metacharacters are inert. Full shell interpretation is available but only
via an explicit ``use_shell: true`` (and the tool is gated on top of that).

These fail if the default silently regains shell interpretation.

Authored by Fable 5 (adversarial seat), 2026-07-02.
"""
from __future__ import annotations

import sys

import pytest

from sanctuary.tools.builtin import _shell_command


# A real, always-present executable, forward-slashed so POSIX tokenization
# keeps the path intact on Windows too.
_PY = sys.executable.replace("\\", "/")


@pytest.mark.asyncio
async def test_empty_command():
    result = await _shell_command({})
    assert not result.success
    assert "No command" in result.error


@pytest.mark.asyncio
async def test_safe_mode_runs_a_program():
    result = await _shell_command({"command": f'"{_PY}" -c "print(123)"'})
    assert result.success, result.error
    assert "123" in result.output["stdout"]


@pytest.mark.asyncio
async def test_safe_mode_does_not_chain_with_ampersands():
    # In safe mode the trailing `&& echo PWNED` must NOT run as a second
    # command -- the tokens are passed to the program as inert argv.
    cmd = f'"{_PY}" -c "print(123)" && echo PWNED'
    result = await _shell_command({"command": cmd})
    assert result.success, result.error
    assert "123" in result.output["stdout"]
    assert "PWNED" not in result.output["stdout"]


@pytest.mark.asyncio
async def test_safe_mode_does_not_substitute_or_pipe():
    # $(...) and | are literal in safe mode; the program sees them as args
    # and no substitution/pipe occurs.
    cmd = f'"{_PY}" -c "import sys; print(len(sys.argv))" $(whoami) | cat'
    result = await _shell_command({"command": cmd})
    assert result.success, result.error
    # argv is [prog-name, extra tokens...]; more than 1 means they arrived as
    # literal arguments rather than being consumed by a shell.
    assert int(result.output["stdout"].strip()) >= 2


@pytest.mark.asyncio
async def test_use_shell_true_enables_shell_features():
    result = await _shell_command({"command": "echo hello", "use_shell": True})
    assert result.success
    assert "hello" in result.output["stdout"]


@pytest.mark.asyncio
async def test_unbalanced_quotes_reported_not_crashed():
    result = await _shell_command({"command": 'echo "unterminated'})
    assert not result.success
    assert "parse" in result.error.lower()
