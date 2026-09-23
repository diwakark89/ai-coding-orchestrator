"""Tests for the stage-progress dot animation (`StageAnimator`)."""

import asyncio
import io
import re

import pytest
from rich.console import Console

from agentflow.ui.progress import StageAnimator

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


async def test_non_terminal_console_prints_once_without_animating() -> None:
    buf = io.StringIO()
    console = Console(force_terminal=False, file=buf)

    async with StageAnimator(console, "Planning", interval=0.01, max_dots=3):
        await asyncio.sleep(0.05)

    assert buf.getvalue().strip() == "Planning..."


async def test_terminal_console_cycles_dots_and_resets() -> None:
    buf = io.StringIO()
    console = Console(force_terminal=True, file=buf, no_color=True, width=80)
    animator = StageAnimator(console, "Planning", interval=0.02, max_dots=3)

    async with animator:
        await asyncio.sleep(0.15)

    output = _strip_ansi(buf.getvalue())
    assert "Planning..." in output  # reached max_dots at least once
    assert animator._task is None
    assert animator._live is None


async def test_cleanup_runs_when_body_raises() -> None:
    buf = io.StringIO()
    console = Console(force_terminal=True, file=buf, no_color=True, width=80)
    animator = StageAnimator(console, "Implementing", interval=0.01, max_dots=2)

    with pytest.raises(RuntimeError, match="boom"):
        async with animator:
            await asyncio.sleep(0.03)
            raise RuntimeError("boom")

    assert animator._task is None
    assert animator._live is None
