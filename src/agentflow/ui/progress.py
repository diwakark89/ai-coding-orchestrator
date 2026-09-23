"""Lightweight text animation for long-running stage subprocess awaits."""

import asyncio

from rich.console import Console
from rich.live import Live


class StageAnimator:
    """Async context manager that animates a stage label while its body awaits.

    Prints `label`, then appends dots one per tick up to `max_dots`, then resets back
    to the bare label and repeats -- e.g. "Planning" -> "Planning." -> ... ->
    "Planning.........." -> "Planning" -> ... -- for as long as the wrapped `async with`
    body is still running. Falls back to a single static print when the console isn't a
    real terminal (redirected stdout, CI, tests), skipping the `Live` display and
    background task entirely.
    """

    def __init__(
        self,
        console: Console,
        label: str,
        *,
        interval: float = 0.5,
        max_dots: int = 10,
    ) -> None:
        self._console = console
        self._label = label
        self._interval = interval
        self._max_dots = max_dots
        self._live: Live | None = None
        self._task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> "StageAnimator":
        if not self._console.is_terminal:
            self._console.print(f"{self._label}...")
            return self
        self._live = Live(self._label, console=self._console, auto_refresh=False)
        self._live.start()
        self._task = asyncio.create_task(self._tick())
        return self

    async def _tick(self) -> None:
        dots = 0
        assert self._live is not None
        while True:
            await asyncio.sleep(self._interval)
            dots = 0 if dots >= self._max_dots else dots + 1
            self._live.update(f"{self._label}{'.' * dots}", refresh=True)

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._task is None:
            return
        try:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass  # a cosmetic animation glitch must never mask the real stage outcome
        finally:
            if self._live is not None:
                self._live.update(self._label, refresh=True)
                self._live.stop()
            self._live = None
            self._task = None
