"""Rich console output and display utilities."""

import sys

from rich.console import Console
from rich.panel import Panel
from rich.text import Text


def configure_utf8_streams() -> None:
    """Ensure stdout and stderr handle UTF-8 characters on platforms with legacy encodings."""
    for stream in (sys.stdout, sys.stderr):
        if stream and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


# Configure UTF-8 encoding on import
configure_utf8_streams()


def get_status_symbol(symbol: str, fallback: str) -> str:
    """Return symbol if encodable in standard output, else ASCII fallback."""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        symbol.encode(encoding)
        return symbol
    except Exception:
        return fallback


CHECKMARK = get_status_symbol("✓", "+")
CROSSMARK = get_status_symbol("✗", "x")
WARNINGMARK = "!"
INFOMARK = get_status_symbol("ℹ", "i")

console = Console(legacy_windows=False)


class ConsoleUI:
    """Console UI helper for formatted output using Rich."""

    def __init__(self, console_instance: Console | None = None) -> None:
        self.console = console_instance or console

    def print_header(self, title: str, subtitle: str | None = None) -> None:
        """Render a formatted application header."""
        header_text = Text(title, style="bold cyan")
        if subtitle:
            header_text.append(f"\n{subtitle}", style="dim")
        self.console.print(Panel(header_text, border_style="cyan", expand=False))

    def print_success(self, message: str) -> None:
        """Print a success line."""
        self.console.print(f"[bold green]{CHECKMARK}[/bold green] {message}")

    def print_warning(self, message: str) -> None:
        """Print a warning line."""
        self.console.print(f"[bold yellow]{WARNINGMARK}[/bold yellow] {message}")

    def print_error(self, message: str) -> None:
        """Print an error line."""
        self.console.print(f"[bold red]{CROSSMARK}[/bold red] {message}")

    def print_info(self, message: str) -> None:
        """Print an informational line."""
        self.console.print(f"[bold blue]{INFOMARK}[/bold blue] {message}")
