"""Progress bar utilities for long-running operations."""

from typing import Callable, Iterator, Optional, TypeVar

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

T = TypeVar("T")


def create_progress_bar(console: Optional[Console] = None) -> Progress:
    """Create a standard progress bar."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    )


def create_spinner(console: Optional[Console] = None) -> Progress:
    """Create a spinner for indeterminate operations."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    )


def progress_iterator(
    items: Iterator[T],
    total: int,
    description: str = "Processing...",
    console: Optional[Console] = None,
) -> Iterator[T]:
    """Wrap an iterator with a progress bar."""
    with create_progress_bar(console) as progress:
        task = progress.add_task(description, total=total)
        for item in items:
            yield item
            progress.advance(task)
