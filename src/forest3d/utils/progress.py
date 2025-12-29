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
    """Create a standard progress bar for Forest3D operations.

    Args:
        console: Optional Rich console for output.

    Returns:
        Configured Progress instance.
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    )


def create_spinner(console: Optional[Console] = None) -> Progress:
    """Create a spinner for indeterminate operations.

    Args:
        console: Optional Rich console for output.

    Returns:
        Configured Progress instance with spinner only.
    """
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
    """Wrap an iterator with a progress bar.

    Args:
        items: Iterator to wrap.
        total: Total number of items.
        description: Progress bar description.
        console: Optional Rich console for output.

    Yields:
        Items from the iterator.
    """
    with create_progress_bar(console) as progress:
        task = progress.add_task(description, total=total)
        for item in items:
            yield item
            progress.advance(task)
