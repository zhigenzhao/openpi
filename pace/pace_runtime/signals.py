"""Signal handling utilities for PACE Runtime.

Provides utilities for handling SLURM signals like SIGTERM
for graceful checkpoint saving before job termination.
"""

from __future__ import annotations

import signal
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable


@dataclass
class SignalState:
    """Track signal state."""

    sigterm_received: bool = False
    sigusr1_received: bool = False
    sigterm_time: datetime | None = None
    sigusr1_time: datetime | None = None


# Global signal state
_signal_state = SignalState()

# Callbacks to run on signal
_sigterm_callbacks: list[Callable[[], None]] = []
_sigusr1_callbacks: list[Callable[[], None]] = []


def _sigterm_handler(signum: int, frame) -> None:
    """Handle SIGTERM signal."""
    _signal_state.sigterm_received = True
    _signal_state.sigterm_time = datetime.now()

    for callback in _sigterm_callbacks:
        try:
            callback()
        except Exception:
            pass  # Don't let callback errors prevent other callbacks


def _sigusr1_handler(signum: int, frame) -> None:
    """Handle SIGUSR1 signal (often used by SLURM for checkpoint request)."""
    _signal_state.sigusr1_received = True
    _signal_state.sigusr1_time = datetime.now()

    for callback in _sigusr1_callbacks:
        try:
            callback()
        except Exception:
            pass


def setup_signal_handlers(
    handle_sigterm: bool = True,
    handle_sigusr1: bool = True,
) -> None:
    """Setup signal handlers for graceful shutdown.

    Args:
        handle_sigterm: Whether to handle SIGTERM.
        handle_sigusr1: Whether to handle SIGUSR1.
    """
    if handle_sigterm:
        signal.signal(signal.SIGTERM, _sigterm_handler)

    if handle_sigusr1:
        signal.signal(signal.SIGUSR1, _sigusr1_handler)


def register_sigterm_callback(callback: Callable[[], None]) -> None:
    """Register a callback to run when SIGTERM is received.

    Args:
        callback: Function to call on SIGTERM.
    """
    _sigterm_callbacks.append(callback)


def register_sigusr1_callback(callback: Callable[[], None]) -> None:
    """Register a callback to run when SIGUSR1 is received.

    Args:
        callback: Function to call on SIGUSR1.
    """
    _sigusr1_callbacks.append(callback)


def should_checkpoint() -> bool:
    """Check if a checkpoint should be saved due to signal.

    Returns True if SIGUSR1 was received (SLURM checkpoint request).

    Returns:
        True if checkpoint should be saved.
    """
    return _signal_state.sigusr1_received


def should_stop() -> bool:
    """Check if training should stop due to signal.

    Returns True if SIGTERM was received.

    Returns:
        True if training should stop.
    """
    return _signal_state.sigterm_received


def clear_checkpoint_request() -> None:
    """Clear the checkpoint request flag after saving."""
    _signal_state.sigusr1_received = False


def get_signal_state() -> SignalState:
    """Get the current signal state.

    Returns:
        Copy of current signal state.
    """
    return SignalState(
        sigterm_received=_signal_state.sigterm_received,
        sigusr1_received=_signal_state.sigusr1_received,
        sigterm_time=_signal_state.sigterm_time,
        sigusr1_time=_signal_state.sigusr1_time,
    )


class GracefulShutdown:
    """Context manager for graceful shutdown handling.

    Example:
        >>> with GracefulShutdown() as shutdown:
        ...     for epoch in range(100):
        ...         train_epoch(epoch)
        ...         if shutdown.should_stop:
        ...             save_checkpoint()
        ...             break
    """

    def __init__(self):
        """Initialize the shutdown handler."""
        self._original_sigterm = None
        self._original_sigusr1 = None

    def __enter__(self) -> GracefulShutdown:
        """Setup signal handlers."""
        self._original_sigterm = signal.signal(signal.SIGTERM, _sigterm_handler)
        self._original_sigusr1 = signal.signal(signal.SIGUSR1, _sigusr1_handler)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Restore original signal handlers."""
        if self._original_sigterm is not None:
            signal.signal(signal.SIGTERM, self._original_sigterm)
        if self._original_sigusr1 is not None:
            signal.signal(signal.SIGUSR1, self._original_sigusr1)

    @property
    def should_stop(self) -> bool:
        """Check if SIGTERM was received."""
        return _signal_state.sigterm_received

    @property
    def should_checkpoint(self) -> bool:
        """Check if SIGUSR1 was received."""
        return _signal_state.sigusr1_received

    def clear_checkpoint_request(self) -> None:
        """Clear checkpoint request after saving."""
        _signal_state.sigusr1_received = False
