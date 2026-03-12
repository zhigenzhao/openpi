"""Launch phases for PACE.

Defines the ordering of CLI argument injection phases.
"""

from __future__ import annotations

from enum import Enum
from typing import List


class LaunchPhase(str, Enum):
    """Phases for CLI argument injection.

    Arguments are assembled in phase order to build the final command.
    """

    PRE_BASE = "pre_base"      # Before base command args
    BASE = "base"              # Base command (from config)
    POST_BASE = "post_base"    # After base, before user args (typically)
    PRE_RESUME = "pre_resume"  # Before resume-related args
    POST_RESUME = "post_resume"  # After resume args
    FINAL = "final"            # Final arguments


# Phase execution order
PHASE_ORDER: list[LaunchPhase] = [
    LaunchPhase.PRE_BASE,
    LaunchPhase.BASE,
    LaunchPhase.POST_BASE,
    LaunchPhase.PRE_RESUME,
    LaunchPhase.POST_RESUME,
    LaunchPhase.FINAL,
]


class PhaseOrder:
    """Helper for working with phase ordering."""

    @staticmethod
    def get_order() -> list[LaunchPhase]:
        """Get phases in execution order."""
        return PHASE_ORDER.copy()

    @staticmethod
    def get_index(phase: LaunchPhase) -> int:
        """Get index of a phase in the order."""
        return PHASE_ORDER.index(phase)

    @staticmethod
    def is_before(phase1: LaunchPhase, phase2: LaunchPhase) -> bool:
        """Check if phase1 comes before phase2."""
        return PHASE_ORDER.index(phase1) < PHASE_ORDER.index(phase2)

    @staticmethod
    def is_after(phase1: LaunchPhase, phase2: LaunchPhase) -> bool:
        """Check if phase1 comes after phase2."""
        return PHASE_ORDER.index(phase1) > PHASE_ORDER.index(phase2)
