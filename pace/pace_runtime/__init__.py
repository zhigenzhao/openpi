"""PACE Runtime - Python helper library for training code.

This package provides utilities for robust checkpoint saving and
training completion marking, designed to work with any training
framework.

Example:
    >>> from pace_runtime import checkpoint_transaction, mark_training_done
    >>>
    >>> with checkpoint_transaction(checkpoint_dir, tag="step_1000") as tx_dir:
    ...     torch.save(model.state_dict(), f"{tx_dir}/model.pt")
    ...     torch.save(optimizer.state_dict(), f"{tx_dir}/optimizer.pt")
    >>>
    >>> mark_training_done(run_dir)
"""

from pace_runtime.checkpointing import (
    checkpoint_transaction,
    find_latest_checkpoint,
    is_checkpoint_valid,
)
from pace_runtime.markers import (
    mark_training_done,
    is_training_done,
    CHECKPOINT_OK,
    TRAINING_DONE,
)

__all__ = [
    "checkpoint_transaction",
    "find_latest_checkpoint",
    "is_checkpoint_valid",
    "mark_training_done",
    "is_training_done",
    "CHECKPOINT_OK",
    "TRAINING_DONE",
]
