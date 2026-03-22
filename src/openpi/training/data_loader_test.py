import dataclasses

import jax
import numpy as np

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_state_restoration():
    """Test that save/restore of data loader state reproduces the correct next batch.

    This mirrors the exact train.py checkpoint/resume workflow:

      # --- original run ---
      data_iter = iter(loader)
      batch = next(data_iter)          # prefetch first batch
      for step in range(num_steps):
          train_step(batch)
          batch = next(data_iter)      # prefetch NEXT batch  ← happens before save
          if should_save(step):
              save_state(loader)       # saved with the prefetched batch already consumed

      # --- resumed run ---
      set_state(loader, saved_state)
      data_iter = iter(loader)
      batch = next(data_iter)          # must equal the prefetched batch from the save point

    The invariant: after restore, the first batch returned must be the same batch
    that was prefetched (but not yet trained on) when the checkpoint was written.
    """
    config = _config.get_config("pi05_gim_dual_tshirt")
    # num_workers=0 keeps the test single-process and avoids worker-spawn issues.
    config = dataclasses.replace(config, batch_size=4, num_workers=0)

    loader = _data_loader.create_data_loader(
        config,
        skip_norm_stats=True,
        shuffle=True,
    )

    # --- simulate a fresh training run ---
    data_iter = iter(loader)
    batch = next(data_iter)  # initial prefetch (train.py line: batch = next(data_iter))

    num_steps = 3
    for _ in range(num_steps):
        # train_step(batch) would go here — we skip the actual training.
        batch = next(data_iter)  # prefetch next batch, exactly as train.py does

    # At this point `batch` holds the prefetched-but-not-yet-trained batch.
    # train.py saves state here (after prefetch, before the next train_step).
    state = loader.get_state()
    assert state is not None, "get_state() returned None before any epoch was started"

    # Record the prefetched batch — this is what the resumed run must reproduce.
    _, actions_expected = batch
    state_expected = batch[0].state

    # --- simulate resume ---
    loader.set_state(state)
    resumed_iter = iter(loader)
    resumed_batch = next(resumed_iter)  # train.py: batch = next(data_iter)

    _, actions_resumed = resumed_batch

    np.testing.assert_array_equal(
        np.asarray(actions_expected),
        np.asarray(actions_resumed),
        err_msg="Actions after restore do not match the prefetched batch from the save point",
    )
    np.testing.assert_array_equal(
        np.asarray(state_expected),
        np.asarray(resumed_batch[0].state),
        err_msg="Robot state after restore does not match the prefetched batch from the save point",
    )
