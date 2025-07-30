from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._rng = rng or jax.random.key(0)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

        start_time = time.monotonic()
        self._rng, sample_rng = jax.random.split(self._rng)
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng, _model.Observation.from_dict(inputs), **self._sample_kwargs),
        }
        # Unbatch and convert to np.ndarray.        # Unbatch and convert to np.ndarray.
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        model_time = time.monotonic() - start_time

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class RTCPolicy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        action_horizon: int = 50,
        prefix_attention_horizon: int = 25,
        inference_delay: int = 5,
        max_guidance_weight: float = 1.0,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self._sample_actions_rtc = nnx_utils.module_jit(model.sample_actions_rtc)
        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._rng = rng or jax.random.key(0)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}

        self.prefix_attention_horizon: int = prefix_attention_horizon
        self.inference_delay: int = inference_delay
        self.max_guidance_weight: float = max_guidance_weight
        self.action_horizon: int = action_horizon
        self._model = model
        self._jit_warmed_up = False
        self._enable_rtc = False

    def _warmup_jit_functions(self, obs_dict: dict):
        """Pre-compile JIT functions using the first real observation to avoid compilation delay."""
        print("[RTCPolicy] Pre-compiling JIT functions with first observation...")

        # Use real observation structure but create a clean copy for warmup
        warmup_obs = {}
        for k, v in obs_dict.items():
            if k not in {"prefix_actions", "inference_delay", "prefix_attention_horizon"}:
                warmup_obs[k] = v

        # Apply transformations and convert to JAX format
        warmup_inputs = jax.tree.map(lambda x: x, warmup_obs)
        warmup_inputs = self._input_transform(warmup_inputs)
        warmup_inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], warmup_inputs)
        warmup_obs_jax = _model.Observation.from_dict(warmup_inputs)

        # Warmup sample_actions (regular inference)
        dummy_rng = jax.random.key(42)
        print("[RTCPolicy] Warming up sample_actions...")
        _ = self._sample_actions(dummy_rng, warmup_obs_jax, **self._sample_kwargs)

        # Warmup sample_actions_rtc (RTC inference)
        print("[RTCPolicy] Warming up sample_actions_rtc...")
        dummy_prefix_actions = jnp.zeros((50, self._model.action_dim))  # Full horizon dummy prefix
        _ = self._sample_actions_rtc(
            dummy_rng,
            warmup_obs_jax,
            dummy_prefix_actions,
            self.inference_delay,
            self.prefix_attention_horizon,
            self.max_guidance_weight,
            **self._sample_kwargs,
        )
        print("[RTCPolicy] JIT warmup complete!")
        self._jit_warmed_up = True

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        # Warm up JIT functions on first call
        if not self._jit_warmed_up:
            self._warmup_jit_functions(obs)

        # Extract RTC-specific parameters from obs
        if obs.get("prefix_actions") is not None:
            prefix_actions_np = obs["prefix_actions"]
            # Pad prefix_actions to full action_horizon
            pad_length = self.action_horizon - prefix_actions_np.shape[0]
            padding = jnp.zeros((pad_length, prefix_actions_np.shape[1]))
            prefix_actions_jax = jnp.concatenate([jnp.asarray(prefix_actions_np), padding], axis=0)
            assert self.prefix_attention_horizon == prefix_actions_np.shape[0]
            assert self.inference_delay == obs.get("inference_delay", 0)
            self._enable_rtc = True
        else:
            prefix_actions_jax = None
            self._enable_rtc = False

        # Create inputs dict without the RTC-specific keys
        rtc_keys = {"prefix_actions", "inference_delay", "prefix_attention_horizon"}
        inputs = {k: v for k, v in obs.items() if k not in rtc_keys}

        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, inputs)
        inputs = self._input_transform(inputs)
        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

        start_time = time.monotonic()
        self._rng, sample_rng = jax.random.split(self._rng)
        if self._enable_rtc:
            outputs = {
                "state": inputs["state"],
                "actions": self._sample_actions_rtc(
                    sample_rng,
                    _model.Observation.from_dict(inputs),
                    prefix_actions_jax,
                    self.inference_delay,
                    self.prefix_attention_horizon,
                    self.max_guidance_weight,
                    **self._sample_kwargs,
                ),
            }
        else:
            # Fallback to the original sample_actions method if prefix_attention_horizon is not set.
            # This is useful for compatibility with older models.
            outputs = {
                "state": inputs["state"],
                "actions": self._sample_actions(
                    sample_rng,
                    _model.Observation.from_dict(inputs),
                    **self._sample_kwargs,
                ),
            }

        # Unbatch and convert to np.ndarray.        # Unbatch and convert to np.ndarray.
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        model_time = time.monotonic() - start_time

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        print(f"[RTC] Inference completed in {model_time * 1000:.2f} ms, outputs: {outputs}")
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
