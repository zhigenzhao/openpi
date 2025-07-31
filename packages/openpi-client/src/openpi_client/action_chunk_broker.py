import concurrent.futures
import threading
from typing import Dict

import numpy as np
import tree
from typing_extensions import override

from openpi_client import base_policy as _base_policy


class ActionChunkBroker(_base_policy.BasePolicy):
    """Wraps a policy to return action chunks one-at-a-time.

    Assumes that the first dimension of all action fields is the chunk size.

    A new inference call to the inner policy is only made when the current
    list of chunks is exhausted.
    """

    def __init__(self, policy: _base_policy.BasePolicy, action_horizon: int):
        self._policy = policy
        self._action_horizon = action_horizon
        self._cur_step: int = 0

        self._last_results: Dict[str, np.ndarray] | None = None

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        if self._last_results is None:
            self._last_results = self._policy.infer(obs)
            self._cur_step = 0

        def slicer(x):
            if isinstance(x, np.ndarray):
                return x[self._cur_step, ...]
            else:
                return x

        results = tree.map_structure(slicer, self._last_results)
        self._cur_step += 1

        if self._cur_step >= self._action_horizon:
            self._last_results = None

        return results

    @override
    def reset(self) -> None:
        self._policy.reset()
        self._last_results = None
        self._cur_step = 0


class RTCActionChunkBroker(_base_policy.BasePolicy):
    """Wraps a policy to return action chunks one-at-a-time.

    Assumes that the first dimension of all action fields is the chunk size.

    A new inference call to the inner policy is only made when the current
    list of chunks is exhausted. The inference is run in a separate thread
    to avoid blocking the main thread.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        action_horizon: int,
        execution_horizon: int,
        inference_delay: int,
    ):
        self._policy = policy
        self._action_horizon = action_horizon
        self._cur_step: int = 0
        self._execution_horizon = execution_horizon
        self._inference_delay = inference_delay
        self._last_results: Dict[str, np.ndarray] | None = None
        self._prefix_actions: np.ndarray | None = None
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._inference_future: concurrent.futures.Future | None = None
        self._lock = threading.Lock()

        self._prefix_weights = get_prefix_weights(
            self._inference_delay, self._action_horizon - self._execution_horizon, self._action_horizon, "linear"
        )
        self._one_minus_prefix_weights = 1.0 - self._prefix_weights

    def _run_inference(self, obs: Dict) -> Dict:
        """Run inference in a separate thread."""
        return self._policy.infer(obs)

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        with self._lock:
            # Check if we need to start a new inference
            if self._last_results is None:
                # If there's already an inference running, wait for it
                if self._inference_future is None:
                    obs["prefix_actions"] = None
                    obs["inference_delay"] = self._inference_delay
                    # Start new inference in background thread
                    self._inference_future = self._executor.submit(self._run_inference, obs)

                self._last_results = self._inference_future.result()
                self._inference_future = None
                self._cur_step = 0

            def slicer(x):
                if isinstance(x, np.ndarray):
                    return x[self._cur_step, ...]
                else:
                    return x

            results = tree.map_structure(slicer, self._last_results)
            print(f"results: {results}, cur_step: {self._cur_step}")
            self._cur_step += 1

            # Check if we need to start the next inference early
            if self._cur_step == self._execution_horizon:
                # state = obs["state"]
                padding_dim = (self._execution_horizon, *self._last_results["actions"].shape[1:])
                self._prefix_actions = np.concatenate(
                    [
                        self._last_results["actions"][self._cur_step :, ...].copy(),
                        np.zeros(padding_dim, dtype=np.float32),
                    ],
                    axis=0,
                )

                self._inference_future = self._executor.submit(self._run_inference, obs)
            elif self._cur_step == self._execution_horizon + self._inference_delay:
                self._last_results = self._inference_future.result()
                new_actions = self._last_results["actions"]
                fixed_actions = (
                    new_actions * self._one_minus_prefix_weights[:, None]
                    + self._prefix_actions * self._prefix_weights[:, None]
                )
                self._last_results["actions"] = fixed_actions
                self._inference_future = None
                self._cur_step = self._inference_delay

            return results

    @override
    def reset(self) -> None:
        with self._lock:
            # Cancel any pending inference
            if self._inference_future is not None:
                self._inference_future.cancel()
                self._inference_future = None

            self._policy.reset()
            self._last_results = None
            self._cur_step = 0

    def __del__(self):
        """Clean up the thread pool when the object is destroyed."""
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)


def get_prefix_weights(start: int, end: int, total: int, schedule: str) -> np.ndarray:
    """With start=2, end=6, total=10, the output will be:
    1  1  4/5 3/5 2/5 1/5 0  0  0  0
        ^              ^
        start           end
    `start` (inclusive) is where the chunk starts being allowed to change. `end` (exclusive) is where the chunk stops
    paying attention to the prefix. if start == 0, then the entire chunk is allowed to change. if end == total, then the
    entire prefix is attended to.

    `end` takes precedence over `start` in the sense that, if `end < start`, then `start` is pushed down to `end`. Thus,
    if `end` is 0, then the entire prefix will always be ignored.
    """
    start = np.minimum(start, end)
    if schedule == "ones":
        w = np.ones(total)
    elif schedule == "zeros":
        w = (np.arange(total) < start).astype(np.float32)
    elif schedule == "linear" or schedule == "exp":
        w = np.clip((start - 1 - np.arange(total)) / (end - start + 1) + 1, 0, 1)
        if schedule == "exp":
            w = w * np.expm1(w) / (np.e - 1)
    else:
        raise ValueError(f"Invalid schedule: {schedule}")
    return np.where(np.arange(total) >= end, 0, w)
