import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_gim_dual_example() -> dict:
    """Creates a random input example for the GIM dual arm policy."""
    return {
        "state": np.random.rand(14),
        "top_image": np.random.randint(256, size=(240, 424, 3), dtype=np.uint8),
        "left_wrist_image": np.random.randint(256, size=(240, 424, 3), dtype=np.uint8),
        "right_wrist_image": np.random.randint(256, size=(240, 424, 3), dtype=np.uint8),
        "task": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class GimDualArmInputs(transforms.DataTransformFn):
    """Converts inputs to the model format for GIM dual arm."""

    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        state = transforms.pad_to_dim(data["state"], self.action_dim)

        base_image = _parse_image(data["top_image"])
        left_wrist_image = _parse_image(data["left_wrist_image"])
        right_wrist_image = _parse_image(data["right_wrist_image"])

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class GimDualArmOutputs(transforms.DataTransformFn):
    """Converts outputs from the model back to GIM dual arm format."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :14])}
