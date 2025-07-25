import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_r1_lite_example() -> dict:
    """Creates a random input example for the R1 Lite policy."""
    return {
        "state": np.random.rand(14),
        "base_image": np.random.randint(256, size=(240, 424, 3), dtype=np.uint8),
        "left_wrist_image": np.random.randint(256, size=(240, 424, 3), dtype=np.uint8),
        "right_wrist_image": np.random.randint(256, size=(240, 424, 3), dtype=np.uint8),
        "task": "drive to the white table",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class GalaxeaR1LiteInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # The action dimension of the model. Will be used to pad state and actions for pi0 model (not pi0-FAST).
    # Do not change this for your own dataset.
    action_dim: int

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        # We only mask padding for pi0 model, not pi0-FAST. Do not change this for your own dataset.
        mask_padding = self.model_type == _model.ModelType.PI0

        # We pad the proprioceptive input to the action dimension of the model.
        # For pi0-FAST, we don't pad the state.
        # Keep this for your own dataset, but if your dataset stores the proprioceptive input
        # in a different key than "state", you should change it below.
        state = transforms.pad_to_dim(data["state"], self.action_dim)

        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "image" or "wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # left wrist image below.
        base_image = _parse_image(data["base_image"])
        left_wrist_image = _parse_image(data["left_wrist_image"])
        right_wrist_image = _parse_image(data["right_wrist_image"])

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                # Mask any non-existent images with False (if ``mask_padding`` is True).
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            # We are padding to the model action dim.
            # For pi0-FAST, this is a no-op (since action_dim = 7).
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        elif "task" in data:
            inputs["prompt"] = data["task"]

        return inputs


@dataclasses.dataclass(frozen=True)
class GalaxeaR1LiteOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For R1 Lite, we return 17 actions: left_arm(6) + left_gripper(1) + right_arm(6) + right_gripper(1) + chassis(3)
        # For your own dataset, replace `17` with the action dimension of your dataset.
        return {"actions": np.asarray(data["actions"][:, :17])}