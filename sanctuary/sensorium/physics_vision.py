"""Perception adapter: a rendered :class:`CameraFrame` -> a visual :class:`Percept`.

The third adapter in the training ground, and the first one whose tensor path
has somewhere to go.

``physics_percepts`` had to leave ``Percept.tensor_data`` empty because Luthi's
trunk has no proprioceptive encoder. Vision is different: ``VisionEncoder``
exists, ``luthi.sanctuary_interface.encode_vision`` is part of the seam, and the
checkpoint was trained on images. So this adapter *does* populate
``tensor_data`` -- with a tensor built to that encoder's actual contract rather
than to a plausible-looking guess.

Why vision and not another channel
----------------------------------

It is the modality JEPA's scale evidence rests on. Per
``LuthiModel/docs/research/2026-08-01_lewm-vs-vjepa2-framing.md``, V-JEPA 2 is
the existence proof that the architecture class survives being large -- 1.2B
params over >1M hours of *video*, then action-conditioned planning on real robot
arms from unlabeled footage. A body that sees is the closest this training
ground gets to the setting the objective is known to work in.

The encoder contract, matched exactly
-------------------------------------

Taken from the training pipeline (``luthi/coco_data.py::_load_image``) rather
than assumed, because a normalization mismatch is silent: the tensor would have
the right shape and dtype, the encoder would accept it, and the model would
simply learn from miscoloured input.

1. uint8 ``[H, W, 3]`` -> float in ``[0, 1]`` (divide by 255)
2. ``[H, W, 3]`` -> ``[3, H, W]`` (channels first)
3. ImageNet normalization: ``(x - mean) / std`` with
   ``mean=(0.485, 0.456, 0.406)``, ``std=(0.229, 0.224, 0.225)``
4. batch dimension -> ``[1, 3, H, W]``, which is what ``encode_vision`` takes

The frame should already be square and at the encoder's ``image_size`` (224 by
default, giving 196 patch tokens at ``patch_size=16``). Rendering at the right
size beats resizing afterwards, so :func:`frame_to_percept` validates rather
than silently rescaling -- a quiet resize is how an aspect-ratio bug survives to
training time.
"""

from __future__ import annotations

import torch

from sanctuary.core.schema import Percept
from sanctuary.physics.backends.mujoco_backend import CameraFrame

__all__ = [
    "frame_to_percept",
    "frame_to_tensor",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "VISUAL",
]

#: Modality tag. Matches the existing ``"visual"`` percepts in the codebase.
VISUAL = "visual"

#: Must stay identical to ``luthi/coco_data.py``. If that changes, this is wrong
#: and nothing will say so -- the shapes still line up.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def frame_to_tensor(frame: CameraFrame, *, expected_size: int | None = 224) -> torch.Tensor:
    """Convert a rendered frame to an ``encode_vision``-ready tensor.

    Args:
        frame: A frame from
            :meth:`~sanctuary.physics.backends.mujoco_backend.MuJoCoPhysicsAuthority.render_camera`.
        expected_size: Square size the encoder was built for. ``None`` skips the
            check, for a checkpoint trained at a different resolution.

    Returns:
        ``[1, 3, H, W]`` float32, ImageNet-normalized.

    Raises:
        ValueError: if the frame is not ``[H, W, 3]``, or is not square at
            ``expected_size``. Both are silent failures otherwise -- the encoder
            would accept a wrong-shaped or wrong-scaled image and simply learn
            from it.
    """
    rgb = frame.rgb
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(
            f"Expected an [H, W, 3] RGB frame, got shape {tuple(rgb.shape)}."
        )

    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    if height != width:
        raise ValueError(
            f"Expected a square frame; got {width}x{height}. Render square "
            f"rather than resizing here -- a silent rescale is how an "
            f"aspect-ratio bug reaches training."
        )
    if expected_size is not None and height != expected_size:
        raise ValueError(
            f"Frame is {width}x{height} but the encoder expects "
            f"{expected_size}x{expected_size}. Render at the encoder's size "
            f"(render_camera(width=..., height=...)) rather than rescaling; "
            f"pass expected_size=None if this checkpoint genuinely differs."
        )

    tensor = torch.from_numpy(rgb).to(torch.float32).div(255.0)
    tensor = tensor.permute(2, 0, 1)

    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    tensor = (tensor - mean) / std

    return tensor.unsqueeze(0)


def frame_to_percept(
    frame: CameraFrame,
    *,
    expected_size: int | None = 224,
    description: str | None = None,
) -> Percept:
    """Turn a rendered frame into a percept carrying the image itself.

    The dual path ``Percept`` documents: ``tensor_data`` holds the encoder-ready
    image for the trunk, and ``content`` holds a short text line for the prompt
    context. The text deliberately describes *the act of seeing* rather than the
    scene -- naming what is in view would be the scaffold doing the model's
    perceptual work for it, and handing it labels it never earned.

    Args:
        frame: The rendered frame.
        expected_size: Passed to :func:`frame_to_tensor`.
        description: Optional extra context for the text line. Use it for the
            camera's role ("forward eye"), not for scene contents.
    """
    tensor = frame_to_tensor(frame, expected_size=expected_size)

    detail = f" ({description})" if description else ""
    content = (
        f"[vision:{frame.camera}]{detail} {frame.width}x{frame.height} view "
        f"at t={frame.time:.2f}s"
    )

    return Percept(
        modality=VISUAL,
        content=content,
        source=f"physics:camera={frame.camera}:step={frame.step}",
        embedding_summary=f"visual field from {frame.camera}",
        tensor_data=tensor,
    )
