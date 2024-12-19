import numpy as np
import torch

from ..utils import system


def score_pixel_score_image(
    pixel_score_image: torch.Tensor,
    mean_spot: torch.Tensor,
    force_cpu: bool = True,
) -> torch.Tensor:
    """
    Computes the OMP spot score image from the pixel score image(s). The final spot score image is the pixel score image
    convolved with the mean spot divided by the mean spot's sum.

    Args:
        pixel_score_image (`(n_batches x im_y x im_x x im_z) tensor[float32]`): OMP pixel scores in a 3D volume. Any
            non-computed or out of bounds pixel scores will be zero.
        mean_spot (`(size_y x size_x x size_z) tensor[float32]`): OMP mean spot shape. This can range from -1 and 1.
        force_cpu (bool): use the CPU only, never the GPU. Default: true.

    Returns:
        (`(n_batches x im_y x im_x x im_z) tensor[float32]`): spot_score_image. OMP spot score for every image pixel, on
            every given batch.
    """
    assert type(pixel_score_image) is torch.Tensor
    assert type(mean_spot) is torch.Tensor
    assert pixel_score_image.dim() == 4
    assert pixel_score_image.shape[0] < 2_000, "More than 2,000 batches given"
    assert (mean_spot >= 0).all()

    run_on = system.get_device(force_cpu)

    score_image = pixel_score_image.detach().clone().to(device=run_on)
    mean_spot = mean_spot.to(device=run_on)

    spot_shape_kernel = mean_spot.detach().clone()
    spot_shape_kernel /= spot_shape_kernel.sum()

    spot_shape_kernel = spot_shape_kernel[np.newaxis, np.newaxis]
    score_image = score_image[:, np.newaxis]
    scores = torch.nn.functional.conv3d(score_image, spot_shape_kernel, padding="same", bias=None)[:, 0]
    scores = scores.cpu().to(dtype=pixel_score_image.dtype)

    return scores
