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
    convolved with the mean spot divided by the mean spot's sum. The outside edges are considered zeros.

    Args:
        pixel_score_image (`(n_batches x im_y x im_x x im_z) tensor[float32]`): OMP pixel scores in a 3D volume. Any
            non-computed or out of bounds pixel scores will be zero.
        mean_spot (`(size_y x size_x x size_z) tensor[float32]`): OMP mean spot shape.
        force_cpu (bool): use the CPU only. Default: true.

    Returns:
        (`(n_batches x im_y x im_x x im_z) tensor[float32]`): spot_score_image. OMP spot score for every image pixel, on
            every given batch.
    """
    assert type(pixel_score_image) is torch.Tensor
    assert type(mean_spot) is torch.Tensor
    assert pixel_score_image.dim() == 4
    assert pixel_score_image.shape[0] < 2_000, "More than 2,000 batches given"
    assert (mean_spot >= 0).all()

    device = system.get_device(force_cpu)

    score_image = pixel_score_image.detach().clone().to(device=device)
    spot_shape_kernel = mean_spot.detach().clone().to(dtype=score_image.dtype, device=device)
    spot_shape_kernel /= spot_shape_kernel.sum()

    spot_shape_kernel = spot_shape_kernel[np.newaxis, np.newaxis]
    score_image = score_image[:, np.newaxis]
    scores = torch.nn.functional.conv3d(score_image, spot_shape_kernel, padding="same", bias=None)[:, 0]

    scores = scores.cpu().to(dtype=pixel_score_image.dtype)
    return scores


def boost_z_edge_spot_scores(spot_score_image: torch.Tensor, mean_spot: torch.Tensor) -> torch.Tensor:
    """
    Along the z axis, the kernel is cut off if a pixel is too close the edge of the z stack. So, these pixel scores are
    boosted. This boosting is not applied along the x or y axes because there are many more x and y pixels and there is
    x/y tile overlap to solve the issue.

    Args:
        spot_score_image (`(n_batches x im_y x im_x x im_z) tensor[float32]`): the OMP spot score for every image pixel,
            on every given batch.
        mean_spot (`(size_y x size_x x size_z) tensor[float32]`): the OMP mean spot shape.

    Returns:
        (`(n_batches x im_y x im_x x im_z) tensor[float32]`): spot_score_image_boosted. The boosted OMP spot score
            image.
    """
    assert type(spot_score_image) is torch.Tensor
    assert type(mean_spot) is torch.Tensor
    assert spot_score_image.dim() == 4
    assert (mean_spot >= 0).all()

    spot_score_image_boosted = spot_score_image.detach().clone()
    spot_shape_kernel = mean_spot.detach().clone().to(dtype=spot_score_image_boosted.dtype)
    spot_shape_kernel /= spot_shape_kernel.sum()

    # FIXME: This algorithm assumes that the mean spot is symmetrical along the middle z plane. Can be made more robust.
    z_edge_size = min(mean_spot.shape[2] // 2, spot_score_image.shape[3])
    z_edge_weightings = torch.zeros((1, 1, 1, z_edge_size), dtype=torch.float32)
    for z_edge in range(z_edge_size):
        z_edge_weightings[0, 0, 0, z_edge] = torch.reciprocal(1 - spot_shape_kernel[:, :, : (z_edge + 1)].sum())
        z_edge_weightings[torch.isinf(z_edge_weightings)] = 0
        if z_edge > 0:
            assert (z_edge_weightings[..., z_edge] >= z_edge_weightings[..., z_edge - 1]).all()

    if z_edge_size > 0:
        spot_score_image_boosted[:, :, :, :z_edge_size] *= torch.flip(z_edge_weightings, [3])
        spot_score_image_boosted[:, :, :, -z_edge_size:] *= z_edge_weightings

    return spot_score_image_boosted
