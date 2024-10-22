from typing import Any, Tuple

import numpy as np
import scipy
import torch

from ..utils import base as utils_base


def compute_mean_spot(
    coefficients: Any,
    spot_positions_yxz: torch.Tensor,
    spot_positions_gene_no: torch.Tensor,
    tile_shape: Tuple[int, int, int],
    spot_shape: Tuple[int, int, int],
) -> torch.Tensor:
    """
    Compute the mean spot from the given positions on the coefficient images in a cuboid local region centred around each
    given spot position. The mean spot is the mean of the image signs in the cuboid region.

    Args:
        coefficients (`(n_pixels x n_genes) scipy.sparse.csr_array`): coefficient images. Any out of bounds
            retrievals around spots are set to zero.
        spot_positions_yxz (`(n_spots x 3) tensor`): every spot position to use to compute the spot. If n_spots is 0,
            a mean spot of zeros is returned.
        spot_positions_gene_no (`(n_spots) tensor[int]`): every spot position's gene number.
        tile_shape (tuple of three ints): the tile's shape in y, x, and z.
        spot_shape (tuple of three ints): spot size in y, x, and z respectively. This is the size of the cuboids
            around each spot position. This must be an odd number in each dimension so the spot position can be centred.

    Returns:
        (`spot_shape tensor[float32]`) mean_spot: the mean of the signs of the coefficient.
    """
    assert type(coefficients) is scipy.sparse.csr_matrix, f"Got type {type(coefficients)}"
    assert type(spot_positions_yxz) is torch.Tensor
    assert spot_positions_yxz.dim() == 2
    n_spots = int(spot_positions_yxz.shape[0])
    assert n_spots > 0
    assert spot_positions_yxz.shape[1] == 3
    assert type(spot_positions_gene_no) is torch.Tensor
    assert spot_positions_gene_no.dim() == 1
    assert spot_positions_yxz.shape[0] == spot_positions_gene_no.shape[0]
    assert type(tile_shape) is tuple
    assert len(tile_shape) == 3
    assert coefficients.shape[0] == np.prod(tile_shape)
    assert type(spot_shape) is tuple
    assert len(spot_shape) == 3
    assert all([type(spot_shape[i]) is int for i in range(3)])
    assert (torch.asarray(spot_shape) % 2 != 0).all(), "spot_shape must be only odd numbers"

    spot_shifts = np.array(utils_base.get_shifts_from_kernel(np.ones(spot_shape)))
    spot_shifts = torch.asarray(spot_shifts).int()
    n_shifts = spot_shifts.size(1)
    # (3, n_shifts)
    spot_shift_positions = spot_shifts + (torch.asarray(spot_shape, dtype=int) // 2)[:, np.newaxis]

    spots = torch.zeros((0, n_shifts)).float()

    for g in spot_positions_gene_no.unique():
        g_coef_image = torch.asarray(coefficients[:, [g]].toarray().reshape(tile_shape, order="F")).float()
        # Pad the coefficient image for out of bound cases.
        g_coef_image = torch.nn.functional.pad(g_coef_image, (0, spot_shape[2], 0, spot_shape[1], 0, spot_shape[0]))
        g_yxz = spot_positions_yxz[spot_positions_gene_no == g].int()
        # (3, n_shifts, n_spots)
        g_spot_positions_yxz = g_yxz.T[:, np.newaxis].repeat_interleave(n_shifts, dim=1)
        g_spot_positions_yxz += spot_shifts[:, :, np.newaxis]
        # (n_shifts, n_spots)
        g_spots = g_coef_image[tuple(g_spot_positions_yxz)].float()
        # (g_n_spots, n_shifts)
        g_spots = g_spots.T

        spots = torch.cat((spots, g_spots), dim=0)

    assert spots.shape == (n_spots, n_shifts)
    mean_spot = torch.zeros(spot_shape).float()
    mean_spot[tuple(spot_shift_positions)] = spots.sign().mean(dim=0)

    return mean_spot


def count_edge_ones(
    spot: torch.Tensor,
) -> int:
    """
    Counts the number of ones on the x and y edges for all z planes.

    Args:
        spot (`(size_y x size_x x size_z) tensor[int]`): OMP spot shape. It is a made up of only zeros and ones.
            Ones indicate where the spot coefficient is likely to be positive.
    """
    assert type(spot) is torch.Tensor
    assert spot.dim() == 3
    assert torch.isin(spot, torch.asarray([0, 1], device=spot.device)).all()

    count = 0
    for z in range(spot.shape[2]):
        count += spot[:, :, z].sum() - spot[1 : spot.shape[0] - 1, 1 : spot.shape[1] - 1, z].sum()
    return int(count)
