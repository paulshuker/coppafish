import numpy as np
import scipy
import torch


def is_duplicate_spot(yxz_global_positions: torch.Tensor, tile_number: int, tile_centres: torch.Tensor) -> torch.Tensor:
    """
    Checks what spot positions are duplicates. A duplicate is defined as any spot that is closer to a different tile
    origin than the one it is assigned to.

    Args:
        yxz_global_positions (`(n_points x 3) tensor[int]`): y, x, and z global positions for each spot.
        tile_number (int): the tile index for all spot positions.
        tile_centres (`(n_tiles x 3) tensor[float]`): each tile's centre in global coordinates.

    Returns:
        (`(n_points) tensor[bool]`): is_duplicate. True for each duplicate spot.
    """
    assert type(yxz_global_positions) is torch.Tensor
    n_points = yxz_global_positions.shape[0]
    assert n_points > 0, "Require at least one spot"
    assert yxz_global_positions.shape[1] == 3
    assert type(tile_number) is int
    assert type(tile_centres) is torch.Tensor
    assert tile_centres.shape[1] == 3
    assert tile_number >= 0 and tile_number < tile_centres.shape[0]

    kdtree = scipy.spatial.KDTree(tile_centres.numpy())
    # Find the nearest tile origin for each spot position.
    # If this is not the tile number assigned to the spot, it is a duplicate.
    closest_tile_numbers = kdtree.query(yxz_global_positions.numpy(), k=1)[1]
    closest_tile_numbers = np.array(closest_tile_numbers)
    closest_tile_numbers = torch.asarray(closest_tile_numbers)
    is_duplicate = closest_tile_numbers != tile_number

    return is_duplicate
