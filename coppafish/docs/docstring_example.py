from typing import Tuple

import numpy as np
import torch
import zarr


def large_function(
    arr_0: np.ndarray[np.float16],
    arr_1: torch.Tensor,
    arr_2: zarr.Array,
    number: float | None = None,
) -> Tuple[zarr.Group, float, int]:
    """
    An description of exactly what the function does. This docstring must contain
    enough detail to make the exact function again, without looking at any code.

    Args:
        arr_0 (`(n_pixels x 3) ndarray[float32]`): a description of arr_0.
        arr_1 (`(n_pixels x n_rounds x n_channels) tensor[uint32]`): a description
            of arr_1.
        arr_2 (`(n_pixels x n_rounds x (n_channels + 1)) zarray[uint16]`): a
            description of arr_2.
        number (float or none, optional): a description of number. Default: none.

    Returns:
        A tuple containing:
            - (zgroup): zgroup_0. A zarr Group containing arrays named zarr_0,
                zarr_1, and zarr_2.
            - (float): variable_0. A description of variable_0.
            - (int): variable_1. A description of variable_1.
    """
    ...
