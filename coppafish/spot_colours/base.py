from typing import Any, Optional, Union

import numpy as np
import torch
import zarr


def convert_coords_to_torch_grid(yxz_coords: torch.Tensor, image_shape: tuple[int, int, int]) -> torch.Tensor:
    """
    Convert typically used y, x, z pixel coordinates into pytorch grid coordinates as defined in pytorch's grid_sample
    function (https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html) with align_corners set to
    True.

    Args:
        yxz_coords(`(... x 3) tensor[float or int]`): y, x, and z positions.
        image_shape (tuple of three ints): the image length that the yxz_coords are relative to in the y, x, and z
            directions respectively.

    Returns:
        `yxz_coords.shape tensor[float32]` yxz_grid_coords: yxz_coords converted to pytorch grid space, ready to use in
            grid_sample.
    """
    assert type(yxz_coords) is torch.Tensor
    assert yxz_coords.numel() > 0
    assert yxz_coords.shape[-1] == 3
    assert type(image_shape) is tuple
    assert len(image_shape) == 3
    assert all([type(length) is int and length > 0 for length in image_shape])

    ndim = yxz_coords.ndim
    # The spacing between two pytorch grid positions that should be 1 pixel separation in yxz coordinate space for each
    # direction.
    grid_step = 2 / (torch.tensor(image_shape).float() - 1)
    grid_step = grid_step.reshape((1,) * (ndim - 1) + (3,))
    yxz_grid = yxz_coords.detach().clone().float()
    yxz_grid *= grid_step
    yxz_grid -= 1
    # Edge case when the image_shape has single pixel dimension(s). All coordinates within said single pixel are set
    # to 0. Otherwise, they are set to -2 so they are out of bounds.
    is_single_pixel_dimension = torch.tensor(image_shape) == 1
    for i in range(3):
        if is_single_pixel_dimension[i]:
            is_within_bound = torch.isclose(yxz_coords[..., i], torch.zeros(1).float())
            yxz_grid[..., i][is_within_bound] = 0
            yxz_grid[..., i][torch.logical_not(is_within_bound)] = -2

    # The grid_sample function places z coordinate at index 0 and y at the last index.
    yxz_grid = yxz_grid[..., [2, 1, 0]]
    return yxz_grid


def apply_flow_new(
    yxz: np.ndarray | torch.Tensor, flow: zarr.Array | np.ndarray, tile: int, r: int
) -> np.ndarray | torch.Tensor:
    """
    Apply the pixel shifts from flow to each yxz positions given. If the yxz positions are not exact integers within
    the flow image, then bilinear interpolation is done. On out-of-bound regions, the flow shift is taken to be the
    same as the nearby, edge pixels.

    Args:
        yxz (`(n_points x 3) ndarray[int or float] or tensor[int or float]`): the yxz coordinates.
        flow (`(n_tiles x n_rounds x 3 x im_y x im_x x im_z) zarray[float] or ndarray[float]`): the optical flow shift
            for each pixel in the image for the y, x, and z directions. yxz positions must be aligned with the flow
            image. I.e. 0, 0, 0 in yxz must be shifted by the flow at 0, 0, 0.
        tile (int): tile index to gather flow for.
        r (int): round index to gather flow for.

    Returns:
        (`(n_points x 3) ndarray[float32] or tensor[float32]`): yxz_flow. yxz coordinates optical flow shifted. Returns
            a tensor if yxz is a tensor.
    """
    assert type(yxz) is np.ndarray or type(yxz) is torch.Tensor
    assert type(flow) is zarr.Array or type(flow) is np.ndarray
    assert type(tile) is int
    assert type(r) is int
    assert flow.shape[0] > 0
    assert flow.shape[1] > 0
    assert flow.shape[2] == 3
    tile_shape = tuple(flow.shape[3:])
    assert tile >= 0 and tile < flow.shape[0]
    assert r >= 0 and r < flow.shape[1], f"Got round {r}, expected r >= 0 and r < {flow.shape[1]}"
    yxz_torch = yxz
    if type(yxz) is np.ndarray:
        yxz_torch = torch.tensor(yxz_torch)
    yxz_torch = yxz_torch.detach().clone().float()
    if yxz_torch.size(0) == 0:
        return yxz_torch
    yxz_min, yxz_max = yxz_torch.min(0).values, yxz_torch.max(0).values
    yxz_min = yxz_min.floor().clamp(min=0).int().tolist()
    yxz_max = (yxz_max.ceil() + 1).clamp(max=torch.tensor(tile_shape)).int().tolist()
    subset_tile_shape: tuple[int] = tuple([yxz_max[i] - yxz_min[i] for i in range(3)])
    flow_torch = flow[tile, r, :, yxz_min[0] : yxz_max[0], yxz_min[1] : yxz_max[1], yxz_min[2] : yxz_max[2]]
    assert flow_torch.shape == (3,) + subset_tile_shape
    flow_torch = torch.from_numpy(flow_torch).float()
    assert yxz_torch.ndim == 2
    assert yxz_torch.shape[1] == 3
    assert flow_torch.ndim == 4, f"{flow_torch.ndim}"
    assert flow_torch.shape[0] == 3

    yxz_torch -= torch.tensor(yxz_min)[None]
    yxz_grid = convert_coords_to_torch_grid(yxz_torch, subset_tile_shape)
    # Input has shape (3, 1, flow.shape[0], flow.shape[1], flow.shape[2]).
    # Grid has shape (3 (repeated thrice), 1, 1, n_points, 3)
    # Result has shape (3, 1, 1, 1, n_points).
    flow_torch = flow_torch[:, None]
    yxz_grid = yxz_grid[None, None, None].repeat_interleave(3, dim=0)
    flow_shifts = torch.nn.functional.grid_sample(flow_torch, yxz_grid, align_corners=True, padding_mode="border")
    flow_shifts = flow_shifts[:, 0, 0, 0]
    yxz_torch += flow_shifts.T + torch.tensor(yxz_min)[None]

    if type(yxz) is np.ndarray:
        yxz_torch = yxz_torch.numpy()
    return yxz_torch


def apply_affine(yxz: torch.Tensor, affine: torch.Tensor) -> torch.Tensor:
    """
    Transform the coordinates yxz based on the affine transform alone. E.g. to find coordinates of spots on the same
    tile but on a different round and channel.

    Args:
        yxz (`(n_points x 3) tensor[float or int]`): y, x, and z coordinates positions to affine transform.
        affine (`(4 x 3) tensor[float]`): affine transform to apply.

    Returns:
        `(n_points x 3) tensor[float32]` yxz_affine: the yxz coordinates affine transformed.
    """
    assert type(yxz) is torch.Tensor
    assert yxz.ndim == 2
    assert yxz.shape[1] == 3
    assert type(affine) is torch.Tensor
    assert affine.shape == (4, 3)
    # apply icp correction
    yxz_transform = torch.cat([yxz, torch.ones(yxz.shape[0], 1)], dim=1).float()
    yxz_transform = yxz_transform @ affine
    return yxz_transform


def get_spot_colours_new_safe(
    nbp_basic_info, yxz: Optional[Union[np.ndarray, torch.Tensor]] = None, *args, **kwargs
) -> np.ndarray:
    """
    A wrapper function for get_spot_colours_new below. See that function for the arguments/details. This function runs
    get_spot_colours_new through multiple calls to avoid memory crashing on large images.

    Args:
        nbp_basic_info (NotebookPage): `basic_info` notebook page.
        yxz (`(n_points x 3) ndarray or tensor`, optional): positions to gather. Default: the entire tile.
        args (tuple): positional arguments.
        kwargs (dict[str, any]): keyword arguments.

    Returns:
        `(n_points x n_rounds x n_channels_use) ndarray` colours: gathered image colours.
    """
    assert type(yxz) is np.ndarray or type(yxz) is torch.Tensor or yxz is None
    tile_shape = (nbp_basic_info.tile_sz, nbp_basic_info.tile_sz, len(nbp_basic_info.use_z))
    if yxz is None:
        yxz = [np.linspace(0, tile_shape[i] - 1, tile_shape[i]) for i in range(3)]
        yxz = np.array(np.meshgrid(*yxz, indexing="ij")).astype(np.int16).reshape((3, -1), order="F").T
    if type(yxz) is np.ndarray:
        yxz = torch.from_numpy(yxz).detach().clone()
    assert yxz.ndim == 2
    assert yxz.shape[1] == 3

    first_compute = True
    z_coords = yxz[:, 2].detach().clone()
    # Sort yxz coordinates to gather based on z plane as we want to gather all of the same z plane at once.
    # This makes for faster disk reading and avoids memory crashing when yxz spans a lot of the tile space.
    _, yxz_sort_indices = z_coords.sort(stable=True)
    yxz_sorted = yxz[yxz_sort_indices]
    for z in z_coords.unique():
        is_z = torch.isclose(yxz_sorted[:, 2], z).nonzero()
        index_min, index_max = is_z[0], is_z[-1] + 1
        i_colours = get_spot_colours_new(yxz_sorted[index_min:index_max], *args, **kwargs)
        if first_compute:
            colours = np.zeros((yxz.shape[0],) + i_colours.shape[1:], i_colours.dtype)
            first_compute = False
        colours[yxz_sort_indices[index_min:index_max]] = i_colours
        del i_colours
    return colours


def get_spot_colours_new(
    yxz: Union[np.ndarray, torch.Tensor],
    image: Union[np.ndarray, zarr.Array],
    flow: Union[np.ndarray, zarr.Array],
    affine: Union[np.ndarray, torch.Tensor],
    tile: int,
    use_rounds: list[int],
    use_channels: list[int],
    output_dtype: np.dtype = np.float32,
    out_of_bounds_value: Any = np.nan,
) -> np.ndarray:
    """
    (Sub)pixel positions are gathered from the given image. First, the given yxz positions are optical flow shifted
    (yxz + flow_shifts). Second, the positions are then affine transformed (yxz_flow @ affine) to get final, float32
    positions for each round/channel combination. These positions are then gathered from image for the given channels
    on all rounds. Subpixel resolution is supported as bilinear interpolation is used to gather the optical flow shifts
    and the final image values.

    Args:
        yxz (`(n_points x 3) ndarray or tensor`): positions to gather.
        image (`(n_tiles x n_rounds x n_channels x im_y x im_x x im_z) ndarray or zarray`): image to gather from.
        flow (`(n_tiles x n_rounds x 3 x im_y x im_x x im_z) ndarray or zarray`): optical flow shifts.
        affine (`(n_tiles x n_rounds x n_channels x 4 x 3) ndarray or tensor`): affine transform.
        tile (int): tile index.
        use_rounds (list of ints): the round indices to use. These rounds must be sequencing rounds.
        use_channels (list of ints): channel indices to use.
        output_dtype (np.dtype, optional): the returned spot colour datatype. Default: float32.
        out_of_bounds_value (any): what to value to set for out of bound spot colours. Default: np.nan.

    Returns:
        `(n_points x n_rounds x n_channels_use) ndarray[output_dtype]`: colours. Gathered image colours.
    """
    # Default value.
    if use_channels is None:
        use_channels = list(range(image.shape[2]))
    # Verify parameters.
    assert type(yxz) is np.ndarray or type(yxz) is torch.Tensor
    assert type(image) is np.ndarray or type(image) is zarr.Array
    assert type(flow) is np.ndarray or type(flow) is zarr.Array
    assert type(affine) is np.ndarray or type(affine) is torch.Tensor
    assert type(tile) is int
    assert type(use_rounds) is list
    assert use_channels is None or type(use_channels) is list
    assert yxz.ndim == 2
    assert yxz.shape[1] == 3
    assert image.ndim == 6
    assert flow.ndim == 6
    assert flow.shape[2] == 3
    assert affine.ndim == 5
    assert affine.shape[3:] == (4, 3)
    assert tile >= 0 and tile < image.shape[0]
    assert len(use_rounds) > 0
    assert all([type(r) is int for r in use_rounds])
    assert all([r >= 0 and r < flow.shape[1] for r in use_rounds]), "Unknown round index in use_rounds"
    assert len(use_channels) > 0
    assert all([type(c) is int for c in use_channels])
    assert all([c >= 0 and c < image.shape[2] for c in use_channels])

    # TODO: GPU support.

    # Prepare variables.
    tile_shape = tuple(image.shape[3:])
    # Pytorch float32 tensors are used whilst computing.
    yxz_torch = yxz
    if type(yxz_torch) is np.ndarray:
        yxz_torch = torch.tensor(yxz_torch)
    yxz_torch = yxz_torch.detach().clone().float()
    affine_torch = affine
    if type(affine_torch) is np.ndarray:
        affine_torch = torch.tensor(affine_torch)
    affine_torch = affine_torch.detach().clone().float()

    # TODO: Vectorise applying flow and affine transforms.
    yxz_t = torch.full((len(use_rounds), len(use_channels), yxz.shape[0], 3), torch.nan, dtype=torch.float32)
    for r in use_rounds:
        # image_tr = np.zeros((len(use_channels), 1) + tile_shape, np.float32)
        yxz_tr = torch.zeros((len(use_channels), yxz.shape[0], 3), dtype=torch.float32)
        # First, apply round r optical flow to the given coordinates.
        yxz_r = apply_flow_new(yxz_torch, flow, tile, r)
        for c_index, c in enumerate(use_channels):
            # For each channel, apply the affine transform to the optical flow shifted yxz coordinates.
            yxz_trc = apply_affine(yxz_r, affine_torch[tile, r, c])
            yxz_tr[c_index] = yxz_trc
            del yxz_trc
        del yxz_r

        yxz_t[r] = yxz_tr
        del yxz_tr

    # Gather the smallest sized cuboid of filter image data to bilinear-interpolate all yxz_t coordinates.
    # This saves tons of disk read time and avoids memory crashing.
    yxz_t_min = yxz_t.min(0)[0].min(0)[0].min(0)[0].floor().clamp(0)
    yxz_t_max = yxz_t.max(0)[0].max(0)[0].max(0)[0].ceil().clamp(yxz_t_min, torch.tensor(tile_shape)).int().tolist()
    yxz_t_min = yxz_t_min.int().tolist()
    yxz_t_max = [min(dim + 1, tile_shape[i]) for i, dim in enumerate(yxz_t_max)]
    subset_tile_shape: tuple[int] = tuple([yxz_t_max[i] - yxz_t_min[i] for i in range(3)])
    subset_tile_shape = tuple([max(shape, 1) for shape in subset_tile_shape])
    image_t = torch.zeros((len(use_rounds), len(use_channels), 1) + subset_tile_shape, dtype=torch.float32)
    for r in use_rounds:
        for c_index, c in enumerate(use_channels):
            image_trc = image[
                tile, r, c, yxz_t_min[0] : yxz_t_max[0], yxz_t_min[1] : yxz_t_max[1], yxz_t_min[2] : yxz_t_max[2]
            ]
            image_trc = torch.from_numpy(image_trc).float()
            image_t[r, c_index, 0] = image_trc

    image_t = image_t.reshape((len(use_rounds) * len(use_channels), 1) + subset_tile_shape)
    yxz_t = yxz_t.reshape((len(use_rounds) * len(use_channels), 1, 1, yxz.shape[0], 3))

    # Convert the yxz coordinates relative to the new subset image.
    yxz_t -= torch.tensor(yxz_t_min)[None, None, None, None]
    grid_t = convert_coords_to_torch_grid(yxz_t, subset_tile_shape)
    del yxz_t

    # Input (image_tr) has shape (n_rounds * n_channels_use, 1, image.shape[0], image.shape[1], image.shape[2]).
    # Grid has shape (n_rounds * n_channels_use, 1, 1, n_points, 3)
    # Result has shape (n_rounds * n_channels_use, 1, 1, 1, n_points).
    colours = torch.nn.functional.grid_sample(image_t, grid_t, align_corners=True)

    # Grid positions that are out of bounds are filled.
    out_of_bounds = (grid_t < -1) | (grid_t > +1)
    out_of_bounds = out_of_bounds.any(4)[:, np.newaxis]
    colours[out_of_bounds] = out_of_bounds_value
    del out_of_bounds, grid_t

    colours = colours[:, 0, 0, 0].reshape((len(use_rounds), len(use_channels), yxz.shape[0]))
    # (n_rounds, n_channels_use, n_points) -> (n_points, n_rounds, n_channels_use).
    colours = colours.swapaxes(0, 2).swapaxes(1, 2)

    colours = colours.numpy()
    colours = colours.astype(output_dtype)

    return colours
