import math as maths
from typing import List, Tuple, Union, Any, Optional

import numpy as np
import torch
import tqdm
import zarr

from .. import utils


def convert_coords_to_torch_grid(yxz_coords: torch.Tensor, image_shape: tuple[int, int, int]) -> torch.Tensor:
    """
    Convert typically used y, x, z pixel coordinates into pytorch grid coordinates as defined in pytorch's grid_sample
    function (https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html) with align_corners set to
    True.

    Args:
        - yxz_coords(`(n_points x 3) tensor[float or int]`): y, x, and z positions.
        - image_shape (tuple of three ints): the image length that the yxz_coords are relative to in the y, x, and z
            directions respectively.

    Returns:
        `(n_points x 3) tensor[float32]` yxz_grid_coords: yxz_coords converted to pytorch grid space, ready to use in
            grid_sample.
    """
    assert type(yxz_coords) is torch.Tensor
    assert yxz_coords.shape[1] == 3
    assert type(image_shape) is tuple
    assert len(image_shape) == 3
    assert all([type(length) is int and length > 0 for length in image_shape])

    # The spacing between two pytorch grid positions that should be 1 pixel separation in yxz coordinate space for each
    # direction.
    grid_step = 2 / (torch.tensor(image_shape).float() - 1)
    yxz_grid = yxz_coords.detach().clone().float()
    yxz_grid *= grid_step[None]
    yxz_grid -= 1
    # The grid_sample function places z coordinate at index 0, with y at the last index.
    yxz_grid = yxz_grid[..., [2, 1, 0]]
    return yxz_grid


def apply_flow_new(
    yxz: Union[np.ndarray, torch.Tensor], flow: Union[zarr.Array, np.ndarray], tile: int, r: int
) -> torch.Tensor:
    """
    Apply the pixel shifts from flow to each yxz positions given. If the yxz positions are not exact integers within
    the flow image, then bilinear interpolation is done. On out-of-bound regions, the flow shift is taken to be the
    same as the nearby, edge pixels.

    Args:
        - yxz (`(n_points x 3) ndarray[int or float] or tensor[int or float]`): the yxz coordinates.
        - flow (`(n_tiles x n_rounds x 3 x im_y x im_x x im_z) zarray[float] or ndarray[float]`): the optical flow
            shift for each pixel in the image for the y, x, and z directions. yxz positions must be aligned with the
            flow image. I.e. 0, 0, 0 in yxz must be shifted by the flow at 0, 0, 0.
        - tile (int): tile index to gather flow for.
        - r (int): round index to gather flow for.

    Returns:
        `(n_points x 3) tensor[float32]` yxz_flow: yxz coordinates optical flow shifted.
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
    # Add one pixel of additional flow retrieval for interpolation.
    yxz_min = (yxz_min - 1).clamp(min=0).int().tolist()
    yxz_max = (yxz_max + 1).clamp(max=torch.tensor(tile_shape)).int().tolist()
    flow_torch = np.zeros(flow.shape[2:], np.float32)
    flow_torch[:, yxz_min[0] : yxz_max[0], yxz_min[1] : yxz_max[1], yxz_min[2] : yxz_max[2]] = flow[
        tile, r, :, yxz_min[0] : yxz_max[0], yxz_min[1] : yxz_max[1], yxz_min[2] : yxz_max[2]
    ]
    flow_torch = torch.tensor(flow_torch)
    assert yxz_torch.ndim == 2
    assert yxz_torch.shape[1] == 3
    assert flow_torch.ndim == 4, f"{flow_torch.ndim}"
    assert flow_torch.shape[0] == 3
    yxz_grid = convert_coords_to_torch_grid(yxz_torch, tile_shape)
    # Input has shape (3, 1, flow.shape[0], flow.shape[1], flow.shape[2]).
    # Grid has shape (3 (repeated thrice), 1, 1, n_points, 3)
    # Result has shape (3, 1, 1, 1, n_points).
    flow_torch = flow_torch[:, None]
    yxz_grid = yxz_grid[None, None, None].repeat_interleave(3, dim=0)
    flow_shifts = torch.nn.functional.grid_sample(flow_torch, yxz_grid, align_corners=True, padding_mode="border")
    flow_shifts = flow_shifts[:, 0, 0, 0]
    yxz_torch += flow_shifts.T
    return yxz_torch


def apply_flow(
    yxz: Union[np.ndarray, torch.Tensor],
    flow: Union[np.ndarray, torch.Tensor],
    top_left: Union[np.ndarray, torch.Tensor] = np.array([0, 0, 0]),
) -> Union[np.ndarray, torch.Tensor]:
    """
    Apply a flow to a set of points. Note that this is applying forward warping, meaning that new_points = points +
    flow. The flow we pass in may be cropped, so if this is the case, to sample the points correctly, we need to make
    the points relative to the top left corner of the cropped image.

    Args:
        yxz: integer points to apply the warp to. (n_points x 3 in yxz coords) (UNSHIFTED)
        flow: flow to apply to the points. (3 x cube_size_y x cube_size_x x cube_size_z) (SHIFTED)
        top_left: the top left corner of the cube in the flow_image. (3 in yxz coords) Default: [0, 0, 0]

    Returns:
        yxz_flow: (float) new points. (n_points x 3 in yxz coords)
    """
    # First, make yxz coordinates relative to the top left corner of the flow image, so that we can sample the shifts
    yxz_relative = yxz - top_left
    y_indices_rel, x_indices_rel, z_indices_rel = yxz_relative.T
    # sample the shifts relative to the top left corner of the flow image
    yxz_shifts = np.array([flow[i, y_indices_rel, x_indices_rel, z_indices_rel] for i in range(3)]).astype(np.float32).T
    # if original coords are torch, make the shifts torch
    if type(yxz) is torch.Tensor:
        yxz_shifts = torch.tensor(yxz_shifts)
    # apply the shifts to the original points
    yxz_flow = yxz + yxz_shifts
    return yxz_flow


def apply_affine(yxz: torch.Tensor, affine: torch.Tensor) -> torch.Tensor:
    """
    Transform the coordinates yxz based on the affine transform alone. E.g. to find coordinates of spots on the same
    tile but on a different round and channel.

    Args:
        - yxz (`(n_points x 3) tensor[float or int]`): y, x, and z coordinates positions to affine transform.
        - affine (`(4 x 3) tensor[float]`): affine transform to apply.

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


def remove_background(spot_colours: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Removes background from spot colours
    Args:
        spot_colours: 'float [n_spots x n_rounds x n_channels_use]' spot colours to remove background from.
    Returns:
        'spot_colours: [n_spots x n_rounds x n_channels_use]' spot colours with background removed.
        background_noise: [n_spots x n_channels_use]' background noise for each spot and channel.
    """
    n_spots = spot_colours.shape[0]
    background_noise = np.percentile(spot_colours, 25, axis=1)
    # Loop through all channels and remove the background from each channel.
    for c in tqdm.tqdm(range(spot_colours.shape[2])):
        background_code = np.zeros(spot_colours[0].shape)
        background_code[:, c] = 1
        # Remove the component of the background from the spot colour for each spot
        spot_colours -= background_noise[:, c][:, None, None] * np.repeat(background_code[None], n_spots, axis=0)

    return spot_colours, background_noise


def get_spot_colours_new_safe(
    nbp_basic_info, yxz: Optional[Union[np.ndarray, torch.Tensor]] = None, *args, **kwargs
) -> np.ndarray:
    """
    A wrapper function for get_spot_colours_new below. See that function for the arguments/details. This function runs
    get_spot_colours_new through multiple calls to avoid memory crashing on large images.

    Args:
        - nbp_basic_info (NotebookPage): `basic_info` notebook page.
        - yxz (`(n_points x 3) ndarray or tensor`, optional): positions to gather. Default: the entire tile.
        - args (tuple): positional arguments.
        - kwargs (dict[str, any]): keyword arguments.
    """
    assert type(yxz) is np.ndarray or type(yxz) is torch.Tensor or yxz is None
    tile_shape = (nbp_basic_info.tile_sz, nbp_basic_info.tile_sz, len(nbp_basic_info.use_z))
    n_channels_use, n_rounds_use = len(nbp_basic_info.use_channels), len(nbp_basic_info.use_rounds)
    if yxz is None:
        yxz = [np.linspace(0, tile_shape[i] - 1, tile_shape[i]) for i in range(3)]
        yxz = np.array(np.meshgrid(*yxz, indexing="ij")).astype(np.int32).T.reshape((-1, 3), order="F")
    assert yxz.ndim == 2
    assert yxz.shape[1] == 3
    batch_size = maths.floor(utils.system.get_available_memory() * 1.3e7 / (n_channels_use * n_rounds_use))
    n_batches = maths.ceil(yxz.shape[0] / batch_size)
    quiet = n_batches == 1
    colours = None
    for i in tqdm.trange(n_batches, desc=f"Getting spot colours", unit="batch", disable=quiet):
        index_min, index_max = i * batch_size, min((i + 1) * batch_size, yxz.shape[0])
        i_colours = get_spot_colours_new(yxz=yxz[index_min:index_max], *args, **kwargs)
        if colours is None:
            colours = i_colours
        else:
            colours = np.append(colours, i_colours, axis=0)
        del i_colours
    return colours


def get_spot_colours_new(
    yxz: Union[np.ndarray, torch.Tensor],
    image: Union[np.ndarray, zarr.Array],
    flow: Union[np.ndarray, zarr.Array],
    affine: Union[np.ndarray, torch.Tensor],
    tile: int,
    use_rounds: list[int],
    use_channels: Optional[list[int]] = None,
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
        - yxz (`(n_points x 3) ndarray or tensor`): positions to gather.
        - image (`(n_tiles x n_rounds x n_channels x im_y x im_x x im_z) ndarray or zarray`): image to gather from.
        - flow (`(n_tiles x n_rounds x 3 x im_y x im_x x im_z) ndarray or zarray`): optical flow shifts.
        - affine (`(n_tiles x n_rounds x n_channels x 4 x 3) ndarray or tensor`): affine transform.
        - tile (int): tile index.
        - use_rounds (list of ints): the round indices to use. These rounds must be sequencing rounds.
        - use_channels (list of ints, optional): channel indices to use. Default: all channels.
        - output_dtype (np.dtype, optional): the returned spot colour datatype. Default: float32.
        - out_of_bounds_value (any): what to value to set for out of bound spot colours. Default: np.nan.

    Returns:
        `(n_points x n_rounds x n_channels_use) ndarray[output_dtype]` colours: gathered image colours.
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
    assert all([r >= 0 and r < flow.shape[1] for r in use_rounds]), f"Cannot use round index {r}"
    assert len(use_channels) > 0
    assert all([type(c) is int for c in use_channels])
    assert all([c >= 0 and c < image.shape[2] for c in use_channels])

    # Prepare variables.
    tile_shape = tuple(image.shape[3:])
    # Pytorch tensors are used throughout and cast to float32 while computing.
    yxz_torch = yxz
    if type(yxz_torch) is np.ndarray:
        yxz_torch = torch.tensor(yxz_torch)
    yxz_torch = yxz_torch.detach().clone().float()
    affine_torch = affine
    if type(affine_torch) is np.ndarray:
        affine_torch = torch.tensor(affine_torch)
    affine_torch = affine_torch.detach().clone().float()

    colours = np.zeros((yxz.shape[0], len(use_rounds), len(use_channels)), output_dtype)
    for r in use_rounds:
        # First, apply round r optical flow to the given coordinates.
        r_yxz = apply_flow_new(yxz_torch, flow, tile, r)
        for c_index, c in enumerate(use_channels):
            # For each channel, apply the affine transform to the optical flow shifted yxz coordinates.
            c_yxz = apply_affine(r_yxz, affine_torch[tile, r, c])
            # Only gather image data that is required by the yxz coordinates.
            c_yxz_min, c_yxz_max = c_yxz.min(0).values, c_yxz.max(0).values
            # Pad the gathered data by one pixel for interpolation.
            c_yxz_min = (c_yxz_min - 1).clamp(min=0).int().tolist()
            c_yxz_max = (c_yxz_max + 1).clamp(max=torch.tensor(tile_shape)).int().tolist()
            image_trc = np.zeros(tile_shape, np.float32)
            image_trc[c_yxz_min[0] : c_yxz_max[0], c_yxz_min[1] : c_yxz_max[1], c_yxz_min[2] : c_yxz_max[2]] = image[
                tile, r, c, c_yxz_min[0] : c_yxz_max[0], c_yxz_min[1] : c_yxz_max[1], c_yxz_min[2] : c_yxz_max[2]
            ]
            image_trc = torch.tensor(image_trc)

            c_yxz_grid = convert_coords_to_torch_grid(c_yxz, tile_shape)
            del c_yxz
            # Input has shape (1, 1, image.shape[0], image.shape[1], image.shape[2]).
            # Grid has shape (1, 1, 1, n_points, 3)
            # Result has shape (1, 1, 1, 1, n_points).
            image_trc = image_trc[None, None]
            colours_trc = torch.nn.functional.grid_sample(image_trc, c_yxz_grid[None, None, None], align_corners=True)
            colours_trc = colours_trc[0, 0, 0, 0]

            # Set out of bound coordinates to out_of_bounds_value.
            is_out_of_bounds = (c_yxz_grid < -1) | (c_yxz_grid > +1)
            colours_trc[is_out_of_bounds.any(1)] = out_of_bounds_value

            colours[:, r, c_index] = colours_trc.numpy().astype(output_dtype)
    return colours


def get_spot_colours(
    image: Union[np.ndarray, zarr.Array],
    flow: Union[np.ndarray, zarr.Array],
    affine_correction: Union[np.ndarray, torch.Tensor],
    yxz_base: Union[np.ndarray, torch.Tensor],
    tile: int,
    output_dtype: torch.dtype = torch.float32,
    fill_value: float = float("nan"),
    use_channels: List[int] = None,
) -> np.ndarray:
    """
    Takes some spots found on the reference round, and computes the corresponding spot intensity
    in specified imaging rounds/channels. The algorithm goes as follows:
    Loop over rounds:
        Apply flow: yxz_flow = yxz_base + flow
        Loop over channels:
            Apply ICP correction: yxz_flow_and_affine = yxz_flow @ icp_correction
            Interpolate spot intensities: spot_colours[:, r, c] = grid_sample(image[r, c], yxz_flow_and_affine)
    The code has been profiled, and any time-consuming operations have been passed to PyTorch and can be run on a GPU.

    - Note: Although yxz is a list of n_spots x 3 and does not need to be made up of intervals, we load the bounding
    box of the image to speed up the loading process and help interpolate points. This means that accessing many random
    points will be slower than accessing a subset of the image at once.

    Args:
        - image: 'float16 memmap [n_tiles x n_rounds x n_channels x im_y x im_x x im_z]' unregistered image data.
        - flow: 'float16 memmap [n_tiles x n_rounds x 3 x im_y x im_x x im_z]' flow data.
        - affine_correction: 'float32 [n_tiles x n_rounds x n_channels x 4 x 3]' affine correction data
        - yxz_base: 'int [n_spots x 3]' spot coordinates, or tuple
        - tile: 'int' tile index to run on.
        - output_dtype: 'dtype' dtype of the output spot colours.
        - fill_value: 'float' value to fill in for out of bounds spots.
        - use_channels: 'List[int]' channels to run on.

    Returns:
        `(n_spots x n_rounds x n_channels) ndarray[output_dtype]' spot_colours: spot colours.
    """
    # Deal with default values.
    if use_channels is None:
        use_channels = list(range(image.shape[2]))
    if type(affine_correction) is np.ndarray:
        affine_correction = torch.tensor(affine_correction, dtype=torch.float32)
    if type(yxz_base) is np.ndarray:
        yxz_base = torch.tensor(yxz_base, dtype=torch.float32)
    n_tiles, n_rounds, n_channels = image.shape[0], flow.shape[1], image.shape[2]
    assert affine_correction.shape[1:] == (
        n_rounds,
        n_channels,
        4,
        3,
    ), f"Expected shape {(n_tiles, n_rounds, n_channels, 4, 3)}, got {affine_correction.shape}"

    # initialize variables
    n_spots, n_use_rounds, n_use_channels = yxz_base.shape[0], flow.shape[1], len(use_channels)
    use_rounds = list(np.arange(n_use_rounds))
    tile_size = torch.tensor(image.shape[3:])
    pad_size = torch.tensor([100, 100, 5])
    spot_colours = torch.full((n_spots, n_use_rounds, n_use_channels), fill_value, dtype=output_dtype)

    # load slices of the images rather than sampling coordinates directly.
    yxz_min, yxz_max = yxz_base.min(axis=0).values.int(), yxz_base.max(axis=0).values.int()
    # pad to ensure that we are able to interpolate the points even if the shifts are large.
    yxz_min, yxz_max = (
        torch.maximum(yxz_min - pad_size, torch.tensor([0, 0, 0])),
        torch.minimum(yxz_max + pad_size, tile_size),
    )
    cube_size = yxz_max - yxz_min
    # load the sliced images for each round and channel (from yxz_min to yxz_max)
    image = np.array(
        [
            [
                image[tile, r, c, yxz_min[0] : yxz_max[0], yxz_min[1] : yxz_max[1], yxz_min[2] : yxz_max[2]]
                for c in use_channels
            ]
            for r in use_rounds
        ]
    )
    flow = np.array(
        [
            flow[tile, r, :, yxz_min[0] : yxz_max[0], yxz_min[1] : yxz_max[1], yxz_min[2] : yxz_max[2]]
            for r in use_rounds
        ]
    )
    # convert to torch tensor
    image = torch.tensor(image, dtype=torch.float32)

    # begin the loop over rounds and channels
    for r in tqdm.tqdm(range(n_use_rounds), total=n_use_rounds, desc="Round Loop"):
        # initialize the coordinates for the round
        yxz_round_r = torch.zeros((n_use_channels, n_spots, 3), dtype=torch.float32)
        # the flow is the same for all channels in the same round. Therefore, only need to read it once.
        # Since flow is cropped, pass the top left corner to this function, so it reads the coords relative to yxz_min.
        yxz_flow = apply_flow(yxz=yxz_base.int(), flow=flow[r], top_left=yxz_min)
        for i, c in enumerate(use_channels):
            # apply the affine transform to the spots
            yxz_round_r[i] = apply_affine(yxz=yxz_flow, affine=affine_correction[tile, r, c])
            # Since image has top left corner yxz_min, must make the sampling points relative to this.
            yxz_round_r[i] -= yxz_min
            # convert tile coordinates [0, cube_size] to coordinates [0, 2]
            yxz_round_r[i] = 2 * yxz_round_r[i] / (cube_size - 1)
            # convert coordinates [0, 2] to coordinates [-1, 1]
            yxz_round_r[i] -= 1
        zxy_round_r = yxz_round_r[:, :, [2, 1, 0]]

        # grid_sample expects image to be input as [N, M, D, H, W] where
        # N = batch size: We set this to n_use_channels,
        # M = number of images to be sampled at the same grid locations: We set this to 1,
        # D = depth, H = height, W = width: We set these to n_y, n_x and n_z respectively.

        # grid_sample expects grid to be input as [N, D', H', W', 3] where
        # N = batch size: We set this to n_use_channels,
        # D' = depth out, H' = height out, W' = width out: We set these to n_spots, 1, 1
        # 3 = 3D coordinates of the points to sample (NOTE: These must be in the order z, x, y).
        # This is NOT included in the documentation, but is inferred from the source code.
        round_r_colours = torch.nn.functional.grid_sample(
            input=image[r, :, None, :, :, :],
            grid=zxy_round_r[:, :, None, None, :],
            mode="bilinear",
            align_corners=True,
            padding_mode="border",
        )

        # grid_sample gives output as [N, M, D', H', W'] as defined above.
        round_r_colours = round_r_colours[:, 0, :, 0, 0]
        spot_colours[:, r, :] = round_r_colours.T

        # Any out of bound grid sample retrievals are set to fill_value.
        is_out_of_bounds = torch.logical_or(zxy_round_r < -1, zxy_round_r > 1).any(dim=2).T
        spot_colours[:, r, :][is_out_of_bounds] = fill_value

    return spot_colours.numpy()
