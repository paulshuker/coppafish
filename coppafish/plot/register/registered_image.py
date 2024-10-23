from typing import Optional

import napari
import numpy as np

from ... import spot_colours
from ...setup.notebook import Notebook


def view_registered_images(
    nb: Notebook,
    tile: Optional[int] = None,
    subset_limits_yxz: Optional[tuple[tuple[int], tuple[int], tuple[int]]] = None,
    show: bool = True,
) -> None:
    """
    View all registered images for a subset of a tile.

    Args:
        - nb (Notebook): the notebook with `register` completed.
        - tile (int, optional): tile index. Default: `nb.basic_info.use_tiles[0]`.
        - subset_limits_yxz (tuple of three tuples of two ints, optional): a tuple containing the minimum and maximum
            (exclusive) coordinates for the given dimension. There are three tuples for y, x, and z axes respectively.
            For example, a 14x10x11 subset could be `subset_limits_yxz = ((2, 16), (0, 10), (5, 16))`. All coordinates
            are relative to the anchor round/channel. The limits are cropped if they go outside of the tile's bounds.
            Default: 400x400x5 subset with bottom-left corner at the origin.
        - show (bool, optional): show the viewer after creating, used for unit testing purposes only. Default: true.
    """
    if type(nb) is not Notebook:
        raise TypeError(f"nb must be a notebook")
    if type(tile) is not int and tile is not None:
        raise TypeError(f"tile must be an int")
    if type(subset_limits_yxz) is not tuple and subset_limits_yxz is not None:
        raise TypeError(f"tile must be an int")
    if tile is None:
        tile: int = nb.basic_info.use_tiles[0]
    if subset_limits_yxz is None:
        subset_limits_yxz = ((0, 400), (0, 400), (0, 5))
    if tile not in nb.basic_info.use_tiles:
        raise ValueError(f"tile must be one of {nb.basic_info.use_tiles}")
    if (
        type(subset_limits_yxz) is not tuple
        or not all([type(limit) is tuple for limit in subset_limits_yxz])
        or not all([all([type(limit_val) is int for limit_val in limit]) for limit in subset_limits_yxz])
    ):
        raise TypeError("subset_limits_yxz must be a tuple containing tuples containing ints")
    tile_shape: tuple[int] = (nb.basic_info.tile_sz, nb.basic_info.tile_sz, len(nb.basic_info.use_z))
    new_subset_limits_yxz = tuple()
    for i in range(3):
        # Fix subset limits to within the images bounds.
        new_min = max(subset_limits_yxz[i][0], 0)
        new_max = min(subset_limits_yxz[i][1], tile_shape[i])
        new_subset_limits_yxz += ((new_min, new_max),)
    subset_limits_yxz = new_subset_limits_yxz
    subset_shape_yxz = tuple([limit[1] - limit[0] for limit in subset_limits_yxz])

    # Gather anchor image.
    anchor_image = nb.filter.images[
        tile,
        nb.basic_info.anchor_round,
        nb.basic_info.anchor_channel,
        subset_limits_yxz[0][0] : subset_limits_yxz[0][1],
        subset_limits_yxz[1][0] : subset_limits_yxz[1][1],
        subset_limits_yxz[2][0] : subset_limits_yxz[2][1],
    ]
    # (im_y, im_x, im_z) -> (im_z, im_y, im_x).
    anchor_image = anchor_image.transpose((2, 0, 1))

    # Gather sequencing and DAPI images.
    use_channels = [nb.basic_info.dapi_channel] + nb.basic_info.use_channels
    yxz = [np.linspace(subset_limits_yxz[i][0], subset_limits_yxz[i][1] - 1, subset_shape_yxz[i]) for i in range(3)]
    yxz = np.array(np.meshgrid(*yxz, indexing="ij")).astype(np.int16).reshape((3, -1), order="F").T
    icp_correction = nb.register.icp_correction
    icp_correction[tile, :, nb.basic_info.dapi_channel] = icp_correction[tile, :, nb.basic_info.anchor_channel]

    images = spot_colours.base.get_spot_colours_new_safe(
        nb.basic_info,
        yxz=yxz,
        image=nb.filter.images,
        flow=nb.register.flow,
        affine=icp_correction,
        tile=tile,
        use_rounds=nb.basic_info.use_rounds,
        use_channels=use_channels,
        output_dtype=np.float32,
        out_of_bounds_value=np.nan,
    )
    images = images.reshape(subset_shape_yxz + images.shape[1:], order="F")
    # (im_y, im_x, im_z, n_rounds, n_channels_use) -> (n_rounds, n_channels_use, im_z, im_y, im_x).
    images = images.transpose((3, 4, 2, 0, 1))

    if not show:
        return

    viewer = napari.Viewer(title=f"Registered Images, Tile {tile}")
    viewer.add_image(anchor_image, name="r=anchor c=anchor", visible=True, rgb=False)

    for r_index, round in enumerate(nb.basic_info.use_rounds):
        for c_index, channel in enumerate(use_channels):
            r_str = str(round)
            if round == nb.basic_info.anchor_round:
                r_str = "anchor"
            c_str = str(channel)
            if channel == nb.basic_info.anchor_channel:
                c_str = "anchor"
            if channel == nb.basic_info.dapi_channel:
                c_str = "dapi"
            viewer.add_image(images[r_index, c_index], name=f"r={r_str} c={c_str}", visible=False, rgb=False)

    napari.run()
