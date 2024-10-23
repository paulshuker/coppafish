import os

import numpy as np
import zarr

from .. import log
from ..call_spots import base as call_spots_base
from ..setup.notebook_page import NotebookPage
from ..spot_colours import base as spot_colours_base


def get_reference_spots(
    nbp_basic: NotebookPage,
    nbp_file: NotebookPage,
    nbp_filter: NotebookPage,
    nbp_find_spots: NotebookPage,
    nbp_register: NotebookPage,
    nbp_stitch: NotebookPage,
) -> NotebookPage:
    """
    This takes each spot found on the reference round/channel and computes the corresponding intensity
    in each of the imaging rounds/channels.

    Args:
        nbp_basic: `basic_info` notebook page.
        nbp_file: `file_names` notebook page.
        nbp_filter: `filter` notebook page.
        nbp_find_spots: 'find_spots' notebook page.
        nbp_register: `register` notebook page.
        nbp_stitch: `stitch` notebook page.

    Returns:
        `NotebookPage[ref_spots]` - Page containing intensity of each reference spot on each imaging round/channel.
    """
    log.info("Get ref spots started")
    # Create a notebook page for ref_spots which stores information like local coords, tile_no of each spot and more.
    nbp = NotebookPage("ref_spots")
    # The code is going to loop through all tiles, as we expect some anchor spots on each tile but r and c should stay
    # fixed as the value of the reference round and reference channel
    tile_origin = nbp_stitch.tile_origin
    r = nbp_basic.anchor_round
    c = nbp_basic.anchor_channel
    use_tiles, use_rounds, use_channels = np.array(nbp_basic.use_tiles), nbp_basic.use_rounds, nbp_basic.use_channels

    # all means all spots found on the reference round / channel
    all_local_yxz = np.zeros((0, 3), dtype=np.int16)
    all_local_tile = np.zeros(0, dtype=np.int16)

    # Loop through tiles and record the local_yxz spots on this tile.
    # We then append this to all_local_yxz and all_local_tile arrays.
    for t in nbp_basic.use_tiles:
        t_local_yxz = nbp_find_spots.spot_yxz[f"t{t}r{r}c{c}"][:]
        if np.shape(t_local_yxz)[0] == 0:
            continue
        all_local_yxz = np.append(all_local_yxz, t_local_yxz, axis=0)
        all_local_tile = np.append(all_local_tile, np.full(t_local_yxz.shape[0], t, dtype=np.int16))

    # find duplicate spots as those detected on a tile which is not tile centre they are closest to
    not_duplicate = call_spots_base.get_non_duplicate(
        tile_origin, list(nbp_basic.use_tiles), nbp_basic.tile_centre, all_local_yxz, all_local_tile
    )

    # nd means all spots that are not duplicate
    nd_local_yxz = all_local_yxz[not_duplicate]
    nd_local_tile = all_local_tile[not_duplicate]

    # Only save used rounds/channels initially
    n_use_rounds, n_use_channels, n_use_tiles = len(use_rounds), len(use_channels), len(use_tiles)
    spot_colours = np.zeros((0, n_use_rounds, n_use_channels), dtype=np.float32)
    local_yxz = np.zeros((0, 3), dtype=np.int16)
    tile = np.zeros(0, dtype=np.int16)
    log.info("Reading in spot_colours for ref_round spots")
    for t in nbp_basic.use_tiles:
        in_tile = nd_local_tile == t
        if np.sum(in_tile) == 0:
            continue
        log.info(f"Tile {np.where(use_tiles==t)[0][0]+1}/{n_use_tiles}")
        log.debug(f"Tile {t} has {nd_local_yxz[in_tile].shape[0]} reference spots")
        colours = spot_colours_base.get_spot_colours_new_safe(
            nbp_basic,
            image=nbp_filter.images,
            flow=nbp_register.flow,
            affine=nbp_register.icp_correction,
            tile=t,
            yxz=nd_local_yxz[in_tile],
            use_rounds=use_rounds,
            use_channels=use_channels,
        )
        valid = ~(np.isnan(colours).any(1).any(1))
        # Zeros in all channels for any round is an invalid spot.
        valid &= ~(np.isclose(colours, 0).all(2).any(1))
        log.debug(f"Valid ref pixel colours: {valid.sum()} out of {valid.size} for tile {t}")
        spot_colours = np.append(spot_colours, colours[valid], axis=0)
        local_yxz = np.append(local_yxz, nd_local_yxz[in_tile][valid], axis=0)
        tile = np.append(tile, np.ones(valid.sum(), dtype=np.int16) * t)

    # Convert the numpy results to zarrays for saving.
    kwargs = dict(chunks=False, zarr_version=2)
    local_yxz = zarr.array(local_yxz, store=os.path.join(nbp_file.output_dir, "local_yxz.zarray"), **kwargs)
    tile = zarr.array(tile, store=os.path.join(nbp_file.output_dir, "tile.zarray"), **kwargs)
    spot_colours = zarr.array(spot_colours, store=os.path.join(nbp_file.output_dir, "spot_colours.zarray"), **kwargs)

    # save spot info to notebook
    nbp.local_yxz = local_yxz
    nbp.tile = tile
    nbp.colours = spot_colours
    log.info("Get ref spots complete")

    return nbp
