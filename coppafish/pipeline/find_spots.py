import math as maths
import os

import numpy as np
import tqdm
import zarr

from .. import log
from ..find_spots import base, detect
from ..setup.config_section import ConfigSection
from ..setup.notebook_page import NotebookPage
from ..utils import indexing


def find_spots(
    config: ConfigSection,
    nbp_basic: NotebookPage,
    nbp_file: NotebookPage,
    nbp_filter: NotebookPage,
) -> NotebookPage:
    """
    Create a point cloud for each filtered image. Results are saved in a new `find_spots` notebook page.

    See `'find_spots'` section in coppafish/setup/notebook_page.py file for description of the variables in the page.

    Args:
        - config (dict): dictionary obtained from `'find_spots'` section of config file.
        - nbp_basic (NotebookPage): `basic_info` notebook page.
        - nbp_file (NotebookPage): `file_names` notebook page.
        - nbp_filter (NotebookPage): `filter` notebook page.

    Returns:
        (NotebookPage) nbp_find_spots: `find_spots` notebook page.
    """
    log.info("Find spots started")

    # Phase 0: Initialisation
    nbp = NotebookPage("find_spots", {config.name: config.to_dict()})
    auto_thresh_multiplier = config["auto_thresh_multiplier"]
    if auto_thresh_multiplier <= 0:
        raise ValueError("The auto_thresh_multiplier in 'find_spots' config must be positive")
    n_z = np.max([1, nbp_basic.is_3d * nbp_basic.nz])
    max_spots = maths.floor(config["max_spots_percent"] * nbp_basic.tile_sz**2 / 100)
    INVALID_AUTO_THRESH = -1
    auto_thresh = np.full(
        (nbp_basic.n_tiles, nbp_basic.n_rounds + nbp_basic.n_extra_rounds, nbp_basic.n_channels),
        fill_value=INVALID_AUTO_THRESH,
        dtype=np.float32,
    )
    group_path = os.path.join(nbp_file.output_dir, "spot_yxz.zgroup")
    spot_yxz = zarr.group(store=group_path, overwrite=True, zarr_version=2)
    spot_no = np.zeros(
        (nbp_basic.n_tiles, nbp_basic.n_rounds + nbp_basic.n_extra_rounds, nbp_basic.n_channels), dtype=np.int32
    )

    # Define use_indices as a [n_tiles x n_rounds x n_channels] boolean array where use_indices[t, r, c] is True if
    # we want to find spots on said tile `t`, round `r`, channel `c`.
    use_indices = np.zeros(
        (nbp_basic.n_tiles, nbp_basic.n_rounds + nbp_basic.use_anchor, nbp_basic.n_channels), dtype=bool
    )
    for t, r, c in indexing.create(
        nbp_basic,
        include_anchor_round=True,
        include_anchor_channel=True,
        include_bad_trc=True,
    ):
        use_indices[t, r, c] = True

    # Phase 2: Detect spots on uncompleted tiles, rounds and channels
    pbar = tqdm.tqdm(total=use_indices.sum(), desc="Finding spots", unit="image")
    for t, r, c in np.argwhere(use_indices).tolist():
        pbar.set_postfix_str(f"{t=}, {r=}, {c=}")
        image_trc = nbp_filter.images[t, r, c].astype(np.float32)

        # Compute the image's auto threshold to detect spots.
        mid_z = image_trc.shape[2] // 2
        auto_thresh[t, r, c] = auto_thresh_multiplier * np.median(np.abs(image_trc[..., mid_z]))

        if auto_thresh[t, r, c] <= 0:
            auto_thresh[t, r, c] = auto_thresh_multiplier

        local_yxz, spot_intensity = detect.detect_spots(
            image_trc,
            auto_thresh[t, r, c].item(),
            remove_duplicates=True,
            radius_xy=config["radius_xy"],
            radius_z=config["radius_z"],
        )
        if r != nbp_basic.anchor_round:
            # On imaging rounds, only keep the highest intensity spots on each z plane.
            local_yxz = base.filter_intense_spots(local_yxz, spot_intensity, n_z, max_spots)

        spot_no[t, r, c] = local_yxz.shape[0]
        log.debug(f"Found {spot_no[t, r, c]} spots on {t=}, {r=}, {c=}")
        # Save results to zarr group.
        trc_yxz = spot_yxz.zeros(f"t{t}r{r}c{c}", chunks=local_yxz.size == 0, shape=local_yxz.shape, dtype=np.int16)
        trc_yxz[:] = local_yxz
        del local_yxz, spot_intensity, trc_yxz
        pbar.update()
    pbar.close()

    # Phase 3: Save results to notebook page
    nbp.auto_thresh = auto_thresh
    nbp.spot_yxz = spot_yxz
    nbp.spot_no = spot_no
    log.info("Find spots complete")

    return nbp
