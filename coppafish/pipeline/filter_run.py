import math as maths
import os
import pickle
from typing import Tuple

import numpy as np
import zarr
from tqdm import tqdm

from .. import log
from ..filter import base as filter_base
from ..filter import deconvolution
from ..setup.config_section import ConfigSection
from ..setup.notebook_page import NotebookPage
from ..utils import indexing, system, tiles_io


def run_filter(
    config: ConfigSection, nbp_file: NotebookPage, nbp_basic: NotebookPage
) -> Tuple[NotebookPage, NotebookPage]:
    """
    Read in extracted raw images, filter them, then re-save in a different location.

    Args:
        config (ConfigSection): config section obtained from 'filter' section of config file.
        nbp_file (NotebookPage): 'file_names' notebook page.
        nbp_basic (NotebookPage): 'basic_info' notebook page.

    Returns tuple containing:
        - NotebookPage: 'filter' notebook page.
        - NotebookPage: 'filter_debug' notebook page.

    Notes:
        - See `'filter'` and `'filter_debug'` sections of `notebook_page.py` file for description of variables.
    """
    filter_config = {config.name: config.to_dict(), "version": {"0": system.get_software_version()}}
    nbp = NotebookPage("filter", filter_config)
    nbp_debug = NotebookPage("filter_debug", filter_config)

    log.debug("Filter started")

    # Remember the config values during a run.
    last_config = filter_config.copy()
    config_path = os.path.join(nbp_file.output_dir, "filter_last_config.pkl")
    if os.path.isfile(config_path):
        with open(config_path, "rb") as config_file:
            last_config = pickle.load(config_file)
    assert type(last_config) is dict
    config_unchanged = filter_config == last_config
    with open(config_path, "wb") as config_file:
        pickle.dump(config, config_file)

    indices = indexing.create(
        nbp_basic,
        include_anchor_round=True,
        include_anchor_channel=True,
        include_dapi_seq=True,
        include_dapi_anchor=True,
        include_bad_trc=False,
    )

    max_ind = np.array(indices).max(0).tolist()
    shape = (max_ind[0] + 1, max_ind[1] + 1, max_ind[2] + 1, nbp_basic.tile_sz, nbp_basic.tile_sz, len(nbp_basic.use_z))
    # Chunks are made into thin rods along the y direction as this is how the images are later gathered in OMP.
    x_length = max(maths.floor(1e6 / (shape[1] * shape[3] * 2)), 1)
    z_length = 1
    while x_length > nbp_basic.tile_sz:
        x_length -= nbp_basic.tile_sz
        z_length += 1
    z_length = min(z_length, shape[5])
    chunks = (1, None, 1, None, x_length, z_length)
    images = zarr.open_array(
        os.path.join(nbp_file.output_dir, "filter_images.zarr"),
        "a",
        shape=shape,
        chunks=chunks,
        fill_value=np.nan,
        zarr_version=2,
        dtype=np.float16,
    )
    if "completed_indices" not in images.attrs:
        images.attrs["completed_indices"] = []
    # Bad trc images are filled with zeros.
    for t, r, c in nbp_basic.bad_trc:
        images[t, r, c] = 0
        images.attrs["completed_indices"] = images.attrs["completed_indices"] + [(t, r, c)]

    wiener_filter = None
    if not os.path.isfile(nbp_file.psf):
        raise FileNotFoundError(f"Could not find the PSF at location {nbp_file.psf}")

    # Put z to last index
    psf = np.load(nbp_file.psf)["arr_0"].astype(np.float32).swapaxes(0, 2)
    if np.max(psf.shape[:2]) < psf.shape[2]:
        log.warn(f"The given PSF has a strange shape of yxz = {psf.shape}")
    # Normalise psf so the min is 0 and the max is 1.
    psf = psf - psf.min()
    psf = psf / psf.max()
    pad_im_shape = (
        np.array([nbp_basic.tile_sz, nbp_basic.tile_sz, len(nbp_basic.use_z)])
        + np.array(config["wiener_pad_shape"]) * 2
    )
    wiener_filter = filter_base.get_wiener_filter(psf, pad_im_shape, config["wiener_constant"])
    nbp_debug.psf = psf

    with tqdm(total=len(indices), desc="Filtering extract images") as pbar:
        for t, r, c in indices:
            if config_unchanged and (t, r, c) in images.attrs["completed_indices"]:
                # Already saved filtered images are not re-filtered.
                pbar.update()
                continue
            file_path_raw = nbp_file.tile_unfiltered[t][r][c]
            raw_image_exists = tiles_io.image_exists(file_path_raw)
            pbar.set_postfix({"round": r, "tile": t, "channel": c})
            assert raw_image_exists, f"Raw, extracted file at\n\t{file_path_raw}\nnot found"

            # Get t, r, c image from raw files
            im_filtered = tiles_io._load_image(file_path_raw)[:]
            # Move to floating point before filtering.
            im_filtered = im_filtered.astype(np.float64)

            # All images are deconvolved, including the DAPI.
            im_filtered = deconvolution.wiener_deconvolve(
                im_filtered, config["wiener_pad_shape"], wiener_filter, config["force_cpu"]
            )
            im_filtered = im_filtered.astype(np.float16)
            images[t, r, c] = im_filtered
            del im_filtered
            images.attrs["completed_indices"] = images.attrs["completed_indices"] + [(t, r, c)]

            pbar.update()

    nbp.images = images
    os.remove(config_path)
    log.debug("Filter complete")

    return nbp, nbp_debug
