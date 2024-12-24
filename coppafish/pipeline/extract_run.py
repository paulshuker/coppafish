import os

import numpy as np
from tqdm import tqdm

from .. import log
from ..extract.raw_jobs import JobsReader
from ..extract.raw_nd2 import Nd2Reader
from ..extract.raw_numpy import NumpyReader
from ..extract.raw_reader import RawReader
from ..setup.config_section import ConfigSection
from ..setup.notebook_page import NotebookPage
from ..utils import indexing, system, tiles_io


def run_extract(config: ConfigSection, nbp_file: NotebookPage, nbp_basic: NotebookPage) -> NotebookPage:
    """
    This reads in images from the raw `nd2` files, filters them and then saves them as zarr array files in the tile
    directory.

    Args:
        config (ConfigSection): dictionary obtained from 'extract' section of config file.
        nbp_file (NotebookPage): 'file_names' notebook page.
        nbp_basic (NotebookPage): 'basic_info' notebook page.

    Returns:
        (NotebookPage) nbp_extract: `extract` notebook page.

    Notes:
        - See `'extract'` sections in `coppafish/setup/notebook_page.py` file for description of the variables in each page.
    """
    nbp = NotebookPage("extract", {config.name: config.to_dict()})
    nbp.num_rotations = config["num_rotations"]

    log.debug("Extraction started")

    raw_extension_readers: dict[str, RawReader] = {
        ".nd2": Nd2Reader(),
        ".npy": NumpyReader(),
        "JOBS": JobsReader(),
    }

    if not os.path.isdir(nbp_file.extract_dir):
        os.mkdir(nbp_file.extract_dir)
    # Save the earliest used coppafish version to extract inside of the extract directory.
    version_path = os.path.join(nbp_file.extract_dir, ".version")
    if os.path.isfile(version_path):
        with open(version_path, "r") as file:
            extract_version = file.readline()
        if extract_version != system.get_software_version():
            log.info(f"Using pre-existing extract results from version {extract_version}")
    else:
        with open(version_path, "w") as file:
            file.write(system.get_software_version())

    indices = indexing.create(
        nbp_basic,
        include_anchor_round=True,
        include_anchor_channel=True,
        include_dapi_seq=True,
        include_dapi_anchor=True,
    )
    indices_t = indexing.unique(indices, axis=0)
    indices_r = indexing.unique(indices, axis=1)
    with tqdm(
        total=len(indices_t) * len(indices_r),
        desc=f"Extracting raw {nbp_file.raw_extension} files",
    ) as pbar:
        for t, _, _ in indices_t:
            for _, r, _ in indices_r:
                pbar.set_postfix({"tile": t, "round": r})

                channels = list(indexing.find_channels_for(indices, tile=t, round=r))
                file_paths = [nbp_file.tile_unfiltered[t][r][c] for c in channels]
                files_exist = [tiles_io.image_exists(file_path) for file_path in file_paths]

                if all(files_exist):
                    pbar.update()
                    continue

                # Has shape (n_channels, im_y, im_x, im_z).
                channel_images = raw_extension_readers[nbp_file.raw_extension].read(nbp_basic, nbp_file, t, r, channels)
                channel_images = channel_images[:, :, :, nbp_basic.use_z]

                for im, c, file_path, file_exists in zip(channel_images, channels, file_paths, files_exist):
                    if file_exists:
                        continue
                    im = im.astype(np.uint16, casting="safe")
                    im = np.rot90(im, k=config["num_rotations"], axes=(0, 1))
                    z_plane_means = im.mean((0, 1))
                    if (z_plane_means < config["z_plane_mean_warning"]).any():
                        log.warn(
                            f"Raw image {t=}, {r=}, {c=} has dim z plane(s) at "
                            + f"{np.where(z_plane_means < config['z_plane_mean_warning'])[0].tolist()}. You may "
                            + f"wish to remove the affected image by setting `bad_trc = {t}, {r}, {c}, ...` in "
                            + "the basic_info config then re-run the pipeline with an empty output directory."
                        )
                    tiles_io._save_image(im, file_path)
                    del im
                pbar.update()
    log.debug("Extraction complete")
    return nbp
