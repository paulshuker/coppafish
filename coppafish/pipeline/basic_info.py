import json
import os

import numpy as np

from .. import log
from ..extract import nd2
from ..setup import tile_details
from ..setup.config import Config
from ..setup.notebook_page import NotebookPage
from ..utils import base as utils_base


def set_basic_info(config: Config) -> NotebookPage:
    """
    Adds info from `'basic_info'` section of config file to notebook page.

    To `basic_info` page, the following is also added:
    `anchor_round`, `n_rounds`, `n_extra_rounds`, `n_tiles`, `n_channels`, `nz`, `tile_sz`, `tilepos_yx`,
    `tilepos_yx_nd2`, `pixel_size_xy`, `pixel_size_z`, `tile_centre`, `use_anchor`.

    See `'basic_info'` sections of `notebook_comments.json` file
    for description of the variables.

    Args:
        - `config` : `dict` - Config dictionary.
    Returns:
        - `NotebookPage[basic_info]` - Page contains information that is used at all stages of the pipeline.
    """
    # Break the page contents up into 2 types, contents that must be read in from the config and those that can
    # be computed from the metadata.
    config_file = config["file_names"]
    config_basic = config["basic_info"]

    # Initialize Notebook
    associated_configs = {config_basic.name: config_basic.to_dict(), config_file.name: config_file.to_dict()}
    nbp = NotebookPage("basic_info", associated_configs)

    # Stage 1: Compute metadata. This is done slightly differently in the 3 cases of different raw extensions
    raw_extension = nd2.get_raw_extension(config_file["input_dir"])
    all_files = []
    for root, _, filenames in os.walk(config_file["input_dir"]):
        for filename in filenames:
            all_files.append(os.path.join(root, filename))
    all_files.sort()
    if raw_extension == ".nd2":
        if config_file["round"] is None and config_file["anchor"] is None:
            log.error(ValueError("config_file['round'] or config_file['anchor'] should not both be left blank"))
        # load in metadata of nd2 file corresponding to first round
        # Allow for degenerate case when only anchor has been provided
        if config_file["round"] is not None:
            first_round_raw = os.path.join(config_file["input_dir"], config_file["round"][0])
        else:
            first_round_raw = os.path.join(config_file["input_dir"], config_file["anchor"])
        metadata = nd2.get_metadata(first_round_raw + raw_extension, config=config)

    elif raw_extension == ".npy":
        # Load in metadata as dictionary from a json file
        metadata_file = [file for file in all_files if file.endswith(".json")][0]
        if metadata_file is None:
            log.error(
                ValueError(
                    "There is no json metadata file in input_dir. This should have been set at the point of "
                    "ND2 extraction to npy."
                )
            )
        metadata = json.load(open(metadata_file))

    elif raw_extension == "jobs":
        metadata = nd2.get_jobs_metadata(all_files, config_file["input_dir"], config=config)
    else:
        log.error(
            ValueError(
                f"config_file['raw_extension'] should be either '.nd2' or '.npy' but it is "
                f"{config_file['raw_extension']}."
            )
        )

    # Stage 2: Read in page contents from config that cannot be computed from metadata.
    # the metadata. First few keys in the basic info page are only variables that the user can influence
    for key, value in list(config_basic.items())[:12]:
        if key == "bad_trc" and value is not None:
            nbp.__setattr__(
                key, tuple([(value[3 * i], value[3 * i + 1], value[3 * i + 2]) for i in range(len(value) // 3)])
            )
            continue
        nbp.__setattr__(key, value)
    if nbp.bad_trc is None:
        del nbp.bad_trc
        nbp.bad_trc = tuple()

    # Stage 3: Fill in all the metadata except xy_pos and nz.
    for key, value in metadata.items():
        if key in ("xy_pos", "nz"):
            continue
        # Set every metadata list to a tuple since lists are not allowed in the notebook.
        if type(value) is list:
            value = np.array(value)
        nbp.__setattr__(key, value)

    # Reverse the tile positions from raw tiles, if true.
    reversed_tilepos_yx_nd2 = nbp.tilepos_yx_nd2
    del nbp.tilepos_yx_nd2
    reversed_tilepos_yx_nd2 = tile_details.reverse_raw_tile_positions(
        reversed_tilepos_yx_nd2, config_basic["reverse_tile_positions_x"], config_basic["reverse_tile_positions_y"]
    )
    nbp.tilepos_yx_nd2 = reversed_tilepos_yx_nd2

    # Stage 4: If anything from the first 12 entries has been left blank, deal with that here.
    # Unfortunately, this is just many if statements as all blank entries need to be handled differently.
    # Notebook doesn't allow us to reset a value once it has been set so must delete and reset.

    # Next condition just says that if we are using the anchor and we don't specify the anchor round we will default it
    # to the final round. Add an extra round for the anchor and reduce the number of non anchor rounds by 1.
    if nbp.use_anchor:
        # Tell software that extra round is just an extra round and reduce the number of rounds
        nbp.n_extra_rounds = 1
        if nbp.anchor_round is None:
            del nbp.anchor_round
            nbp.anchor_round = metadata["n_rounds"]
        if nbp.anchor_channel is None:
            log.error(ValueError("Need to provide an anchor channel if using anchor!"))
    else:
        nbp.n_extra_rounds = 0

    # If no use_tiles given, default to all
    if nbp.use_tiles is None:
        del nbp.use_tiles
        nbp.use_tiles = tuple(np.arange(metadata["n_tiles"]).tolist())

    # If no use_rounds given, replace none with [], unless non jobs and user has provided the rounds in the file names
    if nbp.use_rounds is None:
        del nbp.use_rounds
        if config_file["round"] is not None and raw_extension != "jobs":
            nbp.use_rounds = tuple(np.arange(len(config_file["round"])).tolist())
        else:
            nbp.use_rounds = tuple(np.arange(0, nbp.n_rounds).tolist())

    if nbp.use_channels is None:
        del nbp.use_channels
        nbp.use_channels = tuple(np.arange(metadata["n_channels"]).tolist())
    if len(nbp.use_channels) > 9:
        raise NotImplementedError("There must be 9 or fewer sequencing channels to run coppafish.")

    # If no use_z given, default to all except the first if ignore_first_z_plane = True
    if nbp.use_z is None:
        del nbp.use_z
        use_z = np.arange(int(config_basic["ignore_first_z_plane"]), metadata["nz"]).tolist()
        use_z.sort()
        nbp.use_z = tuple(use_z)
    # This has not been assigned yet but now we can be sure that use_z not None!
    nbp.nz = len(nbp.use_z)
    for i in range(nbp.nz):
        if i == (nbp.nz - 1):
            break
        if abs(nbp.use_z[i] - nbp.use_z[i + 1]) > 1:
            raise ValueError("use_z contains z planes that are not connected. This may cause software instability")

    if nbp.use_dyes is None:
        del nbp.use_dyes
        nbp.use_dyes = utils_base.deep_convert(np.arange(len(nbp.dye_names)).tolist())
        nbp.n_dyes = len(nbp.use_dyes)

    return nbp
