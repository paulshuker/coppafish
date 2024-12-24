import itertools
from typing import List, Optional

import napari

from .. import log
from ..setup import file_names
from ..setup.notebook import Notebook
from ..utils import tiles_io


def view_extracted_images(
    nb: Notebook,
    config_path: str,
    tiles: Optional[List[int]] = None,
    rounds: Optional[List[int]] = None,
    channels: Optional[List[int]] = None,
) -> None:
    """
    View the unfiltered, extracted images.

    Args:
        nb (Notebook): notebook.
        config_path (str): file path to the config file.
        tiles (List[int], optional): tiles to view. Default: all tiles.
        rounds (List[int], optional): rounds to view. Default: all rounds.
        channels (List[int], optional): channels to view. Default: all channels.
    """
    assert nb.has_page("extract"), "Extract must be run first"

    if tiles is None:
        tiles = nb.basic_info.use_tiles.copy()
    if rounds is None:
        rounds = nb.basic_info.use_rounds.copy()
    if channels is None:
        channels = nb.basic_info.use_channels.copy()

    nbp_file = file_names.get_file_names(nb.basic_info, config_path)

    viewer = napari.Viewer(title="Coppafish extracted images")

    for t, r, c in itertools.product(tiles, rounds, channels):
        file_path = nbp_file.tile_unfiltered[t][r][c]
        if not tiles_io.image_exists(file_path):
            log.warn(f"Image at {file_path} not found, skipping")
            continue
        image_trc = tiles_io._load_image(file_path)
        viewer.add_image(image_trc, name=f"{t=}, {r=}, {c=}")

    napari.run()
