import itertools
from typing import List, Optional

import napari
import numpy as np

from ..setup.notebook import Notebook


def view_filtered_images(
    nb: Notebook,
    tiles: Optional[List[int]] = None,
    rounds: Optional[List[int]] = None,
    channels: Optional[List[int]] = None,
    apply_colour_norm_factor: bool = True,
    share_contrast_limits: bool = False,
) -> None:
    """
    View the filtered images.

    Args:
        nb (Notebook): notebook.
        tiles (list of int, optional): tiles to view. Default: first tile.
        rounds (list of int, optional): rounds to view. Default: all rounds.
        channels (list of int, optional): channels to view. Default: all channels.
        apply_colour_norm_factor (bool, optional): apply the colour normalisation factor from call spots, if call spots
            is in the given notebook. Default: true.
        share_contrast_limits (bool, optional): use the same contrast limits for all filtered images shown. Default:
            false.
    """
    assert nb.has_page("filter"), "Filter must be run first"

    if tiles is None:
        tiles = nb.basic_info.use_tiles[:1]
    if rounds is None:
        rounds = nb.basic_info.use_rounds.copy()
    if channels is None:
        channels = nb.basic_info.use_channels.copy()

    factor = np.ones((len(nb.basic_info.use_tiles), len(nb.basic_info.use_rounds), len(nb.basic_info.use_channels)))
    if apply_colour_norm_factor and nb.has_page("call_spots"):
        factor = nb.call_spots.colour_norm_factor.astype(np.float32)

    min = 1e20
    max = -1e20
    images = []
    names = []
    for t, r, c in itertools.product(tiles, rounds, channels):
        image_trc: np.ndarray = nb.filter.images[t, r, c].astype(np.float32)
        # y, x, z -> z, y, x.
        image_trc = image_trc.swapaxes(1, 2).swapaxes(0, 1)
        image_trc *= factor[t, r, nb.basic_info.use_channels.index(c)]
        images.append(image_trc)
        names.append(f"Filter {t=}, {r=}, {c=}")
        image_min = image_trc.min()
        image_max = image_trc.max()
        if image_min < min:
            min = image_min
        if image_max > max:
            max = image_max

    viewer = napari.Viewer(title="Coppafish filtered images")
    limits = None
    for image, name in zip(images, names):
        if share_contrast_limits:
            limits = [min, max]
        viewer.add_image(image, name=name, rgb=False, contrast_limits=limits)

    napari.run()
