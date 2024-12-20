import dask.array
import numpy as np

from ..setup.notebook_page import NotebookPage
from .raw_reader import RawReader


class NumpyReader(RawReader):
    """
    Reader for raw numpy files.

    For example, Robominnie (the pipeline integration tester) uses raw numpy files as input.
    Read ND2 files for the given channels.

    """

    def read(
        self, nbp_basic: NotebookPage, nbp_file: NotebookPage, tile: int, round: int, channels: list[int]
    ) -> np.ndarray:
        """
        Similar to ND2, expect a separate directory for each round. Every directory has one numpy array for each tile
        with shape (1, n_z_planes, n_channels, n_y_pixels, n_x_pixels).

        Returns:
            (`(len(channels) x im_y x im_x x im_z) ndarray`): image. The channel image(s).
        """
        super().read(nbp_basic, nbp_file, tile, round, channels)

        tile_raw = super().get_tile_raw_index(tile, nbp_basic.tilepos_yx_nd2, nbp_basic.tilepos_yx)
        file_path = super().get_round_file_path(nbp_file, round)

        # Has shape (n_channels, im_y, im_x, im_z).
        tile_round_images: np.ndarray = dask.array.from_npy_stack(file_path)[tile_raw].compute()

        tile_round_images = tile_round_images[channels]

        return tile_round_images
