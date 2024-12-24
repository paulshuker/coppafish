import warnings

import nd2
import numpy as np

from ..setup.notebook_page import NotebookPage
from .raw_reader import RawReader


class Nd2Reader(RawReader):
    """
    Reader for raw, ND2 files.
    """

    def read(
        self, nbp_basic: NotebookPage, nbp_file: NotebookPage, tile: int, round: int, channels: list[int]
    ) -> np.ndarray:
        """
        Read ND2 files for the given channels.

        We expect every ND2 file to have the shape (n_tiles, n_z_planes, n_channels, n_y_pixels, n_x_pixels). There
        should be one ND2 file for each round.

        Returns:
            (`(len(channels) x im_y x im_x x im_z) ndarray`): image. The channel image(s).
        """
        super().read(nbp_basic, nbp_file, tile, round, channels)

        tile_raw = super().get_tile_raw_index(tile, nbp_basic.tilepos_yx_nd2, nbp_basic.tilepos_yx)
        file_path = super().get_round_file_path(nbp_file, round)

        with nd2.ND2File(file_path) as images:
            # Hiding a warning (known issue with nd2 package https://github.com/tlambert03/nd2/issues/239).
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="dask.tokenize")
                images = images.to_dask()

        images = images[tile_raw]

        # Convert the dask array into an ndarray as all channels for one tile/round can fit in memory.
        # This is the fastest way because the dask array is not chunked over channels like previously thought.
        images = images.compute()

        images = images[:, channels]

        # Put the z index to the end.
        # zcyx -> czyx -> cyzx -> cyxz where c is the channels index.
        images = images.swapaxes(0, 1).swapaxes(1, 2).swapaxes(2, 3)

        return images
