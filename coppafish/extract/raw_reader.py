import os

import numpy as np

from ..setup.notebook_page import NotebookPage


class RawReader:
    """
    The base raw reader class.

    Every supported raw extension type inherits the RawReader and creates their own way of reading the raw images as
    ndarrays for coppafish extraction.
    """

    def read(
        self, nbp_basic: NotebookPage, nbp_file: NotebookPage, tile: int, round: int, channels: list[int]
    ) -> np.ndarray:
        """
        Base function to read raw files for the given tile, round for the specific channels.

        Args:
            nbp_basic (NotebookPage): `basic_info` notebook page.
            nbp_file (NotebookPage): `file_names` notebook page.
            tile (int): tile index.
            round (int): round index.
            channels (list of int): the channels to gather.

        Returns:
            ((tuple of length `len(channels)`) containing `(im_y x im_x x im_z) ndarrays`): images. The tile/round
                images. Can be any datatype, but we work with uint16 raw files. They are later converted to floating
                point numbers when filtering the images.
        """
        assert type(nbp_basic) is NotebookPage
        assert type(nbp_file) is NotebookPage
        assert type(tile) is int
        assert type(round) is int
        assert type(channels) is list
        assert len(channels) > 0
        assert all([type(c) is int for c in channels])

    def get_tile_raw_index(self, tile_index: int, tile_pos_yx_raw: np.ndarray, tile_pos_yx: np.ndarray) -> int:
        """
        Convert the given, coppafish tile index into the raw tile index. This is required since coppafish uses a
        different tile ordering system to Nikon's ND2 files which we use as raw input.

        Args:
            tile_index (int): the coppafish tile index to convert.
            tile_pos_yx_raw (`(n_tiles x 2) ndarray[int]`): tile_pos_yx_raw[i] contains the y and x position of tile at
                index `i`. i = 0 refers to y, x = 0, 0. i = 1 refers to y, x = 0, 1 if x tile count > 1.
            tile_pos_yx (`(n_tiles x 2) ndarray[int]`): tile_pos_yx[i] contains the y and x position of tile at index
                `i`. i = 0 refers to y, x = [maxY, maxX]. i = 1 refers to y, x = [maxY, maxX - 1] if x tile count > 1.

        Returns:
            (int): the raw tile index.
        """
        assert type(tile_index) is int
        assert type(tile_pos_yx_raw) is np.ndarray
        assert tile_pos_yx_raw.ndim == 2
        assert tile_pos_yx_raw.shape[1] == 2
        assert type(tile_pos_yx) is np.ndarray
        assert tile_pos_yx.ndim == 2
        assert tile_pos_yx.shape[1] == 2

        # Since npy and raw files have different tile positioning, convert tile_pos_yx to raw tile coordinates.
        tile_pos_yx = np.max(tile_pos_yx, axis=0) - tile_pos_yx
        nd2_index = (tile_pos_yx_raw == tile_pos_yx[[tile_index]]).all(1).nonzero()[0][0].item()

        return nd2_index

    def get_round_file_path(self, nbp_file: NotebookPage, round: int) -> str:
        """
        Get the file path for the given round.
        """
        # Always have anchor after imaging rounds.
        round_files = nbp_file.round + [nbp_file.anchor]
        file_path = os.path.join(nbp_file.input_dir, round_files[round])
        if nbp_file.raw_extension == ".nd2":
            file_path += nbp_file.raw_extension

        return file_path

    def get_channel_laser(self, nbp_basic: NotebookPage) -> list[int]:
        """
        Get a list of unique wavelengths in nm for the lasers used.
        """
        assert type(nbp_basic) is NotebookPage

        if nbp_basic.channel_laser is not None:
            channel_laser = list(set(nbp_basic.channel_laser))
        else:
            channel_laser = []
        channel_laser.sort()

        return channel_laser

    def get_channel_cam(self, nbp_basic: NotebookPage) -> list[int]:
        """
        Get a list of unique wavelengths in nm for the camera in all channels.
        """
        assert type(nbp_basic) is NotebookPage

        if nbp_basic.channel_camera is not None:
            channel_camera = list(set(nbp_basic.channel_camera))
        else:
            channel_camera = []
        channel_camera.sort()

        return channel_camera
