from os import path

import nd2
import numpy as np

from ...setup.notebook_page import NotebookPage
from ..raw_nd2 import Nd2Reader
from ..raw_reader import RawReader

FILE_NAME = "dims_z5t3c2y32x32"
SUFFIX = ".nd2"


def test_raw_nd2() -> None:
    # We use a uint16 ND2 file of shape (3, 5, 2, 32, 32) to unit test with.
    nd2_file_path = path.join(path.dirname(__file__), FILE_NAME + SUFFIX)

    assert path.isfile(nd2_file_path)

    all_data = np.array([])
    with nd2.ND2File(nd2_file_path) as file:
        all_data = file.asarray()

    nbp_basic = NotebookPage("basic_info")
    nbp_basic.use_anchor = True
    nbp_basic.tilepos_yx = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0, 2], [1, 2]], int)
    nbp_basic.tilepos_yx_nd2 = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]], int)

    nbp_file = NotebookPage("file_names")
    nbp_file.input_dir = path.dirname(nd2_file_path)
    nbp_file.round = tuple(["wrong_file"] * 2 + [FILE_NAME] + ["wrong_file"] * 5)
    nbp_file.anchor = "anchor"
    nbp_file.raw_extension = SUFFIX

    # This refers to tile index 2 in the raw file tile ordering.
    tile = 1
    round = 2
    channels = [0, 1]

    reader = Nd2Reader()
    assert isinstance(reader, RawReader)
    image = reader.read(nbp_basic, nbp_file, tile, round, channels)

    assert type(image) is np.ndarray
    assert image.ndim == 4
    assert image.shape == (len(channels), 32, 32, 5)
    for c_i, c in enumerate(channels):
        for z in range(5):
            assert (image[c_i, :, :, z] == all_data[2, z, c]).all()

    channels = [1]
    reader = Nd2Reader()
    image = reader.read(nbp_basic, nbp_file, tile, round, channels)

    assert type(image) is np.ndarray
    assert image.ndim == 4
    assert image.shape == (len(channels), 32, 32, 5)
    for c_i, c in enumerate(channels):
        for z in range(5):
            assert (image[c_i, :, :, z] == all_data[2, z, c]).all()
