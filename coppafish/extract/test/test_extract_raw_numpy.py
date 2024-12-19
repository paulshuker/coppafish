import tempfile
from os import path

import dask.array
import numpy as np

from ...setup.notebook_page import NotebookPage
from ..raw_numpy import NumpyReader
from ..raw_reader import RawReader


def test_raw_numpy() -> None:
    n_tiles = 6
    n_rounds = 3
    n_channels = 5
    im_y = 11
    im_x = 13
    im_z = 7

    rng = np.random.RandomState(0)
    all_data = rng.randint(100, size=(n_tiles, n_rounds, n_channels, im_y, im_x, im_z))

    npy_dir = tempfile.TemporaryDirectory(suffix="coppafish")
    npy_dir_name = npy_dir.name

    dask_chunks = (1, n_channels, im_y, im_x, im_z)

    for r in range(n_rounds):
        r_dir = path.join(npy_dir_name, f"{r}")
        image_dask = dask.array.from_array(all_data[:, r], chunks=dask_chunks)
        dask.array.to_npy_stack(r_dir, image_dask)

    nbp_basic = NotebookPage("basic_info")
    nbp_basic.use_anchor = True
    nbp_basic.tilepos_yx = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0, 2], [1, 2]], int)
    nbp_basic.tilepos_yx_nd2 = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]], int)

    nbp_file = NotebookPage("file_names")
    nbp_file.input_dir = npy_dir_name
    nbp_file.round = tuple([f"{r}" for r in range(n_rounds)])
    nbp_file.anchor = "anchor"
    nbp_file.raw_extension = ".npy"

    # This refers to tile index 2 in the raw file tile ordering.
    tile = 1
    round = 2
    channels = [0, 1]

    reader = NumpyReader()
    assert isinstance(reader, RawReader)
    image = reader.read(nbp_basic, nbp_file, tile, round, channels)

    assert type(image) is np.ndarray
    assert image.ndim == 4
    assert image.shape == (len(channels), im_y, im_x, im_z)
    for c_i, c in enumerate(channels):
        for z in range(5):
            assert (image[c_i, :, :, z] == all_data[2, round, c, :, :, z]).all()

    channels = [1, 0, 4]
    reader = NumpyReader()
    image = reader.read(nbp_basic, nbp_file, tile, round, channels)

    assert type(image) is np.ndarray
    assert image.ndim == 4
    assert image.shape == (len(channels), im_y, im_x, im_z)
    for c_i, c in enumerate(channels):
        for z in range(5):
            assert (image[c_i, :, :, z] == all_data[2, round, c, :, :, z]).all()

    npy_dir.cleanup()
