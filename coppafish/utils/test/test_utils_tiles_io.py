import os
import numpy as np

from coppafish import NotebookPage
from coppafish.utils import tiles_io


def test_tiles_io_save_load_tile():
    for file_type in [".npy", ".zarr"]:
        directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), "unit_test_dir")

        if not os.path.isdir(directory):
            os.mkdir(directory)
        rng = np.random.RandomState(0)
        array_1_shape = (3, 3, 4)
        array_1 = rng.randint(2**16, size=array_1_shape, dtype=np.int32)

        nbp_file_3d = NotebookPage("file_names")
        for name, value in {"tile": (((os.path.join(directory, f"array{file_type}"),),),)}.items():
            nbp_file_3d.__setattr__(name, value)

        nbp_basic_3d = NotebookPage("basic_info")
        for name, value in {
            "is_3d": True,
            "anchor_round": 100,
            "dapi_channel": 5,
            "tile_sz": 3,
            "use_z": (0, 1, 2, 3),
            "tile_pixel_value_shift": 0,
            "pre_seq_round": 99,
        }.items():
            nbp_basic_3d.__setattr__(name, value)

        # 3d:
        tiles_io.save_image(nbp_file_3d, nbp_basic_3d, file_type, array_1, 0, 0, 0)
        output = tiles_io.load_image(nbp_file_3d, nbp_basic_3d, file_type, 0, 0, 0, yxz=None, apply_shift=False)
        assert np.allclose(array_1, output), "Loaded in tile does not have the same values as starting tile"
        tiles_io.save_image(nbp_file_3d, nbp_basic_3d, file_type, array_1, 0, 0, 0)
        yxz = (None, None, (1, 2))
        output = tiles_io.load_image(nbp_file_3d, nbp_basic_3d, file_type, 0, 0, 0, yxz=yxz, apply_shift=False)
        assert output.ndim == 3
        assert output.shape == (array_1_shape[0], array_1_shape[1], 1)
        assert np.allclose(array_1[:, :, [1]], output), "Expected a subvolume to be loaded in"
        yxz = ((1, 3), None, (0, 2))
        output = tiles_io.load_image(nbp_file_3d, nbp_basic_3d, file_type, 0, 0, 0, yxz=yxz, apply_shift=False)
        assert output.ndim == 3
        assert output.shape == (2, array_1_shape[1], 2)
        assert np.allclose(array_1[1:3, :, 0:2], output), "Expected a subvolume to be loaded in"


# TODO: get_npy_tile_ind unit tests.
