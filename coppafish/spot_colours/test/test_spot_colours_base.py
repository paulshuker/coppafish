import numpy as np
import torch

from coppafish import NotebookPage
from coppafish.spot_colours import base as spot_colours_base


def test_convert_coords_to_torch_grid() -> None:
    # Test this function by grid sampling an image for all its pixel positions, so no interpolation should be done and
    # the same image should be given back.
    rng = np.random.RandomState(0)
    image_shape = 4, 5, 6
    image = rng.rand(*image_shape)
    yxz_coords = np.meshgrid(
        np.linspace(0, image_shape[0] - 1, image_shape[0]),
        np.linspace(0, image_shape[1] - 1, image_shape[1]),
        np.linspace(0, image_shape[2] - 1, image_shape[2]),
        indexing="ij",
    )
    # NOTE: Order 'F' reshaping must be used in numpy reshape method as this is the same as pytorch's reshape method.
    yxz_coords = np.array(yxz_coords).T.reshape((-1, 3), order="F")
    yxz_coords = torch.tensor(yxz_coords)
    yxz_grid_coords = spot_colours_base.convert_coords_to_torch_grid(yxz_coords, image_shape)
    assert type(yxz_grid_coords) is torch.Tensor
    assert yxz_grid_coords.shape == (np.prod(image_shape).item(), 3)
    yxz_grid_coords = yxz_grid_coords.reshape(image_shape + (3,))[None]
    image = torch.tensor(image).float()
    # Input has shape (1, 1, 4, 5, 6).
    # Grid has shape (1, 4, 5, 6, 3).
    # Result has shape (1, 1, 4, 5, 6).
    grid_image = torch.nn.functional.grid_sample(image[None, None], yxz_grid_coords, align_corners=True)
    grid_image = grid_image[0, 0]
    assert grid_image.shape == image_shape
    assert torch.allclose(grid_image, image)


def test_apply_flow_new() -> None:
    # Check that a simple +1, +2, and -3 shift in each direction works.
    yxz = np.zeros((2, 3), dtype=np.int16)
    yxz[1] = [3, 5, 6]
    n_tiles, n_rounds = 1, 3
    tile = 0
    round = 1
    image_shape = (4, 5, 6)
    flow = np.zeros((n_tiles, n_rounds, 3) + image_shape)
    flow[tile, round, 0] = +1
    flow[tile, round, 1] = +2
    flow[tile, round, 2] = -3
    yxz_expected = yxz.copy()
    yxz_expected[0] = 1, 2, -3
    yxz_expected[1] = 4, 7, 3
    yxz_expected = torch.tensor(yxz_expected).float()
    yxz_flow = spot_colours_base.apply_flow_new(yxz, flow, tile, round)
    assert type(yxz_flow) is np.ndarray
    assert yxz_flow.shape == (2, 3)
    assert torch.allclose(torch.from_numpy(yxz_flow), yxz_expected)
    yxz = torch.tensor(yxz)
    yxz_flow = spot_colours_base.apply_flow_new(yxz, flow, tile, round)
    assert type(yxz_flow) is torch.Tensor
    assert yxz_flow.shape == (2, 3)
    assert torch.allclose(yxz_flow, yxz_expected)


def test_get_spot_colours_new_safe() -> None:
    rng = np.random.RandomState(0)
    n_tiles = 2
    n_rounds = 3
    n_channels = 4
    tile_shape = (5, 5, 7)
    image_shape = (n_tiles, n_rounds, n_channels) + tile_shape
    image = rng.rand(*image_shape).astype(np.float32)
    yxz = np.meshgrid(
        np.linspace(0, tile_shape[0] - 1, tile_shape[0]),
        np.linspace(0, tile_shape[1] - 1, tile_shape[1]),
        np.linspace(0, tile_shape[2] - 1, tile_shape[2]),
        indexing="ij",
    )
    yxz = np.array(yxz).reshape((3, -1), order="F").T
    flow = np.zeros((n_tiles, n_rounds, 3) + tile_shape)
    affine = np.zeros((n_tiles, n_rounds, n_channels, 4, 3))
    affine[:, :, :, :3] = np.eye(3)
    tile = 0
    use_rounds = list(range(n_rounds))
    use_channels = list(range(n_channels))
    nbp_basic = NotebookPage("basic_info")
    nbp_basic.tile_sz = tile_shape[0]
    nbp_basic.use_z = tuple(range(tile_shape[2]))
    output_dtype = np.float32
    colours = spot_colours_base.get_spot_colours_new_safe(
        nbp_basic,
        yxz=yxz,
        image=image,
        flow=flow,
        affine=affine,
        tile=tile,
        use_rounds=use_rounds,
        use_channels=use_channels,
        output_dtype=output_dtype,
    )
    assert type(colours) is np.ndarray
    assert colours.dtype.type is output_dtype
    assert colours.shape == (np.prod(tile_shape), n_rounds, n_channels)
    abs_tol = 1e-7
    for r in range(n_rounds):
        for c in range(n_channels):
            assert np.allclose(colours[:, r, c], image[tile, r, c][tuple(yxz.T.astype(int))], atol=abs_tol)


def test_get_spot_colours_new() -> None:
    # Make sure that when the transforms are all identity the image is returned unchanged.
    rng = np.random.RandomState(0)
    n_tiles = 2
    n_rounds = 3
    n_channels = 4
    tile_shape = (5, 6, 7)
    image_shape = (n_tiles, n_rounds, n_channels) + tile_shape
    image = rng.rand(*image_shape).astype(np.float32)
    yxz = np.meshgrid(
        np.linspace(0, tile_shape[0] - 1, tile_shape[0]),
        np.linspace(0, tile_shape[1] - 1, tile_shape[1]),
        np.linspace(0, tile_shape[2] - 1, tile_shape[2]),
        indexing="ij",
    )
    yxz = np.array(yxz).reshape((3, -1), order="F").T
    flow = np.zeros((n_tiles, n_rounds, 3) + tile_shape)
    affine = np.zeros((n_tiles, n_rounds, n_channels, 4, 3))
    affine[:, :, :, :3] = np.eye(3)
    tile = 0
    use_rounds = list(range(n_rounds))
    use_channels = list(range(n_channels))
    output_dtype = np.float32
    colours = spot_colours_base.get_spot_colours_new(
        yxz, image, flow, affine, tile, use_rounds, use_channels, output_dtype=output_dtype
    )
    assert type(colours) is np.ndarray
    assert colours.dtype.type is output_dtype
    assert colours.shape == (np.prod(tile_shape), n_rounds, n_channels)
    abs_tol = 1e-7
    for r in range(n_rounds):
        for c in range(n_channels):
            assert np.allclose(colours[:, r, c], image[tile, r, c][tuple(yxz.T.astype(int))], atol=abs_tol)
    # NOTE: Reshaping with order F again is correct.
    colours = colours.swapaxes(0, 1).swapaxes(1, 2).reshape((n_rounds, n_channels) + tile_shape, order="F")
    assert np.allclose(colours, image[tile], atol=abs_tol)

    # Try with just gathering a single line of x pixels.
    yxz = np.meshgrid(
        [3],
        np.linspace(0, tile_shape[1] - 1, tile_shape[1]),
        [6],
        indexing="ij",
    )
    yxz = np.array(yxz).reshape((3, -1), order="F").T
    colours = spot_colours_base.get_spot_colours_new(
        yxz, image, flow, affine, tile, use_rounds, use_channels, output_dtype=output_dtype
    )
    assert type(colours) is np.ndarray
    assert colours.shape == (yxz.shape[0], n_rounds, n_channels)
    for r in range(n_rounds):
        for c in range(n_channels):
            assert np.allclose(colours[:, r, c], image[tile, r, c][tuple(yxz.T.astype(int))], atol=abs_tol)

    # Check optical flow shifting is working.
    yxz = np.meshgrid(
        np.linspace(0, tile_shape[0] - 1, tile_shape[0]),
        np.linspace(0, tile_shape[1] - 1, tile_shape[1]),
        np.linspace(0, tile_shape[2] - 1, tile_shape[2]),
        indexing="ij",
    )
    yxz = np.array(yxz).reshape((3, -1), order="F").T
    flow[tile, 0, 0] = 1
    flow[tile, 0, 1] = -1
    flow[tile, 2, 2] = 2
    colours = spot_colours_base.get_spot_colours_new(
        yxz, image, flow, affine, tile, use_rounds, use_channels, output_dtype=output_dtype
    )
    colours = colours.swapaxes(0, 1).swapaxes(1, 2).reshape((n_rounds, n_channels) + tile_shape, order="F")
    # Check out of bounds.
    assert np.isnan(colours[0, :, -1]).all()
    assert np.isnan(colours[0, :, :, 0]).all()
    assert np.isnan(colours[2, :, :, :, -2:]).all()
    assert np.allclose(colours[0, :, :-1, 1:], image[tile, 0, :, 1:, :-1], atol=abs_tol)
    assert np.allclose(colours[1], image[tile, 1], atol=abs_tol)
    assert np.allclose(colours[2, :, :, :, :-2], image[tile, 2, :, :, :, 2:], atol=abs_tol)

    # Check affine transform is working.
    tile_shape = (5, 5, 6)
    flow = np.zeros((n_tiles, n_rounds, 3) + tile_shape)
    image_shape = (n_tiles, n_rounds, n_channels) + tile_shape
    image = rng.rand(*image_shape).astype(np.float32)
    output_dtype = np.float64
    yxz = np.meshgrid(
        np.linspace(0, tile_shape[0] - 1, tile_shape[0]),
        np.linspace(0, tile_shape[1] - 1, tile_shape[1]),
        np.linspace(0, tile_shape[2] - 1, tile_shape[2]),
        indexing="ij",
    )
    yxz = np.array(yxz).reshape((3, -1), order="F").T
    affine = np.zeros((n_tiles, n_rounds, n_channels, 4, 3))
    affine[:, :, :, :3] = np.eye(3)
    affine[tile, 2, 1] = 0
    # Flip the x and y positions on round 2, channel 1.
    affine[tile, 2, 1, 0, 1] = 1
    affine[tile, 2, 1, 1, 0] = 1
    affine[tile, 2, 1, 2, 2] = 1
    # Compress y positions by a factor of 2 on round 1, channel 0.
    affine[tile, 1, 0, 0, 0] = 0.5
    # Stretch z positions by a factor of 2 on round 0, channel 0.
    affine[tile, 0, 0, 2, 2] = 2
    # Shift all x positions by 1.5 pixels on round 2, channel 0.
    affine[tile, 2, 0, 3, 1] = 1.5
    colours = spot_colours_base.get_spot_colours_new(
        yxz, image, flow, affine, tile, use_rounds, use_channels, output_dtype=output_dtype
    )
    assert colours.dtype.type is output_dtype
    colours = colours.swapaxes(0, 1).swapaxes(1, 2).reshape((n_rounds, n_channels) + tile_shape, order="F")
    assert np.allclose(colours[0, 1:4], image[tile, 0, 1:4], atol=abs_tol)
    assert np.allclose(colours[1, 1:4], image[tile, 1, 1:4], atol=abs_tol)
    assert np.allclose(colours[2, 2:4], image[tile, 2, 2:4], atol=abs_tol)
    assert np.allclose(colours[2, 1].swapaxes(0, 1), image[tile, 2, 1], atol=abs_tol)
    assert np.allclose(colours[1, 0][[0, 2, 4]], image[tile, 1, 0, :3], atol=abs_tol)
    assert np.allclose(colours[0, 0, :, :, :3], image[tile, 0, 0][:, :, [0, 2, 4]], atol=abs_tol)
    assert np.isnan(colours[0, 0, :, :, 3:]).all()
    assert np.allclose(colours[2, 0, 0, 0, 0], image[tile, 2, 0, 0, 1:3, 0].mean(), atol=abs_tol)
    assert np.isnan(colours[2, 0, :, 3:]).all()
