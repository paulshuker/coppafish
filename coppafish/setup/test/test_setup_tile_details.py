import numpy as np

from .. import tile_details


def test_reverse_raw_tile_positions() -> None:
    # A single tile position.
    raw_tile_positions = np.array([[0, 1]], int)
    assert (raw_tile_positions == tile_details.reverse_raw_tile_positions(raw_tile_positions, False, False)).all()
    assert (raw_tile_positions == tile_details.reverse_raw_tile_positions(raw_tile_positions, False, True)).all()
    assert (raw_tile_positions == tile_details.reverse_raw_tile_positions(raw_tile_positions, True, False)).all()
    assert (raw_tile_positions == tile_details.reverse_raw_tile_positions(raw_tile_positions, True, True)).all()

    # Test on two tile positions.
    raw_tile_positions = np.array([[0, 0], [0, 1]], int)
    assert (raw_tile_positions == tile_details.reverse_raw_tile_positions(raw_tile_positions, False, False)).all()
    assert (raw_tile_positions == tile_details.reverse_raw_tile_positions(raw_tile_positions, False, True)).all()
    assert (
        np.array([[0, 1], [0, 0]], int) == tile_details.reverse_raw_tile_positions(raw_tile_positions, True, False)
    ).all()
    assert (
        np.array([[0, 1], [0, 0]], int) == tile_details.reverse_raw_tile_positions(raw_tile_positions, True, True)
    ).all()

    raw_tile_positions = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]], int)
    n_tiles = raw_tile_positions.shape[0]

    reverse_x = False
    reverse_y = False
    output = tile_details.reverse_raw_tile_positions(raw_tile_positions, reverse_x, reverse_y)
    assert type(output) is np.ndarray
    assert output.shape == (n_tiles, 2)
    assert (output == raw_tile_positions).all()
    assert raw_tile_positions.dtype == output.dtype

    reverse_x = True
    reverse_y = False
    output = tile_details.reverse_raw_tile_positions(raw_tile_positions, reverse_x, reverse_y)
    assert type(output) is np.ndarray
    assert output.shape == (n_tiles, 2)
    assert (output == np.array([[0, 1], [0, 0], [1, 1], [1, 0], [2, 1], [2, 0]], int)).all()
    assert raw_tile_positions.dtype == output.dtype

    reverse_x = False
    reverse_y = True
    output = tile_details.reverse_raw_tile_positions(raw_tile_positions, reverse_x, reverse_y)
    assert type(output) is np.ndarray
    assert output.shape == (n_tiles, 2)
    assert (output == np.array([[2, 0], [2, 1], [1, 0], [1, 1], [0, 0], [0, 1]], int)).all()
    assert raw_tile_positions.dtype == output.dtype

    reverse_x = True
    reverse_y = True
    output = tile_details.reverse_raw_tile_positions(raw_tile_positions, reverse_x, reverse_y)
    assert type(output) is np.ndarray
    assert output.shape == (n_tiles, 2)
    assert (output == np.array([[2, 1], [2, 0], [1, 1], [1, 0], [0, 1], [0, 0]], int)).all()
    assert raw_tile_positions.dtype == output.dtype
