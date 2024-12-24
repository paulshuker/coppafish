import tempfile

import numpy as np
import zarr

from ...setup.notebook_page import NotebookPage
from .. import base


def test_get_all() -> None:
    temp_dir = tempfile.TemporaryDirectory()

    n_rounds = 3
    n_channels = 4

    rng = np.random.RandomState(0)

    nbp_basic = NotebookPage("basic_info")
    nbp_basic.use_tiles = (0, 2, 3, 5)
    nbp_basic.use_rounds = tuple([r for r in range(n_rounds)])
    nbp_basic.use_channels = tuple([c for c in range(n_channels)])

    nbp_call_spots = NotebookPage("call_spots")
    nbp_call_spots.colour_norm_factor = rng.rand(max(nbp_basic.use_tiles) + 1, n_rounds, n_channels).astype(np.float32)

    results_group = zarr.group(temp_dir.name, zarr_version=2)
    tiles = np.zeros(0, dtype=np.int16)
    local_yxz = np.zeros((0, 3), dtype=np.int16)
    scores = np.zeros(0, dtype=np.float16)
    gene_no = np.zeros(0, dtype=np.int16)
    colours = np.zeros((0, n_rounds, n_channels), dtype=np.float16)

    for t in nbp_basic.use_tiles:
        n_tile_spots = rng.randint(10, 31)

        t_tiles = np.full(n_tile_spots, t, dtype=np.int16)
        t_local_yxz = rng.randint(0, 99, size=(n_tile_spots, 3), dtype=np.int16)
        t_scores = rng.rand(n_tile_spots).astype(np.float16)
        t_gene_no = rng.randint(0, 99, size=(n_tile_spots), dtype=np.int16)
        t_colours = rng.rand(n_tile_spots, n_rounds, n_channels).astype(np.float16)

        tile_group = results_group.create_group(f"tile_{t}")
        tile_group.zeros("local_yxz", shape=t_local_yxz.shape, dtype=np.int16)
        tile_group.zeros("scores", shape=t_scores.shape, dtype=np.float16)
        tile_group.zeros("gene_no", shape=t_gene_no.shape, dtype=np.int16)
        tile_group.zeros("colours", shape=t_colours.shape, dtype=np.float16)

        tile_group.local_yxz[:] = t_local_yxz
        tile_group.scores[:] = t_scores
        tile_group.gene_no[:] = t_gene_no
        tile_group.colours[:] = t_colours

        tiles = np.append(tiles, t_tiles, axis=0)
        local_yxz = np.append(local_yxz, t_local_yxz, axis=0)
        scores = np.append(scores, t_scores, axis=0)
        gene_no = np.append(gene_no, t_gene_no, axis=0)
        colours = np.append(colours, t_colours, axis=0)

    nbp_omp = NotebookPage("omp")
    nbp_omp.results = results_group

    all_scores, all_tiles = base.get_all_scores(nbp_basic, nbp_omp)
    assert type(all_scores) is np.ndarray
    assert all_scores.shape == (tiles.size,)
    assert type(all_tiles) is np.ndarray
    assert np.allclose(all_scores, scores)
    assert np.allclose(all_tiles, tiles)

    expected_intensities = np.abs(colours.astype(np.float32) * nbp_call_spots.colour_norm_factor[tiles]).max(2).min(1)
    expected_intensities = expected_intensities.astype(np.float16)
    all_intensities = base.get_all_intensities(nbp_basic, nbp_call_spots, nbp_omp)
    assert type(all_intensities) is np.ndarray
    assert all_intensities.shape == (tiles.size,)
    assert np.allclose(all_intensities, expected_intensities)

    all_gene_no, all_tiles = base.get_all_gene_no(nbp_basic, nbp_omp)
    assert type(all_gene_no) is np.ndarray
    assert all_gene_no.shape == (tiles.size,)
    assert type(all_tiles) is np.ndarray
    assert np.allclose(all_gene_no, gene_no)
    assert np.allclose(all_tiles, tiles)

    all_local_yxz, all_tiles = base.get_all_local_yxz(nbp_basic, nbp_omp)
    assert type(all_local_yxz) is np.ndarray
    assert all_local_yxz.shape == (tiles.size, 3)
    assert type(all_tiles) is np.ndarray
    assert np.allclose(all_local_yxz, local_yxz)
    assert np.allclose(all_tiles, tiles)

    all_colours, all_tiles = base.get_all_colours(nbp_basic, nbp_omp)
    assert type(all_colours) is np.ndarray
    assert all_colours.shape == (tiles.size, n_rounds, n_channels)
    assert type(all_tiles) is np.ndarray
    assert np.allclose(all_colours, colours)
    assert np.allclose(all_tiles, tiles)

    temp_dir.cleanup()
