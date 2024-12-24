from typing import Tuple

import numpy as np

from ..setup.notebook_page import NotebookPage


def get_all_scores(
    nbp_basic: NotebookPage, nbp_omp: NotebookPage
) -> Tuple[np.ndarray[np.float16], np.ndarray[np.int16]]:
    """
    Get gene scores for every tile, concatenated together.

    Args:
        nbp_basic (notebook page): `basic_info` notebook page.
        nbp_omp (notebook page): `omp` notebook page.

    Returns tuple containing:
        (`(n_spots) ndarray[float16]`): all_scores. All gene scores.
        (`(n_spots) ndarray[int16]`): all_tiles. The tile for each spot.
    """
    assert type(nbp_basic) is NotebookPage
    assert type(nbp_omp) is NotebookPage

    all_scores = np.concatenate([nbp_omp.results[f"tile_{t}/scores"][:] for t in nbp_basic.use_tiles], dtype=np.float16)
    all_tiles = np.concatenate(
        [np.full(nbp_omp.results[f"tile_{t}/scores"].shape[0], t, np.int16) for t in nbp_basic.use_tiles],
        dtype=np.int16,
    )

    return all_scores, all_tiles


def get_all_intensities(
    nbp_basic: NotebookPage, nbp_call_spots: NotebookPage, nbp_omp: NotebookPage
) -> np.ndarray[np.float16]:
    """
    Get all spot intensities for all tiles, concatenated together.

    Args:
        nbp_basic (NotebookPage): `basic_info` notebook page.
        nbp_call_spots (NotebookPage): `call_spots` notebook page.
        nbp_omp (NotebookPage): `omp` notebook page.

    Returns:
        (`(n_spots) ndarray[float16]`): intensities. Every spot's intensity.
    """
    colours, tile_numbers = get_all_colours(nbp_basic, nbp_omp)
    colours = colours.astype(np.float32)
    # OMP's intensity will be a similar scale to prob and anchor if the spot colours are colour normalised too.
    colours *= nbp_call_spots.colour_norm_factor[tile_numbers].astype(np.float32)
    intensities = np.abs(colours).max(2).min(1).astype(np.float16)

    return intensities


def get_all_gene_no(
    nbp_basic: NotebookPage, nbp_omp: NotebookPage
) -> Tuple[np.ndarray[np.int16], np.ndarray[np.int16]]:
    """
    Get gene numbers for every tile, concatenated together.

    Args:
        nbp_basic (notebook page): `basic_info` notebook page.
        nbp_omp (notebook page): `omp` notebook page.

    Returns tuple containing:
            (`(n_spots) ndarray[int16]`): all_gene_no. All gene numbers.
            (`(n_spots) ndarray[int16]`): all_tiles. The tile for each spot.
    """
    assert type(nbp_basic) is NotebookPage
    assert type(nbp_omp) is NotebookPage

    all_gene_no = np.concatenate([nbp_omp.results[f"tile_{t}/gene_no"][:] for t in nbp_basic.use_tiles], dtype=np.int16)
    all_tiles = np.concatenate(
        [np.full(nbp_omp.results[f"tile_{t}/gene_no"].shape[0], t, np.int16) for t in nbp_basic.use_tiles],
        dtype=np.int16,
    )

    return all_gene_no, all_tiles


def get_all_local_yxz(
    nbp_basic: NotebookPage, nbp_omp: NotebookPage
) -> Tuple[np.ndarray[np.int16], np.ndarray[np.int16]]:
    """
    Get spot local positions for every tile, concatenated together.

    Args:
        nbp_basic (notebook page): `basic_info` notebook page.
        nbp_omp (notebook page): `omp` notebook page.

    Returns:
        (`(n_spots) ndarray[int16]`): all_local_yxz. All gene local positions.
        (`(n_spots) ndarray[int16]`): all_tiles. The tile for each spot.
    """
    assert type(nbp_basic) is NotebookPage
    assert type(nbp_omp) is NotebookPage

    all_local_yxz = np.concatenate(
        [nbp_omp.results[f"tile_{t}/local_yxz"][:] for t in nbp_basic.use_tiles], dtype=np.int16
    )
    all_tiles = np.concatenate(
        [np.full(nbp_omp.results[f"tile_{t}/local_yxz"].shape[0], t, np.int16) for t in nbp_basic.use_tiles],
        dtype=np.int16,
    )

    return all_local_yxz, all_tiles


def get_all_colours(
    nbp_basic: NotebookPage, nbp_omp: NotebookPage
) -> Tuple[np.ndarray[np.float16], np.ndarray[np.int16]]:
    """
    Get spot local positions for every tile, concatenated together.

    Args:
        - nbp_basic (notebook page): `basic_info` notebook page.
        - nbp_omp (notebook page): `omp` notebook page.

    Returns:
        (`(n_spots) ndarray[int16]`): all_colours. All spot colours.
        (`(n_spots) ndarray[int16]`): all_tiles. The tile for each spot.
    """
    assert type(nbp_basic) is NotebookPage
    assert type(nbp_omp) is NotebookPage

    all_colours = np.concatenate(
        [nbp_omp.results[f"tile_{t}/colours"][:] for t in nbp_basic.use_tiles], dtype=np.float16
    )
    all_tiles = np.concatenate(
        [np.full(nbp_omp.results[f"tile_{t}/colours"].shape[0], t, np.int16) for t in nbp_basic.use_tiles],
        dtype=np.int16,
    )

    return all_colours, all_tiles
