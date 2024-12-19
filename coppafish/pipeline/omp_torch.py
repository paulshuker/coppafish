import importlib.resources as importlib_resources
import math as maths
import os
import pickle
import platform
from typing import Tuple

import numpy as np
import scipy
import torch
import tqdm
import zarr

from .. import find_spots, log, utils
from ..omp import scores
from ..omp.pixel_scores import PixelScoreSolver
from ..setup.config_section import ConfigSection
from ..setup.notebook_page import NotebookPage
from ..spot_colours import base as spot_colours_base
from ..utils import duplicates, system


def run_omp(
    config: ConfigSection,
    nbp_file: NotebookPage,
    nbp_basic: NotebookPage,
    nbp_extract: NotebookPage,
    nbp_filter: NotebookPage,
    nbp_register: NotebookPage,
    nbp_stitch: NotebookPage,
    nbp_call_spots: NotebookPage,
) -> NotebookPage:
    """
    Run orthogonal matching pursuit (omp) on every pixel to determine a pixel score for each gene at each pixel.

    From these OMP pixel scores, create a spot score at every pixel position by convolving with a given mean spot.

    Detect spots to find final gene reads.

    See `omp` section of file `coppafish/setup/notebook_page.py` for descriptions of the omp variables.

    Args:
        config (ConfigSection): config section for `omp`.
        nbp_file (NotebookPage): `file_names` notebook page.
        nbp_basic (NotebookPage): `basic_info` notebook page.
        nbp_extract (NotebookPage): `extract` notebook page.
        nbp_filter (NotebookPage): `filter` notebook page.
        nbp_register (NotebookPage): `register` notebook page.
        nbp_stitch (NotebookPage): `stitch` notebook page.
        nbp_call_spots (NotebookPage): `call_spots` notebook page.

    Returns:
        `NotebookPage[omp]`: nbp_omp. Page containing gene assignments and info for OMP spots.
    """
    assert type(config) is ConfigSection
    assert type(nbp_file) is NotebookPage
    assert type(nbp_basic) is NotebookPage
    assert type(nbp_extract) is NotebookPage
    assert type(nbp_filter) is NotebookPage
    assert type(nbp_register) is NotebookPage
    assert type(nbp_stitch) is NotebookPage
    assert type(nbp_call_spots) is NotebookPage

    log.info("OMP started")
    log.debug(f"{torch.cuda.is_available()=}")
    log.debug(f"{config['force_cpu']=}")

    omp_config = {config.name: config.to_dict()}
    nbp = NotebookPage("omp", omp_config)

    torch.backends.cudnn.deterministic = True
    if platform.system() != "Windows":
        # Avoids chance of memory crashing on Linux.
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Preparing useful values used during OMP.
    n_genes = nbp_call_spots.bled_codes.shape[0]
    n_rounds_use = len(nbp_basic.use_rounds)
    n_channels_use = len(nbp_basic.use_channels)
    tile_shape: Tuple[int] = nbp_basic.tile_sz, nbp_basic.tile_sz, len(nbp_basic.use_z)
    n_tile_pixels = np.prod(tile_shape).item()
    tile_origins = nbp_stitch.tile_origin.astype(np.float32)
    tile_centres = duplicates.get_tile_centres(nbp_basic.tile_sz, len(nbp_basic.use_z), tile_origins)
    tile_origins = torch.from_numpy(tile_origins)
    n_subset_pixels = config["subset_pixels"]
    n_memory_constant = 7e7 / (n_genes * n_rounds_use * n_channels_use)
    yxz_all = [np.linspace(0, tile_shape[i] - 1, tile_shape[i]) for i in range(3)]
    yxz_all = np.array(np.meshgrid(*yxz_all, indexing="ij")).astype(np.int16).reshape((3, -1), order="F").T
    bled_codes = nbp_call_spots.bled_codes.astype(np.float32)
    assert np.isnan(bled_codes).sum() == 0, "bled codes cannot contain nan values"
    assert np.allclose(np.linalg.norm(bled_codes, axis=(1, 2)), 1), "bled codes must be L2 normalised"
    solver = PixelScoreSolver()
    bg_bled_codes = solver.create_background_bled_codes(n_rounds_use, n_channels_use)
    max_genes = config["max_genes"]
    solver_kwargs = dict(
        bled_codes=bled_codes,
        background_codes=bg_bled_codes,
        maximum_iterations=max_genes,
        dot_product_threshold=config["dot_product_threshold"],
        minimum_intensity=config["minimum_intensity"],
        alpha=config["alpha"],
        beta=config["beta"],
        force_cpu=config["force_cpu"],
    )
    colour_norm_factor = nbp_call_spots.colour_norm_factor.astype(np.float32)
    n_chunk_max = 600_000

    # Remember the latest OMP config values during a run.
    last_omp_config = omp_config.copy()
    config_path = os.path.join(nbp_file.output_dir, "omp_last_config.pkl")
    if os.path.isfile(config_path):
        with open(config_path, "rb") as config_file:
            last_omp_config = pickle.load(config_file)
    assert type(last_omp_config) is dict
    config_unchanged = omp_config == last_omp_config
    with open(config_path, "wb") as config_file:
        pickle.dump(omp_config, config_file)
    del omp_config, last_omp_config

    mean_spot_filepath = nbp_file.omp_mean_spot
    if mean_spot_filepath is None:
        mean_spot_filepath = importlib_resources.files("coppafish.omp").joinpath("mean_spot.npy")
    mean_spot: np.ndarray = np.load(mean_spot_filepath)
    if not np.issubdtype(mean_spot.dtype, np.floating):
        raise ValueError(f"The mean spot at {mean_spot_filepath} must be a float dtype, got {mean_spot.dtype}")
    if mean_spot.ndim != 3:
        raise ValueError(f"Mean spot must have 3 dimensions, got {mean_spot.ndim}")
    if any([(dim % 2 == 0) and (dim > 0) for dim in mean_spot.shape]):
        raise ValueError(f"Mean spot must have all odd dimension shapes, got {mean_spot.shape}")
    nbp.mean_spot = np.array(mean_spot, np.float32)
    mean_spot = torch.from_numpy(nbp.mean_spot)

    # Every tile's results are appended to a zarr.Group. The zarr group is kept in the output directory until OMP is
    # complete, then it is moved into the 'omp' notebook page.
    group_path = os.path.join(nbp_file.output_dir, "results.zgroup")
    results = zarr.group(store=group_path, zarr_version=2)
    tile_exists = [
        f"tile_{t}" in results
        and "colours" in results[f"tile_{t}"]
        and utils.system.get_software_version() == results[f"tile_{t}"].attrs["software_version"]
        for t in nbp_basic.use_tiles
    ]

    for t_index, t in enumerate(nbp_basic.use_tiles):
        if tile_exists[t_index] and config_unchanged:
            log.info(f"OMP is skipping tile {t}, results already found at {nbp_file.output_dir}")
            continue

        # STEP 1: Gather spot colours and compute OMP pixel scores on the entire tile, one subset at a time.
        device = system.get_device(config["force_cpu"])
        postfix = {"tile": t, "device": str(device).upper()}
        spot_colour_kwargs = dict(
            image=nbp_filter.images,
            flow=nbp_register.flow,
            affine=nbp_register.icp_correction,
            tile=t,
            use_rounds=nbp_basic.use_rounds,
            use_channels=nbp_basic.use_channels,
            output_dtype=np.float32,
            out_of_bounds_value=0,
        )
        log.debug(f"Compute pixel scores, tile {t} started")
        # The tile's pixel score results are stored as a list of scipy sparse matrices. Each item is a specific subset
        # that was run. Appending them all together is done on demand later as it is computationally expensive to do
        # this while as a sparse matrix. Most pixel scores in each row are zeroes (this is because rows go over all
        # genes in the panel, most pixels only assign one or two genes), so a csr matrix is appropriate.
        pixel_scores: list[scipy.sparse.csr_matrix] = []
        index_subset, index_min, index_max = 0, 0, 0
        log.debug(f"OMP {max_genes=}")
        log.debug(f"OMP {n_subset_pixels=}")

        with tqdm.tqdm(total=n_tile_pixels, desc="Computing pixel scores", unit="pixel", postfix=postfix) as pbar:
            while index_min < n_tile_pixels:
                if n_subset_pixels is None:
                    index_max += maths.floor(utils.system.get_available_memory(device) * n_memory_constant)
                else:
                    index_max += n_subset_pixels
                index_max = min(index_max, n_tile_pixels)
                index_max = max(index_max, index_min + 1)

                yxz_subset = yxz_all[index_min:index_max]
                colour_subset = spot_colours_base.get_spot_colours_new_safe(nbp_basic, yxz_subset, **spot_colour_kwargs)
                colour_subset *= colour_norm_factor[[t]]
                intensity = np.abs(colour_subset.copy()).max(2).min(1)
                is_intense = intensity >= config["minimum_intensity"]
                del intensity

                pixel_scores_subset = np.zeros((index_max - index_min, n_genes), np.float32)
                if is_intense.sum() > 0:
                    pixel_scores_subset[is_intense] = solver.solve(colour_subset[is_intense], **solver_kwargs)
                del colour_subset, is_intense

                pixel_scores_subset = scipy.sparse.csr_matrix(pixel_scores_subset)
                pixel_scores.append(pixel_scores_subset.copy())
                del pixel_scores_subset
                pbar.update(index_max - index_min)
                index_min = index_max
                index_subset += 1
        log.debug(f"Compute pixel scores, tile {t} complete")

        tile_results = results.create_group(f"tile_{t}", overwrite=True)
        tile_results.attrs["software_version"] = utils.system.get_software_version()
        t_spots_local_yxz = tile_results.zeros(
            "local_yxz", overwrite=True, shape=(0, 3), chunks=(n_chunk_max, 3), dtype=np.int16
        )
        t_spots_tile = tile_results.zeros("tile", overwrite=True, shape=0, chunks=(n_chunk_max,), dtype=np.int16)
        t_spots_gene_no = tile_results.zeros("gene_no", overwrite=True, shape=0, chunks=(n_chunk_max,), dtype=np.int16)
        t_spots_score = tile_results.zeros("scores", overwrite=True, shape=0, chunks=(n_chunk_max,), dtype=np.float16)

        batch_size = int(2e6 * utils.system.get_available_memory(device) // n_tile_pixels)
        batch_size = max(batch_size, 1)
        log.debug(f"Gene batch size: {batch_size}")
        gene_batches = [
            [g for g in range(b * batch_size, min((b + 1) * batch_size, n_genes))]
            for b in range(maths.ceil(n_genes / batch_size))
        ]
        for gene_batch in tqdm.tqdm(gene_batches, desc="Scoring/detecting spots", unit="gene batch", postfix=postfix):
            # STEP 2: Score every gene's pixel score image.
            g_pixel_image = torch.full((len(gene_batch),) + tile_shape, torch.nan, dtype=torch.float32)
            for g_i, g in enumerate(gene_batch):
                g_pixel_image[g_i] = torch.from_numpy(
                    np.vstack([subset[:, [g]].toarray() for subset in pixel_scores]).reshape(tile_shape, order="F")
                )
            g_score_image = scores.score_pixel_score_image(g_pixel_image, mean_spot, config["force_cpu"])
            del g_pixel_image
            g_score_image = g_score_image.to(dtype=torch.float16)

            # STEP 3: Detect genes as score local maxima.
            for g_i, g in enumerate(gene_batch):
                g_spot_local_positions, g_spot_scores = find_spots.detect.detect_spots(
                    g_score_image[g_i],
                    config["score_threshold"],
                    radius_xy=config["radius_xy"],
                    radius_z=config["radius_z"],
                    remove_duplicates=True,
                )
                g_spot_local_positions = torch.from_numpy(g_spot_local_positions).to(torch.int16)
                g_spot_scores = torch.from_numpy(g_spot_scores)
                n_g_spots = g_spot_scores.size(0)
                if n_g_spots == 0:
                    continue

                # Delete any spot positions that are duplicates.
                g_spot_global_positions = g_spot_local_positions.detach().clone().float()
                g_spot_global_positions += tile_origins[[t]]
                is_duplicate = duplicates.is_duplicate_spot(g_spot_global_positions, t, tile_centres)
                g_spot_local_positions = g_spot_local_positions[~is_duplicate]
                g_spot_scores = g_spot_scores[~is_duplicate]
                del g_spot_global_positions, is_duplicate

                g_spot_scores = g_spot_scores.to(torch.float16)
                n_g_spots = g_spot_scores.size(0)
                if n_g_spots == 0:
                    continue
                log.debug(f"{n_g_spots=}")
                g_spots_tile = torch.full((n_g_spots,), t).to(torch.int16)
                g_spots_gene_no = torch.full((n_g_spots,), g).to(torch.int16)

                # Append new results.
                g_spot_local_positions = g_spot_local_positions.numpy()
                g_spot_scores = g_spot_scores.numpy()
                g_spots_tile = g_spots_tile.numpy()
                g_spots_gene_no = g_spots_gene_no.numpy()
                t_spots_local_yxz.append(g_spot_local_positions, axis=0)
                t_spots_score.append(g_spot_scores, axis=0)
                t_spots_tile.append(g_spots_tile, axis=0)
                t_spots_gene_no.append(g_spots_gene_no, axis=0)
                del g_spot_local_positions, g_spot_scores, g_spots_tile, g_spots_gene_no
        if t_spots_tile.size == 0:
            raise ValueError(
                f"No OMP spots found on tile {t}. Please check that registration and call spots is working. "
                + "If so, consider adjusting OMP config parameters."
            )
        # For each detected spot, save the image intensity at its location, without background fitting.
        log.info("Gathering final spot colours")
        t_local_yxzs = t_spots_local_yxz[:]
        t_spots_colours = tile_results.zeros(
            "colours",
            shape=(t_spots_tile.size, n_rounds_use, n_channels_use),
            chunks=(n_chunk_max, 1, 1),
            dtype=np.float16,
        )
        t_spots_colours[:] = spot_colours_base.get_spot_colours_new_safe(
            nbp_basic, t_local_yxzs, **spot_colour_kwargs
        ).astype(np.float16)
        del t_spots_local_yxz, t_spots_tile, t_spots_gene_no, t_spots_score, t_spots_colours, t_local_yxzs, tile_results
        log.debug("Gathering final spot colours complete")

    os.remove(config_path)

    nbp.results = results
    log.info("OMP complete")

    return nbp
