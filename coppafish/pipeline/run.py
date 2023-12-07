import os
import tqdm
import joblib
import warnings
import numpy as np
from scipy import sparse
import numpy.typing as npt
from typing import Optional, Union

from .. import setup, utils
from ..find_spots import check_spots
from ..call_spots import base as call_spots_base
from .register import preprocessing
from . import basic_info
from . import call_reference_spots
from . import extract_run
from . import scale_run
from . import stitch
from . import find_spots
from . import register
from . import get_reference_spots
from . import omp


def run_pipeline(config_file: str, overwrite_ref_spots: bool = False, parallel: bool = False, n_jobs: int = 8
                 ) -> setup.Notebook:
    """
    Bridge function to run every step of the pipeline.

    Args:
        config_file: Path to config file.
        overwrite_ref_spots: Only used if *Notebook* contains *ref_spots* but not *call_spots* page.
            If `True`, the variables:

            * `gene_no`
            * `score`
            * `score_diff`
            * `intensity`

            in `nb.ref_spots` will be overwritten if they exist. If this is `False`, they will only be overwritten
            if they are all set to `None`, otherwise an error will occur.
        parallel: Boolean, if 'True' will run the pipeline in parallel by splitting the data into tiles and running
            each tile in parallel.
        n_jobs: number of joblib threads to run
    
    Returns:
        Notebook: notebook containing all information gathered during the pipeline.
    """
    nb = initialize_nb(config_file)
    if not parallel:
        run_tile_indep_pipeline(nb)
        run_stitch(nb)
        run_reference_spots(nb, overwrite_ref_spots)
        run_omp(nb)
    else:
        #TODO: Add run_scale before extract is run
        config_files = setup.split_config(config_file)
        nb_list = [initialize_nb(f) for f in config_files]
        joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(run_extract)(n) for n in nb_list)
        nb = setup.merge_notebooks(nb_list, master_nb=nb)
        run_find_spots(nb)
        run_register(nb)
        run_stitch(nb)
        run_register(nb, overwrite_ref_spots)
        run_omp(nb)

    return nb


def run_tile_indep_pipeline(nb: setup.Notebook, run_tile_by_tile: bool = False) -> None:
    """
    Run tile-independent pipeline processes.

    Args:
        nb (Notebook): notebook containing 'basic_info' and 'file_names' pages.
        run_tile_by_tile (bool, optional): run each tile on a separate notebook through 'find_spots' and 'register', 
            then merge them together. Default: false.
    """
    run_scale(nb)
    if run_tile_by_tile and nb.basic_info.n_tiles > 1:
        print("Running tile by tile...")
        # Load one tile image into memory to run in both find_spots and register
        nb_tiles = setup.notebook.split_by_tiles(nb)
        for i, tile in enumerate(nb.basic_info.use_tiles):
            nb_tile = nb_tiles[i]
            #TODO: Instead of extract writing to disk then loading from disk again, make `run_extract` return `image_t`
            # as well as writing to disk. This would be optimal.
            run_extract(nb_tile)
            image_t = utils.tiles_io.load_tile(
                nb.file_names, nb.basic_info, nb_tile.extract.file_type, tile, apply_shift=False, 
            )
            nb_tile = run_find_spots(nb_tile, image_t)
        nb = setup.merge_notebooks(nb_tiles, nb)
        #TODO: Place run_register within the t loop and run tile by tile, inputting image_t as a parameter
        run_register(nb)
    if not run_tile_by_tile:
        run_extract(nb)
        run_find_spots(nb)
        run_register(nb)


def initialize_nb(config_file: str) -> setup.Notebook:

    """
    Quick function which creates a `Notebook` and adds `basic_info` page before saving.
    `file_names` page will be added automatically as soon as `basic_info` page is added.
    If `Notebook` already exists and contains these pages, it will just be returned.

    Args:
        config_file: Path to config file.

    Returns:
        `Notebook` containing `file_names` and `basic_info` pages.
    """
    nb = setup.Notebook(config_file=config_file)

    config = nb.get_config()

    if not nb.has_page("basic_info"):
        nbp_basic = basic_info.set_basic_info_new(config)
        nb += nbp_basic
    else:
        warnings.warn('basic_info', utils.warnings.NotebookPageWarning)
    return nb


def run_scale(nb: setup.Notebook) -> None:
    """
    This runs the `scale` step of the pipeline to produce the scale factors to use during extraction.
    
    `scale` page is added to the `Notebook` before saving.

    Args:
        nb (Notebook): `Notebook` containing `file_names` and `basic_info` pages.
    """
    if not nb.has_page("scale"):
        config = nb.get_config()
        nbp = scale_run.compute_scale(
            config['scale'], nb.file_names, nb.basic_info, 
        )
        nb += nbp
    else:
        warnings.warn('scale', utils.warnings.NotebookPageWarning)
    

def run_extract(nb: setup.Notebook) -> None:
    """
    This runs the `extract_and_filter` step of the pipeline to produce the tiff files in the tile directory.

    `extract` and `extract_debug` pages are added to the `Notebook` before saving.

    If `Notebook` already contains these pages, it will just be returned.

    Args:
        nb: `Notebook` containing `file_names`, `basic_info` and `scale` pages.
    """
    if not all(nb.has_page(["extract", "extract_debug"])):
        config = nb.get_config()
        nbp, nbp_debug = extract_run.extract_and_filter(config['extract'], nb.file_names, nb.basic_info, nb.scale)
        nb += nbp
        nb += nbp_debug
    else:
        warnings.warn('extract', utils.warnings.NotebookPageWarning)
        warnings.warn('extract_debug', utils.warnings.NotebookPageWarning)


def run_find_spots(
    nb: setup.Notebook, image_t: Optional[npt.NDArray[np.uint16]] = None) -> Union[None, setup.NotebookPage]:
    """
    This runs the `find_spots` step of the pipeline to produce point cloud from each tiff file in the tile directory.

    `find_spots` page added to the `Notebook` before saving if image_t is not given.

    If `Notebook` already contains this page, it will just be returned.

    Args:
        nb: `Notebook` containing `extract` page.
        image_t (`(n_rounds x n_channels x ny x nz (x nz)) ndarray[uint16]`, optional): image pixels for tile 
            specified in `nb`. If given, find_spots will be run on this single tile only and return the find_spots 
            NotebookPage, not adding it to the Notebook. Default: not given.
        
    Returns:
        NoteBook containing 'find_spots' page.
    """
    if image_t is not None:
        assert len(nb.basic_info.use_tiles) == 1, "Can only run on a single-tile notebook when `image_t` is given"

    if not nb.has_page("find_spots"):
        config = nb.get_config()
        nbp = find_spots.find_spots(
            config['find_spots'], nb.file_names, nb.basic_info, nb.extract, nb.extract.auto_thresh, image_t, 
        )
        nb += nbp
        # error if too few spots - may indicate tile or channel which should not be included
        check_spots.check_n_spots(nb)
    else:
        warnings.warn('find_spots', utils.warnings.NotebookPageWarning)
    return nb


def run_stitch(nb: setup.Notebook) -> None:
    """
    This runs the `stitch` step of the pipeline to produce origin of each tile
    such that a global coordinate system can be built. Also saves stitched DAPI and reference channel images.

    `stitch` page added to the `Notebook` before saving.

    If `Notebook` already contains this page, it will just be returned.
    If stitched images already exist, they won't be created again.

    Args:
        nb: `Notebook` containing `find_spots` page.
    """
    config = nb.get_config()
    if not nb.has_page("stitch"):
        nbp_debug = stitch.stitch(config['stitch'], nb.basic_info, nb.find_spots.spot_yxz, nb.find_spots.spot_no)
        nb += nbp_debug
    else:
        warnings.warn('stitch', utils.warnings.NotebookPageWarning)
    # Two conditions below:
    # 1. Check if there is a big dapi_image
    # 2. Check if there is NOT a file in the path directory for the dapi image
    if nb.file_names.big_dapi_image is not None and not os.path.isfile(nb.file_names.big_dapi_image):
        # save stitched dapi
        # Will load in from nd2 file if nb.extract_debug.r_dapi is None i.e. if no DAPI filtering performed.
        utils.tiles_io.save_stitched(nb.file_names.big_dapi_image, nb.file_names, nb.basic_info, nb.extract, 
                                nb.stitch.tile_origin, nb.basic_info.anchor_round,
                                nb.basic_info.dapi_channel, nb.extract_debug.r_dapi is None,
                                config['stitch']['save_image_zero_thresh'], config['extract']['num_rotations'])

    if nb.file_names.big_anchor_image is not None and not os.path.isfile(nb.file_names.big_anchor_image):
        # save stitched reference round/channel
        utils.tiles_io.save_stitched(nb.file_names.big_anchor_image, nb.file_names, nb.basic_info, nb.extract, 
                                     nb.stitch.tile_origin, nb.basic_info.anchor_round, nb.basic_info.anchor_channel, 
                                     False, config['stitch']['save_image_zero_thresh'], 
                                     config['extract']['num_rotations'])


def run_register(nb: setup.Notebook, image_t: Optional[npt.NDArray[np.uint16]] = None) -> None:
    """
    This runs the `register_initial` step of the pipeline to find shift between ref round/channel to each imaging round
    for each tile. It then runs the `register` step of the pipeline which uses this as a starting point to get
    the affine transforms to go from the ref round/channel to each imaging round/channel for every tile.

    `register_initial`, `register` and `register_debug` pages are added to the `Notebook` before saving.

    If `Notebook` already contains these pages, it will just be returned.
    
    If `image_t` is given, register will run on the single tile given.

    Args:
        nb: `Notebook` containing `extract` page.
        image_t (`(n_rounds x n_channels x ny x nz (x nz)) ndarray[uint16]`, optional): image for tile 
            `nb[basic_info][use_tile]`. Default: not given.
    """
    if image_t is not None:
        assert len(nb.basic_info.use_tiles) == 1, "Can only run on a single-tile notebook when `image_t` is given"
        NotImplementedError("Cannot run register tile by tile yet")

    config = nb.get_config()
    # if not all(nb.has_page(["register", "register_debug"])):
    if not nb.has_page("register"):
        nbp, nbp_debug = register.register(
            nb.basic_info, 
            nb.file_names, 
            nb.extract, 
            nb.find_spots, 
            config['register'], 
            np.pad(nb.basic_info.tilepos_yx, ((0, 0), (0, 1)), mode='constant', constant_values=1), 
            pre_seq_blur_radius=0, 
        )
        nb += nbp
        nb += nbp_debug
        # Save reg images
        round_registration_channel = config['register']['round_registration_channel']
        for t in nb.basic_info.use_tiles:
            use_rounds_with_preseq = nb.basic_info.use_rounds + [nb.basic_info.pre_seq_round] * nb.basic_info.use_preseq
            with tqdm.tqdm(total=len(use_rounds_with_preseq), desc="Saving registration images") as pbar:
                for r in use_rounds_with_preseq:
                    pbar.set_postfix({'tile': t, 'round': r})
                    if round_registration_channel is not None:
                        preprocessing.generate_reg_images(nb, t, r, round_registration_channel)
                    if round_registration_channel is None:
                        preprocessing.generate_reg_images(nb, t, r, nb.basic_info.anchor_channel)
                    pbar.update(1)
            with tqdm.tqdm(total=len(nb.basic_info.use_channels), desc="Saving registration images") as pbar:
                for c in nb.basic_info.use_channels:
                    pbar.set_postfix({'tile': t, 'channel': c})
                    preprocessing.generate_reg_images(nb, t, 3, c)
                    pbar.update(1)
            if round_registration_channel is not None:
                preprocessing.generate_reg_images(nb, t, nb.basic_info.anchor_round, round_registration_channel)
            preprocessing.generate_reg_images(nb, t, nb.basic_info.anchor_round, nb.basic_info.anchor_channel)
    else:
        warnings.warn('register', utils.warnings.NotebookPageWarning)
        warnings.warn('register_debug', utils.warnings.NotebookPageWarning)


def run_reference_spots(nb: setup.Notebook, overwrite_ref_spots: bool = False) -> None:
    """
    This runs the `reference_spots` step of the pipeline to get the intensity of each spot on the reference
    round/channel in each imaging round/channel. The `call_spots` step of the pipeline is then run to produce the
    `bleed_matrix`, `bled_code` for each gene and the gene assignments of the spots on the reference round.

    `ref_spots` and `call_spots` pages are added to the Notebook before saving.

    If `Notebook` already contains these pages, it will just be returned.

    Args:
        nb: `Notebook` containing `stitch` and `register` pages.
        overwrite_ref_spots: Only used if *Notebook* contains *ref_spots* but not *call_spots* page.
            If `True`, the variables:

            * `gene_no`
            * `score`
            * `score_diff`
            * `intensity`

            in `nb.ref_spots` will be overwritten if they exist. If this is `False`, they will only be overwritten
            if they are all set to `None`, otherwise an error will occur.
    """
    if not nb.has_page('ref_spots'):
        nbp = get_reference_spots.get_reference_spots(nb.file_names, nb.basic_info, nb.find_spots, nb.extract,
                                  nb.stitch.tile_origin, nb.register.transform)
        nb += nbp  # save to Notebook with gene_no, score, score_diff, intensity = None.
                   # These will be added in call_reference_spots
    else:
        warnings.warn('ref_spots', utils.warnings.NotebookPageWarning)
    if not nb.has_page("call_spots"):
        config = nb.get_config()
        nbp, nbp_ref_spots = call_reference_spots.call_reference_spots(config['call_spots'], nb.file_names, 
                                                                       nb.basic_info, nb.ref_spots, nb.extract, 
                                                                       transform=nb.register.transform, 
                                                                       overwrite_ref_spots=overwrite_ref_spots)
        nb += nbp
    else:
        warnings.warn('call_spots', utils.warnings.NotebookPageWarning)


def run_omp(nb: setup.Notebook) -> None:
    """
    This runs the orthogonal matching pursuit section of the pipeline as an alternate method to determine location of
    spots and their gene identity.
    It achieves this by fitting multiple gene bled codes to each pixel to find a coefficient for every gene at
    every pixel. Spots are then local maxima in these gene coefficient images.

    `omp` page is added to the Notebook before saving.

    Args:
        nb: `Notebook` containing `call_spots` page.
    """
    if not nb.has_page("omp"):
        config = nb.get_config()
        # Use tile with most spots on to find spot shape in omp
        spots_tile = np.sum(nb.find_spots.spot_no, axis=(1, 2))
        tile_most_spots = nb.basic_info.use_tiles[np.argmax(spots_tile[nb.basic_info.use_tiles])]
        nbp = omp.call_spots_omp(config['omp'], nb.file_names, nb.basic_info, nb.extract, nb.call_spots, 
                                 nb.stitch.tile_origin, nb.register.transform, tile_most_spots)
        nb += nbp

        # Update omp_info files after omp notebook page saved into notebook
        # Save only non-duplicates - important spot_coefs saved first for exception at start of call_spots_omp
        # which can deal with case where duplicates removed from spot_coefs but not spot_info.
        # After re-saving here, spot_coefs[s] should be the coefficients for gene at nb.omp.local_yxz[s]
        # i.e. indices should match up.
        spot_info = np.load(nb.file_names.omp_spot_info)
        not_duplicate = call_spots_base.get_non_duplicate(nb.stitch.tile_origin, nb.basic_info.use_tiles, 
                                                          nb.basic_info.tile_centre, spot_info[:, :3], spot_info[:, 6])
        spot_coefs = sparse.load_npz(nb.file_names.omp_spot_coef)
        sparse.save_npz(nb.file_names.omp_spot_coef, spot_coefs[not_duplicate])
        np.save(nb.file_names.omp_spot_info, spot_info[not_duplicate])

        # only raise error after saving to notebook if spot_colors have nan in wrong places.
        utils.errors.check_color_nan(nbp.colors, nb.basic_info)
    else:
        warnings.warn('omp', utils.warnings.NotebookPageWarning)
