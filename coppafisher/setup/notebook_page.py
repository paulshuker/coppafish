import copy
import json
import os
import shutil
import tempfile
import time
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import zarr

from ..utils import base as utils_base
from ..utils import system as utils_system


# NOTE: Every method and variable with an underscore at the start should not be accessed externally.
class NotebookPage:
    """
    Every notebook page contains variable names and their associated values.

    A notebook page never communicates with a notebook. But, the notebook communicates with notebook pages. There are
    only a handful of valid page names. Each page can only contain specific variable names, these are stored in the
    attribute _options. Each variable can only hold specific datatypes given in _options. If a variable is set to a
    wrong type, PageTypeError is raised.
    """

    def get_page_name(self) -> str:
        return self._name

    _name: str
    name: str = property(get_page_name)

    # Attribute names allowed to be set inside the notebook page that are not in _options.
    _valid_attribute_names = ("_name", "_time_created", "_version", "_associated_configs")

    _associated_configs: Dict[str, Dict[str, Any]]

    def get_associated_configs(self) -> Dict[str, Dict[str, Any]]:
        return self._associated_configs

    associated_configs: Dict[str, Dict[str, Any]] = property(get_associated_configs)

    _metadata_name: str = "_metadata.json"

    _page_name_key: str = "page_name"

    _time_created: float

    def get_time_created(self) -> float:
        return self._time_created

    time_created: float = property(get_time_created)

    _time_created_key: str = "time_created"
    _version: str
    _version_key: str = "version"
    _associated_config_key: str = "associated_configs"

    def get_version(self) -> str:
        return self._version

    version: str = property(get_version)

    # Each page variable is given a list. The list contains a datatype(s) in the first index followed by a description.
    # A variable can be allowed to take multiple datatypes by separating them with an ' or '. Check the supported
    # types by looking at the function _is_types at the end of this file. The 'tuple' is a special datatype that can be
    # nested. For example, tuple[tuple[int]] is a valid datatype. Also, when a `tuple` type variable is returned by a
    # page, it actually gives a nested `list` instead. This is for backwards compatibility reasons. Modifying the list
    # (like using the `.append` method) will not change the variable saved to disk.
    _datatype_separator: str = " or "
    _datatype_nest_start: str = "["
    _datatype_nest_end: str = "]"
    _options: Dict[str, Dict[str, list]] = {
        "basic_info": {
            "anchor_channel": [
                "int or none",
                "Channel in anchor used. None if anchor not used.",
            ],
            "anchor_round": [
                "int or none",
                "Index of anchor round (typically the first round after imaging rounds so `anchor_round = n_rounds`)."
                + "`None` if anchor not used.",
            ],
            "dapi_channel": [
                "int or none",
                "Channel in anchor round that contains *DAPI* images. `None` if no *DAPI*.",
            ],
            "use_channels": [
                "tuple[int] or none",
                "n_use_channels. Channels in imaging rounds to use throughout pipeline.",
            ],
            "use_rounds": ["tuple[int] or none", "n_use_rounds. Imaging rounds to use throughout pipeline."],
            "use_z": ["tuple[int] or none", "z planes used to make tile *npy* files"],
            "use_tiles": [
                "tuple[int] or none",
                "n_use_tiles tiles to use throughout pipeline."
                + "For an experiment where the tiles are arranged in a $4 \\times 3$ ($n_y \\times n_x$) grid, "
                + "tile indices are indicated as below:"
                + "\n"
                + "| 2  | 1  | 0  |"
                + "\n"
                + "| 5  | 4  | 3  |"
                + "\n"
                + "| 8  | 7  | 6  |"
                + "\n"
                + "| 11 | 10 | 9  |",
            ],
            "use_dyes": ["tuple[int] or none", "n_use_dyes dyes to use when assigning spots to genes."],
            "dye_names": [
                "tuple[str] or none",
                "Names of all dyes so for gene with code $360...$,"
                + "gene appears with `dye_names[3]` in round $0$, `dye_names[6]` in round $1$, `dye_names[0]`"
                + " in round $2$ etc. `none` if each channel corresponds to a different dye.",
            ],
            "is_3d": [
                "bool",
                "`True` if *3D* pipeline used, `False` if *2D*",
            ],
            "channel_camera": [
                "ndarray[int]",
                "`channel_camera[i]` is the wavelength in *nm* of the camera on channel $i$."
                + " Empty array if `dye_names = none`.",
            ],
            "channel_laser": [
                "ndarray[int]",
                "`channel_laser[i]` is the wavelength in *nm* of the laser on channel $i$."
                + "`none` if `dye_names = none`.",
            ],
            "n_extra_rounds": [
                "int",
                "Number of non-imaging rounds, typically 1 if using anchor and 0 if not.",
            ],
            "n_rounds": [
                "int",
                "Number of imaging rounds in the raw data",
            ],
            "tile_sz": [
                "int",
                "$yx$ dimension of tiles in pixels",
            ],
            "n_tiles": [
                "int",
                "Number of tiles in the raw data",
            ],
            "n_channels": [
                "int",
                "Number of channels in the raw data",
            ],
            "nz": [
                "int",
                "Number of z-planes used to make the *npy* tile images (can be different from number in raw data).",
            ],
            "n_dyes": [
                "int",
                "Number of dyes used",
            ],
            "tile_centre": [
                "ndarray[float]",
                "`[y, x, z]` location of tile centre in units of `[yx_pixels, yx_pixels, z_pixels]`."
                + "For *2D* pipeline, `tile_centre[2] = 0`",
            ],
            "tilepos_yx_nd2": [
                "ndarray[int]",
                "[n_tiles x 2] `tilepos_yx_nd2[i, :]` is the $yx$ position of tile with *fov* index $i$ in the *nd2*"
                + "file. Index 0 refers to `YX = [0, 0]`"
                + "Index 1 refers to `YX = [0, 1]` if `MaxX > 0`. "
                + "The order can be changed based on config section basic_info options `reverse_tile_positions_x` and "
                + "`reverse_tile_positions_y`.",
            ],
            "tilepos_yx": [
                "ndarray[int]",
                "[n_tiles x 2] `tilepos_yx[i, :]` is the $yx$ position of tile with tile directory (*npy* files) "
                + "index $i$. Equally, `tilepos_yx[use_tiles[i], :]` is $yx$ position of tile `use_tiles[i]`."
                + "Index 0 refers to `YX = [MaxY, MaxX]`"
                + "Index 1 refers to `YX = [MaxY, MaxX - 1]` if `MaxX > 0`",
            ],
            "pixel_size_xy": [
                "float",
                "$yx$ pixel size in microns",
            ],
            "pixel_size_z": [
                "float",
                "$z$ pixel size in microns",
            ],
            "use_anchor": [
                "bool",
                "whether or not to use anchor",
            ],
            "bad_trc": [
                "tuple[tuple[int]] or none",
                "Tuple of bad tile, round, channel combinations. If a tile, round, channel combination is in this,"
                + "it will not be used in the pipeline.",
            ],
        },
        "file_names": {
            "input_dir": [
                "dir",
                "Where raw *nd2* files are",
            ],
            "output_dir": [
                "dir",
                "Where notebook is saved",
            ],
            "extract_dir": [
                "dir",
                "Where extract, unfiltered image files are saved",
            ],
            "round": [
                "tuple[file]",
                "n_rounds names of *nd2* files for the imaging rounds. If not using, will be an empty list.",
            ],
            "anchor": [
                "str or none",
                "Name of *nd2* file for the anchor round. `none` if anchor not used",
            ],
            "raw_extension": [
                "str",
                "*.nd2* or *.npy* indicating the data type of the raw data.",
            ],
            "raw_metadata": [
                "str or none",
                "If `raw_extension = .npy`, this is the name of the *json* file in `input_dir` which contains the "
                + "required metadata extracted from the initial *nd2* files."
                + "I.e. it is the output of *coppafisher/utils/nd2/save_metadata*",
            ],
            "code_book": [
                "file",
                "Text file which contains the codes indicating which dye to expect on each round for each gene",
            ],
            "psf": [
                "file",
                "*npy* file location indicating the average spot shape" + "This will have the shape `n_z x n_y x n_x`.",
            ],
            "tile_unfiltered": [
                "tuple[tuple[tuple[file]]]",
                "List of string arrays [n_tiles][(n_rounds + n_extra_rounds) {x n_channels if 3d}]"
                + "`tile[t][r][c]` is the [extract][file_type] unfiltered file containing all z planes for tile $t$, "
                + "round $r$, channel $c$",
            ],
            "fluorescent_bead_path": [
                "str or none",
                "Path to *nd2* file containing fluorescent beads. `none` if not used.",
            ],
            "initial_bleed_matrix": [
                "dir or none",
                "Location of initial bleed matrix file. If `none`, then use the default bleed matrix",
            ],
            "omp_mean_spot": [
                "file or none",
                "Location of the OMP mean spot .npy file. If `none`, then the default mean spot is used",
            ],
        },
        "extract": {
            "num_rotations": [
                "int",
                "The number of 90 degree anti-clockwise rotations applied to every image.",
            ],
        },
        "filter": {
            "images": [
                "zarray[float16]",
                "`(n_tiles x (n_rounds + n_extra_rounds) x n_channels x tile_sz x tile_sz x len(use_z))` zarray float16. "
                + "All microscope images after filtering (deblurring) is applied.",
            ],
        },
        "filter_debug": {
            "psf": [
                "ndarray[float]",
                "Numpy float array [psf_shape[0] x psf_shape[1] x psf_shape[2]] or None (psf_shape is in config file)"
                + "Average shape of spot from individual raw spot images normalised so max is 1 and min is 0."
                + "`None` if not applying the Wiener deconvolution.",
            ],
        },
        "find_spots": {
            "auto_thresh": [
                "ndarray[float32]",
                "`(n_tiles x (n_rounds + n_extra_rounds) x n_channels) ndarray`"
                + "`auto_thresh[t, r, c]` is the intensity threshold for tile $t$, round $r$, channel $c$ "
                + "used for spot detection.",
            ],
            "spot_no": [
                "ndarray[int32]",
                "Numpy array [n_tiles x (n_rounds + n_extra_rounds) x n_channels]"
                + "`spot_no[t, r, c]` is the number of spots found on tile $t$, round $r$, channel $c$",
            ],
            "spot_yxz": [
                "zgroup",
                "A zarr group containing all spot detection positions relative to their tile's bottom left corner. For "
                + "each tile/round/channel combination, a `(n_trc_spots x 3) zarray[int16]` is stored. Each "
                + "tile/round/channel index is uniquely saved. For example, tile 0, round 1, channel 2 is labelled "
                + "t0r1c2 so it can be gathered as ndarray[int1e] by nb.find_spots.spot_yxz['t0r1c2'][:]",
            ],
        },
        "stitch": {
            "tile_origin": [
                "ndarray[float]",
                "Numpy array (n_tiles x 3)"
                + "`tile_origin[t,:]` is the bottom left $yxz$ coordinate of tile $t$."
                + "$yx$ coordinates in yx-pixels and z coordinate in z-pixels."
                + "nan is populated in places where a tile is not used in the pipeline.",
            ],
            "shifts": [
                "ndarray[float]",
                "Numpy array (n_tiles x n_tiles x 3)"
                + "`shifts[t1, t2, :]` is the $yxz$ shift from tile $t1$ to tile $t2$."
                + "Zeros are populated in places where shift is not calculated, i.e. if tiles are not adjacent,"
                + "or if one of the tiles is not used in the pipeline.",
            ],
            "scores": [
                "ndarray[float]",
                "Numpy array [n_tiles x n_tiles]"
                + "`scores[t1, t2]` is the score of the shift from tile $t1$ to tile $t2$. "
                + "Zeros are populated in places where the shift is not calculated, i.e. if tiles are not adjacent "
                + "or if one of the tiles is not used in the pipeline.",
            ],
            "dapi_image": [
                "zarray[float16]",
                "float16 array (im_z x im_y x im_x). "
                + "Fused large dapi image created by merging all tiles together after stitch shifting is applied.",
            ],
            "anchor_image": [
                "zarray[float16]",
                "float16 array (im_z x im_y x im_x). "
                + "Fused large anchor image created by merging all tiles together after stitch shifting is applied.",
            ],
        },
        "register": {
            "flow": [
                "zarray[float16]",
                "n_tiles x n_rounds x 3 x tile_sz x tile_sz x len(use_z)",
                "The optical flow shifts for each image pixel after smoothing. The third axis is for the different "
                + "image directions. 0 is the y shifts, 1 is the x shifts, 2 is the z shifts. "
                + "flow[t, r] takes the anchor image to t/r image.",
            ],
            "correlation": [
                "zarray[float16]",
                "n_tiles x n_rounds x tile_sz x tile_sz x len(use_z)",
                "The optical flow correlations.",
            ],
            "flow_raw": [
                "zarray[float16]",
                "n_tiles x n_rounds x 3 x tile_sz x tile_sz x len(use_z)",
                "The optical flow shifts for each image pixel before smoothing. The third axis is for the different "
                + "image directions. 0 is the y shifts, 1 is the x shifts, 2 is the z shifts.",
            ],
            "icp_correction": [
                "ndarray[float64]",
                "Numpy float array [n_tiles x n_rounds x n_channels x 4 x 3]"
                + "yxz affine corrections to be applied after the warp.",
            ],
            "anchor_images": [
                "zarray[uint8]",
                "Numpy uint8 array `(n_tiles x 2 x im_y x im_x x im_z)`"
                + "A subset of the anchor image after all image registration is applied. "
                + "The second axis is for the channels. 0 is the dapi channel, 1 is the anchor reference channel.",
            ],
            "round_images": [
                "zarray[uint8]",
                "Numpy uint8 array `(n_tiles x n_rounds x 3 x im_y x im_x x im_z)`"
                + "A subset of the anchor image after all image registration is applied. "
                + "The third axis is for the registration step. 0 is before register, 1 is after optical flow, 2 is "
                + "after optical flow and ICP",
            ],
            "channel_images": [
                "zarray[uint8]",
                "Numpy uint8 array `(n_tiles x n_channels x 3 x im_y x im_x x im_z)`"
                + "The third axis is for the registration step. 0 is before register, 1 is after optical flow, 2 is "
                + "after optical flow and ICP",
            ],
        },
        "register_debug": {
            "channel_transform_initial": [
                "ndarray[float]",
                "Numpy float array [n_channels x 4 x 3]"
                + "Initial affine transform to go from the ref round/channel to each imaging channel.",
            ],
            "round_correction": [
                "ndarray[float]",
                "Numpy float array [n_tiles x n_rounds x 4 x 3]"
                + "yxz affine corrections to be applied after the warp.",
            ],
            "channel_correction": [
                "ndarray[float]",
                "Numpy float array [n_tiles x n_channels x 4 x 3]"
                + "yxz affine corrections to be applied after the warp.",
            ],
            "n_matches_round": [
                "ndarray[int]",
                "Numpy int array [n_tiles x n_rounds x n_icp_iters]"
                + "Number of matches found for each iteration of icp for the round correction.",
            ],
            "mse_round": [
                "ndarray[float]",
                "Numpy float array [n_tiles x n_rounds x n_icp_iters]"
                + "Mean squared error for each iteration of icp for the round correction.",
            ],
            "converged_round": [
                "ndarray[bool]",
                "Numpy boolean array [n_tiles x n_rounds]"
                + "Whether the icp algorithm converged for the round correction.",
            ],
            "n_matches_channel": [
                "ndarray[int]",
                "Numpy int array [n_tiles x n_channels x n_icp_iters]"
                + "Number of matches found for each iteration of icp for the channel correction.",
            ],
            "mse_channel": [
                "ndarray[float]",
                "Numpy float array [n_tiles x n_channels x n_icp_iters]"
                + "Mean squared error for each iteration of icp for the channel correction.",
            ],
            "converged_channel": [
                "ndarray[bool]",
                "Numpy boolean array [n_tiles x n_channels]"
                + "Whether the icp algorithm converged for the channel correction.",
            ],
        },
        "ref_spots": {
            "local_yxz": [
                "zarray[int16]",
                "Numpy array [n_spots x 3]. "
                + "`local_yxz[s]` are the $yxz$ coordinates of spot $s$ found on `tile[s]`, `ref_round`, `ref_channel`."
                + "To get `global_yxz`, add `nb.stitch.tile_origin[tile[s]]`.",
            ],
            "tile": [
                "zarray[int16]",
                "Numpy array [n_spots]. Tile each spot was found on.",
            ],
            "colours": [
                "zarray[float32]",
                "Numpy array [n_spots x n_rounds x n_channels]. "
                + "`[s, r, c]` is the intensity of spot $s$ in round $r$, channel $c$.",
            ],
        },
        "call_spots": {
            "gene_names": [
                "ndarray[str]",
                "Numpy string array [n_genes]" + "Names of all genes in the code book provided.",
            ],
            "gene_codes": [
                "ndarray[int32]",
                "Numpy integer array [n_genes x n_rounds]"
                + "`gene_codes[g, r]` indicates the dye that should be present for gene $g$ in round $r$.",
            ],
            "colour_norm_factor": [
                "ndarray[float32]",
                "Numpy float array [n_tiles x n_rounds x n_channels_use]"
                + "Normalisation factor for each tile, round, channel. This is multiplied by colours to equalise "
                "intensities across tiles, rounds and channels and to make the intensities of each dye as close as "
                "possible to pre-specified target values.",
            ],
            "initial_scale": [
                "ndarray[float32]",
                "Numpy float array [n_tiles x n_rounds x n_channels_use]"
                + "Initial scaling factor for each tile, round, channel. This is multiplied by colours to equalise "
                "intensities across tiles, rounds and channels.",
            ],
            "rc_scale": [
                "ndarray[float32]",
                "Numpy float array [n_rounds x n_channels_use]"
                + "colour norm factor is a product of 2 scales. The first is the target scale which is the scale "
                + "that maximises similarity between tile independent free bled codes and the target values",
            ],
            "tile_scale": [
                "ndarray[float32]",
                "Numpy float array [n_tiles x n_rounds x n_channels_use]"
                + "colour norm factor is a product of 2 scales. The second is the homogeneous scale which is the "
                + "scale that maximises similarity between tile dependent free bled codes and the target bled codes. "
                "In doing so, we make the tile dependent codes as close as possible to the tile independent codes "
                "(ie: we homogenise these codes).",
            ],
            "free_bled_codes": [
                "ndarray[float32]",
                "Numpy float array [n_genes x n_tiles x n_rounds x n_channels_use]"
                + "free_bled_codes[g, t] is approximately the mean of all spots assigned to gene g in tile t with high "
                "probability. It is not quite the mean because we have a prior that the channel vector for each "
                "round will be mostly parallel to the expected dye code for that gene in that round, so this is "
                "taken into account.",
            ],
            "free_bled_codes_tile_independent": [
                "ndarray[float32]",
                "Numpy float array [n_genes x n_rounds x n_channels_use]"
                + "Tile independent free bled codes. free_bled_codes_tile_independent[g] is approximately the mean "
                "of all spots assigned to gene g in all tiles with high probability. It is not quite the mean because"
                " we have a prior that the channel vector for each round will be mostly parallel to "
                "the expected dye code for that gene in that round, so this is taken into account.",
            ],
            "bled_codes": [
                "ndarray[float32]",
                "Numpy float array [n_genes x n_rounds x n_channels_use]"
                + "bled_codes[g, r, c] = target_scale[r, c] * free_bled_codes_tile_independent[g, r, c], "
                "meaning that these codes are scaled versions of the tile independent free bled codes that are "
                "scaled to make the intensities of each dye as close as possible to pre-specified target values.",
            ],
            "bleed_matrix_raw": [
                "ndarray[float32]",
                "Numpy float array [n_dyes x n_channels_use]"
                + "These are the dye codes obtained from an image of each dye alone, outside of any tissue.",
            ],
            "bleed_matrix_initial": [
                "ndarray[float32]",
                "Numpy float array [n_dyes x n_channels_use]"
                + "bleed_matrix_initial[d] is a vector of length n_channels_use that gives the expected intensity of "
                "dye d in each channel. This initial guess is obtained from a SVD of spots which belong to dye d "
                "with high probability. It differs from the final bleed matrix by a scale factor and by the spots "
                " used to calculate it.",
            ],
            "bleed_matrix": [
                "ndarray[float32]",
                "Numpy float array [n_dyes x n_channels_use]"
                + "bleed_matrix[d] is a vector of length n_channels_use that gives the expected intensity of dye d in "
                "each channel. This is the final bleed matrix and is obtained by computing the probabilities of "
                " scaled spots against the target bled codes.",
            ],
            "dot_product_gene_no": [
                "zarray[int16]",
                "Numpy array [n_spots]. Gene number assigned to each spot. `None` if not assigned.",
            ],
            "dot_product_gene_score": [
                "zarray[float16]",
                "Numpy float array [n_spots]. `score[s]' is the highest gene coef of spot s.",
            ],
            "gene_probabilities": [
                "zarray[float32]",
                "Numpy float array [n_spots x n_genes]. `gene_probabilities[s, g]` is the probability that spot $s$ "
                + "belongs to gene $g$.",
            ],
            "gene_probabilities_initial": [
                "zarray[float32]",
                "Numpy float array [n_spots x n_genes]. `gene_probabilities_initial[s, g]` is the probability that spot"
                + " $s$ belongs to gene $g$ after only initial scaling compared against the raw bleed matrix.",
            ],
            "intensity": [
                "zarray[float32]",
                "Numpy float32 array [n_spots]. "
                + "$\\chi_s = \\underset{r}{\\mathrm{median}}(\\max_c\\zeta_{s_{rc}})$"
                + "where $\\pmb{\\zeta}_s=$ `colors[s, r]*colour_norm_factor[r]`.",
            ],
        },
        "omp": {
            "mean_spot": [
                "ndarray[float32]",
                "Numpy float32 array [im_y x im_x x im_z]. "
                + "The mean spot used to compute the final OMP spot scores",
            ],
            "results": [
                "zgroup",
                "A zarr group containing all OMP spots. Each tile's results are separated into subgroups. "
                + "For example, you can access tile 0's subgroup by doing `nb.omp.results['tile_0']`. Each tile "
                + "subgroup contains 4 zarr arrays: local_yxz, scores, gene_no, and colours. Each has dtype int16, "
                + "float16, int16, and float16 respectively. Each has shape (n_spots, 3), (n_spots), (n_spots), "
                + "(n_spots, n_rounds_use, n_channels_use) respectively. "
                + "To gather tile 0's spot's local_yxz's into memory, do `nb.omp.results['tile_0/local_yxz'][:]`. "
                + "The local_yxz positions are relative to the tile. Converting these to global spot positions "
                + "requires adding the tile_origin from the 'stitch' page.",
            ],
        },
        "thresholds": {
            "intensity": [
                "float",
                "Final accepted reference and OMP spots require `intensity > thresholds[intensity]`."
                + "This is copied from `config[thresholds]` and if not given there, will be set to "
                + "`nb.call_spots.gene_efficiency_intensity_thresh`."
                + "intensity for a really intense spot is about 1 so intensity_thresh should be less than this.",
            ],
            "score_ref": [
                "float",
                "Final accepted reference spots are those which pass `quality_threshold` which is:"
                + ""
                + "`nb.ref_spots.score > thresholds[score_ref]` and `intensity > thresholds[intensity]`."
                + ""
                + "This is copied from `config[thresholds]`."
                + "Max score is 1 so `score_ref` should be less than this.",
            ],
            "score_omp": [
                "float",
                "Final accepted *OMP* spots are those which pass `quality_threshold` which is:"
                + ""
                + "`score > thresholds[score_omp]` and `intensity > thresholds[intensity]`."
                + ""
                + "`score` is given by:"
                + ""
                + "`score = (score_omp_multiplier * n_neighbours_pos + n_neighbours_neg) / "
                + "(score_omp_multiplier * n_neighbours_pos_max + n_neighbours_neg_max)`."
                + ""
                + "This is copied from `config[thresholds]`."
                + "Max score is 1 so `score_thresh` should be less than this.",
            ],
            "score_omp_multiplier": [
                "float",
                "Final accepted OMP spots are those which pass quality_threshold which is:"
                + ""
                + "`score > thresholds[score_omp]` and `intensity > thresholds[intensity]`."
                + ""
                + "score is given by:"
                + ""
                + "`score = (score_omp_multiplier * n_neighbours_pos + n_neighbours_neg) / "
                + "(score_omp_multiplier * n_neighbours_pos_max + n_neighbours_neg_max)`."
                + ""
                + "This is copied from `config[thresholds]`.",
            ],
        },
        # For unit testing only.
        "debug": {
            "a": ["int"],
            "b": ["float"],
            "c": ["bool"],
            "d": ["tuple[int]"],
            "e": ["tuple[tuple[float]]"],
            "f": ["int or float"],
            "g": ["none"],
            "h": ["float or none"],
            "i": ["str"],
            "j": ["ndarray[float]"],
            "k": ["ndarray[int]"],
            "l": ["ndarray[bool]"],
            "m": ["ndarray[str]"],
            "n": ["ndarray[uint]"],
            "o": ["zarray[float]"],
            "p": ["zgroup"],
        },
    }
    _type_suffixes: Dict[str, str] = {
        "int": ".json",
        "float": ".json",
        "str": ".json",
        "bool": ".json",
        "file": ".json",
        "dir": ".json",
        "tuple": ".json",
        "none": ".json",
        "ndarray": ".npz",
        "zarray": ".zarray",
        "zgroup": ".zgroup",
    }

    def __init__(self, page_name: str, associated_config: Dict[str, Dict[str, Any]] = None) -> None:
        """
        Initialise a new, empty notebook page.

        Args:
            page_name (str): the notebook page name. Must exist within _options in the notebook page class.
            associated_config (dict, optional): dictionary containing string keys of config section names. Values are
                the config's dictionary. Default: empty dictionary.

        Notes:
            - The way that the notebook handles zarr arrays is special since they must not be kept in memory. To give
                the notebook page a zarr variable, you must give a zarr.Array class for the array. The array must be
                kept on disk, so you can save the array anywhere to disk initially that is outside of the
                notebook/notebook page. Then, when the notebook page is complete and saved, the zarr array is moved by
                the page into the page's directory. Therefore, a zarr array is never put into memory. When an existing
                zarr array is accessed in a page, it gives you the zarr.Array class, which can then be put into memory
                as a numpy array when indexed.
        """
        if associated_config is None:
            associated_config = {}
        assert type(associated_config) is dict
        for key in associated_config:
            assert type(key) is str
            assert type(associated_config[key]) is dict
            for subkey in associated_config[key]:
                assert type(subkey) is str

        if page_name not in self._options.keys():
            raise ValueError(f"Could not find _options for page called {page_name}")
        self._name = page_name
        self._time_created = time.time()
        self._version = utils_system.get_software_version()
        self._associated_configs = copy.deepcopy(associated_config)
        self._sanity_check_options()

    def save(self, page_directory: str, /) -> None:
        """
        Save the notebook page to the given directory. If the directory already exists, do not overwrite it.
        """
        if os.path.isdir(page_directory):
            return
        if len(self.get_unset_variables()) > 0:
            raise ValueError(
                f"Cannot save unfinished page {self._name}. "
                + f"Variable(s) {self._get_unset_variables()} not assigned yet."
            )

        os.mkdir(page_directory)
        metadata_path = self._get_metadata_path(page_directory)
        self._save_metadata(metadata_path)
        for name in self._get_variables().keys():
            value = self.__getattribute__(name)
            types_as_str: str = self._get_variables()[name][0]
            self._save_variable(name, value, types_as_str, page_directory)

    def load(self, page_directory: str, /) -> None:
        """
        Load all variables from inside the given directory. All variables already set inside of the page are
        overwritten.
        """
        if not os.path.isdir(page_directory):
            raise FileNotFoundError(f"Could not find page directory at {page_directory} to load from")

        metadata_path = self._get_metadata_path(page_directory)
        self._load_metadata(metadata_path)
        for name in self._get_variables().keys():
            self.__setattr__(name, self._load_variable(name, page_directory))

    def get_unset_variables(self) -> Tuple[str]:
        """
        Return a tuple of all variable names that have not been set to a valid value in the notebook page.
        """
        unset_variables = []
        for variable_name in self._get_variables().keys():
            try:
                self.__getattribute__(variable_name)
            except AttributeError:
                unset_variables.append(variable_name)
        return tuple(unset_variables)

    def resave(self, page_directory: str, /) -> None:
        """
        Re-save all variables in the given page directory based on the variables in memory.
        """
        assert type(page_directory) is str
        if not os.path.isdir(page_directory):
            raise SystemError(f"No page directory at {page_directory}")
        if len(os.listdir(page_directory)) == 0:
            raise SystemError(f"Page directory at {page_directory} is empty")
        if len(self.get_unset_variables()) > 0:
            raise ValueError(
                f"Cannot re-save a notebook page at {page_directory} when it has not been completed yet. "
                + f"The variable(s) {', '.join(self.get_unset_variables())} are not assigned."
            )

        temp_directories: List[tempfile.TemporaryDirectory] = []
        for variable_name, description in self._get_variables().items():
            suffix = self._type_str_to_suffix(description[0].split(self._datatype_separator)[0])
            variable_path = self._get_variable_path(page_directory, variable_name, suffix)

            if suffix in (".zarray", ".zgroup"):
                # Zarr files are saved outside the page during re-save as they are not kept in memory.
                temp_directory = tempfile.TemporaryDirectory()
                temp_zarr_path = os.path.join(temp_directory.name, f"{variable_name}.{suffix}")
                temp_directories.append(temp_directory)
                shutil.move(variable_path, temp_zarr_path)
                if suffix == ".zarray":
                    self.__setattr__(variable_name, zarr.open_array(temp_zarr_path))
                elif suffix == ".zgroup":
                    self.__setattr__(variable_name, zarr.open_group(temp_zarr_path))
                continue

            os.remove(variable_path)

        shutil.rmtree(page_directory)
        self.save(page_directory)
        for temp_directory in temp_directories:
            temp_directory.cleanup()

    def __gt__(self, variable_name: str) -> None:
        """
        Print a variable's description by doing `notebook_page > "variable_name"`.
        """
        assert type(variable_name) is str

        if variable_name not in self._get_variables().keys():
            print(f"No variable named {variable_name}")
            return

        variable_desc = "No description"
        valid_types = self._get_expected_types(variable_name)
        if len(self._get_variables()[variable_name]) > 1:
            variable_desc = "".join(self._get_variables()[variable_name][1:])
        print(f"Variable {variable_name}:")
        print(f"\tPage: {self._name}")
        print(f"\tValid type(s): {valid_types}")
        print(f"\tDescription: {variable_desc}")

    def __setattr__(self, name: str, value: Any, /) -> None:
        """
        Deals with syntax `notebook_page.name = value`.
        """
        if name in self._valid_attribute_names:
            object.__setattr__(self, name, value)
            return

        if name not in self._get_variables().keys():
            raise NameError(f"Cannot set variable {name} in {self._name} page. It is not inside _options")
        expected_types = self._get_expected_types(name)
        if not self._is_types(value, expected_types):
            added_msg = ""
            if type(value) is np.ndarray or type(value) is zarr.Array:
                added_msg += f" with dtype {value.dtype.type}"
            msg = f"Cannot set variable {name} to type {type(value)}{added_msg}. Expected type(s) {expected_types}"
            raise PageTypeError(msg)

        object.__setattr__(self, name, value)

    def __getattribute__(self, name: str, /) -> Any:
        """
        Deals with syntax 'value = notebook_page.name' when `name` exists in the page already.
        """
        result = object.__getattribute__(self, name)
        if type(result) is tuple:
            result = utils_base.deep_convert(result, list)
        elif type(result) is np.ndarray:
            result = result.copy()
        return result

    def get_variable_count(self) -> int:
        return len(self._get_variables())

    def _get_variables(self) -> Dict[str, List[str]]:
        # Variable refers to variables that are set during the pipeline, not metadata.
        return self._options[self._name]

    def _save_metadata(self, file_path: str) -> None:
        if os.path.isfile(file_path):
            raise SystemError(f"Metadata file at {file_path} already exists")

        metadata = {
            self._page_name_key: self._name,
            self._time_created_key: self._time_created,
            self._version_key: self._version,
            self._associated_config_key: self._associated_configs,
        }
        with open(file_path, "x") as file:
            file.write(json.dumps(metadata, indent=4))

    def _load_metadata(self, file_path: str) -> None:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Metadata file at {file_path} not found")

        metadata: dict[str, Any] = {}
        with open(file_path, "r") as file:
            metadata = json.loads(file.read())
            assert type(metadata) is dict
        self._name = metadata[self._page_name_key]
        self._time_created = metadata[self._time_created_key]
        self._version = metadata[self._version_key]
        self._associated_configs = metadata[self._associated_config_key]

    def _get_metadata_path(self, page_directory: str) -> str:
        return os.path.join(page_directory, self._metadata_name)

    def _get_page_directory(self, in_directory: str) -> str:
        return os.path.join(in_directory, self._name)

    def _get_expected_types(self, name: str) -> str:
        return self._get_variables()[name][0]

    def _save_variable(self, name: str, value: Any, types_as_str: str, page_directory: str) -> None:
        file_suffix = self._type_str_to_suffix(types_as_str.split(self._datatype_separator)[0])
        new_path = self._get_variable_path(page_directory, name, file_suffix)

        if file_suffix == ".json":
            with open(new_path, "x") as file:
                file.write(json.dumps({"value": value}, indent=4))
        elif file_suffix == ".npz":
            value.setflags(write=False)
            np.savez_compressed(new_path, value)
        elif file_suffix == ".zarray":
            if type(value) is not zarr.Array:
                raise PageTypeError(f"Variable {name} is of type {type(value)}, expected zarr.Array")
            old_path = os.path.abspath(value.store.path)
            shutil.move(old_path, new_path)
            new_array = zarr.open_array(store=new_path, mode="r+")
            new_array.read_only = True
            self.__setattr__(name, new_array)
        elif file_suffix == ".zgroup":
            if type(value) is not zarr.Group:
                raise PageTypeError(f"Variable {name} is of type {type(value)}, expected zarr.Group")
            old_path = os.path.abspath(value.store.path)
            shutil.move(old_path, new_path)
            new_group = zarr.open_group(store=new_path, mode="r")
            self.__setattr__(name, new_group)
        else:
            raise NotImplementedError(f"File suffix {file_suffix} is not supported")

    def _load_variable(self, name: str, page_directory: str) -> Any:
        types_as_str = self._get_variables()[name][0].split(self._datatype_separator)
        file_suffix = self._type_str_to_suffix(types_as_str[0])
        file_path = self._get_variable_path(page_directory, name, file_suffix)
        if not os.path.exists(file_path):
            raise SystemError(f"Failed to find variable path: {file_path}")

        if file_suffix == ".json":
            with open(file_path, "r") as file:
                value = json.loads(file.read())["value"]
                # A JSON file does not support saving tuples, they must be converted back to tuples here.
                if type(value) is list:
                    value = utils_base.deep_convert(value)
            return value
        elif file_suffix == ".npz":
            return np.load(file_path)["arr_0"]
        elif file_suffix == ".zarray":
            return zarr.open_array(file_path)
        elif file_suffix == ".zgroup":
            return zarr.open_group(file_path)
        else:
            raise NotImplementedError(f"File suffix {file_suffix} is not supported")

    def _get_variable_path(self, page_directory: str, variable_name: str, suffix: str) -> str:
        assert type(page_directory) is str
        assert type(variable_name) is str
        assert type(suffix) is str

        return str(os.path.abspath(os.path.join(page_directory, f"{variable_name}{suffix}")))

    def _sanity_check_options(self) -> None:
        # Only multiple datatypes can be options for the same variable if they save to the same save file type. So, a
        # variable's type cannot be "ndarray[int] or zarr" because they save into different file types.
        for page_name, page_options in self._options.items():
            for var_name, var_list in page_options.items():
                unique_suffixes = set()
                types_as_str = var_list[0]
                for type_as_str in types_as_str.split(self._datatype_separator):
                    unique_suffixes.add(self._type_str_to_suffix(type_as_str))
                if len(unique_suffixes) > 1:
                    raise PageTypeError(
                        f"Variable {var_name} in page {page_name} has incompatible types: "
                        + f"{' and '.join(unique_suffixes)} in _options"
                    )

    def _type_str_to_suffix(self, type_as_str: str) -> str:
        return self._type_suffixes[type_as_str.split(self._datatype_nest_start)[0]]

    def _is_types(self, value: Any, types_as_str: str) -> bool:
        valid_types: List[str] = types_as_str.split(self._datatype_separator)
        for type_str in valid_types:
            if self._is_type(value, type_str):
                return True
        return False

    def _is_type(self, value: Any, type_as_str: str) -> bool:
        if self._datatype_separator in type_as_str:
            raise ValueError(f"Type {type_as_str} in _options cannot contain the phrase {self._datatype_separator}")

        if type_as_str == "none":
            return value is None
        elif type_as_str == "int":
            return type(value) is int
        elif type_as_str == "float":
            return type(value) is float
        elif type_as_str == "str":
            return type(value) is str
        elif type_as_str == "bool":
            return type(value) is bool
        elif type_as_str == "file":
            return type(value) is str
        elif type_as_str == "dir":
            return type(value) is str
        elif type_as_str == "tuple":
            return type(value) is tuple
        elif type_as_str.startswith("tuple"):
            if type(value) is not tuple:
                return False
            if len(value) == 0:
                return True
            else:
                for subvalue in value:
                    if not self._is_type(
                        subvalue, type_as_str[len("tuple" + self._datatype_nest_start) : -len(self._datatype_nest_end)]
                    ):
                        return False
                return True
        elif type_as_str.startswith("ndarray"):
            return self._is_ndarray_of_dtype(value, self._get_dtypes_in_type_str(type_as_str))
        elif type_as_str.startswith("zarray"):
            return self._is_zarray_of_dtype(value, self._get_dtypes_in_type_str(type_as_str))
        elif type_as_str == "zgroup":
            return type(value) is zarr.Group
        else:
            raise PageTypeError(f"Unexpected type '{type_as_str}' found in _options in NotebookPage class")

    def _is_ndarray_of_dtype(self, variable: Any, valid_dtypes: Tuple[np.dtype], /) -> bool:
        assert type(valid_dtypes) is tuple

        return type(variable) is np.ndarray and isinstance(variable.dtype.type(), valid_dtypes)

    def _is_zarray_of_dtype(self, variable: Any, valid_dtypes: Tuple[np.dtype], /) -> bool:
        assert type(valid_dtypes) is tuple

        return type(variable) is zarr.Array and isinstance(variable.dtype.type(), valid_dtypes)

    def _get_dtypes_in_type_str(self, type_as_str: str) -> Tuple[Union[np.dtype, str, bool]]:
        assert type(type_as_str) is str

        if not (self._datatype_nest_start in type_as_str and self._datatype_nest_end in type_as_str):
            raise ValueError(
                f"Type {type_as_str} needs a dtype between {self._datatype_nest_start} and {self._datatype_nest_end}"
            )
        dtype_in_str = type_as_str[type_as_str.index(self._datatype_nest_start) + 1 :]
        dtype_in_str = dtype_in_str[: dtype_in_str.index(self._datatype_nest_end)]
        # TODO: Remove support for ambiguous "int", "uint", and "float" numpy dtypes.
        if dtype_in_str == "int":
            return (np.int16, np.int32, np.int64)
        elif dtype_in_str == "int16":
            return (np.int16,)
        elif dtype_in_str == "int32":
            return (np.int32,)
        elif dtype_in_str == "int64":
            return (np.int64,)
        elif dtype_in_str == "uint":
            return (np.uint16, np.uint32, np.uint64)
        elif dtype_in_str == "uint8":
            return (np.uint8,)
        elif dtype_in_str == "uint16":
            return (np.uint16,)
        elif dtype_in_str == "uint32":
            return (np.uint32,)
        elif dtype_in_str == "uint64":
            return (np.uint64,)
        elif dtype_in_str == "float":
            return (np.float16, np.float32, np.float64)
        elif dtype_in_str == "float16":
            return (np.float16,)
        elif dtype_in_str == "float32":
            return (np.float32,)
        elif dtype_in_str == "float64":
            return (np.float64,)
        elif dtype_in_str == "str":
            return (str, np.str_)
        elif dtype_in_str == "bool":
            return (bool, np.bool_)
        else:
            raise ValueError(f"Unknown datatype {dtype_in_str} in {type_as_str} for a notebook page variable")


class PageTypeError(Exception):
    def __init__(self, msg: str):
        """
        Error raised because the notebook page was given a wrong variable type.

        Args:
            - msg (str): the error message.
        """
        super().__init__(msg)
