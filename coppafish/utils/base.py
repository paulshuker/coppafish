import os
import tqdm
import inspect
import itertools
import numpy as np
from pathlib import PurePath
import numpy.typing as npt
from typing import Union, Dict, List, Optional

from ..setup.notebook import Notebook
from .. import log


def get_function_name() -> str:
    """
    Get the name of the function that called this function.

    Returns:
        str: function name.
    """
    return str(inspect.stack()[1][3])


def set_notebook_output_dir(notebook_path: str, new_output_dir: str) -> None:
    """
    Changes the notebook variables to use the given `output_dir`, then re-saves the notebook.

    Args:
        notebook_path (str): path to notebook.
        new_dir (str): new output directory.
    """
    assert os.path.isfile(notebook_path), f"Notebook at {notebook_path} not found"
    assert os.path.isdir(new_output_dir), f"{new_output_dir} directory not found"

    nb = Notebook(notebook_path)
    # Set the copied notebook variables to the right output directory.
    nb.file_names.finalized = False
    del nb.file_names.output_dir
    nb.file_names.output_dir = new_output_dir

    old_name = PurePath(nb.file_names.spot_details_info).name
    del nb.file_names.spot_details_info
    nb.file_names.spot_details_info = os.path.join(new_output_dir, old_name)
    old_name = PurePath(nb.file_names.psf).name
    del nb.file_names.psf
    nb.file_names.psf = os.path.join(new_output_dir, old_name)

    old_name = PurePath(nb.file_names.omp_spot_shape).name
    del nb.file_names.omp_spot_shape
    nb.file_names.omp_spot_shape = os.path.join(new_output_dir, old_name)
    old_name = PurePath(nb.file_names.omp_spot_shape_float).name
    del nb.file_names.omp_spot_shape_float
    nb.file_names.omp_spot_shape_float = os.path.join(new_output_dir, old_name)
    old_name = PurePath(nb.file_names.omp_spot_info).name
    del nb.file_names.omp_spot_info
    nb.file_names.omp_spot_info = os.path.join(new_output_dir, old_name)
    old_name = PurePath(nb.file_names.omp_spot_coef).name
    del nb.file_names.omp_spot_coef
    nb.file_names.omp_spot_coef = os.path.join(new_output_dir, old_name)

    old_name = PurePath(nb.file_names.big_dapi_image).name
    del nb.file_names.big_dapi_image
    nb.file_names.big_dapi_image = os.path.join(new_output_dir, old_name)
    old_name = PurePath(nb.file_names.big_anchor_image).name
    del nb.file_names.big_anchor_image
    nb.file_names.big_anchor_image = os.path.join(new_output_dir, old_name)

    nb.file_names.finalized = True
    nb.save(notebook_path)


def set_notebook_tile_dir(notebook_path: str, new_tile_dir: str) -> None:
    """
    Changes the notebook variables to use the given `tile_dir`, then re-saves the notebook.

    Args:
        notebook_path (str): path to notebook.
        new_tile_dir (str): new tile directory.
    """
    assert os.path.isfile(notebook_path), f"Notebook {notebook_path} not found"
    if not os.path.isdir(new_tile_dir):
        log.warn(f"New tile directory {new_tile_dir} does not exist. Continuing anyway...")

    new_tile_dir = os.path.normpath(new_tile_dir)

    nb = Notebook(notebook_path)
    nb.file_names.finalized = False

    del nb.file_names.tile_dir
    nb.file_names.tile_dir = os.path.join(new_tile_dir, "filter")
    del nb.file_names.tile_unfiltered_dir
    nb.file_names.tile_unfiltered_dir = os.path.join(new_tile_dir, "extract")
    old_name = PurePath(nb.file_names.scale).name
    del nb.file_names.scale
    nb.file_names.scale = os.path.join(new_tile_dir, old_name)
    old_tile: list = nb.file_names.tile.copy()
    old_tile_unfiltered = nb.file_names.tile_unfiltered.copy()
    new_tile = old_tile.copy()
    new_tile_unfiltered = old_tile_unfiltered.copy()
    for i, j, k in itertools.product(range(len(old_tile)), range(len(old_tile[0])), range(len(old_tile[0][0]))):
        old_tile_ijk = os.path.normpath(old_tile[i][j][k])
        new_tile[i][j][k] = os.path.join(nb.file_names.tile_dir, PurePath(old_tile_ijk).name)
        old_tile_unfiltered_ijk = os.path.normpath(old_tile_unfiltered[i][j][k])
        new_tile_unfiltered[i][j][k] = os.path.join(
            nb.file_names.tile_unfiltered_dir, PurePath(old_tile_unfiltered_ijk).name
        )
    del nb.file_names.tile
    nb.file_names.tile = new_tile
    del nb.file_names.tile_unfiltered
    nb.file_names.tile_unfiltered = new_tile_unfiltered

    nb.file_names.finalized = True
    nb.save(notebook_path)


def round_any(x: Union[float, npt.NDArray], base: float, round_type: str = "round") -> Union[float, npt.NDArray]:
    """
    Rounds `x` to the nearest multiple of `base` with the rounding done according to `round_type`.

    Args:
        x: Number or array to round.
        base: Rounds `x` to nearest integer multiple of value of `base`.
        round_type: One of the following, indicating how to round `x` -

            - `'round'`
            - `'ceil'`
            - `'floor'`

    Returns:
        Rounded version of `x`.

    Example:
        ```
        round_any(3, 5) = 5
        round_any(3, 5, 'floor') = 0
        ```
    """
    if round_type == "round":
        return base * np.round(x / base)
    elif round_type == "ceil":
        return base * np.ceil(x / base)
    elif round_type == "floor":
        return base * np.floor(x / base)
    else:
        log.error(
            ValueError(
                f"round_type specified was {round_type} but it should be one of the following:\n" f"round, ceil, floor"
            )
        )


def setdiff2d(array1: npt.NDArray[np.float_], array2: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    """
    Finds all unique elements in `array1` that are also not in `array2`. Each element is appended along the first axis.
    E.g.

    - If `array1` has `[4,0]` twice, `array2` does not have `[4,0]`, returned array will have `[4,0]` once.

    - If `array1` has `[4,0]` twice, `array2` has `[4,0]` once, returned array will not have `[4,0]`.

    Args:
        array1: `float [n_elements1 x element_dim]`.
        array2: `float [n_elements2 x element_dim]`.

    Returns:
        `float [n_elements_diff x element_dim]`.
    """
    set1 = set([tuple(x) for x in array1])
    set2 = set([tuple(x) for x in array2])
    return np.array(list(set1 - set2))


def expand_channels(array: npt.NDArray[np.float_], use_channels: List[int], n_channels: int) -> npt.NDArray[np.float_]:
    """
    Expands `array` to have `n_channels` channels. The `i`th channel from `array` is placed into the new channel index
    `use_channels[i]` in the new array. Any channels unset in the new array are set to zeroes.

    Args:
        array (`[n1 x n2 x ... x n_k x n_channels_use] ndarray[float]`): array to expand.
        use_channels (`list` of `int`): list of channels to use from `array`.
        n_channels (int): Number of channels to expand `array` to.

    Returns:
        (`[n1 x n2 x ... x n_k x n_channels_use] ndarray[float]`): expanded_array copy.
    """
    assert len(use_channels) <= array.shape[-1], "use_channels is greater than the number of channels found in `array`"
    assert n_channels >= array.shape[-1], "Require n_channels >= the number of channels currently in `array`"

    old_array_shape = np.array(array.shape)
    new_array_shape = old_array_shape.copy()
    new_array_shape[-1] = n_channels
    expanded_array = np.zeros(new_array_shape)

    for i, channel in enumerate(use_channels):
        expanded_array[..., channel] = array[..., i]

    return expanded_array


def reed_solomon_codes(n_genes: int, n_rounds: int, n_channels: Optional[int] = None) -> Dict:
    """
    Generates random gene codes based on reed-solomon principle, using the lowest degree polynomial possible for the
    number of genes needed. The `i`th gene name will be `gene_i`. We assume that `n_channels` is the number of unique
    dyes, each dye is labelled between `(0, n_channels]`.

    Args:
        n_genes (int): number of unique gene codes to generate.
        n_rounds (int): number of sequencing rounds.
        n_channels (int, optional): number of channels. Default: same as `n_rounds`.

    Returns:
        Dict (str: str): gene names as keys, gene codes as values.

    Raises:
        ValueError: if all gene codes produced cannot be unique.

    Notes:
        See [wikipedia](https://en.wikipedia.org/wiki/Reed%E2%80%93Solomon_error_correction) for more details.
    """
    if n_channels is None:
        n_channels = n_rounds
    assert n_rounds > 1, "Require at least two rounds"
    assert n_channels > 1, "Require at least two channels"
    assert n_genes > 0, "Require at least one gene"
    assert n_channels < 10, "n_channels >= 10 is not supported"

    verbose = n_genes > 10
    degree = 0
    # Find the smallest degree polynomial required to produce `n_genes` unique gene codes. We use the smallest degree
    # polynomial because this will have the smallest amount of overlap between gene codes
    while True:
        max_unique_codes = int(n_rounds**degree - n_rounds)
        if max_unique_codes >= n_genes:
            break
        degree += 1
        if degree == 20:
            log.error(ValueError("Polynomial degree required is too large for generating the gene codes"))
    # Create a `degree` degree polynomial, where each coefficient goes between (0, n_rounds] to generate each unique
    # gene code
    codes = dict()
    # Index 0 is for constant, index 1 for linear coefficient, etc..
    most_recent_coefficient_set = np.array(np.zeros(degree + 1))
    for n_gene in tqdm.trange(n_genes, ascii=True, unit="Codes", desc="Generating gene codes", disable=not verbose):
        # Find the next coefficient set that works, which is not just constant across all rounds (like a background
        # code)
        while True:
            # Iterate to next working coefficient set, by mod n_channels addition
            most_recent_coefficient_set[0] += 1
            for i in range(most_recent_coefficient_set.size):
                if most_recent_coefficient_set[i] >= n_channels:
                    # Cycle back around to 0, then add one to next coefficient
                    most_recent_coefficient_set[i] = 0
                    most_recent_coefficient_set[i + 1] += 1
            if np.all(most_recent_coefficient_set[1 : degree + 1] == 0):
                continue
            break
        # Generate new gene code
        new_code = ""
        gene_name = f"gene_{n_gene}"
        for r in range(n_rounds):
            result = 0
            for j in range(degree + 1):
                result += most_recent_coefficient_set[j] * r**j
            result = int(result)
            result %= n_channels
            new_code += str(result)
        # Add new code to dictionary
        codes[gene_name] = new_code
    values = list(codes.values())
    if len(values) != len(set(values)):
        # Not every gene code is unique
        log.error(
            ValueError(
                f"Could not generate {n_genes} unique gene codes with {n_rounds} rounds/dyes. "
                + "Maybe try decreasing the number of genes or increasing the number of rounds."
            )
        )
    return codes
