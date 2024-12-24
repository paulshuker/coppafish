import math as maths
from collections.abc import Callable, Iterable
from typing import Any, Dict, Optional, Tuple

import numpy as np
import tqdm
import zarr

from .. import log


def deep_convert(value: Iterable[Any], conversion: Callable = tuple, /) -> Tuple[Any]:
    """
    Convert the iterable and all nested iterables inside into datatype specified by the given conversion function.
    The function does not try to convert strings, numpy arrays and zarrays, even though they are iterable.

    Args:
        - value (Iterable): the iterable value to convert.
        - conversion (Callable): a function capable of being input an iterable and converting it to another iterable.
            Common examples are `tuple` and `list`.
    """
    assert hasattr(value, "__iter__"), "value must be iterable to convert to a tuple"

    result = [None] * len(value)
    for i, subvalue in enumerate(value):
        iterable = hasattr(subvalue, "__iter__")
        iterable = iterable and type(subvalue) is not str and type(subvalue) is not np.ndarray
        iterable = iterable and type(subvalue) is not zarr.Array
        if iterable:
            result[i] = deep_convert(subvalue, conversion)
        else:
            result[i] = subvalue
    result = conversion(result)
    return result


def reed_solomon_codes(n_genes: int, n_rounds: int, n_dyes: Optional[int] = None) -> Dict[str, str]:
    """
    Generates random gene codes based on reed-solomon principle, using the lowest degree polynomial possible for the
    number of genes needed. The `i`th gene name will be `gene_i`. We assume that `n_dyes` is the number of unique dyes,
    each dye is labelled between `(0, n_dyes]`.

    Args:
        n_genes (int): number of unique gene codes to generate.
        n_rounds (int): number of sequencing rounds.
        n_dyes (int, optional): number of dyes. Default: same as `n_rounds`.

    Returns:
        Dict (str: str): gene names as keys, gene codes as values.

    Raises:
        ValueError: if all gene codes produced cannot be unique.

    Notes:
        See [wikipedia](https://en.wikipedia.org/wiki/Reed%E2%80%93Solomon_error_correction) for more details.
    """
    if n_dyes is None:
        n_dyes = n_rounds
    assert n_rounds > 1, "Require at least two rounds"
    assert n_dyes > 1, "Require at least two dyes"
    assert n_genes > 0, "Require at least one gene"
    assert n_dyes < 10, "n_dyes >= 10 is not yet supported. Please raise an issue if required"

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
            # Iterate to next working coefficient set, by mod n_dyes addition
            most_recent_coefficient_set[0] += 1
            for i in range(most_recent_coefficient_set.size):
                if most_recent_coefficient_set[i] >= n_dyes:
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
            result %= n_dyes
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


def estimate_runtime() -> None:
    """
    Asks the user for relevant questions to estimate the pipeline run-time for coppafish to complete.
    """
    n_sequence_rounds = int(eval(input("Number of sequencing rounds: ")))
    n_rounds = n_sequence_rounds + 1
    n_sequence_channels = int(eval(input("Number of sequencing channels: ")))
    n_channels = n_sequence_channels + 1
    n_genes = int(eval(input("Gene panel size: ")))
    n_tile_pixels = int(eval(input("Number of pixels in one tile: ")))
    n_tiles = int(eval(input("Number of tiles: ")))
    has_gpu = input("Do you have a GPU available? (y/n): ")
    if has_gpu not in ("y", "n"):
        raise ValueError("Must answer y or n")
    has_gpu = has_gpu == "y"
    # All times are in minutes.
    extract_compress_time = 7.3e-11 * n_tile_pixels * n_rounds * n_channels * n_tiles
    extract_read_time = 6.1e-9 * n_tile_pixels * n_rounds * n_tiles
    extract_time = maths.ceil(extract_compress_time + extract_read_time)
    filter_time = maths.ceil(6.2e-10 * n_tile_pixels * n_rounds * n_channels * n_tiles)
    find_spots_time = maths.ceil(9.2e-10 * n_tile_pixels * n_rounds * n_sequence_channels * n_tiles)
    register_time = maths.ceil(7.1e-10 * n_tile_pixels * n_rounds * n_channels * n_tiles)
    stitch_time = maths.ceil(1e-9 * n_tile_pixels * n_tiles)
    get_ref_spots_time = maths.ceil(2.3e-10 * n_tile_pixels * n_sequence_rounds * n_sequence_channels * n_tiles)
    call_spots_time = maths.ceil(2.5e-15 * n_tile_pixels * n_sequence_channels * n_sequence_rounds * n_genes * n_tiles)
    if not has_gpu:
        omp_colour_time = 3e-9 * n_tile_pixels * n_sequence_rounds * n_sequence_channels * n_tiles
        omp_compute_time = 3e-13 * n_tile_pixels * n_sequence_rounds * n_sequence_channels * n_genes * n_tiles
        # Time taken to compute the OMP mean spot and spot.
        omp_mean_spot_time = 2e-10 * n_tile_pixels * n_genes
        # Time taken to place OMP coefficients into the scipy sparse matrix.
        omp_sparse_time = 1.3e-10 * n_tile_pixels * n_genes * n_tiles
        omp_score_time = 2.6e-10 * n_tile_pixels * n_genes * n_tiles
    else:
        # TODO: Update the GPU compute times.
        omp_colour_time = 3e-9 * n_tile_pixels * n_sequence_rounds * n_sequence_channels * n_tiles
        omp_compute_time = 3e-13 * n_tile_pixels * n_sequence_rounds * n_sequence_channels * n_genes * n_tiles
        # Time taken to compute the OMP mean spot and spot.
        omp_mean_spot_time = 2e-10 * n_tile_pixels * n_genes
        # Time taken to place OMP coefficients into the scipy sparse matrix.
        omp_sparse_time = 1.3e-10 * n_tile_pixels * n_genes * n_tiles
        omp_score_time = 2.6e-10 * n_tile_pixels * n_genes * n_tiles
    omp_time = maths.ceil(omp_colour_time + omp_compute_time + omp_mean_spot_time + omp_sparse_time + omp_score_time)

    print(f"GPU: {has_gpu}")
    print(f"Extract time: {extract_time} minutes")
    print(f"Filter time: {filter_time} minutes")
    print(f"Find spots time: {find_spots_time} minutes")
    print(f"Register time: {register_time} minutes")
    print(f"Stitch time: {stitch_time} minutes")
    print(f"Get reference spots time: {get_ref_spots_time} minutes")
    print(f"Call spots time: {call_spots_time} minutes")
    print(f"OMP time: {omp_time} minutes")
    total_time = (
        extract_time
        + filter_time
        + find_spots_time
        + register_time
        + stitch_time
        + get_ref_spots_time
        + call_spots_time
        + omp_time
    )
    print("-" * 50)
    print(f"Total time: {total_time // 60} hours ({total_time} minutes)")
