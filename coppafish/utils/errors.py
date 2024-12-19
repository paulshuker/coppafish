from typing import Tuple

import numpy as np
import scipy


def compare_spots(
    spot_positions_0: np.ndarray,
    spot_gene_indices_0: np.ndarray,
    spot_positions_1: np.ndarray,
    spot_gene_indices_1: np.ndarray,
    distance_threshold: float,
) -> Tuple[np.ndarray[np.int8], int]:
    """
    Compare two collections of spots and assign each spot in the 0 collection either true positive, wrong positive,
    false positive. If a spot in collection 0 is matched to a spot in collection 1 if close together and of the same
    gene index, then this spot is a true positive. Any remaining spots are matched to close together spots in
    collection 1 also left unmatched and assigned as wrong positives. Remaining spots in collection 0 are false
    positives. If there are still unassigned spots in collection 1, these add to the false negative count. When
    multiple matches are found, the matched spots are the ones appearing first in the arrays for deterministic
    results. Collection 1 acts as the ground truth dataset.

    Args:
        - (`(n_spots_0 x 3) ndarray[float]`) spot_positions_0: spot collection 0 positions. spot_positions_0[i] is the
            ith spot's y, x, and z position.
        - (`(n_spots_0) ndarray[int]`) spot_gene_indices_0: spot gene indices.
        - (`(n_spots_1 x 3) ndarray[float]`) spot_positions_1: spot collection 1 positions.
        - (`(n_spots_1) ndarray[int]`) spot_gene_indices_1: spot collection 1 gene indices.
        - (float) distance_threshold: spot's are matched if their distance is no greater than distance_threshold.

    Returns:
        - spot_assignments (`(n_spots_0) ndarray[int8]): each collection 0 spot is given a label. 0 represents a true
            positive, 1 represents a wrong positive, 2 represents a false positive.
        - n_false_negatives (int): the number of false negative spots.
    """
    assert type(spot_positions_0) is np.ndarray
    assert spot_positions_0.ndim == 2
    assert spot_positions_0.shape[1] == 3
    assert type(spot_gene_indices_0) is np.ndarray
    assert spot_gene_indices_0.ndim == 1
    assert spot_gene_indices_0.shape[0] == spot_positions_0.shape[0]
    assert type(spot_positions_1) is np.ndarray
    assert spot_positions_1.ndim == 2
    assert spot_positions_1.shape[1] == 3
    assert type(spot_gene_indices_1) is np.ndarray
    assert spot_gene_indices_1.ndim == 1
    assert spot_gene_indices_1.shape[0] == spot_positions_1.shape[0]
    assert type(distance_threshold) is float
    assert distance_threshold > 0

    spot_assignments = np.zeros_like(spot_gene_indices_0, np.int8) - 1
    n_spots_0 = spot_positions_0.shape[0]
    n_spots_1 = spot_positions_1.shape[0]
    unmatched_spot_0 = np.ones(n_spots_0, bool)
    unmatched_spot_1 = np.ones(n_spots_1, bool)

    # Loop through spots, find matches of the same gene index.
    for i in range(n_spots_0):
        i_gene = spot_gene_indices_0[i]
        if unmatched_spot_1.sum() == 0:
            break
        spot_positions_1_unmatched = np.full_like(spot_positions_1, -999_999, np.float32)
        spot_positions_1_unmatched[unmatched_spot_1] = spot_positions_1[unmatched_spot_1]
        kdtree = scipy.spatial.KDTree(spot_positions_1_unmatched)
        matching_spots_1 = kdtree.query_ball_point(spot_positions_0[i], r=distance_threshold, workers=-1)
        if len(matching_spots_1) == 0:
            continue
        is_matching_gene_1 = np.zeros_like(spot_gene_indices_1, bool)
        is_matching_gene_1[matching_spots_1] = spot_gene_indices_1[matching_spots_1] == i_gene
        if is_matching_gene_1.sum() == 0:
            continue
        first_matching_spot_1 = is_matching_gene_1.nonzero()[0][0]
        unmatched_spot_0[i] = False
        unmatched_spot_1[first_matching_spot_1] = False
        spot_assignments[i] = 0

    # Now loop through unmatched 0 spots again. Any matching spots with wrong gene are now considered wrong positives.
    for i in unmatched_spot_0.nonzero()[0]:
        i_gene = spot_gene_indices_0[i]
        spot_positions_1_unmatched = np.full_like(spot_positions_1, -999_999, np.float32)
        spot_positions_1_unmatched[unmatched_spot_1] = spot_positions_1[unmatched_spot_1]
        kdtree = scipy.spatial.KDTree(spot_positions_1_unmatched)
        matching_spots_1 = kdtree.query_ball_point(spot_positions_0[i], r=distance_threshold, workers=-1)
        if len(matching_spots_1) == 0:
            spot_assignments[i] = 2
            continue
        is_matching_gene_1 = spot_gene_indices_1[matching_spots_1] == i_gene
        assert is_matching_gene_1.sum() == 0
        first_matching_spot_1 = matching_spots_1[0]
        unmatched_spot_0[i] = False
        unmatched_spot_1[first_matching_spot_1] = False
        spot_assignments[i] = 1

    false_negative_count = unmatched_spot_1.sum()

    return spot_assignments, false_negative_count
