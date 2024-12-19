import numpy as np

from ..omp import base as omp_base
from ..setup.notebook import Notebook


def get_intensity_thresh(nb: Notebook) -> float:
    """
    Gets threshold for intensity from parameters in `config file` or Notebook.

    Args:
        nb: Notebook containing at least the `call_spots` page.

    Returns:
        intensity threshold
    """
    if nb.has_page("thresholds"):
        intensity_thresh = nb.thresholds.intensity
    else:
        intensity_thresh = nb.call_spots.abs_intensity_percentile[50]
    return intensity_thresh


def quality_threshold(
    nb: Notebook, method: str = "omp", intensity_thresh: float = 0, score_thresh: float = 0
) -> np.ndarray:
    """
    Indicates which spots pass both the score and intensity quality thresholding.

    Args:
        nb: Notebook containing at least the `ref_spots` page.
        method: `'ref'` or `'omp'` or 'prob' indicating which spots to consider.
        intensity_thresh: Intensity threshold for spots included.
        score_thresh: Score threshold for spots included.

    Returns:
        `bool [n_spots]` indicating which spots pass quality thresholding.

    """
    if method.lower() != "omp" and method.lower() != "anchor" and method.lower() != "prob":
        raise ValueError(f"method must be 'omp', 'prob', or 'anchor' but {method} given.")
    method_omp = method.lower() == "omp"
    method_anchor = method.lower() == "anchor"
    method_prob = method.lower() == "prob"

    # No intensity threshold is available for omp.
    if method_omp:
        score, _ = omp_base.get_all_scores(nb.basic_info, nb.omp)
    elif method_anchor:
        score = nb.call_spots.dot_product_gene_score
    elif method_prob:
        score = np.max(nb.call_spots.gene_probabilities, axis=1)
    if method_omp:
        intensity = np.ones_like(score)
    else:
        intensity = nb.call_spots.intensity
    qual_ok = np.array([score > score_thresh, intensity > intensity_thresh]).all(axis=0)
    return qual_ok
