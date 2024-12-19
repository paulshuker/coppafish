import os
from numbers import Number

import numpy as np
import pandas as pd

from ..results.base import MethodData
from ..setup.notebook import Notebook


def export_to_pciseq(
    nb: Notebook,
    method: str,
    score_thresh: float | None = None,
    intensity_thresh: float | None = None,
) -> str:
    """
    Saves a .csv files containing gene spot information compatible with pciSeq. The csv contains:

    - Gene: Name of gene each spot was assigned to.
    - y: y coordinate of each spot in stitched coordinate system.
    - x: x coordinate of each spot in stitched coordinate system.
    - z_stack: z coordinate of each spot in stitched coordinate system (in units of z-pixels).
    - score: the spot's score.

    Only spots which pass the thresholds are saved.

    One .csv file is saved for each method: *omp*, *anchor*, and *prob*. The .csv file is saved into the directory
    where the notebook is located.

    Args:
        nb (Notebook): the experiment's coppafish outputted Notebook, must have completed at least up to call spots.
        method (str): gene calling method. Can be 'omp', 'anchor', or 'prob'.
        score_thresh (float, optional): only include spots with score >= score_thresh. Default: 0.
        intensity_thresh (float, optional): only include spots with colour intensity >= intensity_thresh. Default: 0

    Returns:
        (str): csv_file_path. The file path to the saved csv file.
    """
    assert type(nb) is Notebook
    assert type(method) is str
    if method not in ("prob", "anchor", "omp"):
        raise ValueError(f"Unknown method: {method}")
    if score_thresh is None:
        score_thresh = 0
        print("Using no score threshold")
    if intensity_thresh is None:
        intensity_thresh = 0
        print("Using no intensity threshold")
    assert isinstance(intensity_thresh, Number), "Thresholds must be numbers"
    assert isinstance(score_thresh, Number), "Thresholds must be numbers"
    intensity_thresh = float(intensity_thresh)
    score_thresh = float(score_thresh)

    nbp_omp = None
    if nb.has_page("omp"):
        nbp_omp = nb.omp

    spot_data = MethodData(method, nb.basic_info, nb.stitch, nb.ref_spots, nb.call_spots, nbp_omp)
    spot_data.remove_data_at(np.logical_or(spot_data.score < score_thresh, spot_data.intensity < intensity_thresh))

    file_path = os.path.join(os.path.dirname(nb.directory), f"pciseq_{method}.csv")
    if os.path.isfile(file_path):
        print(f"WARNING: file {file_path} already exists, it will be overwritten")
    # Save the global yxz positions with their corresponding gene indices.
    df_to_export = pd.DataFrame()
    df_to_export["Gene"] = nb.call_spots.gene_names[spot_data.gene_no]
    df_to_export["y"] = spot_data.yxz[:, 0]
    df_to_export["x"] = spot_data.yxz[:, 1]
    df_to_export["z_stack"] = spot_data.yxz[:, 2]
    df_to_export["score"] = spot_data.score
    df_to_export.to_csv(file_path, mode="w", index=False)

    print(f"pciSeq file saved for method {method} at " + file_path)

    return file_path
