import numpy as np

from ..omp import base as omp_base
from ..setup.notebook_page import NotebookPage


# Each instance of this class holds data on a specific gene calling method.
class MethodData:
    """
    Spot results for a particular gene calling method.

    The way that data is saved to the notebook is different for each gene calling method. Therefore, we create this
    class as a common interface for gathering and filtering spot results. It is used by multiple post-pipeline functions
    like export_to_pciseq and the Viewer.
    """

    MAX_CHECK_COUNT = 10_000
    _ATTRIBUTE_NAMES = ("tile", "local_yxz", "yxz", "gene_no", "score", "colours", "intensity", "indices")

    method: str
    tile: np.ndarray
    local_yxz: np.ndarray
    yxz: np.ndarray
    gene_no: np.ndarray
    score: np.ndarray
    colours: np.ndarray
    intensity: np.ndarray
    # We keep track of the spots' indices relative to the notebook since we will cut out spots that are part of
    # invisible genes to improve performance.
    indices: np.ndarray

    def __init__(
        self,
        method: str,
        nbp_basic: NotebookPage,
        nbp_stitch: NotebookPage,
        nbp_ref_spots: NotebookPage,
        nbp_call_spots: NotebookPage,
        nbp_omp: NotebookPage | None,
    ) -> None:
        """
        Gather all spot data for a particular gene calling method that is stored in self.

        Args:
            method (str): gene calling method to gather. Can be 'prob', 'anchor', or 'omp'.
            nbp_basic (NotebookPage): `basic_info` notebook page.
            nbp_stitch (NotebookPage): `stitch` notebook page.
            nbp_ref_spots (NotebookPage): `ref_spots` notebook page.
            nbp_call_spots (NotebookPage): `call_spots` notebook page.
            nbp_omp (NotebookPage or none): `omp` notebook page.
        """
        assert type(method) is str
        assert method in ("prob", "anchor", "omp"), f"Unknown method {method}"
        assert type(nbp_ref_spots) is NotebookPage
        assert type(nbp_call_spots) is NotebookPage
        assert type(nbp_omp) is NotebookPage or nbp_omp is None

        self.method = method

        if method == "prob":
            self.score = nbp_call_spots.gene_probabilities[:].max(1)
        if method == "anchor":
            self.score = nbp_call_spots.dot_product_gene_score[:]
        if method in ("prob", "anchor"):
            self.tile = nbp_ref_spots.tile[:]
            self.local_yxz = nbp_ref_spots.local_yxz[:].astype(np.int16)
            self.yxz = self.local_yxz.astype(np.float32) + nbp_stitch.tile_origin[self.tile]
            self.gene_no = np.argmax(nbp_call_spots.gene_probabilities[:], 1).astype(np.int16)
            self.colours = nbp_ref_spots.colours[:].astype(np.float32)
            self.intensity = nbp_call_spots.intensity[:]
        if method == "omp":
            if nbp_omp is None:
                raise ValueError(f"{method} requires the omp notebook page")
            self.local_yxz, self.tile = omp_base.get_all_local_yxz(nbp_basic, nbp_omp)
            self.yxz = self.local_yxz.astype(np.float32) + nbp_stitch.tile_origin[self.tile]
            self.gene_no = omp_base.get_all_gene_no(nbp_basic, nbp_omp)[0].astype(np.int16)
            self.score = omp_base.get_all_scores(nbp_basic, nbp_omp)[0]
            self.colours = omp_base.get_all_colours(nbp_basic, nbp_omp)[0].astype(np.float32)
            self.intensity = omp_base.get_all_intensities(nbp_basic, nbp_call_spots, nbp_omp)

        self.indices = np.linspace(0, self.score.size - 1, self.score.size, dtype=np.uint32)

        # Sanity check spot data.
        self.check_variables()

    def remove_data_at(self, remove: np.ndarray[bool]) -> None:
        """
        Delete the i'th spot data if remove[i] == True.
        """
        assert type(remove) is np.ndarray
        assert remove.ndim == 1
        assert remove.size == self.tile.size
        keep_sum = (~remove).sum().item()
        for var_name in self._ATTRIBUTE_NAMES:
            self.__setattr__(var_name, self.__getattribute__(var_name)[~remove])
            assert self.__getattribute__(var_name).shape[0] == keep_sum
        self.check_variables()

    def check_variables(self) -> None:
        assert all([type(self.__getattribute__(var_name)) is np.ndarray] for var_name in self._ATTRIBUTE_NAMES)
        assert self.tile.ndim == 1
        assert self.tile.shape[0] >= 0
        assert self.local_yxz.ndim == 2
        assert self.local_yxz.shape[0] >= 0
        assert self.local_yxz.shape[1] == 3
        assert self.gene_no.ndim == 1
        assert self.gene_no.shape[0] >= 0
        assert self.score.ndim == 1
        assert self.score.shape[0] >= 0
        assert self.intensity.ndim == 1
        assert self.intensity.shape[0] >= 0
        assert self.indices.ndim == 1
        assert self.indices.shape[0] >= 0
        if self.tile.size > np.iinfo(self.indices.dtype).max:
            raise ValueError(f"Too many spots in {self.method} to index with uint32")
        assert (
            self.tile.size
            == self.local_yxz.shape[0]
            == self.yxz.shape[0]
            == self.gene_no.size
            == self.score.size
            == self.colours.shape[0]
            == self.intensity.size
            == self.indices.size
        )

        # Check values on a subset.
        assert (self.tile[: self.MAX_CHECK_COUNT] >= 0).all()
        assert (self.gene_no[: self.MAX_CHECK_COUNT] >= 0).all()
        assert (self.score[: self.MAX_CHECK_COUNT] >= 0).all()
        assert (self.intensity[: self.MAX_CHECK_COUNT] >= 0).all()
        assert (self.indices[: self.MAX_CHECK_COUNT] >= 0).all()
