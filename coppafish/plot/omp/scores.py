import warnings

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import mplcursors
import numpy as np

from coppafish.omp import coefs
from coppafish.setup.notebook import NotebookPage
from coppafish.spot_colours import base as spot_colours_base
from coppafish.plot.results_viewer.subplot import Subplot


class ViewOMPDotProductScores(Subplot):
    def __init__(
        self,
        nbp_basic: NotebookPage,
        nbp_filter: NotebookPage,
        nbp_register: NotebookPage,
        nbp_call_spots: NotebookPage,
        nbp_omp: NotebookPage,
        spot_local_yxz: np.ndarray,
        spot_tile: int,
        show: bool = True,
    ):
        """
        View a spot's gene dot product scores on each OMP iteration. A slider is used to switch between OMP iteration
        number.

        Args:
            nbp_basic (NotebookPage): `basic_info` notebook page.
            nbp_filter (NotebookPage): `filter` notebook page.
            nbp_register (NotebookPage): `register` notebook page.
            nbp_call_spots (NotebookPage): `call_spots` notebook page.
            nbp_omp (NotebookPage): `omp` notebook page.
            spot_local_yxz (`(3) ndarray[int]`): the spot's local position relative to its tile's bottom-left corner.
            spot_tile (int-like): the spot's tile index.
            show (bool, optional): display the plot once built. False is useful when unit testing. Default: true.
        """
        assert type(nbp_basic) is NotebookPage
        assert type(nbp_filter) is NotebookPage
        assert type(nbp_register) is NotebookPage
        assert type(nbp_call_spots) is NotebookPage
        assert type(nbp_omp) is NotebookPage
        assert type(spot_local_yxz) is np.ndarray
        assert spot_local_yxz.shape == (3,)

        config = nbp_omp.associated_configs["omp"]
        n_rounds_use = len(nbp_basic.use_rounds)
        n_channels_use = len(nbp_basic.use_channels)
        self.dp_thresh = config["dp_thresh"]

        # image_colours has shape (1, n_rounds_use, n_channels_use).
        image_colours = spot_colours_base.get_spot_colours_new(
            image=nbp_filter.images,
            flow=nbp_register.flow,
            affine=nbp_register.icp_correction,
            yxz=spot_local_yxz[None],
            tile=int(spot_tile),
            use_rounds=nbp_basic.use_rounds,
            use_channels=nbp_basic.use_channels,
            out_of_bounds_value=0,
        )
        omp_solver = coefs.CoefficientSolverOMP()
        bled_codes = nbp_call_spots.bled_codes.astype(np.float32)
        bg_bled_codes = omp_solver.create_background_bled_codes(n_rounds_use, n_channels_use)
        _, self.dp_scores = omp_solver.compute_omp_coefficients(
            pixel_colours=image_colours,
            bled_codes=bled_codes,
            background_codes=bg_bled_codes,
            colour_norm_factor=nbp_call_spots.colour_norm_factor[[spot_tile]].astype(np.float32),
            maximum_iterations=config["max_genes"],
            dot_product_threshold=self.dp_thresh,
            normalisation_shift=config["lambda_d"],
            return_dp_scores=True,
        )
        n_iterations = len(self.dp_scores)
        assert n_iterations > 0
        self.iteration = 1
        n_genes_all = self.dp_scores[0].shape[1]
        self.gene_names_all: list[str] = nbp_call_spots.gene_names.tolist()
        for i in range(n_genes_all - len(self.gene_names_all)):
            self.gene_names_all.append(f"bg_{i}")

        self.fig, axes = plt.subplots(2, 1, gridspec_kw={"height_ratios": [6, 1]}, figsize=(15, 7))
        self.plot_ax: plt.Axes = axes[0]
        self.plot_ax.hlines(self.dp_thresh, 0, n_genes_all, colors="red", linestyles="dashed")
        self.bar_colour = "blue"
        self.bars = self.plot_ax.bar(
            [i + 0.5 for i in range(n_genes_all)],
            [1] * n_genes_all,
            color=self.bar_colour,
            align="center",
            width=0.9,
        )
        # Show the gene's name when hovering over its bar.
        cursor = mplcursors.cursor(self.bars, hover=True)
        cursor.connect("add", lambda sel: sel.annotation.set_text(f"Value: {self.gene_names_all[sel.index]}"))
        cursor.connect("remove", lambda sel: sel.annotation.set_text(""))
        self.plot_ax.set_xlim(0, n_genes_all)
        self.plot_ax.set_title(
            f"Pixel {tuple(spot_local_yxz)}, Tile {spot_tile} Gene Dot Product Scores\n(0, "
            + f"{n_genes_all - n_channels_use}) are gene indices, ({n_genes_all - n_channels_use}, {n_genes_all}) are "
            + f"background genes"
        )
        self.plot_ax.set_ylabel("Gene Score")
        max_score = 1
        for dp_scores in self.dp_scores:
            max_score = np.max([max_score, dp_scores.max()])
        self.plot_ax.set_ylim(0, max_score)
        slider_ax: plt.Axes = axes[1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.iter_slider = Slider(
                slider_ax,
                label="Iteration",
                valmin=1,
                valmax=n_iterations,
                valstep=[i + 1 for i in range(n_iterations)],
                valinit=self.iteration,
            )
        self.iter_slider.on_changed(self.iteration_changed)
        self.iteration_changed()

        if show:
            self.fig.show()

    def draw_data(self) -> None:
        dp_scores = self.dp_scores[self.iteration - 1][0]
        for bar, score in zip(self.bars, dp_scores):
            bar.set_height(score)
            bar.set_color(self.bar_colour)
            if score >= dp_scores.max() and score > self.dp_thresh:
                bar.set_color("green")
        self.fig.canvas.draw_idle()

    def iteration_changed(self, _=None) -> None:
        self.iteration = int(self.iter_slider.val)
        self.draw_data()
