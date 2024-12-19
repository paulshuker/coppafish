import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from ...omp.pixel_scores import PixelScoreSolver
from ...setup.config import Config
from ...setup.notebook_page import NotebookPage
from ..results_viewer.subplot import Subplot


class ViewOMPColourSum(Subplot):
    def __init__(
        self,
        nbp_basic: NotebookPage,
        nbp_call_spots: NotebookPage,
        nbp_omp: NotebookPage | None,
        method: str,
        local_yxz: np.ndarray[int],
        spot_tile: int,
        spot_colour: np.ndarray[float],
        show: bool = True,
    ) -> None:
        """
        Show the weighted gene bled codes that are summed together by OMP to try and produce the total pixel's colour
        after completing all iterations. It also displays the residual colour left behind that is unaccounted for.

        Args:
            nbp_basic (NotebookPage): `basic_info` notebook page.
            nbp_filter (NotebookPage): `filter` notebook page.
            nbp_register (NotebookPage): `register` notebook page.
            nbp_call_spots (NotebookPage): `call_spots` notebook page.
            nbp_omp (NotebookPage or none): `omp` notebook page or none.
            method (str): gene calling method.
            local_yxz (`(3) ndarray[int]`): the pixel position relative to its tile's bottom-left corner.
            spot_tile (int-like): tile index the pixel is on.
            spot_colour (`(n_rounds_use x n_channels_use) ndarray[float]`): the spot's colour.
            show (bool, optional): display the plot once built. False is useful when unit testing. Default: true.
        """
        n_rounds_use = len(nbp_basic.use_rounds)
        n_channels_use = len(nbp_basic.use_channels)
        min_intensity = Config.get_default_for("omp", "minimum_intensity")
        alpha = Config.get_default_for("omp", "alpha")
        beta = Config.get_default_for("omp", "beta")
        max_genes = Config.get_default_for("omp", "max_genes")
        dot_product_threshold = Config.get_default_for("omp", "dot_product_threshold")
        self.gene_names = nbp_call_spots.gene_names
        if nbp_omp is not None:
            min_intensity = float(nbp_omp.associated_configs["omp"]["minimum_intensity"])
            alpha = float(nbp_omp.associated_configs["omp"]["alpha"])
            beta = float(nbp_omp.associated_configs["omp"]["beta"])
            max_genes = int(nbp_omp.associated_configs["omp"]["max_genes"])
            dot_product_threshold = float(nbp_omp.associated_configs["omp"]["dot_product_threshold"])

        self.colour = spot_colour.copy().astype(np.float32)
        self.colour *= nbp_call_spots.colour_norm_factor[spot_tile].astype(np.float32)
        omp_solver = PixelScoreSolver()
        bled_codes = nbp_call_spots.bled_codes.astype(np.float32)
        bg_bled_codes = omp_solver.create_background_bled_codes(n_rounds_use, n_channels_use)
        pixel_scores, gene_weights, gene_residuals = omp_solver.solve(
            pixel_colours=self.colour[np.newaxis],
            bled_codes=bled_codes,
            background_codes=bg_bled_codes,
            maximum_iterations=max_genes,
            dot_product_threshold=dot_product_threshold,
            minimum_intensity=min_intensity,
            alpha=alpha,
            beta=beta,
            return_all_weights=True,
            return_all_residuals=True,
        )
        self.pixel_score = pixel_scores[0]
        self.gene_weight = gene_weights[0]
        self.assigned_genes: np.ndarray[int] = (~np.isnan(self.gene_weight)).nonzero()[0]
        self.gene_weight = self.gene_weight[self.assigned_genes]
        # Has shape (n_genes_assigned, n_rounds_use, n_channels_use).
        self.gene_residuals = gene_residuals[0][self.assigned_genes]
        self.pixel_score = self.pixel_score[self.assigned_genes]
        n_iterations = self.assigned_genes.size
        if n_iterations == 0:
            return

        column_count = max(2, n_iterations + 2)
        self.fig, self.axes = plt.subplots(
            2,
            column_count,
            figsize=(column_count * 2.8, 5.8),
            width_ratios=[3 for _ in range(column_count - 1)] + [1],
            layout="constrained",
        )
        self.assigned_bled_codes: np.ndarray = nbp_call_spots.bled_codes[self.assigned_genes].astype(np.float32)
        # Weight the bled codes.
        self.assigned_bled_codes *= self.gene_weight[:, np.newaxis, np.newaxis]

        abs_max = np.abs(self.assigned_bled_codes).max()
        abs_max = np.max([abs_max, np.abs(self.colour).max()])
        abs_max = np.max([abs_max, np.abs(self.gene_residuals).max()]).item()

        self.cmap = mpl.cm.seismic
        self.norm = mpl.colors.Normalize(vmin=-abs_max, vmax=abs_max)
        self.fig.colorbar(
            mpl.cm.ScalarMappable(cmap=self.cmap, norm=self.norm), cax=None, ax=(self.axes[0, -1], self.axes[1, -1])
        )
        self.draw_data()
        self.fig.suptitle(f"{method.capitalize()} spot at {tuple(local_yxz)} OMP key colours")
        if show:
            self.fig.show()

    def draw_data(self) -> None:
        for ax in self.axes.ravel():
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        for i, g in enumerate(self.assigned_genes):
            w_str = "{:.3f}".format(self.gene_weight[i])
            c_str = "{:.3f}".format(self.pixel_score[i])
            self.axes[0, i].set_title(f"{g}: {self.gene_names[g]}\nweight: {w_str}\npixel score: {c_str}")
            self.axes[0, i].imshow(self.assigned_bled_codes[i].T, cmap=self.cmap, norm=self.norm)

        for i in range(self.axes.shape[1] - 1):
            self.axes[1, i].set_xlabel("Round")

        self.axes[0, 0].set_ylabel("Channel")
        self.axes[1, 0].set_ylabel("Channel")

        # Plot residual colours for each gene.
        for i, g in enumerate(self.assigned_genes):
            self.axes[1, i].set_title(
                r"(Spot colour - bled codes)$\times\epsilon^2$" + f"\nexcept {self.gene_names[g]}"
            )
            self.axes[1, i].imshow(self.gene_residuals[i].T, cmap=self.cmap, norm=self.norm)

        self.axes[1, -2].set_title("Spot colour")
        self.axes[1, -2].imshow(self.colour.T, cmap=self.cmap, norm=self.norm)

        self.fig.canvas.draw_idle()
