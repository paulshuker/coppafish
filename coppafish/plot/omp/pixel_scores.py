import enum
import importlib.resources as importlib_resources
import warnings
from typing import Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.widgets import Button, Slider

from coppafish.omp import scores as omp_scores
from coppafish.omp.pixel_scores import PixelScoreSolver
from coppafish.plot.results_viewer.subplot import Subplot
from coppafish.setup.config import Config
from coppafish.setup.notebook import NotebookPage
from coppafish.spot_colours import base as spot_colours_base


class ViewOMPPixelScoreImage(Subplot):
    class Options(enum.Enum):
        PIXEL_SCORES = enum.auto()
        SCORES = enum.auto()
        ITERATIONS = enum.auto()

    def __init__(
        self,
        nbp_basic: NotebookPage,
        nbp_filter: NotebookPage,
        nbp_register: NotebookPage,
        nbp_call_spots: NotebookPage,
        nbp_omp: NotebookPage | None,
        local_yxz: np.ndarray[int],
        spot_tile: int,
        spot_no: int,
        spot_gene_no: int,
        spot_colour: np.ndarray,
        method: str,
        im_size: int = 8,
        z_planes: Tuple[int] = (-2, -1, 0, 1, 2),
        show: bool = True,
    ):
        """
        Display omp pixel scores around the local neighbourhood of a found spot position.

        Args:
            nbp_basic (NotebookPage): `basic_info` notebook page.
            nbp_filter (NotebookPage): `filter` notebook page.
            nbp_register (NotebookPage): `register` notebook page.
            nbp_call_spots (NotebookPage): `call_spots` notebook page.
            nbp_omp (NotebookPage or none): `omp` notebook page or none.
            local_yxz (`(3) ndarray[int]`): the pixel position relative to its tile's bottom-left corner.
            spot_tile (int-like): tile index the pixel is on.
            spot_no (int-like or none): spot index to be plotted.
            spot_gene_no (int-like or none): spot's gene index. The index will be a selectable option on the slider if
                given. None if not given.
            spot_colour (`(n_rounds_use x n_channels_use) ndarray[float]`): the spot's colour.
            method (str): gene calling method.
            im_size (int, optional): number of pixels out from the central pixel to plot in x and y to create the
                square images. Default: 8.
            z_planes (tuple of int, optional): z planes to show. 0 is the central z plane. Default: (-2, -1, 0, 1, 2).
            show (bool, optional): display the plot once built. False is useful when unit testing. Default: true.
        """
        assert len(z_planes) > 3
        n_rounds_use, n_channels_use = len(nbp_basic.use_rounds), len(nbp_basic.use_channels)
        min_intensity = 0.0
        alpha = Config.get_default_for("omp", "alpha")
        beta = Config.get_default_for("omp", "beta")
        max_genes = Config.get_default_for("omp", "max_genes")
        dot_product_threshold = Config.get_default_for("omp", "dot_product_threshold")
        mean_spot_filepath = importlib_resources.files("coppafish.omp").joinpath("mean_spot.npy")
        mean_spot: np.ndarray = np.load(mean_spot_filepath).astype(np.float32)
        if nbp_omp is not None:
            min_intensity = float(nbp_omp.results[f"tile_{spot_tile}"].attrs["minimum_intensity"])
            alpha = float(nbp_omp.associated_configs["omp"]["alpha"])
            beta = float(nbp_omp.associated_configs["omp"]["beta"])
            max_genes = int(nbp_omp.associated_configs["omp"]["max_genes"])
            dot_product_threshold = float(nbp_omp.associated_configs["omp"]["dot_product_threshold"])
            mean_spot = nbp_omp.mean_spot.astype(np.float32)
        yxz_min = local_yxz.copy() + np.array([-im_size, -im_size, min(z_planes)], int)
        yxz_max = local_yxz.copy() + np.array([im_size, im_size, max(z_planes)], int) + 1
        image_shape = tuple((yxz_max - yxz_min).tolist())
        yxz = np.meshgrid(
            *[np.linspace(yxz_min[i], yxz_max[i] - 1, yxz_max[i] - yxz_min[i]) for i in range(3)], indexing="ij"
        )
        yxz = np.array(yxz).reshape((3, -1), order="F")
        yxz = yxz.T
        colours = spot_colours_base.get_spot_colours_new(
            yxz,
            nbp_filter.images,
            nbp_register.flow,
            nbp_register.icp_correction,
            int(spot_tile),
            nbp_basic.use_rounds,
            nbp_basic.use_channels,
            out_of_bounds_value=0,
        )
        colours *= nbp_call_spots.colour_norm_factor[[spot_tile]].astype(np.float32)
        bled_codes = nbp_call_spots.bled_codes.astype(np.float32)
        solver = PixelScoreSolver()
        pixel_scores = solver.solve(
            colours,
            bled_codes,
            solver.create_background_bled_codes(n_rounds_use, n_channels_use),
            max_genes,
            dot_product_threshold,
            min_intensity,
            alpha,
            beta,
        )
        shape = image_shape + (-1,)
        pixel_scores = pixel_scores.reshape(shape, order="F")
        selectable_genes = set(((~np.isclose(pixel_scores, 0)).sum((0, 1, 2)) > 3).nonzero()[0].tolist())
        iteration_counts = (~np.isclose(pixel_scores, 0)).sum(3)
        self.method = method
        self.spot_no = spot_no
        self.z_planes = z_planes
        self.image_shape = image_shape
        self.spot_gene_no = spot_gene_no
        selectable_genes.add(self.spot_gene_no)
        self.selected_gene = spot_gene_no
        self.gene_names = nbp_call_spots.gene_names
        pixel_scores = pixel_scores.transpose((3, 0, 1, 2))
        self.pixel_scores = pixel_scores
        self.iteration_counts = iteration_counts
        self.score_images = omp_scores.score_pixel_score_image(
            torch.from_numpy(pixel_scores), torch.from_numpy(mean_spot)
        )
        scores = omp_scores.score_pixel_score_image(torch.from_numpy(pixel_scores), torch.from_numpy(mean_spot))[
            :, image_shape[0] // 2, image_shape[1] // 2, image_shape[2] // 2
        ]
        self.scores = scores.numpy()

        self.fig, self.axes = plt.subplots(2, len(z_planes) + 1, height_ratios=(6, 1))
        self.fig.set_figwidth(14)
        self.fig.set_figheight(8)
        for ax in self.axes.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
        for c in range(self.axes.shape[1]):
            ax: plt.Axes = self.axes[1, c]
            ax.spines.clear()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.gene_slider = Slider(
                self.axes[1, 0],
                label="Gene",
                valmin=min(selectable_genes),
                valmax=max(selectable_genes),
                valstep=list(selectable_genes),
                valinit=self.selected_gene,
            )
        self.gene_slider.on_changed(self.gene_selected_updated)
        self.button_colour = "red"
        self.button_colour_press = "green"
        self.pixel_score_button = Button(self.axes[1, 1], "Pixel Scores", hovercolor="0.275")
        self.score_button = Button(self.axes[1, 2], "Final Scores", hovercolor="0.275")
        self.iter_count_button = Button(self.axes[1, 3], "Iteration Counts", hovercolor="0.275")
        self.reset_gene_button = Button(self.axes[1, 4], "Spot Gene", hovercolor="0.275")
        self.pixel_score_button.on_clicked(self.pressed_pixel_score_button)
        self.score_button.on_clicked(self.pressed_score_button)
        self.iter_count_button.on_clicked(self.pressed_iter_button)
        self.reset_gene_button.on_clicked(self.pressed_reset_gene)
        self.pressed_pixel_score_button()

        if show:
            self.fig.show()

    def draw_data(self) -> None:
        title = f"OMP {self.gene_names[self.selected_gene].item()} Pixel Scores end_msg"
        if self.selected_button == self.Options.ITERATIONS:
            title = "OMP Iteration Counts"
        score_str = "{:.3f}".format(self.scores[self.selected_gene])
        title = title.replace("end_msg", f"near {self.method.capitalize()} Spot {self.spot_no}, Score {score_str}")
        self.fig.suptitle(title)
        data = []
        for k in range(len(self.z_planes)):
            if self.selected_button == self.Options.PIXEL_SCORES:
                z_data = self.pixel_scores[self.selected_gene, :, :, k]
                abs_max = np.abs(self.pixel_scores).max()
            elif self.selected_button == self.Options.SCORES:
                z_data = self.score_images[self.selected_gene, :, :, k]
            elif self.selected_button == self.Options.ITERATIONS:
                z_data = self.iteration_counts[:, :, k]
            else:
                raise ValueError(f"Unknown option: {self.selected_button}")
            data.append(z_data)
        if self.selected_button != self.Options.PIXEL_SCORES:
            abs_max = np.max(np.abs(data.copy()))
        cmap = mpl.cm.seismic
        norm = mpl.colors.Normalize(vmin=-abs_max, vmax=abs_max)
        for k, z_plane in enumerate(self.z_planes):
            ax: plt.Axes = self.axes[0, k]
            ax.clear()
            # Two perpendicular lines to help see where the central pixel is.
            ax.hlines(
                self.image_shape[0] / 2 - 0.5,
                -1,
                self.image_shape[1] + 1,
                colors="green",
                linewidth=0.3,
            )
            ax.vlines(
                self.image_shape[1] / 2 - 0.5,
                -1,
                self.image_shape[0] + 1,
                colors="green",
                linewidth=0.3,
            )
            if z_plane == 0:
                ax.set_title("Central Plane")
            else:
                ax.set_title(f"{'+' if z_plane > 0 else '-'}{abs(z_plane)}")
            im = ax.imshow(data[k], cmap=cmap, norm=norm, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
        ax_cbar: plt.Axes = self.axes[0, -1]
        ax_cbar.clear()
        self.fig.colorbar(im, cax=ax_cbar)
        self.fig.canvas.draw_idle()

    def gene_selected_updated(self, _=None) -> None:
        self.selected_gene = int(self.gene_slider.val)
        if self.selected_button == self.Options.ITERATIONS:
            return
        self.draw_data()

    def pressed_pixel_score_button(self, _=None) -> None:
        self.selected_button = self.Options.PIXEL_SCORES
        self.draw_data()
        self.pixel_score_button.label.set_color(self.button_colour_press)
        self.score_button.label.set_color(self.button_colour)
        self.iter_count_button.label.set_color(self.button_colour)
        self.gene_slider.set_active(True)

    def pressed_score_button(self, _=None) -> None:
        self.selected_button = self.Options.SCORES
        self.draw_data()
        self.pixel_score_button.label.set_color(self.button_colour)
        self.score_button.label.set_color(self.button_colour_press)
        self.iter_count_button.label.set_color(self.button_colour)
        self.gene_slider.set_active(True)

    def pressed_iter_button(self, _=None) -> None:
        self.selected_button = self.Options.ITERATIONS
        self.draw_data()
        self.pixel_score_button.label.set_color(self.button_colour)
        self.score_button.label.set_color(self.button_colour)
        self.iter_count_button.label.set_color(self.button_colour_press)
        self.gene_slider.set_active(False)

    def pressed_reset_gene(self, _=None) -> None:
        self.gene_slider.set_val(self.spot_gene_no)
