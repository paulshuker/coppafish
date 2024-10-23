from typing import Tuple, Union
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, Slider
import numpy as np
import torch

from coppafish.omp import coefs, scores_torch
from coppafish.setup.notebook import NotebookPage
from coppafish.spot_colours import base as spot_colours_base
from coppafish.plot.results_viewer.subplot import Subplot


class ViewOMPImage(Subplot):
    def __init__(
        self,
        nbp_basic: NotebookPage,
        nbp_filter: NotebookPage,
        nbp_register: NotebookPage,
        nbp_call_spots: NotebookPage,
        nbp_omp: NotebookPage,
        local_yxz: np.ndarray,
        tile: int,
        spot_no: int,
        spot_gene_no: int,
        method: str,
        im_size: int = 8,
        z_planes: Tuple[int] = (-2, -1, 0, 1, 2),
        init_select_gene: Union[int, None] = None,
        show: bool = True,
    ):
        """
        Display omp coefficients of all genes around the local neighbourhood of a pixel position.

        Args:
            nbp_basic (NotebookPage): `basic_info` notebook page.
            nbp_filter (NotebookPage): `filter` notebook page.
            nbp_register (NotebookPage): `register` notebook page.
            nbp_call_spots (NotebookPage): `call_spots` notebook page.
            nbp_omp (NotebookPage): `omp` notebook page.
            local_yxz (`(3) ndarray[int]`): the pixel position relative to its tile's bottom-left corner.
            tile (int-like): tile index the pixel is on.
            spot_no (int-like or none): spot index to be plotted.
            spot_gene_no (int-like or none): spot's gene index. The index will be a selectable option on the slider if
                given. None if not given.
            method (str): gene calling method.
            im_size (int, optional): number of pixels out from the central pixel to plot to create the square images.
            z_planes (tuple of int, optional): z planes to show. 0 is the central z plane.
            init_select_gene (int, optional): gene number to display initially. Default: the highest scoring gene.
            show (bool, optional): display the plot once built. False is useful when unit testing. Default: true.
        """
        assert type(nbp_basic) is NotebookPage
        assert type(nbp_filter) is NotebookPage
        assert type(nbp_register) is NotebookPage
        assert type(nbp_call_spots) is NotebookPage
        assert type(nbp_omp) is NotebookPage
        assert type(local_yxz) is np.ndarray
        assert local_yxz.shape == (3,)
        assert type(int(tile)) is int
        assert type(int(spot_no)) is int
        assert type(int(spot_gene_no)) is int
        assert type(method) is str
        assert type(im_size) is int
        assert im_size >= 0
        assert type(z_planes) is tuple
        assert init_select_gene is None or type(init_select_gene) is int
        assert type(show) is bool

        plt.style.use("dark_background")

        config = nbp_omp.associated_configs["omp"]

        coord_min = (local_yxz - im_size).tolist()
        coord_min[2] = local_yxz[2].item() + min(z_planes)
        coord_max = (local_yxz + im_size + 1).tolist()
        coord_max[2] = local_yxz[2].item() + max(z_planes) + 1
        yxz = [np.arange(coord_min[i], coord_max[i]) for i in range(3)]
        yxz = np.array(np.meshgrid(*[np.arange(coord_min[i], coord_max[i]) for i in range(3)])).reshape((3, -1)).T

        spot_shape_yxz = tuple([coord_max[i] - coord_min[i] for i in range(3)])
        central_yxz = tuple(torch.asarray(spot_shape_yxz)[np.newaxis].T.int() // 2)
        n_rounds_use, n_channels_use = len(nbp_basic.use_rounds), len(nbp_basic.use_channels)
        image_colours = spot_colours_base.get_spot_colours_new(
            image=nbp_filter.images,
            flow=nbp_register.flow,
            affine=nbp_register.icp_correction,
            yxz=yxz,
            tile=int(tile),
            use_rounds=nbp_basic.use_rounds,
            use_channels=nbp_basic.use_channels,
            out_of_bounds_value=0,
        ).reshape(
            (
                spot_shape_yxz
                + (
                    n_rounds_use,
                    n_channels_use,
                )
            )
        )
        assert not np.allclose(image_colours, 0)
        image_colours = image_colours.reshape((-1, n_rounds_use, n_channels_use))
        bled_codes = nbp_call_spots.bled_codes.astype(np.float32)
        n_genes = bled_codes.shape[0]
        assert (~np.isnan(bled_codes)).all(), "bled codes cannot contain nan values"
        assert np.allclose(np.linalg.norm(bled_codes, axis=(1, 2)), 1), "bled codes must be L2 normalised"
        omp_solver = coefs.CoefficientSolverOMP()
        bg_bled_codes = omp_solver.create_background_bled_codes(n_rounds_use, n_channels_use)
        coefficient_image = omp_solver.compute_omp_coefficients(
            pixel_colours=image_colours,
            bled_codes=bled_codes,
            background_codes=bg_bled_codes,
            colour_norm_factor=nbp_call_spots.colour_norm_factor[[tile]].astype(np.float32),
            maximum_iterations=config["max_genes"],
            dot_product_threshold=config["dp_thresh"],
            normalisation_shift=config["lambda_d"],
        )
        coefficient_image = torch.asarray(coefficient_image).T.reshape(
            (len(nbp_call_spots.gene_names),) + spot_shape_yxz
        )

        self.scores = []
        for g in range(coefficient_image.shape[0]):
            self.scores.append(
                scores_torch.score_coefficient_image(coefficient_image[[g]], torch.asarray(nbp_omp.mean_spot))[0][
                    central_yxz
                ].item()
            )
        self.scores = np.array(self.scores, np.float32)

        self.coefficient_image: np.ndarray = coefficient_image.numpy()

        central_pixel = np.array(self.coefficient_image.shape[1:]) // 2
        central_pixels = np.ix_(range(n_genes), [central_pixel[0]], [central_pixel[1]], [central_pixel[2]])
        gene_is_selectable = ~np.isclose(self.coefficient_image[central_pixels].ravel(), 0)
        gene_is_selectable[spot_gene_no] = True
        if init_select_gene is not None:
            gene_is_selectable[init_select_gene] = True
        assert gene_is_selectable.ndim == 1

        self.gene_names = nbp_call_spots.gene_names
        self.z_planes = z_planes
        self.selectable_genes = np.where(gene_is_selectable)[0]
        if init_select_gene is None:
            self.selected_gene = self.selectable_genes[np.argmax(self.scores[self.selectable_genes])].item()
        else:
            self.selected_gene = init_select_gene
        self.iteration_count_image = (~np.isclose(self.coefficient_image, 0)).astype(int).sum(0)
        self.mid_z = -min(self.z_planes)
        self.function_coefficients = False
        self.show_iteration_counts = False
        self.draw_canvas()
        if show:
            self.fig.show()

    def draw_canvas(self) -> None:
        self.fig, self.axes = plt.subplots(
            nrows=2,
            ncols=len(self.z_planes) + 1,
            squeeze=False,
            gridspec_kw={"width_ratios": [5] * len(self.z_planes) + [1] * 1, "height_ratios": [6, 1]},
            layout="constrained",
        )
        # Keep widgets in self otherwise they will get garbage collected and not respond to clicks anymore.
        ax_slider: plt.Axes = self.axes[1, 0]
        # Ignore the user warnings that occurs when there is only one gene in the slider.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.gene_slider = Slider(
                ax_slider,
                label="Gene",
                valmin=self.selectable_genes.min(),
                valmax=self.selectable_genes.max(),
                valstep=self.selectable_genes,
                valinit=self.selected_gene,
            )
        self.gene_slider.on_changed(self.gene_selected_updated)
        ax_iteration_count = self.axes[1, 2]
        self.show_iteration_count_button = CheckButtons(
            ax_iteration_count,
            ["Show iteration counts"],
            actives=[self.show_iteration_counts],
            frame_props={"edgecolor": "white", "facecolor": "white"},
            check_props={"facecolor": "black"},
        )
        self.show_iteration_count_button.on_clicked(self.show_iteration_count_changed)
        self.draw_data()

    def draw_data(self) -> None:

        if self.show_iteration_counts:
            cmap = mpl.cm.viridis
            image_data = self.iteration_count_image
            norm = mpl.colors.Normalize(vmin=0, vmax=self.iteration_count_image.max())
            title = "OMP Iteration Count"
        else:
            cmap = mpl.cm.PiYG
            image_data = self.coefficient_image[self.selected_gene]
            abs_max = np.abs(image_data).max()
            norm = mpl.colors.Normalize(vmin=-abs_max, vmax=abs_max)
            title = "OMP Coefficients\n"
            title += f"Gene {self.selected_gene} {self.gene_names[self.selected_gene]}\n"
            title += f" Score: {str(self.scores[self.selected_gene])[:4]}"

        for ax in self.axes[0]:
            ax.clear()
        all_spines = ("top", "bottom", "left", "right")
        for ax in self.axes[1]:
            for spine in all_spines:
                ax.spines[spine].set_visible(False)
            ax.set_xticks([], [])
            ax.set_yticks([], [])
        self.fig.suptitle(title)
        self.fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=self.axes[0, -1],
            orientation="vertical",
            label="",
        )
        for i, z_plane in enumerate(self.z_planes):
            ax: plt.Axes = self.axes[0, i]
            ax.clear()
            ax.imshow(image_data[:, :, self.mid_z + z_plane], cmap=cmap, norm=norm)
            ax_title = "Central plane"
            if z_plane < 0:
                ax_title = f"- {abs(z_plane)}"
            if z_plane > 0:
                ax_title = f"+ {abs(z_plane)}"
            ax.set_title(ax_title)
        self.gene_slider.active = not self.show_iteration_counts
        self.fig.canvas.draw_idle()

    def show_iteration_count_changed(self, _) -> None:
        self.show_iteration_counts = self.show_iteration_count_button.get_status()[0]
        self.draw_data()

    def gene_selected_updated(self, _) -> None:
        self.selected_gene = int(self.gene_slider.val)
        self.draw_data()
