from typing import List, Optional, Tuple, Union

import matplotlib as mpl
from matplotlib import cm
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RangeSlider
import mplcursors
import numpy as np
import zarr

from ..results_viewer.subplot import Subplot
from ...call_spots.dot_product import gene_prob_score
from ...setup.notebook import Notebook
from ...spot_colours import base as spot_colours_base


class ColorPlotBase:
    def __init__(
        self,
        images: List,
        norm_factor: Optional[Union[np.ndarray, List]],
        subplot_row_columns: Optional[List] = None,
        fig_size: Optional[Tuple] = None,
        subplot_adjust: Optional[List] = None,
        cbar_pos: Optional[List] = None,
        slider_pos: Optional[List] = None,
        button_pos: Optional[List] = None,
    ):
        """
        This is the base class for plots with multiple subplots and with a slider to change the colour axis and a button
        to change the normalisation.
        After initialising, the function `change_norm()` should be run to plot normalised images.
        This will change `self.method` from `'raw'` to `'norm'`.

        Args:
            images: `float [n_images]`
                Each image is `n_y x n_x (x n_z)`. This is the normalised image.
                There will be a subplot for each image and if it is 3D, the first z-plane will be set as the
                starting data, `self.im_data` while the full 3d data will be saved as `self.im_data_3d`.
            norm_factor: `float [n_images]`
                `norm_factor[i]` is the value to multiply `images[i]` to give raw image.
                `norm_factor[i]` is either an integer or an array of same dimensions as `image[i]`.
                If a single `norm_factor` given, assume same for each image.
            subplot_row_columns: `[n_rows, n_columns]`
                The subplots will be arranged into `n_rows` and `n_columns`.
                If not given, `n_columns` will be 1.
            fig_size: `[width, height]`
                Size of figure to plot in inches.
                If not given, will be set to `(9, 5)`.
            subplot_adjust: `[left, right, bottom, top]`
                The position of the sides of the subplots in the figure.
                I.e., we don't want subplot to overlap with cbar, slider or buttom and this ensures that.
                If not given, will be set to `[0.07, 0.775, 0.095, 0.94]`.
            cbar_pos: `[left, bottom, width, height]`
                Position of colour axis.
                If not given, will be set to `[0.9, 0.15, 0.03, 0.8]`.
            slider_pos: `[left, bottom, width, height]`
                Position of slider that controls colour axis.
                If not given, will be set to `[0.85, 0.15, 0.01, 0.8]`.
            button_pos: `[left, bottom, width, height]`
                Position of button which triggers change of normalisation.
                If not given, will be set to `[0.85, 0.02, 0.1, 0.05]`.
        """
        plt.style.use("dark_background")
        self.n_images = len(images)
        if subplot_row_columns is None:
            subplot_row_columns = [self.n_images, 1]
        # Default positions
        if fig_size is None:
            fig_size = (9, 5)
        if subplot_adjust is None:
            subplot_adjust = [0.07, 0.775, 0.095, 0.94]
        if cbar_pos is None:
            cbar_pos = [0.9, 0.15, 0.03, 0.8]
        if slider_pos is None:
            self.slider_pos = [0.85, 0.15, 0.01, 0.8]
        else:
            self.slider_pos = slider_pos
        if button_pos is None:
            button_pos = [0.85, 0.02, 0.1, 0.05]
        if not isinstance(norm_factor, list):
            # allow for different norm for each image
            if norm_factor is None:
                self.colour_norm = None
            else:
                self.colour_norm = [
                    norm_factor,
                ] * self.n_images
        else:
            self.colour_norm = norm_factor
        self.im_data = [val for val in images]  # put in order channels, rounds
        self.method = "norm" if self.colour_norm is not None else "raw"
        if self.colour_norm is None:
            self.caxis_info = {"norm": {}}
        else:
            self.caxis_info = {"norm": {}, "raw": {}}
        for key in self.caxis_info:
            if key == "raw":
                im_data = self.im_data
                self.caxis_info[key]["format"] = "%.0f"
            else:
                im_data = [self.im_data[i] * self.colour_norm[i] for i in range(self.n_images)]
                self.caxis_info[key]["format"] = "%.2f"
            self.caxis_info[key]["min"] = np.min([im.min() for im in im_data] + [-1e-20])
            self.caxis_info[key]["max"] = np.max([im.max() for im in im_data] + [1e-20])
            self.caxis_info[key]["max"] = np.max([self.caxis_info[key]["max"], -self.caxis_info[key]["min"]])
            # have equal either side of zero so small negatives don't look large
            self.caxis_info[key]["min"] = -self.caxis_info[key]["max"]
            self.caxis_info[key]["clims"] = [self.caxis_info[key]["min"], self.caxis_info[key]["max"]]
            # cmap_norm is so cmap is white at 0.
            self.caxis_info[key]["cmap_norm"] = mpl.colors.TwoSlopeNorm(
                vmin=self.caxis_info[key]["min"], vcenter=0, vmax=self.caxis_info[key]["max"]
            )

        self.fig, self.ax = plt.subplots(
            subplot_row_columns[0], subplot_row_columns[1], figsize=fig_size, sharex=True, sharey=True
        )
        if self.n_images == 1:
            self.ax = [self.ax]  # need it to be a list
        elif subplot_row_columns[0] > 1 and subplot_row_columns[1] > 1:
            self.ax = self.ax.flatten()  # only have 1 ax index
        oob_axes = np.arange(self.n_images, subplot_row_columns[0] * subplot_row_columns[1])
        if oob_axes.size > 0:
            for i in oob_axes:
                self.fig.delaxes(self.ax[i])  # delete excess subplots
            self.ax = self.ax[: self.n_images]
        self.fig.subplots_adjust(
            left=subplot_adjust[0], right=subplot_adjust[1], bottom=subplot_adjust[2], top=subplot_adjust[3]
        )
        self.im = [None] * self.n_images
        if self.im_data[0].ndim == 3:
            # For 3D data, start by showing just the first plane
            self.im_data_3d = self.im_data.copy()
            self.im_data = [val[:, :, 0] for val in self.im_data_3d]
            if self.colour_norm is not None:
                self.colour_norm_3d = self.colour_norm.copy()
                self.colour_norm = [val[:, :, 0] for val in self.colour_norm_3d]
        else:
            self.im_data_3d = None
            self.colour_norm_3d = None
        # initialise plots with a zero array
        for i in range(self.n_images):
            self.im[i] = self.ax[i].imshow(
                np.zeros(self.im_data[0].shape[:2]),
                cmap="seismic",
                aspect="auto",
                norm=self.caxis_info[self.method]["cmap_norm"],
            )
        cbar_ax = self.fig.add_axes(cbar_pos)  # left, bottom, width, height
        self.fig.colorbar(self.im[0], cax=cbar_ax)

        self.slider_ax = self.fig.add_axes(self.slider_pos)
        self.colour_slider = None
        if self.colour_norm is not None:
            self.norm_button_colour = "red"
            self.norm_button_colour_press = "white"
            if self.method == "raw":
                current_colour = self.norm_button_colour
            else:
                current_colour = self.norm_button_colour_press
            self.norm_button_ax = self.fig.add_axes(button_pos)
            self.norm_button = Button(self.norm_button_ax, "Norm", hovercolor="0.275")
            self.norm_button.label.set_color(current_colour)
            self.norm_button.on_clicked(self.change_norm)
        # set default to show normalised images
        self.change_norm()  # initialise with method = 'norm'

    def change_clim(self, val: List):
        """
        Function triggered on change of colour axis slider.

        Args:
            val: `[min_caxis, max_caxis]`
                colour axis of plots will be changed to these values.
        """
        if val[0] >= 0:
            # cannot have positive lower bound with diverging colourmap
            val[0] = -1e-20
        if val[1] <= 0:
            # cannot have negative upper bound with diverging colourmap
            val[1] = 1e-20
        self.caxis_info[self.method]["clims"] = val
        for im in self.im:
            im.set_clim(val[0], val[1])
        self.im[-1].axes.figure.canvas.draw()

    def change_norm(self, event=None):
        """
        Function triggered on press of normalisation button.
        Will either multiply or divide each image by the relevant `colour_norm`.
        """
        # need to make new slider at each button press because min/max will change
        self.slider_ax.remove()
        self.slider_ax = self.fig.add_axes(self.slider_pos)
        if self.colour_norm is not None:
            self.method = "norm" if self.method == "raw" else "raw"  # change to the other method
            if self.method == "raw":
                # Change colour of text when button pressed
                self.norm_button.label.set_color(self.norm_button_colour_press)
            else:
                self.norm_button.label.set_color(self.norm_button_colour)
        for i in range(self.n_images):
            # change image to different normalisation and change clim
            self.im[i].set_data(self.im_data[i] * self.colour_norm[i] if self.method == "norm" else self.im_data[i])
            self.im[i].set_norm(self.caxis_info[self.method]["cmap_norm"])
            self.im[i].set_clim(self.caxis_info[self.method]["clims"][0], self.caxis_info[self.method]["clims"][1])

        self.colour_slider = RangeSlider(
            ax=self.slider_ax,
            label="Clim",
            valmin=self.caxis_info[self.method]["min"],
            valmax=self.caxis_info[self.method]["max"],
            valinit=self.caxis_info[self.method]["clims"],
            orientation="vertical",
            valfmt=self.caxis_info[self.method]["format"],
        )
        self.colour_slider.on_changed(self.change_clim)
        self.im[-1].axes.figure.canvas.draw()


class ViewSpotColourAndCode(Subplot):
    def __init__(
        self,
        spot_no: int,
        spot_score: float,
        spot_tile: int,
        spot_colour: np.ndarray,
        gene_bled_code: np.ndarray,
        gene_index: int,
        gene_name: str,
        colour_norm_factor: np.ndarray,
        use_channels: list[int],
        method: str,
        show: bool = True,
    ):
        """
        Viewer subplot diagnostic to compare a spot's colour to the calculated gene bled code (`bled_code`). After
        background removal (if enabled), the spot's colour is divided by its L2 norm over all its values (all
        rounds/channels).

        Args:
            spot_no (int): index of spot in the notebook (number between 0 and n_spots - 1).
            spot_score (float): score of spot gene assignment.
            spot_tile (int): index of tile spot is on.
            spot_colour (`(n_rounds x n_channels_use) ndarray[float]`): spot colour before background removal.
            gene_bled_code (`(n_rounds x n_channels_use) ndarray[float]`): the spot's gene's final bled code.
            gene_index (int): the spot's gene's index.
            gene_name (str): the spot's gene's name.
            colour_norm_factor (`(n_tiles x n_rounds x n_channels_use) ndarray[float32]`): normalisation factor for
                each tile, round, and channel that is applied to colours.
            use_channels (list of int): sequencing channels used.
            method (str): spot's method. Can be 'anchor', 'omp' or 'prob'.
            show (bool, optional): show the plot after creating. Turn off for unit testing. Default: true.

        Notes:
            - Keep the class instance in a named variable when running this subplot. This ensures that the UI buttons
                continue to allow interaction.
        """
        assert method.lower() in ["anchor", "omp", "prob"], "method must be 'anchor', 'omp' or 'prob'"

        self.use_colour_norm_factor = True
        self.remove_background = True
        self.l2_normalise = True

        self.spot_tile = spot_tile
        self.gene_bled_code = gene_bled_code.astype(np.float32).copy()
        self.colour_norm_factor = colour_norm_factor.astype(np.float32).copy()
        self.spot_colour = spot_colour.astype(np.float32).copy()
        self.fig, self.axes = plt.subplots(2, 1, squeeze=False, sharex=True, sharey=True, layout="constrained")
        self.fig.supxlabel("Round")

        abs_max = np.max(
            [
                np.abs(self.spot_colour.copy()).max(),
                (np.abs(self.spot_colour.copy() * self.colour_norm_factor[spot_tile])).max(),
                np.abs(self.gene_bled_code.copy()).max(),
            ]
        )

        self.cmap = mpl.cm.seismic
        self.norm = mpl.colors.Normalize(vmin=-abs_max, vmax=abs_max)

        # Spot colour image.
        ax: plt.Axes = self.axes[0, 0]
        self.colour_im = ax.imshow(spot_colour.T, cmap=self.cmap, norm=self.norm)
        ax.set_title(
            f"Spot Colour\n{method.capitalize()} index {spot_no}, gene {gene_index} {gene_name}, "
            + f"score: {'{:.2f}'.format(spot_score)}"
        )
        ax.set_ylabel("Channel")
        ax.set_yticks(range(len(use_channels)), use_channels)

        # Predicted gene bled code image.
        ax: plt.Axes = self.axes[1, 0]
        im = ax.imshow(gene_bled_code.T, cmap=self.cmap, norm=self.norm)
        ax.set_title(f"Gene {gene_index}: {gene_name} predicted bled code")
        ax.set_xticks(range(spot_colour.shape[0]), range(spot_colour.shape[0]))
        ax.set_ylabel("Channel")
        ax.set_yticks(range(len(use_channels)), use_channels)

        # Colour bar on right.
        cbar_pos = [0.85, 0.4, 0.09, 0.5]  # left, bottom, width, height
        self.cbar_ax = self.fig.add_axes(cbar_pos)
        self.cbar = plt.colorbar(im, cax=self.cbar_ax, orientation="vertical", label="Intensity")

        self.button_colour_not_pressed = "red"
        self.button_colour_pressed = "green"
        self.use_colour_norm_button_ax = self.fig.add_axes([0.85, 0.25, 0.1, 0.05])
        self.use_colour_norm_button = Button(self.use_colour_norm_button_ax, "Colour Norm Factor", hovercolor="0.275")
        self.use_colour_norm_button.label.set_color(self.button_colour_pressed)
        self.use_colour_norm_button.on_clicked(self.change_use_colour_norm)
        self.background_button_ax = self.fig.add_axes([0.85, 0.05, 0.1, 0.05])
        self.background_button = Button(self.background_button_ax, "Background", hovercolor="0.275")
        self.background_button.label.set_color(self.button_colour_pressed)
        self.background_button.on_clicked(self.change_background)
        self.norm_button_ax = self.fig.add_axes([0.85, 0.15, 0.1, 0.05])
        self.norm_button = Button(self.norm_button_ax, "L2 Normalise", hovercolor="0.275")
        self.norm_button.label.set_color(self.button_colour_pressed)
        self.norm_button.on_clicked(self.change_norm)

        self.plot_colour()

        if show:
            self.fig.show()

    def plot_colour(self) -> None:
        plot_spot_colour = self.spot_colour.copy()
        if self.use_colour_norm_factor:
            plot_spot_colour *= self.colour_norm_factor[self.spot_tile]
        if self.remove_background:
            plot_spot_colour -= np.percentile(plot_spot_colour, 25, axis=0, keepdims=True)
        if self.l2_normalise:
            plot_spot_colour /= np.linalg.norm(plot_spot_colour, axis=(0, 1), keepdims=True)

        self.colour_im.set_data(plot_spot_colour.T)
        abs_max = np.max([np.abs(plot_spot_colour).max(), np.abs(self.gene_bled_code).max()])
        self.colour_im.set_clim(-abs_max, abs_max)
        self.fig.canvas.draw()

    def change_use_colour_norm(self, _=None) -> None:
        """
        Function triggered on press of colour norm factor button. Will either remove/add colour normalisation factor to
        spot_colour
        """
        self.use_colour_norm_factor = not self.use_colour_norm_factor
        if self.use_colour_norm_factor:
            self.use_colour_norm_button.label.set_color(self.button_colour_pressed)
        else:
            self.use_colour_norm_button.label.set_color(self.button_colour_not_pressed)
        self.plot_colour()

    def change_background(self, _=None) -> None:
        """
        Function triggered on press of background button. Will either remove/add background contribution to spot_colour
        """
        self.remove_background = not self.remove_background
        if self.remove_background:
            self.background_button.label.set_color(self.button_colour_pressed)
        else:
            self.background_button.label.set_color(self.button_colour_not_pressed)
        self.plot_colour()

    def change_norm(self, _=None) -> None:
        """
        Function triggered on press of l2 normalise button. Will either remove/add l2 normalisation of spot_colour.
        """
        self.l2_normalise = not self.l2_normalise
        if self.l2_normalise:
            self.norm_button.label.set_color(self.button_colour_pressed)
        else:
            self.norm_button.label.set_color(self.button_colour_not_pressed)
        self.plot_colour()


class ViewSpotColourRegion(Subplot):
    def __init__(
        self,
        spot_no: int,
        spot_score: float,
        spot_local_yxz: np.ndarray,
        spot_tile: int,
        gene_index: int,
        gene_name: str,
        filter_images: zarr.Array,
        flow: zarr.Array,
        affine: zarr.Array,
        colour_norm_factor: np.ndarray,
        use_rounds: list[int],
        use_channels: list[int],
        method: str,
        show: bool = True,
    ):
        """
        Build a grid of pyplot imshows for each channel and round showing the registered, filtered image intensity in a
        2D neighbourhood around the spot in Y and X. Out of bounds pixels are not plotted.

        Args:
            spot_no (int): index of spot in the notebook (number between 0 and n_spots - 1).
            spot_score (float): score of spot gene assignment.
            spot_local_yxz (`(3) ndarray[int]`): the spot's local y, x, and z position relative to the spot's tile's
                bottom-left corner.
            spot_tile (int): index of tile spot is on.
            gene_index (int): the spot's gene's index.
            gene_name (str): the spot's gene's name.
            filter_images (`(n_tiles x n_rounds x n_channels) zarray`): the filtered images.
            flow (): the register optical flow results.
            affine (): the register affine corrections.
            colour_norm_factor (`(n_tiles x n_rounds x n_channels_use) ndarray[float32]`): normalisation factor for
                each tile, round, and channel that is applied to colours.
            use_rounds (list of int): sequencing rounds.
            use_channels (list of int): sequencing channels.
            method (str): spot's method. Can be 'anchor', 'omp' or 'prob'.
            show (bool, optional): show the plot after creating. Turn off for unit testing. Default: true.
        """
        assert method.lower() in ["anchor", "omp", "prob"], "method must be 'anchor', 'omp' or 'prob'"
        self.local_region_shape_yx = (7, 7)
        assert all([shape % 2 == 1 for shape in self.local_region_shape_yx]), "Must be odd numbers only"
        self.use_colour_norm_factor = True
        self.remove_background = False
        self.l2_normalise = False

        self.n_rounds = len(use_rounds)
        self.n_channels = len(use_channels)
        Y = np.linspace(0, self.local_region_shape_yx[0] - 1, self.local_region_shape_yx[0])
        Y -= self.local_region_shape_yx[0] // 2
        Y += spot_local_yxz[0]
        X = np.linspace(0, self.local_region_shape_yx[1] - 1, self.local_region_shape_yx[1])
        X -= self.local_region_shape_yx[1] // 2
        X += spot_local_yxz[1]
        yxz_local_region = np.array(np.meshgrid(Y, X, [spot_local_yxz[2]], indexing="ij"), int)
        # Becomes shape (n_pixels, 3).
        yxz_local_region = yxz_local_region.reshape((3, -1), order="F").T
        colours = spot_colours_base.get_spot_colours_new(
            yxz_local_region, filter_images, flow, affine, int(spot_tile), use_rounds, use_channels
        )
        self.colours = colours.reshape(self.local_region_shape_yx + (self.n_rounds, self.n_channels), order="F")

        self.spot_tile = spot_tile
        self.colour_norm_factor = colour_norm_factor.astype(np.float32).copy()

        self.fig, self.axes = plt.subplots(self.n_channels, self.n_rounds, squeeze=False, sharex=True, sharey=True)
        self.fig.suptitle(
            f"Spot Colour Region\n{method.capitalize()} index {spot_no}, gene {gene_index} {gene_name}, score: "
            + f"{'{:.2f}'.format(spot_score)}"
        )
        self.fig.supxlabel("Round")

        # Spot colour images.
        abs_max = 0
        plot_colours = np.zeros(self.local_region_shape_yx + (self.n_rounds, self.n_channels), float)
        for c in range(self.n_channels):
            for r in range(self.n_rounds):
                rc_colours = self.colours[:, :, r, c].copy()
                if self.use_colour_norm_factor:
                    rc_colours *= self.colour_norm_factor[self.spot_tile, r, c]
                plot_colours[:, :, r, c] = rc_colours
        if self.remove_background:
            plot_colours -= np.percentile(plot_colours, 25, axis=3, keepdims=True)
        if self.l2_normalise:
            plot_colours /= np.linalg.norm(plot_colours, axis=(2, 3), keepdims=True)
        self.fig.supxlabel("Round")
        self.fig.supylabel("Channel")

        abs_max = np.abs(plot_colours).max()
        self.cmap = mpl.cm.seismic
        self.norm = mpl.colors.Normalize(vmin=-abs_max, vmax=+abs_max)

        for c in range(self.n_channels):
            for r in range(self.n_rounds):
                ax: plt.Axes = self.axes[c, r]
                ax.set_xticks([])
                ax.set_yticks([])
                im = ax.imshow(plot_colours[:, :, r, c].T, cmap=self.cmap, norm=self.norm)
        # Colour bar on right.
        cbar_pos = [0.88, 0.075, 0.06, 0.85]  # left, bottom, width, height
        self.cbar_ax = self.fig.add_axes(cbar_pos)
        self.cbar = plt.colorbar(im, cax=self.cbar_ax, pad=0.1, orientation="vertical", label="Intensity")

        if show:
            self.fig.show()


# We are now going to create a new class that will allow us to view the spots used to calculate the gene efficiency
# for a given gene. This will be useful for checking that the spots used are representative of the gene as a whole.
class GeneEfficiencyViewer:
    def __init__(self, nb: Notebook, mode: str = "prob", score_threshold: float = 0):
        """
        Diagnostic to show the n_genes x n_rounds gene efficiency matrix as a heatmap.

        Args:
            nb: Notebook containing experiment details. Must have run at least as far as `call_reference_spots`.
        """
        plt.style.use("dark_background")
        # Get gene probabilities and number of spots for each gene
        if mode == "omp":
            use_tiles = nb.basic_info.use_tiles
            gene_no = np.concatenate([nb.omp.results[f"tile_{t}"].gene_no for t in use_tiles], axis=0)
            score = np.concatenate([nb.omp.results[f"tile_{t}"].scores for t in use_tiles], axis=0)
        elif mode == "anchor":
            gene_no = nb.call_spots.dot_product_gene_no
            score = nb.call_spots.dot_product_gene_score
        else:
            gene_no = np.argmax(nb.call_spots.gene_probabilities, axis=1)
            score = np.max(nb.call_spots.gene_probabilities, axis=1)

        # Count the number of spots for each gene
        n_spots = np.zeros(nb.call_spots.gene_names.shape[0], dtype=int)
        for i in range(nb.call_spots.gene_names.shape[0]):
            n_spots[i] = np.sum((gene_no == i) & (score > score_threshold))

        # add attributes
        self.nb = nb
        self.n_genes = nb.call_spots.gene_names.shape[0]
        self.mode = mode
        self.n_spots = n_spots
        self.fig, self.ax = plt.subplots(figsize=(10, 10))

        # set location of axes
        self.ax.set_position([0.1, 0.1, 0.7, 0.8])
        gene_efficiency = np.linalg.norm(nb.call_spots.free_bled_codes_tile_independent, axis=2)
        self.ax.imshow(
            gene_efficiency, cmap="viridis", vmin=0, vmax=gene_efficiency.max(), aspect="auto", interpolation="none"
        )
        self.ax.set_xlabel("Round")
        self.ax.set_ylabel("Gene")
        self.ax.set_xticks(ticks=np.arange(gene_efficiency.shape[1]))
        self.ax.set_yticks([])

        # add colorbar
        self.ax.set_title("Gene Efficiency")
        cax = self.fig.add_axes([0.9, 0.1, 0.03, 0.8])
        cbar = self.fig.colorbar(self.ax.images[0], cax=cax)
        cbar.set_label("Gene Efficiency")

        # Adding gene names to y-axis would be too crowded. We will use mplcursors to show gene name of gene[r] when
        # hovering over row r of the heatmap. This means we need to only extract the y position of the mouse cursor.
        mplcursors.cursor(self.ax, hover=True).connect("add", lambda sel: self.plot_gene_name(sel.index[0]))
        # 2. Allow genes to be selected by clicking on them
        mplcursors.cursor(self.ax, hover=False).connect(
            "add", lambda sel: GeneSpotsViewer(nb, gene_index=sel.index[0], mode=mode, score_threshold=score_threshold)
        )
        # 3. We would like to add a white rectangle around the observed spot when we hover over it. We will
        # use mplcursors to do this. We need to add a rectangle to the plot when hovering over a gene.
        # We also want to turn off annotation when hovering over a gene so we will use the `hover=False` option.
        mplcursors.cursor(self.ax, hover=2).connect("add", lambda sel: self.add_rectangle(sel.index[0]))

        plt.show()

    def add_rectangle(self, index):
        # We need to remove any existing rectangles from the plot
        index = max(0, index)
        index = min(index, self.n_genes - 1)
        for rectangle in self.ax.patches:
            rectangle.remove()
        # We can then add a new rectangle to the plot
        self.ax.add_patch(Rectangle((-0.5, index - 0.5), self.nb.basic_info.n_rounds, 1, fill=False, edgecolor="white"))

    def plot_gene_name(self, index):
        # We need to remove any existing gene names from the plot
        index = max(0, index)
        index = min(index, self.n_genes - 1)
        for text in self.ax.texts:
            text.remove()
        # We can then add a new gene name to the top right of the plot in size 20 font
        self.ax.text(
            0.95,
            1.05,
            self.nb.call_spots.gene_names[index] + f" ({self.n_spots[index]} spots)",
            transform=self.ax.transAxes,
            size=20,
            horizontalalignment="right",
            verticalalignment="top",
            color="white",
        )


class GeneSpotsViewer:
    def __init__(self, nb: Notebook, gene_index: int = 0, mode: str = "prob", score_threshold: float = 0):
        """
        Diagnostic to show the spots used to calculate the gene efficiency for a given gene.
        Args:
            nb: Notebook containing experiment details. Must have run at least as far as `call_reference_spots`.
            gene_index: Index of gene to be plotted.
            mode: `'prob'` or `'anchor'` or `'omp'`.
                Which method of gene assignment used.
            score_threshold: Minimum score for a spot to be considered.

        """
        plt.style.use("dark_background")
        assert mode.lower() in ["prob", "anchor", "omp"], "mode must be 'prob', 'anchor' or 'omp'"
        assert nb.has_page("call_spots"), "Notebook must have run at least as far as `call_spots`"

        # Add attributes
        self.nb = nb
        self.mode = mode
        self.gene_index = gene_index
        self.score_threshold = score_threshold
        # Load spots
        self.spots, self.score, self.tile, self.spot_index, self.bled_code = None, None, None, None, None
        self.scatter_button, self.view_scatter_codes_button = None, None
        self.load_spots(gene_index)
        # Now initialise the plot, adding fig and ax attributes to the class
        self.fig, self.ax = plt.subplots(2, 1, figsize=(15, 10))
        self.fig_scatter, self.ax_scatter = None, None
        self.plot()
        plt.show()

    def load_spots(self, gene_index: int):
        # initialise variables
        nb = self.nb
        colour_norm = nb.call_spots.colour_norm_factor

        # get spots for gene gene_index
        if self.mode == "omp":
            use_tiles = nb.basic_info.use_tiles
            spots = np.concatenate([nb.omp.results[f"tile_{t}"].colours for t in use_tiles], axis=0)
            gene_no = np.concatenate([nb.omp.results[f"tile_{t}"].gene_no for t in use_tiles], axis=0)
            score = np.concatenate([nb.omp.results[f"tile_{t}"].scores for t in use_tiles], axis=0)
            tile = np.concatenate(
                [t * np.ones(nb.omp.results[f"tile_{t}"].scores.shape[0], dtype=int) for t in use_tiles]
            )
            spot_index = np.concatenate([np.arange(nb.omp.results[f"tile_{t}"].scores.shape[0]) for t in use_tiles])
        elif self.mode == "anchor":
            spots = nb.ref_spots.colours[:]
            gene_no = nb.call_spots.dot_product_gene_no[:]
            score = nb.call_spots.dot_product_gene_score[:]
            tile = nb.ref_spots.tile[:]
            spot_index = np.arange(nb.ref_spots.colours.shape[0])
        else:
            spots = nb.ref_spots.colours[:]
            gene_no = np.argmax(nb.call_spots.gene_probabilities[:], axis=1)
            score = np.max(nb.call_spots.gene_probabilities[:], axis=1)
            tile = nb.ref_spots.tile[:]
            spot_index = np.arange(nb.ref_spots.colours.shape[0])

        # get spots for gene gene_index with score > score_threshold for current mode and valid spots
        invalid = np.any(np.isnan(spots), axis=(1, 2))
        mask = (gene_no == gene_index) & (score > self.score_threshold) & (~invalid)
        spots = spots[mask] * colour_norm[tile[mask]]
        spots = spot_colours_base.remove_background(spots)[0]
        score = score[mask]
        # order spots by scores
        permutation = np.argsort(score)[::-1]
        spots = spots[permutation]
        score = score[permutation]
        spot_index = spot_index[mask][permutation]
        tile = tile[mask][permutation]

        # add attributes
        self.spots = spots.reshape(spots.shape[0], -1)
        self.score = score
        self.bled_code = nb.call_spots.bled_codes[gene_index].reshape(1, -1)
        self.spot_index = spot_index
        self.tile = tile

    def plot(self):
        for a in self.ax:
            a.clear()
        # Now we can plot the spots. We want to create 2 subplots. One with the spots observed and one with the expected
        # spots.
        vmin, vmax = np.percentile(self.spots, [3, 97])
        gene_code = self.nb.call_spots.gene_codes[self.gene_index]
        # we are going to find the mean cosine angle between observed and expected spots in each round
        n_rounds, n_channels = len(self.nb.basic_info.use_rounds), len(self.nb.basic_info.use_channels)
        mean_cosine = np.zeros(n_rounds)
        for r in range(n_rounds):
            colours_r = self.spots[:, r * n_channels : (r + 1) * n_channels].copy()
            colours_r /= np.linalg.norm(colours_r, axis=1)[:, None]
            bled_code_r = self.bled_code[0, r * n_channels : (r + 1) * n_channels].copy()
            bled_code_r /= np.linalg.norm(bled_code_r)
            mean_cosine[r] = np.mean(colours_r @ bled_code_r, axis=0)

        # We can then plot the spots observed and the spots expected.
        self.ax[0].imshow(self.spots, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto", interpolation="none")
        self.ax[1].imshow(self.bled_code, cmap="viridis", aspect="auto", interpolation="none")
        # We can then add titles and axis labels to the subplots.
        names = ["observed spots", "bled code"]
        for i, a in enumerate(self.ax):
            a.set_title(names[i])
            if i == 1:
                a.set_xlabel("Gene Colour")
            a.set_ylabel("Spot")
            # add x ticks of the round number
            n_rounds, n_channels = len(self.nb.basic_info.use_rounds), len(self.nb.basic_info.use_channels)
            x_tick_loc = np.arange(n_channels // 2, n_channels * n_rounds, n_channels)
            if i == 0:
                x_tick_label = [f"R {r} \n {str(np.around(mean_cosine[r], 2))}" for r in self.nb.basic_info.use_rounds]
            else:
                x_tick_label = [f"R {r}" for r in self.nb.basic_info.use_rounds]
            a.set_xticks(x_tick_loc, x_tick_label)
            a.set_yticks([])

        # We would like to add red vertical lines to show the start of each round.
        for j, a in enumerate(self.ax):
            for i in range(self.nb.basic_info.n_rounds):
                a.axvline(i * len(self.nb.basic_info.use_channels) - 0.5, color="r")

        # Set supertitle, colorbar and show plot
        self.fig.suptitle(
            f"Method: {self.mode}, Gene: {self.nb.call_spots.gene_names[self.gene_index]}, "
            f"(Code: {gene_code}) \n Score Threshold: {self.score_threshold:.2f}, "
            f"N: {self.spots.shape[0]}"
        )

        self.add_main_widgets()
        plt.show()

    def secondary_plot(self, event=None):
        # calculate probability of gene assignment and plot this against the score
        bled_codes = self.nb.call_spots.bled_codes
        n_genes, n_rounds, n_channels = bled_codes.shape
        kappa = np.log(1 + n_genes // 75) + 2
        gene_probs = gene_prob_score(
            spot_colours=self.spots.reshape(-1, n_rounds, n_channels), bled_codes=bled_codes, kappa=kappa
        )[:, self.gene_index]
        self.fig_scatter, self.ax_scatter = plt.subplots()
        spot_brightness = np.linalg.norm(self.spots, axis=1)
        self.ax_scatter.scatter(x=gene_probs, y=self.score, alpha=0.5, c=spot_brightness, cmap="viridis")
        self.ax_scatter.set_xlabel("Gene Probability")
        self.ax_scatter.set_ylabel("Gene Score")
        self.ax_scatter.set_title(
            f"Gene Probability vs Gene Score ({self.mode}) for Gene "
            f"{self.nb.call_spots.gene_names[self.gene_index]}"
        )
        # add colorbar
        cbar = self.fig_scatter.colorbar(cm.ScalarMappable(norm=None, cmap="viridis"), ax=self.ax_scatter)
        cbar.set_label("Spot Brightness")
        self.add_secondary_widgets(gene_probs)
        plt.show()

    def add_main_widgets(self):
        # Initialise buttons and cursors
        # 1. We would like each row of the plot to be clickable, so that we can view the observed spot.
        mplcursors.cursor(self.ax[0], hover=False).connect(
            "add",
            lambda sel: ViewSpotColourAndCode(
                self.nb, self.spot_index[sel.index[0]], tile=self.tile[sel.index[0]], method=self.mode
            ),
        )
        # 2. We would like to add a white rectangle around the observed spot when we hover over it
        mplcursors.cursor(self.ax[0], hover=2).connect("add", lambda sel: self.add_rectangle(sel.index[0]))
        # 3. add a button to view a scatter plot of score vs probability
        scatter_button_ax = self.fig.add_axes([0.925, 0.1, 0.05, 0.05])
        self.scatter_button = Button(scatter_button_ax, "S", hovercolor="0.275")
        self.scatter_button.on_clicked(self.secondary_plot)

    def add_secondary_widgets(self, gene_probs):
        # this functions adds widgets to the scatter plot figure and axes
        # 1. add a button to view the gene code
        view_code_button_ax = self.fig_scatter.add_axes([0.925, 0.1, 0.05, 0.05])
        self.view_scatter_codes_button = Button(view_code_button_ax, "C", hovercolor="0.275")
        self.view_scatter_codes_button.on_clicked(lambda event: self.view_scatter_codes(gene_probs, event))

    def add_rectangle(self, index):
        # We need to remove any existing rectangles from the plot
        index = max(0, index)
        index = min(index, self.spots.shape[0] - 1)
        for rectangle in self.ax[0].patches:
            rectangle.remove()
        # We can then add a new rectangle to the plot
        self.ax[0].add_patch(
            Rectangle(
                (-0.5, index - 0.5),
                self.nb.basic_info.n_rounds * len(self.nb.basic_info.use_channels),
                1,
                fill=False,
                edgecolor="white",
            )
        )

    def view_scatter_codes(self, gene_probs: np.ndarray, event=None):
        # this function will grab all visible spots and plot them in a new figure
        # get visible spots by the visible bounding box of ax_scatter
        bottom, top = self.ax_scatter.get_ylim()
        left, right = self.ax_scatter.get_xlim()
        visible_spots = np.where(
            (self.score >= bottom) & (self.score <= top) & (gene_probs >= left) & (gene_probs <= right)
        )[0]
        # plot these spots (if there are any, and not too many)
        if len(visible_spots) == 0:
            print("No spots in visible range")
        elif len(visible_spots) > 10:
            print("Too many spots to view")
        else:
            for s in visible_spots:
                ViewSpotColourAndCode(self.nb, self.spot_index[s], tile=self.tile[s], method=self.mode)


class ViewScalingAndBGRemoval:
    """
    This function will plot isolated spots raw, scaled and bg removed to show the effect of the bg removal and scaling.
    """

    def __init__(self, nb):
        plt.style.use("dark_background")
        self.nb = nb
        spot_tile = nb.ref_spots.tile[:]
        n_spots = spot_tile.shape[0]
        n_rounds, n_channels_use = len(nb.basic_info.use_rounds), len(nb.basic_info.use_channels)
        norm_factor = nb.call_spots.colour_norm_factor

        # get spot colours raw, no_bg and normed_no_bg
        spot_colour_raw = nb.ref_spots.colours[:].copy()
        spot_colour_normed = spot_colour_raw * norm_factor[spot_tile]
        bg = np.repeat(np.percentile(spot_colour_normed, 25, axis=1)[:, None, :], n_rounds, axis=1)
        spot_colour_normed_no_bg = spot_colour_normed - bg

        # Finally, we need to reshape the spots to be n_spots x n_rounds * n_channels. Since we want the channels to be
        # in consecutive blocks of size n_rounds, we can reshape by first switching the round and channel axes.
        # also order the spots by background noise in descending order
        max_spots = 10_000
        background_noise = np.sum(abs(bg), axis=(1, 2))
        colours = [spot_colour_raw, spot_colour_normed, spot_colour_normed_no_bg]
        for i, c in enumerate(colours):
            c = c[np.argsort(background_noise)[::-1]]
            c = c.transpose(0, 2, 1)
            c = c.reshape(n_spots, -1)
            colours[i] = c

        # We're going to make a little viewer to show spots before and after background subtraction and normalisation
        fig, ax = plt.subplots(2, 3, figsize=(10, 5))
        for i, c in enumerate(colours):
            min_intensity, max_intensity = np.percentile(c, [1, 99])
            ax[0, i].imshow(
                c[:max_spots],
                aspect="auto",
                vmin=min_intensity,
                vmax=max_intensity,
                interpolation="none",
            )
            ax[0, i].set_title(["Raw", "Scaled", "Scaled + BG Removed"][i])

        for i, c in enumerate(colours):
            bright_colours = np.percentile(c, 95, axis=0)
            bright_colours = bright_colours.reshape(n_channels_use, n_rounds).flatten()
            ax[1, i].plot(bright_colours, color="white")
            ax[1, i].set_ylim(0, np.max(bright_colours) * 1.1)
            ax[1, i].set_title("95th Percentile Brightness")

        for i, j in np.ndindex(2, 3):
            ax[i, j].set_xticks(
                [k * n_rounds + n_rounds // 2 for k in range(n_channels_use)], nb.basic_info.use_channels
            )
            if i == 0:
                ax[i, j].set_yticks([])
            # separate channels with a horizontal line
            for k in range(1, n_channels_use):
                ax[i, j].axvline(k * n_rounds - 0.5, color="Red", linestyle="--")

        # Add a title
        fig.suptitle("BG Removal + Scaling")

        plt.show()

    # add slider to allow us to vary value of interp between 0 and 1 and update plot
    # def add_hist_widgets(self):
    #     Add a slider on the right of the figure allowing the user to choose the percentile of the histogram
    #     to use as the maximum intensity. This slider should be the same dimensions as the colorbar and should
    #     be in the same position as the colorbar. We should slide vertically to change the percentile.
    # self.ax_slider = self.fig.add_axes([0.94, 0.15, 0.02, 0.6])
    # self.slider = Slider(self.ax_slider, 'Interpolation Coefficient', 0, 1, valinit=0, orientation='vertical')
    # self.slider.on_changed(lambda val: self.update_hist(int(val)))
    # TODO: Add 2 buttons, one for separating normalisation by channel and one for separating by round and channel
