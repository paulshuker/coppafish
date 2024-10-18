import importlib.resources as importlib_resources
from os import path
import time
from typing import Optional
import warnings

from PyQt5.QtCore import QLoggingCategory
from PyQt5.QtWidgets import QComboBox, QPushButton
import matplotlib as mpl
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import napari
import napari.components
import napari.components.viewer_model
from napari.layers import Points
import napari.layers
import napari.settings
from napari.utils.events import Selection
import numpy as np
import pandas as pd
from qtpy.QtCore import Qt
from superqt import QDoubleRangeSlider, QDoubleSlider

from . import legend_new
from .subplot import Subplot
from ..call_spots import bleed_matrix, spot_colours
from ..omp import ViewOMPImage
from ...omp import base as omp_base
from ...setup.notebook import Notebook, NotebookPage
from ...utils import system as utils_system
from .hotkeys_new import Hotkey


class Viewer:
    # Constants:
    _required_page_names: tuple[str] = ("basic_info", "filter", "register", "stitch", "ref_spots", "call_spots")
    _method_to_string: dict[str, str] = {"prob": "Probability", "anchor": "Anchor", "omp": "OMP"}
    _gene_legend_order_by_options: tuple[str] = ("row", "colour")
    _starting_score_thresholds: dict[str, tuple[float, float | None]] = {
        "prob": (0.5, None),
        "anchor": (0.5, None),
        "omp": (0.3, None),
    }
    _default_spot_size: float = 8.0
    _max_open_subplots: int = 5

    # Data:
    nbp_basic: NotebookPage
    nbp_filter: NotebookPage
    nbp_register: NotebookPage
    nbp_stitch: NotebookPage
    nbp_ref_spots: NotebookPage
    nbp_call_spots: NotebookPage
    nbp_omp: NotebookPage | None
    background_image: np.ndarray | None
    mip_background_image: np.ndarray | None
    background_image_layer: napari.layers.Image | None
    max_intensity_project: bool
    spot_data: dict[str, "Viewer.MethodData"]
    genes: tuple["Viewer.Gene"]
    selected_method: str
    selected_spot: int | None
    keep_zs: np.ndarray[bool]
    keep_scores: np.ndarray[bool]
    keep_intensities: np.ndarray[bool]
    keep_genes: np.ndarray[bool]
    z: int
    z_thick: float
    score_threshs: dict[str, tuple[float, float]]
    intensity_threshs: dict[str, tuple[float, float]]
    spot_size: float
    hotkeys: tuple[Hotkey]
    open_subplots: list[Subplot | Figure]
    closing: bool

    # UI variables:
    legend: legend_new.Legend
    point_layers: dict[str, Points]
    method_combo_box: QComboBox
    z_thick_slider: QDoubleSlider
    score_slider: QDoubleRangeSlider
    intensity_slider: QDoubleRangeSlider

    def __init__(
        self,
        nb: Optional[Notebook] = None,
        gene_marker_filepath: Optional[str] = None,
        gene_legend_order_by: str = "colour",
        background_image: Optional[str] = "dapi",
        background_image_colour: str = "gray",
        nbp_basic: Optional[NotebookPage] = None,
        nbp_filter: Optional[NotebookPage] = None,
        nbp_register: Optional[NotebookPage] = None,
        nbp_stitch: Optional[NotebookPage] = None,
        nbp_ref_spots: Optional[NotebookPage] = None,
        nbp_call_spots: Optional[NotebookPage] = None,
        nbp_omp: Optional[NotebookPage] = None,
        show: bool = True,
    ):
        """
        Instantiate a Viewer based on the given output data. The data can be given as a notebook or all the required
        notebook pages (used for unit testing).

        Args:
            nb (Notebook, optional): the notebook to visualise. Must have completed up to `call_spots` at least. If
                none, then all nbp_* notebook pages must be given except nbp_omp which is optional. Default: none.
            gene_marker_filepath (str, optional): the file path to the gene marker file. Default: use the default gene
                marker at coppafish/plot/results_viewer/gene_color.csv.
            gene_legend_order_by (str, optional): how to order the genes in the legend. Use "row" to order genes row by
                row in the gene marker file. "colour" will group genes based on their colourRGB's, each colour group is
                sorted by hue. Each gene name in a colour group is sorted alphabetically. Default: "colour".
            background_image (str or none, optional): what to use as the background image, can be none, "dapi" or a
                file path to a .npy file or .npz file. The array at a file path must be a numpy array of shape
                `(im_y x im_x)` or `(im_z x im_y x im_x)` If set to a .npz file path, the background image must be
                located at key 'arr_0'. Set to None for no background image. Default: "dapi".
            background_image_colour (str, optional): the napari colour mapping used for the background image. Default:
                "gray".
            nbp_basic (NotebookPage, optional): `basic_info` notebook page. Default: not given.
            nbp_filter (NotebookPage, optional): `filter` notebook page. Default: not given.
            nbp_register (NotebookPage, optional): `register` notebook page. Default: not given.
            nbp_stitch (NotebookPage, optional): `stitch` notebook page. Default: not given.
            nbp_ref_spots (NotebookPage, optional): `ref_spots` notebook page. Default: not given.
            nbp_call_spots (NotebookPage, optional): `call_spots` notebook page. Default: not given.
            nbp_omp (NotebookPage, optional): `omp` notebook page. OMP is not a required page. Default: not given.
            show (bool, optional): show the viewer once it is built. False for unit testing. Default: true.
        """
        assert type(nb) is Notebook or nb is None
        assert type(gene_marker_filepath) is str or gene_marker_filepath is None
        if gene_marker_filepath is not None and not path.isfile(gene_marker_filepath):
            raise FileNotFoundError(f"Could not find gene marker filepath at {gene_marker_filepath}")
        self.legend = legend_new.Legend()
        if gene_legend_order_by not in self.legend.order_by_options:
            raise ValueError(f"gene_legend_order_by must be one of {self.legend.order_by_options}")
        if background_image is not None and type(background_image) is not str:
            raise TypeError(f"background_image must be type str, got type {type(background_image)}")
        if background_image is not None and background_image not in ("dapi",) and type(background_image) is not str:
            raise ValueError(f"Unknown given background_image: {background_image} of type {type(background_image)}")
        if type(background_image) is str and background_image not in ("dapi",) and not path.isfile(background_image):
            raise FileNotFoundError(f"No background_image file at {background_image}")
        assert type(nbp_basic) is NotebookPage or nbp_basic is None
        assert type(nbp_filter) is NotebookPage or nbp_filter is None
        assert type(nbp_register) is NotebookPage or nbp_register is None
        assert type(nbp_stitch) is NotebookPage or nbp_stitch is None
        assert type(nbp_ref_spots) is NotebookPage or nbp_ref_spots is None
        assert type(nbp_call_spots) is NotebookPage or nbp_call_spots is None
        assert type(nbp_omp) is NotebookPage or nbp_omp is None
        assert type(show) is bool
        self.show = show
        self.closing = False
        self.z = None
        if nb is not None:
            if not all([nb.has_page(name) for name in self._required_page_names]):
                raise ValueError(f"The notebook requires pages {', '.join(self._required_page_names)}")
            self.nbp_basic = nb.basic_info
            self.nbp_filter = nb.filter
            self.nbp_register = nb.register
            self.nbp_stitch = nb.stitch
            self.nbp_ref_spots = nb.ref_spots
            self.nbp_call_spots = nb.call_spots
            self.nbp_omp = None
            if nb.has_page("omp"):
                self.nbp_omp = nb.omp
        else:
            self.nbp_basic = nbp_basic
            self.nbp_filter = nbp_filter
            self.nbp_register = nbp_register
            self.nbp_stitch = nbp_stitch
            self.nbp_ref_spots = nbp_ref_spots
            self.nbp_call_spots = nbp_call_spots
            self.nbp_omp = nbp_omp
            del nbp_basic, nbp_filter, nbp_register, nbp_stitch, nbp_ref_spots, nbp_call_spots, nbp_omp
        assert self.nbp_basic is not None
        assert self.nbp_filter is not None
        assert self.nbp_register is not None
        assert self.nbp_stitch is not None
        assert self.nbp_ref_spots is not None
        assert self.nbp_call_spots is not None
        del nb

        plt.style.use("dark_background")

        # Suppress any PyQt5 warnings.
        QLoggingCategory.setFilterRules("*.debug=false\n" + "*.warning=false\n" + "qt.qpa.*.warning=false")

        start_time = time.time()

        # Gather all spot data and keep in self.
        print("Gathering spot data")
        spot_data: dict[str, Viewer.MethodData] = {}
        spot_data["prob"] = self.MethodData()
        spot_data["prob"].tile = self.nbp_ref_spots.tile[:]
        spot_data["prob"].local_yxz = self.nbp_ref_spots.local_yxz[:].astype(np.int16)
        spot_data["prob"].yxz = (
            spot_data["prob"].local_yxz.astype(np.float32) + self.nbp_stitch.tile_origin[spot_data["prob"].tile]
        )
        spot_data["prob"].gene_no = np.argmax(self.nbp_call_spots.gene_probabilities[:], 1).astype(np.int16)
        spot_data["prob"].score = self.nbp_call_spots.gene_probabilities[:].max(1)
        spot_data["prob"].colours = self.nbp_ref_spots.colours[:].astype(np.float32)
        spot_data["prob"].intensity = self.nbp_call_spots.intensity[:]
        spot_data["anchor"] = self.MethodData()
        spot_data["anchor"].tile = spot_data["prob"].tile.copy()
        spot_data["anchor"].local_yxz = self.nbp_ref_spots.local_yxz[:].astype(np.int16)
        spot_data["anchor"].yxz = spot_data["prob"].yxz.copy()
        spot_data["anchor"].gene_no = self.nbp_call_spots.dot_product_gene_no[:]
        spot_data["anchor"].score = self.nbp_call_spots.dot_product_gene_score[:]
        spot_data["anchor"].colours = spot_data["prob"].colours.copy()
        spot_data["anchor"].intensity = self.nbp_call_spots.intensity[:]
        self.selected_method = "anchor"
        self.selected_spot = None
        if self.nbp_omp is not None:
            spot_data["omp"] = self.MethodData()
            spot_data["omp"].local_yxz, spot_data["omp"].tile = omp_base.get_all_local_yxz(self.nbp_basic, self.nbp_omp)
            spot_data["omp"].yxz = (
                spot_data["omp"].local_yxz.astype(np.float32) + self.nbp_stitch.tile_origin[spot_data["omp"].tile]
            )
            spot_data["omp"].gene_no = omp_base.get_all_gene_no(self.nbp_basic, self.nbp_omp)[0].astype(np.int16)
            spot_data["omp"].score = omp_base.get_all_scores(self.nbp_basic, self.nbp_omp)[0]
            spot_data["omp"].colours = omp_base.get_all_colours(self.nbp_basic, self.nbp_omp)[0].astype(np.float32)
            # OMP's intensity will be a similar scale to prob and anchor if the spot colours are colour normalised too.
            colours_normed = spot_data["omp"].colours * self.nbp_call_spots.colour_norm_factor[spot_data["omp"].tile]
            spot_data["omp"].intensity = np.median(colours_normed.max(-1), 1)
            self.selected_method = "omp"
        for method in spot_data.keys():
            spot_data[method].indices = np.linspace(
                0, spot_data[method].score.size - 1, spot_data[method].score.size, dtype=np.uint32
            )
        self.spot_data = spot_data
        # Sanity check spot data.
        for data in self.spot_data.values():
            data.check_variables()

        self.genes = self._create_gene_list(gene_marker_filepath)
        if len(self.genes) == 0:
            raise ValueError(f"None of your genes names are found in the gene marker file at {gene_marker_filepath}")

        # Remove spots that are not for a gene in the gene legend for a performance boost and simplicity.
        for method in self.spot_data.keys():
            spot_gene_numbers = self.spot_data[method].gene_no.copy()
            gene_indices = np.array([g.notebook_index for g in self.genes])
            spot_is_invisible = (spot_gene_numbers[:, None] != gene_indices[None]).all(1)
            self.spot_data[method].remove_data_at(spot_is_invisible)

        plt.style.use("dark_background")
        # + 1 for the gene legend.
        plt.rcParams["figure.max_open_warning"] = self._max_open_subplots + 1
        self.viewer = None
        if self.show:
            viewer_kwargs = dict(title=f"Coppafish {utils_system.get_software_version()} Viewer", show=False)
            self.viewer = napari.Viewer(**viewer_kwargs)

        print("Building gene legend")
        self.legend.create_gene_legend(self.genes, gene_legend_order_by)
        self.legend.canvas.mpl_connect("button_press_event", self.legend_clicked)
        if self.viewer_exists():
            self.viewer.window.add_dock_widget(self.legend.canvas, name="Gene Legend", area="left")
        self._update_gene_legend()

        print("Placing background image")
        self._place_background(background_image, background_image_colour)

        print("Building UI")
        self._build_UI()

        print("Placing spots")
        self.point_layers = {}
        for method in self.spot_data.keys():
            spot_gene_numbers = self.spot_data[method].gene_no.copy()
            gene_indices = np.array([g.notebook_index for g in self.genes])
            gene_symbols = np.array([g.symbol_napari for g in self.genes])
            gene_colours = np.array([g.colour for g in self.genes], np.float16)
            saved_gene_indices = (spot_gene_numbers[:, None] == gene_indices[None]).nonzero()[1]
            spot_symbols = gene_symbols[saved_gene_indices]
            spot_colours = gene_colours[saved_gene_indices]
            if self.viewer_exists():
                # Points are 2D to improve performance.
                self.point_layers[method] = self.viewer.add_points(
                    self.spot_data[method].yxz[:, :2],
                    symbol=spot_symbols,
                    face_color=spot_colours,
                    size=self.spot_size,
                    name=self._method_to_string[method],
                    ndim=2,
                    out_of_slice_display=False,
                    visible=False,
                )
                self.point_layers[method].mode = "PAN_ZOOM"
                # Know when a point is selected.
                self.point_layers[method].events.current_symbol.connect(self.selected_spot_changed)
        # Display the correct spot data based on current thresholds.
        if self.z is None:
            self.z = 0
        self._update_all_keep()
        self.update_viewer_data()
        if self.viewer_exists():
            self.viewer.reset_view()

        print(f"Connecting hotkeys")
        self.hotkeys = (
            Hotkey(
                "View hotkeys",
                "h",
                "",
                lambda _: self._add_subplot(self.view_hotkeys()),
                "Help",
                False,
            ),
            Hotkey(
                "Toggle background",
                "i",
                "Toggle the background image on and off",
                self.toggle_background,
                "Visual",
                False,
            ),
            Hotkey(
                "Toggle max intensity projection (default: off)",
                "o",
                "Toggle the background image's max intensity projection along z",
                self.toggle_max_intensity_project,
                "Visual",
                False,
            ),
            Hotkey(
                "View Bleed Matrix",
                "b",
                "Display the bleed matrix. This is an estimation of how each dye expresses itself in each channel",
                lambda _: self._add_subplot(self.view_bleed_matrix()),
                "General Diagnostics",
                requires_selection=False,
            ),
            Hotkey(
                "View Gene Bled Codes",
                "g",
                "Display the gene bled codes. Choose gene by inputting gene index or name",
                lambda _: self._add_subplot(self.view_gene_bled_codes()),
                "General Diagnostics",
                requires_selection=False,
            ),
            Hotkey(
                "View spot colour and code",
                "c",
                "Show the selected spot's colour and predicted bled code",
                lambda _: self._add_subplot(self.view_spot_colour_and_code()),
                "General Diagnostics",
            ),
            Hotkey(
                "View spot colour region",
                "r",
                "Show the selected spot's colour in a local region centred around it",
                lambda _: self._add_subplot(self.view_spot_colour_region()),
                "General Diagnostics",
            ),
            Hotkey(
                "View OMP Coefficients",
                "v",
                "Show the OMP coefficients around the selected spot's local region",
                lambda _: self._add_subplot(self.view_omp_coefficients()),
                "OMP",
            ),
        )
        # Hotkeys can be connected to a function when they occur.
        for hotkey in self.hotkeys:
            if hotkey.invoke is None or not self.viewer_exists():
                continue
            self.viewer.bind_key(hotkey.key_press)(hotkey.invoke)

        # Give the Viewer a larger window.
        if self.viewer_exists():
            self.viewer.window.resize(1400, 900)
            self.viewer.window.activate()

        # When subplots open, some of them need to be kept within the Viewer class to avoid garbage collection.
        # The garbage collection breaks the UI elements like buttons and sliders.
        self.open_subplots = list()

        end_time = time.time()
        print(f"Viewer built in {'{:.1f}'.format(end_time - start_time)}s")

        if self.viewer_exists():
            self.viewer.show()
            try:
                napari.run()
            except KeyboardInterrupt as exception:
                # When keyboard interrupted, close the viewer down properly before interrupting.
                print("Closing")
                self.close()
                raise exception

    def selected_spot_changed(self) -> None:
        self._set_status_to("")
        selected_data: Selection = self.point_layers[self.selected_method].selected_data
        self.selected_spot = selected_data.active
        if self.selected_spot is None:
            return
        index, _, local_yxz, tile, gene_no, score, _ = self._get_selection_data()
        message = (
            f"Selected spot: {index} at {tuple(local_yxz)}, tile {tile}, gene {gene_no}: "
            + f"{self.nbp_call_spots.gene_names[gene_no]}, score {score}"
        )
        print(message)
        self._set_status_to(message)

    def legend_clicked(self, event: mpl.backend_bases.MouseEvent) -> None:
        if event.inaxes != self.legend.canvas.axes:
            # Click event did not occur within the legend axes.
            return
        closest_gene_index = self.legend.get_closest_gene_index_to(event.xdata, event.ydata)
        if closest_gene_index is None:
            return
        closest_gene = self.genes[closest_gene_index]
        if event.button.name == "LEFT":
            # Toggle the gene on and off that was clicked on.
            closest_gene.active = not closest_gene.active
        elif event.button.name == "RIGHT":
            already_isolated = all([not gene.active for gene in self.genes if gene != closest_gene])
            if closest_gene.active:
                if already_isolated:
                    for gene in self.genes:
                        gene.active = True
                else:
                    for gene in self.genes:
                        gene.active = False
                    closest_gene.active = True
            else:
                for gene in self.genes:
                    gene.active = True
        else:
            return
        self._update_gene_keep()
        self.update_viewer_data()
        self._update_gene_legend()

    def z_slider_changed(self, _) -> None:
        # Called when the user changes the z slider in the napari viewer.
        if self.closing:
            # For some god forsaken reason this function is sometimes called when closing the viewer...
            # This is probably an issue I should raise on napari's github if I can make it simple & reproducible.
            return
        new_z = self.viewer.dims.current_step[0]
        if new_z == self.z:
            return
        self.z = new_z
        z_coords = self.spot_data[self.selected_method].local_yxz[:, 2]
        self.keep_zs = ((self.z - self.z_thick) <= z_coords) & (z_coords <= (self.z + self.z_thick))
        self.update_viewer_data()

    def method_changed(self) -> None:
        new_selected_method = list(self.spot_data.keys())[self.method_combo_box.currentIndex()]
        # Only update data if the method changed value.
        if new_selected_method == self.selected_method:
            return
        self.selected_method = new_selected_method
        self.update_widget_values()
        self._update_all_keep()
        self.update_viewer_data()
        self.clear_spot_selections()
        # Put the user back to pan/zoom mode.
        self.viewer.camera.interactive = True
        print(f"Method: {self.selected_method}")

    def z_thick_changed(self) -> None:
        new_z_thickness = self.z_thick_slider.value()
        # Only update data if the slider changed value.
        if new_z_thickness == self.z_thick:
            return
        self.z_thick = new_z_thickness
        self._update_z_keep()
        self.update_viewer_data()
        print(f"Z Thickness: {self.z_thick}")

    def score_thresholds_changed(self) -> None:
        new_score_thresholds = self.score_slider.value()
        if new_score_thresholds == self.score_threshs[self.selected_method]:
            return
        self.score_threshs[self.selected_method] = new_score_thresholds
        self._update_score_keep()
        self.update_viewer_data()
        print(f"Score thresholds: {self.score_threshs[self.selected_method]}")

    def intensity_thresholds_changed(self) -> None:
        new_intensity_thresholds = self.intensity_slider.value()
        if new_intensity_thresholds == self.intensity_threshs[self.selected_method]:
            return
        self.intensity_threshs[self.selected_method] = new_intensity_thresholds
        self._update_intensity_keep()
        self.update_viewer_data()
        print(f"Intensity thresholds: {self.intensity_threshs[self.selected_method]}")

    def marker_size_changed(self) -> None:
        new_spot_size = self.marker_size_slider.value()
        if new_spot_size == self.spot_size:
            return
        self.spot_size = new_spot_size
        self.set_spot_size_to(self.spot_size)

    def contrast_limits_changed(self) -> None:
        new_contrast_limits = self.contrast_slider.value()
        if new_contrast_limits == self.contrast_limits[self.max_intensity_project]:
            return
        self.contrast_limits[self.max_intensity_project] = new_contrast_limits
        self.background_image_layer.contrast_limits = self.contrast_limits[self.max_intensity_project]

    def update_widget_values(self) -> None:
        """
        Called when the method changes. The function refreshes the widget selected values since each method remembers
        different thresholds for convenience.
        """
        self.score_slider.setValue(self.score_threshs[self.selected_method])
        self.intensity_slider.setValue(self.intensity_threshs[self.selected_method])

    def update_viewer_data(self) -> None:
        """
        Called when the viewed spot data has changed. This happens when the selected method, z thickness, intensity
        threshold, or score threshold changes. Called when the Viewer first opens too.
        """
        self.clear_spot_selections()
        if not self.viewer_exists():
            return
        for method in self.spot_data.keys():
            if method != self.selected_method:
                self.point_layers[method].visible = False
                continue
            keep = self.keep_scores & self.keep_intensities & self.keep_zs & self.keep_genes
            self.point_layers[method].shown = keep

            self.point_layers[method].visible = True
            # To allow points on the method layer to be selected, the layer must be selected.
            self.viewer.layers.selection.active = self.point_layers[method]

    def clear_spot_selections(self) -> None:
        if not self.viewer_exists():
            return
        for method in self.spot_data.keys():
            if self.point_layers[method].selected_data.active is None:
                continue
            self.point_layers[method].selected_data.clear()
        self._set_status_to("")

    def set_spot_size_to(self, new_size: float) -> None:
        """
        Update the spot sizes in the napari Viewer. This is purely visual.

        Args:
            - new_size (float): the new spot size.
        """
        for method in self.spot_data.keys():
            self.point_layers[method].size = new_size

    # ========== HOTKEY FUNCTIONS ==========
    def view_hotkeys(self) -> Subplot:
        self._free_subplot_spaces()
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.set_title("Hotkeys", fontdict={"size": 20})
        ax.set_axis_off()
        text = ""
        unique_sections = []
        for hotkey in self.hotkeys:
            if hotkey.section not in unique_sections:
                unique_sections.append(hotkey.section)
        first_section = unique_sections[0]
        for section in unique_sections:
            if section != first_section:
                text += "\n"
            text += section.capitalize() + "\n"
            section_hotkeys = [hotkey for hotkey in self.hotkeys if hotkey.section == section]
            for hotkey in section_hotkeys:
                text += str(hotkey) + "\n"
        ax.text(0.5, 0.5, text, size=12, va="center", ha="center")
        if self.show:
            fig.show()
        return fig

    def view_spot_colour_and_code(self) -> Subplot | None:
        if self.selected_spot is None:
            return
        self._free_subplot_spaces()
        index, _, _, tile, gene_no, score, colour = self._get_selection_data()
        return spot_colours.ViewSpotColourAndCode(
            index,
            score,
            tile,
            colour,
            self.nbp_call_spots.bled_codes[gene_no],
            gene_no,
            self.nbp_call_spots.gene_names[gene_no],
            self.nbp_call_spots.colour_norm_factor,
            self.nbp_basic.use_channels,
            self.selected_method,
            show=self.show,
        )

    def view_bleed_matrix(self) -> Subplot:
        self._free_subplot_spaces()
        return bleed_matrix.ViewBleedMatrix(self.nbp_basic, self.nbp_call_spots, show=self.show)

    def view_gene_bled_codes(self) -> Subplot:
        self._free_subplot_spaces()
        return bleed_matrix.ViewBledCodes(self.nbp_basic, self.nbp_call_spots, show=self.show)

    def view_spot_colour_region(self) -> Subplot | None:
        if self.selected_spot is None:
            return
        self._free_subplot_spaces()
        index, _, local_yxz, tile, gene_no, score, _ = self._get_selection_data()
        return spot_colours.ViewSpotColourRegion(
            index,
            score,
            local_yxz,
            tile,
            gene_no,
            self.nbp_call_spots.gene_names[gene_no],
            self.nbp_filter.images,
            self.nbp_register.flow,
            self.nbp_register.icp_correction,
            self.nbp_call_spots.colour_norm_factor,
            self.nbp_basic.use_rounds,
            self.nbp_basic.use_channels,
            self.selected_method,
            show=self.show,
        )

    def view_omp_coefficients(self) -> Subplot | None:
        if self.selected_spot is None:
            return
        if self.nbp_omp is None:
            return
        self._free_subplot_spaces()
        spot_data = self.spot_data[self.selected_method]
        return ViewOMPImage(
            self.nbp_basic,
            self.nbp_filter,
            self.nbp_register,
            self.nbp_call_spots,
            self.nbp_omp,
            spot_data.local_yxz[self.selected_spot],
            spot_data.tile[self.selected_spot],
            spot_data.indices[self.selected_spot],
            spot_data.gene_no[self.selected_spot],
            self.selected_method,
            show=self.show,
        )

    def toggle_background(self, _=None) -> None:
        if not self.viewer_exists():
            return
        if self.background_image_layer is None:
            return
        self.background_image_layer.visible = not self.background_image_layer.visible

    def toggle_max_intensity_project(self, _=None) -> None:
        if not self.viewer_exists():
            return
        if self.background_image_layer is None:
            return
        self.max_intensity_project = not self.max_intensity_project
        data = self.background_image
        if self.max_intensity_project:
            data = self.mip_background_image
        print(f"Max Intensity Projection: {self.max_intensity_project}")
        self.background_image_layer.data = data
        self.background_image_layer.contrast_limits = self.contrast_limits[self.max_intensity_project]
        self.contrast_slider.setValue(self.background_image_layer.contrast_limits)

    # ======================================

    def viewer_exists(self) -> bool:
        return self.viewer is not None

    def close_all_subplots(self) -> None:
        """
        Close the currently open subplots. Saves memory and avoid warnings.
        """
        for _ in self.open_subplots:
            self._close_oldest_subplot()

    def close(self) -> None:
        """
        Close the entire Viewer.
        """
        if not self.viewer_exists():
            return
        self.close_all_subplots()
        self.closing = True
        self.viewer.close()
        del self.viewer

    def _place_background(self, image: Optional[str], colour_map: str) -> None:
        z_count = max(self.nbp_basic.use_z)
        self.background_image_layer = None
        self.max_intensity_project = False
        if image == "dapi":
            self.background_image = self.nbp_stitch.dapi_image[:]
            self.background_image_layer = self.background_image
        elif type(image) is str and image.endswith(".npy"):
            if not path.isfile(image):
                raise FileNotFoundError(f"Cannot find background image of file path: {image}")
            self.background_image: np.ndarray = np.load(image)
            if self.background_image.ndim not in (2, 3):
                raise ValueError(f"Unexpected background image dimension count: {self.background_image.ndim}")
            if self.background_image.ndim == 2:
                self.background_image = self.background_image[None].repeat(z_count, 0)
            self.background_image_layer = self.background_image
        elif type(image) is str and image.endswith(".npz"):
            if not path.isfile(image):
                raise FileNotFoundError(f"Cannot find background image of file path: {image}")
            self.background_image: np.ndarray = np.load(image)["arr_0"]
            if self.background_image.ndim not in (2, 3):
                raise ValueError(f"Unexpected background image dimension count: {self.background_image.ndim}")
            if self.background_image.ndim == 2:
                self.background_image = self.background_image[None].repeat(z_count, 0)
            self.background_image_layer = self.background_image
        elif type(image) is str:
            raise ValueError(
                f"Unexpected background_image: {image}. " + "The image must end with .npy or .npz or be equal to dapi"
            )
        if self.background_image_layer is not None:
            z_count = self.background_image.shape[0]
            self.mip_background_image = self.background_image.copy().max(0, keepdims=True).repeat(z_count, 0)
            # Keep the max intensity projected background image in self.
            self.mip_background_image = self.mip_background_image
            if not self.viewer_exists():
                return
            self.background_image_layer = self.viewer.add_image(
                self.background_image, rgb=False, axis_labels=("Z", "Y", "X"), colormap=colour_map
            )
        elif self.viewer_exists():
            # Place a blank, 3D image to make the napari Viewer have the z slider.
            self.viewer.add_image(np.zeros((z_count, 1, 1), dtype=np.int8), rgb=False)

    def _build_UI(self) -> None:
        min_yxz = np.array([0, 0, 0], np.float32)
        max_yxz = np.array([self.nbp_basic.tile_sz, self.nbp_basic.tile_sz, max(self.nbp_basic.use_z)], np.float32)
        max_score = 1.0
        max_intensity = 1.0
        for method in self.spot_data.keys():
            method_max_score = self.spot_data[method].score.max()
            if method_max_score > max_score:
                max_score = method_max_score
            method_max_intensity = self.spot_data[method].intensity.max()
            if method_max_intensity > max_intensity:
                max_intensity = method_max_intensity
            method_min_yxz = self.spot_data[method].yxz.min(0)
            method_max_yxz = self.spot_data[method].yxz.max(0)
            min_yxz = min_yxz.clip(max=method_min_yxz)
            max_yxz = max_yxz.clip(min=method_max_yxz)

        # Initial UI Widget values.
        self.z_thick: float = 1.0
        self.score_threshs = {method: self._starting_score_thresholds[method] for method in self.spot_data.keys()}
        for method, score_thresh in self.score_threshs.items():
            if score_thresh[1] is None:
                self.score_threshs[method] = (score_thresh[0], max_score)
        self.intensity_threshs = {method: (0.0, max_intensity) for method in self.spot_data.keys()}
        self.spot_size = self._default_spot_size

        if self.viewer_exists():
            # Method selection as a dropdown box containing every gene call method available.
            self.method_combo_box = QComboBox()
            for method in self.spot_data.keys():
                self.method_combo_box.addItem(self._method_to_string[method])
            self.method_combo_box.setCurrentText(self._method_to_string[self.selected_method])
            self.method_combo_box.currentIndexChanged.connect(self.method_changed)
            # Z thickness slider.
            self.z_thick_slider = QDoubleSlider(Qt.Orientation.Horizontal)
            self.z_thick_slider.setRange(0, max_yxz[2] - min_yxz[2])
            self.z_thick_slider.setValue(self.z_thick)
            self.z_thick_slider.sliderReleased.connect(self.z_thick_changed)
            # Score slider. Keep a separate score threshold for each method.
            self.score_slider = QDoubleRangeSlider(Qt.Orientation.Horizontal)
            self.score_slider.setRange(0, max_score)
            self.score_slider.setValue(self.score_threshs[self.selected_method])
            self.score_slider.sliderReleased.connect(self.score_thresholds_changed)
            # Intensity slider. Keep a separate intensity threshold for each method.
            self.intensity_slider = QDoubleRangeSlider(Qt.Orientation.Horizontal)
            self.intensity_slider.setRange(0, max_intensity)
            self.intensity_slider.setValue(self.intensity_threshs[self.selected_method])
            self.intensity_slider.sliderReleased.connect(self.intensity_thresholds_changed)
            # Marker size slider. For visuals only.
            self.marker_size_slider = QDoubleSlider(Qt.Orientation.Horizontal)
            self.marker_size_slider.setRange(self.spot_size / 4, self.spot_size * 4)
            self.marker_size_slider.setValue(self.spot_size)
            self.marker_size_slider.sliderReleased.connect(self.marker_size_changed)
            self.viewer.window.add_dock_widget(self.method_combo_box, area="left", name="Gene Call Method")
            self.viewer.window.add_dock_widget(self.z_thick_slider, area="left", name="Z Thickness")
            self.viewer.window.add_dock_widget(self.score_slider, area="left", name="Score Thresholds")
            self.viewer.window.add_dock_widget(self.intensity_slider, area="left", name="Intensity Thresholds")
            self.viewer.window.add_dock_widget(self.marker_size_slider, area="left", name="Marker Size")
        if self.background_image_layer is not None:
            # Background image contrast limits for Max Intensity Projection (MIP) true and MIP false.
            starting_limits = (self.background_image.min(), self.background_image.max())
            self.contrast_limits = {False: starting_limits, True: starting_limits}
            if self.viewer_exists():
                self.contrast_slider = QDoubleRangeSlider(Qt.Orientation.Horizontal)
                self.contrast_slider.setRange(*self.contrast_limits[self.max_intensity_project])
                self.contrast_slider.setValue(self.contrast_limits[self.max_intensity_project])
                self.contrast_slider.sliderReleased.connect(self.contrast_limits_changed)
                self.viewer.window.add_dock_widget(self.contrast_slider, area="left", name="Background Contrast")
        if self.viewer_exists():
            # View hotkeys button.
            self.view_hotkeys_button = QPushButton(text="Hotkeys")
            self.view_hotkeys_button.clicked.connect(self.view_hotkeys)
            self.viewer.window.add_dock_widget(self.view_hotkeys_button, area="left", name="Help")
            # Hide the layer list and layer controls.
            # FIXME: This leads to a future deprecation warning. Napari will hopefully add a proper way of doing this
            # in napari >= 0.6.0.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                # Turn off layer list and layer controls.
                self.viewer.window.qt_viewer.dockLayerList.hide()
                self.viewer.window.qt_viewer.dockLayerControls.hide()

            self.z = self.viewer.dims.current_step[0]
            # Connect to z slider changing event.
            self.viewer.dims.events.current_step.connect(self.z_slider_changed)

    def _add_subplot(self, subplot: Figure | Subplot | None) -> None:
        if subplot is None:
            return
        self.open_subplots.append(subplot)

    def _get_selection_data(self) -> tuple[np.ndarray]:
        # Get the currently selected spot's data.
        assert self.selected_spot is not None
        spot_data = self.spot_data[self.selected_method]
        index = spot_data.indices[self.selected_spot]
        local_yxz = spot_data.local_yxz[self.selected_spot]
        yxz = spot_data.yxz[self.selected_spot]
        tile = spot_data.tile[self.selected_spot]
        gene_no = spot_data.gene_no[self.selected_spot]
        score = spot_data.score[self.selected_spot]
        colour = spot_data.colours[self.selected_spot]

        return index, yxz, local_yxz, tile, gene_no, score, colour

    def _create_gene_list(self, gene_marker_filepath: Optional[str] = None) -> tuple["Viewer.Gene"]:
        """
        Create a tuple of genes from the notebook to store information about each gene. This will be saved at
        `self.genes`. Each element of the tuple will be a Viewer.Gene class object. So it will contain the name, colour
        and symbols for each gene.

        Args:
            - gene_marker_file (str, optional), path to csv file containing marker and color for each gene. There must
                be 6 columns in the csv file with the following headers (comma separated)
                    * ID - int, unique number for each gene, in ascending order
                    * GeneNames - str, name of gene with first letter capital
                    * ColorR - float, Rgb color for plotting
                    * ColorG - float, rGb color for plotting
                    * ColorB - float, rgB color for plotting
                    * napari_symbol - str, symbol used to plot in napari
                All RGB values must be between 0 and 1. The first line must be the heading names. Default: use the
                default gene marker file found at coppafish/plot/results_viewer/gene_color.csv.

        Returns:
            (tuple of Viewer.Gene) genes: every genes Gene object.
        """
        if gene_marker_filepath is None:
            gene_marker_filepath = importlib_resources.files("coppafish.plot.results_viewer").joinpath("gene_color.csv")
        if not path.isfile(gene_marker_filepath):
            raise FileNotFoundError(f"Could not find gene marker file at {gene_marker_filepath}")
        gene_legend_info = pd.read_csv(gene_marker_filepath)
        legend_gene_names = gene_legend_info["GeneNames"].values
        genes: list[Viewer.Gene] = []
        invisible_genes = []

        # Create a list of genes with the relevant information. If the gene is not in the gene marker file, it will not
        # be added to the list.
        for i, g in enumerate(self.nbp_call_spots.gene_names):
            if g not in legend_gene_names:
                invisible_genes.append(g)
                continue
            colour = gene_legend_info[gene_legend_info["GeneNames"] == g][["ColorR", "ColorG", "ColorB"]].values[0]
            symbol_napari = gene_legend_info[gene_legend_info["GeneNames"] == g]["napari_symbol"].values[0]
            new_gene: Viewer.Gene = self.Gene(
                name=g, notebook_index=i, colour=colour, symbol_napari=symbol_napari, active=True
            )
            genes.append(new_gene)

        # Warn if any genes are not in the gene marker file.
        if invisible_genes:
            gene_string = "'" + "', '".join(invisible_genes) + "'"
            print(f"Gene(s) {gene_string} are not in the gene marker file and will not be plotted.")

        return tuple(genes)

    def _update_all_keep(self) -> None:
        self._update_z_keep()
        self._update_score_keep()
        self._update_intensity_keep()
        self._update_gene_keep()

    def _update_z_keep(self) -> None:
        z_coords = self.spot_data[self.selected_method].yxz[:, 2]
        min_z = self.z - self.z_thick
        max_z = self.z + self.z_thick
        self.keep_zs = (z_coords >= min_z) & (z_coords <= max_z)

    def _update_score_keep(self) -> None:
        scores = self.spot_data[self.selected_method].score
        score_threshs = self.score_threshs[self.selected_method]
        self.keep_scores = (scores >= score_threshs[0]) & (scores <= score_threshs[1])

    def _update_intensity_keep(self) -> None:
        intensities = self.spot_data[self.selected_method].intensity
        intensity_threshs = self.intensity_threshs[self.selected_method]
        self.keep_intensities = (intensities >= intensity_threshs[0]) & (intensities <= intensity_threshs[1])

    def _update_gene_keep(self) -> None:
        gene_numbers = self.spot_data[self.selected_method].gene_no
        active_gene_numbers = np.array([gene.notebook_index for gene in self.genes if gene.active], np.int16)
        self.keep_genes = (gene_numbers[:, np.newaxis] == active_gene_numbers[np.newaxis]).any(1)

    def _free_subplot_spaces(self, n_free_spaces: int = 1) -> None:
        """
        If there are too many subplots open, then the oldest subplots are closed until there is n_free_spaces free
        spaces.
        """
        while (len(self.open_subplots) + n_free_spaces) > self._max_open_subplots:
            self._close_oldest_subplot()

    def _close_oldest_subplot(self) -> None:
        subplot = self.open_subplots.pop(0)
        if isinstance(subplot, Subplot):
            subplot.close()
        elif isinstance(subplot, Figure):
            plt.close(subplot)
        else:
            raise TypeError(f"Unkown subplot type: {type(subplot)}")

    def _update_gene_legend(self) -> None:
        # Called when the gene selection has changed by user input
        self.legend.update_selected_legend_genes([g.active for g in self.genes])

    def _set_status_to(self, message: str) -> None:
        # Sets the status bar of the viewer to a new message.
        self.viewer.status = message

    # A nested class. Each instance of this class holds data on a specific gene calling method.
    class MethodData:
        _attribute_names = ("tile", "local_yxz", "yxz", "gene_no", "score", "colours", "intensity", "indices")
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

        def remove_data_at(self, remove: np.ndarray[bool]) -> None:
            """
            Delete the i'th spot data if remove[i] == True.
            """
            assert type(remove) is np.ndarray
            assert remove.ndim == 1
            assert remove.size == self.tile.size
            keep_sum = (~remove).sum().item()
            for var_name in self._attribute_names:
                self.__setattr__(var_name, self.__getattribute__(var_name)[~remove])
                assert self.__getattribute__(var_name).shape[0] == keep_sum
            self.check_variables()

        def check_variables(self) -> None:
            assert all([type(self.__getattribute__(var_name)) is np.ndarray] for var_name in self._attribute_names)
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

    class Gene:
        def __init__(
            self,
            name: str,
            notebook_index: int,
            colour: np.ndarray,
            symbol_napari: str,
            active: bool = True,
        ):
            """
            Instantiate data for a single gene.

            Args:
                name: (str) gene name.
                notebook_index: (int) index of the gene within the notebook.
                colour: (np.ndarray) of shape (3,) with the RGB colour of the gene.
                symbol_napari: (str) symbol used to plot in napari.
                active: (bool, optional) whether the gene is currently visible in the Viewer. Used for toggling gene
                    visibility.
            """
            self.name = name
            self.notebook_index = notebook_index
            self.colour = colour
            self.symbol_napari = symbol_napari
            self.active = active
