import importlib.resources as importlib_resources
from os import path
import time
from typing import Any, Optional
import warnings

from PyQt5.QtWidgets import QComboBox, QPushButton
import matplotlib as mpl
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
from ..call_spots import spot_colours
from ..omp import ViewOMPImage
from ...omp import base as omp_base
from ...setup.notebook import Notebook, NotebookPage
from ...utils import system as utils_system
from .hotkeys_new import Hotkey


class Viewer:
    # Constants:
    _required_page_names: tuple[str] = ("basic_info", "filter", "register", "stitch", "ref_spots", "call_spots")
    _method_to_string: dict[str, str] = {"prob": "Probability", "anchor": "Anchor", "omp": "OMP"}
    _starting_score_thresholds: dict[str, tuple[float]] = {"prob": (0.5, 1.0), "anchor": (0.5, 1.0), "omp": (0.4, 1.0)}
    _default_spot_size: float = 4.0

    # Data:
    nbp_basic: NotebookPage
    nbp_filter: NotebookPage
    nbp_register: NotebookPage
    nbp_stitch: NotebookPage
    nbp_ref_spots: NotebookPage
    nbp_call_spots: NotebookPage
    nbp_omp: NotebookPage | None
    background_image: napari.layers.Image | None
    spot_data: dict[str, "Viewer.MethodData"]
    genes: tuple["Viewer.Gene"]
    selected_method: str
    selected_spot: int | None
    z: int
    z_thick: float
    score_threshs: dict[str, tuple[float, float]]
    intensity_threshs: dict[str, tuple[float, float]]
    spot_size: float
    hotkeys: tuple[Hotkey]
    open_subplots: list[Any]

    # UI variables:
    # viewer: napari.Viewer
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
        background_image: Optional[str] = "dapi",
        background_image_colour: str = "gray",
        background_image_max_intensity_projection: bool = False,
        background_image_downsample_factor: int = 3,
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
        notebook pages (useful for unit testing).

        Args:
            - nb (Notebook, optional): the notebook to visualise. Must have completed up to `call_spots` at least. If
                none, then all nbp_* notebook pages must be given except nbp_omp which is optional. Default: none.
            - gene_marker_filepath (str, optional): the file path to the gene marker file. Default: use the default
                gene marker at coppafish/plot/results_viewer/gene_color.csv.
            - background_image (str or none, optional): what to use as the background image, can be "dapi" or None. Set
                to None for no background image. Default: "dapi".
            - nbp_basic (NotebookPage, optional): `basic_info` notebook page. Default: not given.
            - nbp_filter (NotebookPage, optional): `filter` notebook page. Default: not given.
            - nbp_register (NotebookPage, optional): `register` notebook page. Default: not given.
            - nbp_stitch (NotebookPage, optional): `stitch` notebook page. Default: not given.
            - nbp_ref_spots (NotebookPage, optional): `ref_spots` notebook page. Default: not given.
            - nbp_call_spots (NotebookPage, optional): `call_spots` notebook page. Default: not given.
            - nbp_omp (NotebookPage, optional): `omp` notebook page. OMP is not a required page. Default: not given.
            - show (bool, optional): show the viewer once it is built. False for unit testing. Default: true.
        """
        assert type(nb) is Notebook or nb is None
        assert type(gene_marker_filepath) is str or gene_marker_filepath is None
        if gene_marker_filepath is not None and not path.isfile(gene_marker_filepath):
            raise FileNotFoundError(f"Could not find gene marker filepath at {gene_marker_filepath}")
        if background_image is not None and type(background_image) is not str:
            raise TypeError(f"background_image must be type str, got type {type(background_image)}")
        if background_image is not None and background_image not in ("dapi",):
            raise ValueError(f"Unknown given background_image: {background_image}")
        assert type(nbp_basic) is NotebookPage or nbp_basic is None
        assert type(nbp_filter) is NotebookPage or nbp_filter is None
        assert type(nbp_register) is NotebookPage or nbp_register is None
        assert type(nbp_stitch) is NotebookPage or nbp_stitch is None
        assert type(nbp_ref_spots) is NotebookPage or nbp_ref_spots is None
        assert type(nbp_call_spots) is NotebookPage or nbp_call_spots is None
        assert type(nbp_omp) is NotebookPage or nbp_omp is None
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

        start_time = time.time()

        # Gather all spot data and keep in self.
        print("Gathering spot data")
        spot_data: dict[str, Viewer.MethodData] = {}
        spot_data["prob"] = self.MethodData()
        spot_data["prob"].tile = self.nbp_ref_spots.tile[:]
        spot_data["prob"].local_yxz = self.nbp_ref_spots.local_yxz[:].astype(np.float32)
        spot_data["prob"].yxz = spot_data["prob"].local_yxz + self.nbp_stitch.tile_origin[spot_data["prob"].tile]
        spot_data["prob"].gene_no = np.argmax(self.nbp_call_spots.gene_probabilities[:], 1).astype(np.int16)
        spot_data["prob"].score = self.nbp_call_spots.gene_probabilities[:].max(1)
        spot_data["prob"].colours = self.nbp_ref_spots.colours[:].astype(np.float32)
        spot_data["prob"].intensity = self.nbp_call_spots.intensity[:]
        spot_data["anchor"] = self.MethodData()
        spot_data["anchor"].tile = spot_data["prob"].tile.copy()
        spot_data["anchor"].local_yxz = self.nbp_ref_spots.local_yxz[:].astype(np.float32)
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
            spot_data[method].indices = np.linspace(0, spot_data[method].score.size - 1, spot_data[method].score.size)
        self.spot_data = spot_data
        # Sanity check spot data.
        for data in self.spot_data.values():
            data.check_variables()

        min_yxz = np.array([0, 0, 0], np.float32)
        max_yxz = np.array([self.nbp_basic.tile_sz, self.nbp_basic.tile_sz, max(self.nbp_basic.use_z)], np.float32)
        max_score = 1.0
        max_intensity = 1.0
        for method in self.spot_data.keys():
            method_max_score = spot_data[method].score.max()
            if method_max_score > max_score:
                max_score = method_max_score
            method_max_intensity = spot_data[method].intensity.max()
            if method_max_intensity > max_intensity:
                max_intensity = method_max_intensity
            method_min_yxz = spot_data[method].yxz.min(0)
            method_max_yxz = spot_data[method].yxz.max(0)
            min_yxz = min_yxz.clip(max=method_min_yxz)
            max_yxz = max_yxz.clip(min=method_max_yxz)

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
        self.viewer = napari.Viewer(title=f"Coppafish {utils_system.get_software_version()} Viewer", show=False)

        print("Building gene legend")
        self.legend = legend_new.Legend()
        self.legend.create_gene_legend(self.genes)
        self.legend.canvas.mpl_connect("button_press_event", self.legend_clicked)
        self.viewer.window.add_dock_widget(self.legend.canvas, name="Gene Legend", area="left")
        self._update_gene_legend()

        print("Building UI")
        # Method selection as a dropdown box containing every gene call method available.
        self.method_combo_box = QComboBox()
        for method in self.spot_data.keys():
            self.method_combo_box.addItem(self._method_to_string[method])
        self.method_combo_box.setCurrentText(self._method_to_string[self.selected_method])
        self.method_combo_box.currentIndexChanged.connect(self.method_changed)
        self.viewer.window.add_dock_widget(self.method_combo_box, area="left", name="Gene Call Method")
        # Z thickness slider.
        self.z_thick: float = 1.0
        self.z_thick_slider = QDoubleSlider(Qt.Orientation.Horizontal)
        self.z_thick_slider.setRange(0, max_yxz[2] - min_yxz[2])
        self.z_thick_slider.setValue(self.z_thick)
        self.z_thick_slider.sliderReleased.connect(self.z_thick_changed)
        self.viewer.window.add_dock_widget(self.z_thick_slider, area="left", name="Z Thickness")
        # Score slider. Keep a separate score threshold for each method.
        self.score_threshs = {method: self._starting_score_thresholds[method] for method in self.spot_data.keys()}
        self.score_slider = QDoubleRangeSlider(Qt.Orientation.Horizontal)
        self.score_slider.setRange(0, max_score)
        self.score_slider.setValue(self.score_threshs[self.selected_method])
        self.score_slider.sliderReleased.connect(self.score_thresholds_changed)
        self.viewer.window.add_dock_widget(self.score_slider, area="left", name="Score Thresholds")
        # Intensity slider. Keep a separate intensity threshold for each method.
        self.intensity_threshs = {method: (0.0, max_intensity) for method in self.spot_data.keys()}
        self.intensity_slider = QDoubleRangeSlider(Qt.Orientation.Horizontal)
        self.intensity_slider.setRange(0, max_intensity)
        self.intensity_slider.setValue(self.intensity_threshs[self.selected_method])
        self.intensity_slider.sliderReleased.connect(self.intensity_thresholds_changed)
        self.viewer.window.add_dock_widget(self.intensity_slider, area="left", name="Intensity Thresholds")
        # Marker size slider. For visuals only.
        self.spot_size = self._default_spot_size
        self.marker_size_slider = QDoubleSlider(Qt.Orientation.Horizontal)
        self.marker_size_slider.setRange(1.0, self.spot_size * 5)
        self.marker_size_slider.setValue(self.spot_size)
        self.marker_size_slider.sliderReleased.connect(self.marker_size_changed)
        self.viewer.window.add_dock_widget(self.marker_size_slider, area="left", name="Marker Size")
        # View hotkeys button.
        self.view_hotkeys_button = QPushButton(text="Hotkeys")
        self.view_hotkeys_button.clicked.connect(self.view_hotkeys)
        self.viewer.window.add_dock_widget(self.view_hotkeys_button, area="left", name="Help")
        # Hide the layer list and layer controls.
        # FIXME: This leads to a future deprecation warning. Napari will hopefully add a proper way of doing this in
        # >= 0.6.0.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            # Turn off layer list and layer controls.
            self.viewer.window.qt_viewer.dockLayerList.hide()
            self.viewer.window.qt_viewer.dockLayerControls.hide()

        print("Placing background image")
        # TODO: Place the background image layer.
        self.background_image = None
        if background_image == "dapi":
            self.background_image = self.viewer.add_image(self.nbp_stitch.dapi_image[:])

        if self.background_image is None:
            # Place a blank, 3D image to make the napari Viewer have the z slider.
            blank_image = np.zeros((max(self.nbp_basic.use_z), 1, 1), dtype=np.int8)
            # Image has shape (z, y, x)
            self.viewer.add_image(blank_image)

        self.z = self.viewer.dims.current_step[0]
        # Connect to z slider changing event.
        self.viewer.dims.events.current_step.connect(self.z_slider_changed)

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
        # Now display the correct spot data based on current thresholds.
        self.update_viewer_data()
        self.viewer.reset_view()

        print(f"Connecting hotkeys")
        self.hotkeys = (
            Hotkey("View hotkeys", "h", "", self.view_hotkeys, "Help"),
            Hotkey(
                "Toggle background", "i", "Toggle the background image on and off", self.toggle_background, "Visual"
            ),
            Hotkey(
                "View spot colour and code",
                "c",
                "Show the selected spot's colour and predicted bled code",
                self.view_spot_colour_and_code,
                "General Diagnostics",
            ),
            Hotkey(
                "View OMP Coefficients",
                "o",
                "Show the OMP coefficients around the selected spot's local region",
                self.view_omp_coefficients,
                "OMP",
            ),
            # Hotkey("", "", ""),
            # Hotkey("", "", ""),
            # Hotkey("", "", ""),
            # Hotkey("", "", ""),
            # Hotkey("", "", ""),
            # Hotkey("", "", ""),
            # Hotkey("", "", ""),
        )
        # Hotkeys can be connected to a function when they occur.
        for hotkey in self.hotkeys:
            if hotkey.invoke is None:
                continue
            self.viewer.bind_key(hotkey.key_press)(hotkey.invoke)

        # When subplots open, some of them need to be kept within the Viewer class to avoid garbage collection.
        # The garbage collection breaks the UI elements like buttons and sliders.
        self.open_subplots = list()

        end_time = time.time()
        print(f"Viewer built in {'{:.1f}'.format(end_time - start_time)}s")

        if show:
            # Give the Viewer a larger window.
            self.viewer.window.resize(1400, 900)
            self.viewer.show()
            self.viewer.window.activate()
            napari.run()

    def selected_spot_changed(self) -> None:
        selected_data: Selection = self.point_layers[self.selected_method].selected_data
        self.selected_spot = selected_data.active
        if self.selected_spot is None:
            return
        print(f"Selected spot: {self.selected_spot}")

    def legend_clicked(self, event: mpl.backend_bases.MouseEvent) -> None:
        if event.inaxes != self.legend.canvas.axes:
            # Click event did not occur within the legend axes.
            return
        left_click = event.button.name == "LEFT"
        closest_gene_index = self.legend.get_closest_gene_index_to(event.xdata, event.ydata)
        if closest_gene_index is None:
            return
        closest_gene = self.genes[closest_gene_index]
        if left_click:
            # Toggle the gene on and off that was clicked on.
            closest_gene.active = not closest_gene.active
        else:
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
        self._update_gene_legend()
        self.update_viewer_data()

    def z_slider_changed(self, _) -> None:
        # Called when the user changes the z slider in the napari viewer.
        new_z = self.viewer.dims.current_step[0]
        if new_z == self.z:
            return
        self.z = new_z
        self.update_viewer_data()

    def method_changed(self) -> None:
        new_selected_method = list(self.spot_data.keys())[self.method_combo_box.currentIndex()]
        # Only update data if the method changed value.
        if new_selected_method == self.selected_method:
            return
        self.selected_method = new_selected_method
        self.update_widget_values()
        self.update_viewer_data()
        print(f"Method: {self.selected_method}")

    def z_thick_changed(self) -> None:
        new_z_thickness = self.z_thick_slider.value()
        # Only update data if the slider changed value.
        if new_z_thickness == self.z_thick:
            return
        self.z_thick = new_z_thickness
        self.update_viewer_data()
        print(f"Z Thickness: {self.z_thick}")

    def score_thresholds_changed(self) -> None:
        new_score_thresholds = self.score_slider.value()
        if new_score_thresholds == self.score_threshs[self.selected_method]:
            return
        self.score_threshs[self.selected_method] = new_score_thresholds
        self.update_viewer_data()
        print(f"Score thresholds: {self.score_threshs[self.selected_method]}")

    def intensity_thresholds_changed(self) -> None:
        new_intensity_thresholds = self.intensity_slider.value()
        if new_intensity_thresholds == self.intensity_threshs[self.selected_method]:
            return
        self.intensity_threshs[self.selected_method] = new_intensity_thresholds
        self.update_viewer_data()
        print(f"Intensity thresholds: {self.intensity_threshs[self.selected_method]}")

    def marker_size_changed(self) -> None:
        new_spot_size = self.marker_size_slider.value()
        if new_spot_size == self.spot_size:
            return
        self.spot_size = new_spot_size
        self.set_spot_size_to(self.spot_size)
        print(f"Marker size: {self.spot_size}")

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
        for method in self.spot_data.keys():
            if method != self.selected_method:
                self.point_layers[method].visible = False
                continue
            scores = self.spot_data[method].score
            intensities = self.spot_data[method].intensity
            gene_numbers = self.spot_data[method].gene_no
            score_threshs = self.score_threshs[method]
            keep_score = (scores >= score_threshs[0]) & (scores <= score_threshs[1])
            intensity_threshs = self.intensity_threshs[method]
            keep_intensity = (intensities >= intensity_threshs[0]) & (intensities <= intensity_threshs[1])
            z_coords = self.spot_data[method].yxz[:, 2]
            min_z = self.z - self.z_thick
            max_z = self.z + self.z_thick
            keep_z = (z_coords >= min_z) & (z_coords <= max_z)
            active_gene_numbers = np.array([gene.notebook_index for gene in self.genes if gene.active], np.int16)
            keep_gene = (gene_numbers[:, np.newaxis] == active_gene_numbers[np.newaxis]).any(1)
            keep = keep_score & keep_intensity & keep_z & keep_gene
            self.point_layers[method].visible = True
            self.point_layers[method].shown = keep
            # To allow the points on the method layer to be selectable, the layer must be selected.
            self.viewer.layers.selection.active = self.point_layers[method]

    def set_spot_size_to(self, new_size: float) -> None:
        """
        Update the spot sizes in the napari Viewer. This is purely visual.

        Args:
            - new_size (float): the new spot size.
        """
        for method in self.spot_data.keys():
            self.point_layers[method].size = new_size

    # HOTKEY FUNCTIONS:
    def view_hotkeys(self, _=None) -> None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
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
        fig.show()

    def view_spot_colour_and_code(self, _=None) -> None:
        if self.selected_spot is None:
            return
        spot_data = self.spot_data[self.selected_method]
        gene_number = spot_data.gene_no[self.selected_spot]
        spot_colours.ViewSpotColourAndCode(
            spot_data.indices[self.selected_spot],
            spot_data.score[self.selected_spot],
            spot_data.tile[self.selected_spot],
            spot_data.colours[self.selected_spot],
            self.nbp_call_spots.bled_codes[gene_number],
            gene_number,
            self.nbp_call_spots.gene_names[gene_number],
            self.nbp_call_spots.colour_norm_factor,
            self.nbp_basic.use_channels,
            self.selected_method,
        )

    def view_omp_coefficients(self, _=None) -> None:
        if self.selected_spot is None:
            return
        if self.nbp_omp is None:
            return
        spot_data = self.spot_data[self.selected_method]
        self.open_subplots.append(
            ViewOMPImage(
                self.nbp_basic,
                self.nbp_filter,
                self.nbp_register,
                self.nbp_call_spots,
                self.nbp_omp,
                spot_data.local_yxz[self.selected_spot],
                spot_data.tile[self.selected_spot],
                spot_data.indices[self.selected_spot],
                self.selected_method,
            )
        )

    def toggle_background(self, _=None) -> None:
        if self.background_image is None:
            return
        self.background_image.visible = not self.background_image.visible

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

    def _update_gene_legend(self) -> None:
        # Called when the gene selection has changed by user input
        self.legend.update_selected_legend_genes([g.active for g in self.genes])

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

        def remove_data_at(self, mask: np.ndarray[bool]) -> None:
            """
            Delete the i'th spot data if mask[i] == True.
            """
            assert type(mask) is np.ndarray
            assert mask.ndim == 1
            assert mask.size == self.tile.size
            for var_name in self._attribute_names:
                self.__setattr__(var_name, self.__getattribute__(var_name)[~mask])
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
                - name: (str) gene name.
                - notebook_index: (int) index of the gene within the notebook.
                - colour: (np.ndarray) of shape (3,) with the RGB colour of the gene. (int8) or None (if not in
                    gene marker file).
                - symbol_napari: (str) symbol used to plot in napari. (Used in the viewer) or None (if not in
                    gene marker file).
                - active: (bool, optional) whether the gene is currently visible in the Viewer. This allows the user to
                    switch genes off by clicking on the gene legend. Default: true.
            """
            self.name = name
            self.notebook_index = notebook_index
            self.colour = colour
            self.symbol_napari = symbol_napari
            self.active = active
