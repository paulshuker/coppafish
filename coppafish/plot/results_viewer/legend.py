import colorsys
import math as maths

import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvasHeadless
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# A headless (Agg) backend version of MplCanvas, this is required for unit testing the gene legend.
class MplCanvasHeadless(FigureCanvasHeadless):
    def __init__(self, _=None):
        fig = Figure()
        fig.set_size_inches(5, 4)
        self.axes = fig.subplots(1, 1)
        super(MplCanvasHeadless, self).__init__(fig)


class MplCanvas(FigureCanvas):
    def __init__(self, _=None):
        fig = Figure()
        fig.set_size_inches(5, 4)
        self.axes = fig.subplots(1, 1)
        super(MplCanvas, self).__init__(fig)


class Legend:
    _max_columns: int = 4
    _x_separation: float = 0.25
    _text_scatter_separation: float = 0.02
    _unselected_opacity: float = 0.25
    _selected_opacity: float = 1.0
    _selection_radius: float = 0.25
    _order_by_options: tuple[str] = ("row", "colour")
    _plot_index_to_gene_index: np.ndarray[int]
    # A conversion from a napari marker to a matplotlib marker equivalent.
    _napari_to_mpl_marker: dict[str, str] = {
        "cross": "+",
        "disc": "o",
        "square": "s",
        "triangle_up": "^",
        "triangle_down": "v",
        "hbar": "_",
        "vbar": "|",
        "star": "*",
        "arrow": ">",
        "ring": "o",  # A ring is the same as disc but the face colour is set to None with only an edge colour.
        "clobber": r"$\clubsuit$",
        "x": "x",
        "diamond": "d",
    }

    def __init__(self) -> None:
        self._selection_radius_squared = self._selection_radius**2

    def get_order_by_options(self) -> tuple[str]:
        return self._order_by_options

    order_by_options: tuple[str] = property(get_order_by_options)

    def create_gene_legend(self, genes: tuple, order_by: str):
        """
        Create a new gene legend.
        """
        assert order_by in self._order_by_options

        if matplotlib.get_backend() == "Agg":
            self.canvas = MplCanvasHeadless()
        else:
            self.canvas = MplCanvas()
        self.scatter_axes = []
        # Gene scatter points are populated within a bounding box of -1 to 1 in both x and y directions.
        X, Y = [], []
        active_genes = [gene for gene in genes if gene.active]
        active_count = len(active_genes)
        self._plot_index_to_gene_index = np.linspace(0, active_count - 1, active_count, dtype=int)
        if order_by == "colour":
            index_min = 0
            active_colours = np.array([gene.colour for gene in genes if gene.active])
            for unique_colour in self._hue_sort(np.unique(active_colours, axis=0)):
                unique_colour_indices = (active_colours == unique_colour).all(1).nonzero()[0]
                unique_colour_names = [active_genes[i].name.lower() for i in unique_colour_indices]
                names_sorted_indices = np.argsort(unique_colour_names)
                # The unique colour genes are then sorted by their names alphabetically.
                unique_colour_indices = unique_colour_indices[names_sorted_indices]
                index_max = index_min + len(unique_colour_names)
                self._plot_index_to_gene_index[index_min:index_max] = unique_colour_indices
                index_min = index_max
        row_count = maths.ceil(active_count / self._max_columns)
        text_kwargs = dict(fontsize=5 + 20 / maths.sqrt(active_count), ha="left", va="center", c="grey")
        assert np.unique(self._plot_index_to_gene_index).size == self._plot_index_to_gene_index.size
        active_genes = [active_genes[i] for i in self._plot_index_to_gene_index]
        for i, gene in enumerate(active_genes):
            x = (i % self._max_columns) * self._x_separation
            y = 1 - (i - (i % self._max_columns)) / row_count
            self.canvas.axes.text(x + self._text_scatter_separation, y, gene.name, **text_kwargs)
            marker = self._napari_to_mpl_marker[gene.symbol_napari]
            scatter_kwargs = dict()
            scatter_kwargs["s"] = 20 + 10 / maths.sqrt(active_count)
            if gene.symbol_napari == "ring":
                scatter_kwargs["facecolor"] = "none"
                scatter_kwargs["edgecolor"] = gene.colour
            self.scatter_axes.append(self.canvas.axes.scatter(x, y, marker=marker, color=gene.colour, **scatter_kwargs))
            X.append(x)
            Y.append(y)
        self.canvas.axes.set_title("Gene Legend")
        self.canvas.axes.set_xlim(min(X), max(X) + self._text_scatter_separation)
        self.canvas.axes.set_ylim(min(Y) - 0.03, max(Y) + 0.03)
        self.canvas.axes.set_xticks([])
        self.canvas.axes.set_yticks([])
        self.canvas.axes.spines.clear()
        self.X = np.array(X, np.float32)
        self.Y = np.array(Y, np.float32)

    def update_selected_legend_genes(self, active_genes: list[bool]) -> None:
        """
        Update which genes are currently selected in the viewer. A selected gene is given a high opacity.
        """
        active_genes_sorted = [active_genes[i] for i in self._plot_index_to_gene_index]
        for scatter_ax, active in zip(self.scatter_axes, active_genes_sorted):
            if active:
                scatter_ax.set_alpha(self._selected_opacity)
            else:
                scatter_ax.set_alpha(self._unselected_opacity)
        self.canvas.draw_idle()

    def get_closest_gene_index_to(self, x: float, y: float) -> int | None:
        """
        Find the gene index of the closest gene to the given 2d position.

        Args:
            x (float-like): x position.
            y (float-like): y position.

        Returns:
            (int or none): gene_index. The closest gene index. None if no gene is nearby.
        """
        radii = (self.X - x) ** 2 + (self.Y - y) ** 2
        min_radius = np.min(radii)
        if min_radius <= self._selection_radius_squared:
            plot_index = np.argmin(radii).item()
            return self._plot_index_to_gene_index.tolist()[plot_index]
        return None

    def _hue_sort(self, colours: np.ndarray[float]) -> np.ndarray[float]:
        """
        The given colours are sorted based on their hues.

        Args:
            colours (`(n_colours x 3) ndarray[float]`): each colours RGB value, ranging from 0 to 1.

        Returns:
            (`(n_colours x 3) ndarray[float]`): Each colour, now sorted by hue from lowest to highest.
        """
        assert type(colours) is np.ndarray
        assert colours.ndim == 2
        assert colours.shape[0] > 0
        assert colours.shape[1] == 3

        hues = []
        for colour in colours:
            assert colour.shape == (3,)
            hues.append(colorsys.rgb_to_hsv(colour[0].item(), colour[1].item(), colour[2].item())[0])
        return colours[np.argsort(hues)]
