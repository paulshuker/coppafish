import math as maths

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np


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
        "clobber": "$\clubsuit$",
        "x": "x",
        "diamond": "d",
    }

    def create_gene_legend(self, genes: tuple):
        """
        Create a new gene legend.
        """
        plt.style.use("dark_background")

        self.canvas = MplCanvas()
        self.scatter_axes = []
        # Gene scatter points are populated within a bounding box of -1 to 1 in both x and y directions.
        X, Y = [], []
        active_genes = [gene for gene in genes if gene.active]
        active_count = len(active_genes)
        row_count = maths.ceil(active_count / self._max_columns)
        text_kwargs = dict(fontsize=5 + 20 / maths.sqrt(active_count), ha="left", va="center", c="grey")
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
        for scatter_ax, active in zip(self.scatter_axes, active_genes):
            if active:
                scatter_ax.set_alpha(self._selected_opacity)
            else:
                scatter_ax.set_alpha(self._unselected_opacity)
        self.canvas.draw_idle()

    def get_closest_gene_index_to(self, x: float, y: float) -> int | None:
        """
        Find the gene index of the closest gene to the given 2d position.

        Args:
            - x (float-like): x position.
            - y (float-like): y position.

        Returns:
            (int or none) gene_index: the closest gene index. None if no gene is nearby.
        """
        # TODO: Implement.
        radii = (self.X - x) ** 2 + (self.Y - y) ** 2
        min_radius = np.min(radii)
        if min_radius <= self._selection_radius**2:
            return np.argmin(radii).item()
        return None
