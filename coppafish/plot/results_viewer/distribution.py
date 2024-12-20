import math as maths

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

from .subplot import Subplot


class ViewScoreIntensityDistributions(Subplot):
    # The number of bins within a region of 0 to 1 in score or intensity values.
    _bin_density = 100

    def __init__(self, method: str, scores: np.ndarray[float], intensities: np.ndarray[float], show: bool = True):
        """
        Show how the spot scores and intensities distribute for the specific method given.

        Args:
            method (str): the selected method. Must be 'prob', 'anchor', or 'omp'.
            scores (`(n_spots) ndarray[float]`): the scores for the method to view. These should be between 0 and 1.
            intensities (`(n_spots) ndarray[float]`): the intensities for the method's spots.
            show (bool, optional): show the plot after building it. Default: true.
        """
        assert method in ("prob", "anchor", "omp")
        assert scores.ndim == 1
        assert scores.size > 0
        assert (scores >= 0).all()
        assert intensities.ndim == 1
        assert intensities.size > 0
        assert scores.size == intensities.size

        self.log_counts = True
        self.fig, self.axes = plt.subplots(1, 2, gridspec_kw={"width_ratios": [5, 1]})
        self.fig.suptitle(f"{method.capitalize()} Score/Intensity Distributions")
        self.score_range = [0.0, max(scores.max().item(), 1.0)]
        score_bins = maths.ceil((self.score_range[1] - self.score_range[0]) * self._bin_density)
        self.intensity_range = [0.0, max(intensities.max().item(), 1.0)]
        intensity_bins = maths.ceil((self.intensity_range[1] - self.intensity_range[0]) * self._bin_density)
        H, self.xedges, self.yedges = np.histogram2d(
            scores, intensities, bins=(score_bins, intensity_bins), range=[self.score_range, self.intensity_range]
        )
        self.H = H.T
        # Logarithmic scaling to help see low count features in the plot.
        self.H_log = np.log1p(self.H)
        self.draw_data()

        # Check box to toggle logarithmic counting.
        self.button_colour_not_pressed = "red"
        self.button_colour_pressed = "green"
        rax = plt.axes([0.1, 0.8, 0.1, 0.05])  # Position of the checkbox
        self.checkbox = Button(rax, "Log Counts", hovercolor="0.275")
        self.checkbox.on_clicked(self.toggle_log_counts)
        self.checkbox.label.set_color(self.button_colour_pressed)

        if show:
            self.fig.show()

    def draw_data(self) -> None:
        ax: plt.Axes = self.axes[0]
        ax.clear()
        ax.grid(False)
        ax.set_xlim(*self.score_range)
        ax.set_xlabel("Score")
        ax.set_ylim(*self.intensity_range)
        ax.set_ylabel("Intensity")
        data = self.H
        if self.log_counts:
            data = self.H_log
        # We allow non-square pixels so the subplot can correctly fill the plot space even if intensity has many more
        # bins than score for example.
        im = ax.imshow(
            data,
            interpolation="nearest",
            origin="lower",
            extent=(self.xedges[0], self.xedges[-1], self.yedges[0], self.yedges[-1]),
            aspect="auto",
        )
        cbar_ax: plt.Axes = self.axes[1]
        cbar_ax.clear()
        label = "Count"
        if self.log_counts:
            label = "Ln (1 + count)"
        self.fig.colorbar(im, cax=cbar_ax, label=label)
        self.fig.canvas.draw()

    def toggle_log_counts(self, _=None) -> None:
        self.log_counts = not self.log_counts
        if self.log_counts:
            self.checkbox.label.set_color(self.button_colour_pressed)
        else:
            self.checkbox.label.set_color(self.button_colour_not_pressed)
        self.draw_data()
