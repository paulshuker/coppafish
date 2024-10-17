from collections.abc import Callable
from typing import Optional


class Hotkey:
    def __init__(
        self,
        name: str,
        key_press: str,
        description: str,
        invoke: Optional[Callable],
        section: str,
        requires_selection: bool = True,  # If true, requires a selected spot in the viewer to work.
    ):
        self.name = name
        self.key_press = key_press
        self.description = description
        self.invoke = invoke
        self.section = section
        self.requires_selection = requires_selection

    def __str__(self):
        msg = f"("
        if self.requires_selection:
            msg += "Select Spot, "
        msg += f"Press {self.key_press.lower().replace('-', ' + ')}) {self.name}"
        if self.description:
            msg += f": {self.description}"
        return msg

    # view_hotkeys = "Shift-k"
    # switch_zoom_select = "Space"
    # remove_background = "i"
    # view_bleed_matrix = "b"
    # view_background_norm = "n"
    # view_bleed_matrix_calculation = "Shift-b"
    # view_bled_codes = "g"
    # view_all_gene_scores = "Shift-h"
    # view_gene_efficiency = "e"
    # # view_gene_counts = "Shift-g"
    # view_histogram_scores = "h"
    # view_scaled_k_means = "k"
    # view_colour_and_codes = "c"
    # view_spot_intensities = "s"
    # view_spot_colours_and_weights = "d"
    # view_intensity_from_colour = "Shift-i"
    # view_omp_coef_image = "o"
    # # view_omp_pixel_colours = "p"
