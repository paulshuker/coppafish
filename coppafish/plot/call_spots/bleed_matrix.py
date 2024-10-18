import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.widgets import TextBox
import numpy as np

from ..results_viewer.subplot import Subplot
from ...setup.notebook_page import NotebookPage


class ViewBleedMatrix(Subplot):
    def __init__(self, nbp_basic_info: NotebookPage, nbp_call_spots: NotebookPage, show: bool = True):
        """
        Diagnostic to plot `nb.call_spots.bleed_matrix`.

        Args:
            nbp_basic_info (NotebookPage): `basic_info` notebook page.
            nbp_call_spots (NotebookPage): `call_spots` notebook page.
            show (bool, optional): show the figure after creating. Set false for unit testing. Default: true.
        """
        bleed_matrix_raw = nbp_call_spots.bleed_matrix_raw
        bleed_matrix_initial = nbp_call_spots.bleed_matrix_initial
        bleed_matrix = nbp_call_spots.bleed_matrix

        # create figure
        self.fig, self.ax = plt.subplots(1, 3, figsize=(15, 5))
        self.ax[0].imshow(bleed_matrix_raw.T, cmap="viridis")
        self.ax[0].set_title("Raw Bleed Matrix")
        self.ax[1].imshow(bleed_matrix_initial.T, cmap="viridis")
        self.ax[1].set_title("Initial Bleed Matrix")
        self.ax[2].imshow(bleed_matrix.T, cmap="viridis")
        self.ax[2].set_title("Final Bleed Matrix")

        # add x and y labels and ticks
        dye_names = nbp_basic_info.dye_names
        use_channels = nbp_basic_info.use_channels
        for i in range(3):
            self.ax[i].set_xticks(ticks=np.arange(len(dye_names)), labels=dye_names, rotation=45)
            self.ax[i].set_yticks(ticks=np.arange(len(use_channels)), labels=use_channels)
            self.ax[i].set_xlabel("Dye")
            self.ax[i].set_ylabel("Channel")

        # Add super title.
        self.fig.suptitle("Bleed Matrix")

        if show:
            self.fig.show()


class ViewBledCodes(Subplot):
    def __init__(self, nbp_basic_info: NotebookPage, nbp_call_spots: NotebookPage, show: bool = True):
        """
        Diagnostic pyplot to display gene bled codes.

        Args:
            nbp_basic_info (NotebookPage): `basic_info` notebook page.
            nbp_call_spots (NotebookPage): `call_spots` notebook page.
            show (bool, optional): show the figure after creating. Set false for unit testing. Default: true.
        """
        assert type(nbp_basic_info) is NotebookPage
        assert type(nbp_call_spots) is NotebookPage

        self.gene_names: np.ndarray[str] = nbp_call_spots.gene_names
        self.gene_names_lower: np.ndarray[str] = np.char.lower(nbp_call_spots.gene_names)
        self.bled_codes: np.ndarray[float] = nbp_call_spots.bled_codes

        self.fig, self.ax = plt.subplots(1, 1)
        self.fig.subplots_adjust(bottom=0.2)
        cmap = mpl.cm.seismic
        norm = mpl.colors.Normalize(vmin=-np.abs(self.bled_codes).max(), vmax=np.abs(self.bled_codes).max())
        self.im = self.ax.imshow(np.zeros_like(self.bled_codes[0].T), cmap=cmap, norm=norm)
        self.fig.colorbar(self.im, ax=self.ax, orientation="vertical", fraction=0.03, pad=0.04)
        self.ax.set_xticks(list(range(self.bled_codes.shape[1])), nbp_basic_info.use_rounds)
        self.ax.set_xticks(list(range(self.bled_codes.shape[2])), nbp_basic_info.use_channels)

        # Text box for changing the gene to look at.
        axbox = self.fig.add_axes([0.2, 0.05, 0.6, 0.075])
        self.text_box = TextBox(
            axbox, label="Gene index\nor name", color="darkgray", hovercolor="gray", textalignment="center"
        )
        self.text_box.on_submit(self.text_box_changed_to)
        self.text_box.set_val(self.gene_names[0])

        if show:
            self.fig.show()

    def text_box_changed_to(self, expression: str) -> None:
        try:
            gene_no = int(eval(expression))
        except NameError or ValueError:
            gene_name = str(expression)
            gene_no = (self.gene_names == gene_name).nonzero()[0]
            gene_no_2 = (self.gene_names == gene_name.lower()).nonzero()[0]
            if gene_no.size == 0 and gene_no_2.size == 0:
                return
            if gene_no.size == 1:
                gene_no = int(gene_no.item())
            elif gene_no_2.size == 1:
                gene_no = int(gene_no_2.item())
            else:
                return
        if gene_no < 0 or gene_no >= self.bled_codes.shape[0]:
            return
        self.im.set_data(self.bled_codes[gene_no].T)
        self.fig.suptitle(f"Gene {gene_no}: {self.gene_names[gene_no]} bled code")
        self.fig.canvas.draw_idle()
