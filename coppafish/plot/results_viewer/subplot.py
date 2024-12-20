import matplotlib.pyplot as plt
from matplotlib.figure import Figure


# Every Viewer subplot inherits the Subplot class. This is so we can have common functions for each subplot for the
# Viewer to use, like a close method.
class Subplot:
    fig: Figure

    def close(self) -> None:
        plt.close(self.fig)
