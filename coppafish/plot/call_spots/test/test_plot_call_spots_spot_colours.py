import matplotlib.pyplot as plt
import numpy as np

from coppafish.plot.call_spots import spot_colours


def test_ViewSpotColourAndCode() -> None:
    rng = np.random.RandomState(0)

    n_tiles = 3
    n_rounds = 7
    n_channels_use = 9
    use_channels = list(range(n_channels_use))
    use_channels[-1] = 15
    spot_no = 0
    spot_score = 0.6
    spot_tile = 1
    spot_colour = rng.rand(n_rounds, n_channels_use)
    spot_colour[:, 1] = 1.2
    gene_bled_code = rng.rand(n_rounds, n_channels_use)
    gene_name = "gene_name"
    gene_index = 6
    method = "omp"
    colour_norm_factor = rng.rand(n_tiles, n_rounds, n_channels_use) * 3
    for t in range(n_tiles):
        if t == spot_tile:
            continue
        colour_norm_factor[t] = 0

    spot_colour_init = spot_colour.copy()
    gene_bled_code_init = gene_bled_code.copy()
    view = spot_colours.ViewSpotColourAndCode(
        spot_no,
        spot_score,
        spot_tile,
        spot_colour,
        gene_bled_code,
        gene_index,
        gene_name,
        colour_norm_factor,
        use_channels,
        method,
        show=False,
    )
    assert type(view) is spot_colours.ViewSpotColourAndCode
    assert np.allclose(spot_colour_init, spot_colour), f"spot_colour was modified"
    assert np.allclose(gene_bled_code_init, gene_bled_code), f"gene_bled_code was modified"
    view.plot_colour()
    view.change_background()
    view.change_use_colour_norm()
    view.change_norm()
    view.change_background()
    view.change_norm()
    view.change_use_colour_norm()
    view.change_use_colour_norm()
    plt.close(view.fig)
    del view
