import numpy as np
from matplotlib import pyplot as plt


def view_free_and_constrained_bled_codes(
    free_bled_codes_tile_indep: np.ndarray,
    bled_codes: np.ndarray,
    rc_scale: np.ndarray,
    gene_names: np.ndarray,
    n_spots: np.ndarray,
) -> None:
    """
    Function to plot the free and constrained bleed codes for each gene.
    Args:
        free_bled_codes_tile_indep: np.ndarray [n_genes x n_rounds x n_channels_use]
            The free bled codes.
        bled_codes: np.ndarray [n_genes x n_rounds x n_channels_use]
            The constrained bled codes.
        rc_scale: np.ndarray [n_rounds x n_channels_use]
            The scale factor for each round and channel.
        gene_names: np.ndarray [n_genes]
            The gene names.
        n_spots: np.ndarray [n_genes]
            The number of spots for each gene.
    """
    n_columns = 9
    n_genes, n_rounds, n_channels_use = free_bled_codes_tile_indep.shape
    n_rows = n_genes // n_columns + 1

    fig, ax = plt.subplots(n_rows, n_columns)
    codes = np.zeros((n_genes, n_channels_use, 2 * n_rounds + 1)) * np.nan
    # fill in the codes
    for g in range(n_genes):
        codes[g, :, :n_rounds] = free_bled_codes_tile_indep[g].T
        codes[g, :, -n_rounds:] = bled_codes[g].T
    # fill in the image grid
    for g in range(n_genes):
        row, col = g // n_columns, g % n_columns
        ax[row, col].imshow(codes[g], cmap="viridis")
        ax[row, col].set_title(f"{gene_names[g]} ({n_spots[g]})", fontsize=8)
        ax[row, col].axis("off")
    for g in range(n_genes, n_rows * n_columns - 1):
        row, col = g // n_columns, g % n_columns
        ax[row, col].axis("off")

    # add the round_channel scale
    ax[-1, -1].imshow(rc_scale.T, cmap="viridis")
    ax[-1, -1].set_title("round_channel scale", fontsize=8)
    ax[-1, -1].set_xlabel("Round", fontsize=8)
    ax[-1, -1].set_ylabel("Channel", fontsize=8)
    ax[-1, -1].set_xticks([])
    ax[-1, -1].set_yticks([])

    # add title
    plt.suptitle("Free (left) and round_channel (right) bleed codes")
    plt.show()


def view_tile_bled_codes(
    free_bled_codes: np.ndarray,
    free_bled_codes_tile_indep: np.ndarray,
    gene_names: np.ndarray,
    use_tiles: np.ndarray,
    gene: int,
) -> None:
    """
    Function to plot the free bled codes for each tile for a given gene.
    Args:
        free_bled_codes: np.ndarray [n_genes x n_tiles x n_rounds x n_channels_use]
            The free bled codes.
        free_bled_codes_tile_indep: np.ndarray [n_genes x n_rounds x n_channels_use]
            The tile independent free bled codes.
        gene_names: np.ndarray [n_genes]
            The gene names.
        use_tiles: np.ndarray [n_tiles]
            The tiles to use.
        gene: int
            The gene to plot.
    """
    n_columns = 4
    _, _, n_rounds, n_channels_use = free_bled_codes.shape
    n_tiles = len(use_tiles)
    n_rows = int(np.ceil(n_tiles / n_columns)) + 1

    fig, ax = plt.subplots(n_rows, n_columns)
    for i, t in enumerate(use_tiles):
        row, col = i // n_columns, i % n_columns
        ax[row, col].imshow(free_bled_codes[gene, t].T, cmap="viridis")
        ax[row, col].set_title(f"Tile {t}")
        ax[row, col].set_xlabel("Round")
        ax[row, col].set_ylabel("Channel")
        ax[row, col].set_xticks([])
        ax[row, col].set_yticks([])
    for i in range(n_tiles, n_rows * n_columns - 1):
        row, col = i // n_columns, i % n_columns
        ax[row, col].axis("off")

    # add the tile independent bled code
    ax[-1, -1].imshow(free_bled_codes_tile_indep[gene].T, cmap="viridis")
    ax[-1, -1].set_title("Tile independent")
    ax[-1, -1].set_xlabel("Round")
    ax[-1, -1].set_ylabel("Channel")
    ax[-1, -1].set_xticks([])
    ax[-1, -1].set_yticks([])

    plt.suptitle(f"Free bled codes for gene {gene_names[gene]}")
    plt.show()


def view_rc_scale_regression(
    rc_scale: np.ndarray,
    gene_codes: np.ndarray,
    d_max: np.ndarray,
    target_values: np.ndarray,
    free_bled_codes_tile_indep: np.ndarray,
    n_spots: np.ndarray,
    use_channels: tuple,
) -> None:
    """
    Plotter to show the regression of the round_channel scale factor for each round and channel.
    Args:
        rc_scale: np.ndarray [n_rounds x n_channels_use]
            The round_channel scale factor.
        gene_codes: np.ndarray [n_genes x n_rounds]
            gene_codes[g, r] is the expected dye for gene g in round r.
        d_max: np.ndarray [n_channels_use]
            d_max[c] is the dye with the highest expression in channel c.
        target_values: np.ndarray [n_dyes]
            target_values[d] is the target value for dye d in its brightest channel.
        free_bled_codes_tile_indep: np.ndarray [n_genes x n_rounds x n_channels_use]
            The tile independent free bled codes.
        n_spots: np.ndarray [n_genes]
            The number of spots for each gene.
        use_channels: np.ndarray [n_channels_use]

    """
    n_genes, n_rounds, n_channels_use = free_bled_codes_tile_indep.shape
    fig, ax = plt.subplots(n_channels_use, 1)
    for c in range(n_channels_use):
        x, y, y_scaled, s = [], [], [], []
        for r in range(n_rounds):
            # get data
            relevant_genes = np.where(gene_codes[:, r] == d_max[c])[0]
            n_spots_rc = n_spots[relevant_genes]
            new_y_vals = free_bled_codes_tile_indep[relevant_genes, r, c]
            new_y_scaled_vals = free_bled_codes_tile_indep[relevant_genes, r, c] * rc_scale[r, c]
            # append data
            y.append(new_y_vals)
            y_scaled.append(new_y_scaled_vals)
            s.append(np.sqrt(n_spots_rc))
            x.append(np.repeat(2 * r, len(relevant_genes)))
        # convert data to numpy arrays
        x, y, y_scaled, s = np.concatenate(x), np.concatenate(y), np.concatenate(y_scaled), np.concatenate(s)
        jitter = np.random.normal(0, 0.1, len(x))
        ax[c].scatter(x + jitter, y, s=s, c="cyan", alpha=0.5)
        ax[c].scatter(x + 1 + jitter, y_scaled, s=s, c="red", alpha=0.5)
        # add a horizontal line for the target value
        ax[c].axhline(target_values[d_max[c]], color="white", linestyle="--")
        ax[c].set_xticks([])
        max_val = max(np.max(free_bled_codes_tile_indep), np.max(free_bled_codes_tile_indep * rc_scale))
        ax[c].set_yticks(np.round([0, target_values[d_max[c]], max_val], 2))
        ax[c].set_xlim(-1, 2 * n_rounds)
        ax[c].set_ylim(0, max_val)
        ax[c].set_ylabel(f"Channel {use_channels[c]}")
        # add text to the right hand side of each row
        ax[c].text(2 * n_rounds + 0.25, 1, f"mean scale = {np.mean(rc_scale[:, c]) :.2f}")
        if c == n_channels_use - 1:
            ax[c].set_xlabel("Round")
            ax[c].set_xticks(np.arange(0, 2 * n_rounds, 2), labels=np.arange(n_rounds))

    plt.suptitle(
        "Each scale $V_{rc}$ is chosen so that the loss function "
        "$L(V_{rc}) = \sum_{g \in G_{rc}} \sqrt{N_g}(E_{grc}V_{rc} - T_{d_{max}(c)})^2$ is minimised, \n"
        "where $G_{rc}$ is the set of genes with gene code $d_{max}(c)$ in round r and $E_g$ is the "
        "tile-independent bled code for gene g. Cyan points are raw values, red points are scaled values."
    )
    # add another figure to show the round_channel scale
    fig2, ax2 = plt.subplots(1, 1)
    ax2.imshow(rc_scale, cmap="viridis")
    ax2.set_title("round_channel scale")
    ax2.set_ylabel("Round")
    ax2.set_xlabel("Channel")
    ax2.set_yticks(np.arange(n_rounds))
    ax2.set_xticks(np.arange(n_channels_use), labels=use_channels)
    # add colorbar
    cbar = plt.colorbar(ax2.imshow(rc_scale, cmap="viridis"), ax=ax2)
    cbar.set_label("Scale factor")
    plt.show()


def view_tile_scale_regression(
    tile_scale: np.ndarray,
    gene_codes: np.ndarray,
    d_max: np.ndarray,
    bled_codes: np.ndarray,
    free_bled_codes: np.ndarray,
    n_spots: np.ndarray,
    t: int,
    use_channels: tuple,
) -> None:
    """
    Plotter to show the regression of the tile scale factor for tile t for all rounds and channels.

    Args:
        tile_scale: np.ndarray [n_tiles x n_rounds x n_channels_use]
            The tile scale factor.
        gene_codes: np.ndarray [n_genes x n_rounds]
            gene_codes[g, r] is the expected dye for gene g in round r.
        d_max: np.ndarray [n_channels_use]
            d_max[c] is the dye with the highest expression in channel c.
        bled_codes: np.ndarray [n_genes x n_rounds x n_channels_use]
            The target bled codes. bled_codes[g, r, c] = free_bled_codes_tile_indep[g, r, c] * rc_scale[r, c]
        free_bled_codes: np.ndarray [n_genes x n_tiles x n_rounds x n_channels_use]
            The free bled codes.
        n_spots: np.ndarray [n_genes]
            The number of spots for each gene.
        t: int
            The tile to plot.
        use_channels: tuple
            The channels to use.

    """
    use_channels = list(use_channels)
    n_tiles, n_rounds, n_channels_use = tile_scale.shape
    fig, ax = plt.subplots(n_rounds, n_channels_use)
    for r, c in np.ndindex(n_rounds, n_channels_use):
        relevant_genes = np.where(gene_codes[:, r] == d_max[c])[0]
        n_spots_rc = n_spots[relevant_genes]
        x = free_bled_codes[relevant_genes, t, r, c]
        y = bled_codes[relevant_genes, r, c]
        sizes = np.sqrt(n_spots_rc)
        ax[r, c].scatter(x, y, s=sizes)
        ax[r, c].plot(x, tile_scale[t, r, c] * x, color="red")
        ax[r, c].set_title(f"r {r}, c {use_channels[c]}", fontsize=8)
        ax[r, c].set_xticks([])
        ax[r, c].set_yticks([])
        ax[r, c].set_xlim(0, free_bled_codes[:, t].max())
        ax[r, c].set_ylim(0, bled_codes.max())
        if r == n_rounds - 1:
            ax[r, c].set_xlabel("$D_{gtrc}$")
        if c == 0:
            ax[r, c].set_ylabel("$K_{grc}$")
    plt.suptitle(
        "Each tile scale $Q_{trc}$ is chosen so that the loss function "
        "$L(Q_{trc}) = \sum_{g \in G_{rc}} \sqrt{N_g}(D_{gtrc}Q_{trc} - K_{grc})^2$ is minimised, \n"
        "where $G_{rc}$ is the set of genes with gene code $d_{max}(c)$ in round r, $K_{grc} = E_{grc}V_{rc}$ "
        "is the constrained bled code for gene g, $D_g$ is the tile-dependent bled code for gene g, "
        "and $V_{rc}$ is the round/channel scale factor."
    )

    plt.show()


def view_scale_factors(
    tile_scale: np.ndarray, rc_scale: np.ndarray, use_tiles: tuple, use_rounds: tuple, use_channels: tuple
) -> None:
    """
    Function to plot the tile scale factors for each tile, round and channel.
    Args:
        tile_scale: np.ndarray [n_tiles x n_rounds x n_channels_use]
            The tile scale factors.
        rc_scale: np.ndarray [n_rounds x n_channels_use]
            The round_channel scale factors.
        use_tiles: tuple
            The tiles to use.
        use_rounds: tuple
            The rounds to use.
        use_channels:  tuple
            The channels to use.
    """
    use_tiles, use_rounds, use_channels = list(use_tiles), list(use_rounds), list(use_channels)
    tile_scale = tile_scale[use_tiles]
    relative_scale = tile_scale / rc_scale[None, :, :]
    n_tiles, n_rounds, n_channels_use = tile_scale.shape
    tile_scale = tile_scale.reshape((n_tiles * n_rounds, n_channels_use))
    relative_scale = relative_scale.reshape((n_tiles * n_rounds, n_channels_use))
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(rc_scale, cmap="viridis")
    ax[0].set_xticks(np.arange(n_channels_use), labels=use_channels)
    ax[0].set_yticks(np.arange(n_rounds), labels=use_rounds)
    ax[0].set_title("round/channel factors $V_{rc}$")
    ax[0].set_xlabel("Channel")
    ax[0].set_ylabel("Round")
    plt.colorbar(ax[0].imshow(rc_scale, cmap="viridis"), ax=ax[0])
    for i, scale in enumerate([tile_scale, relative_scale]):
        i += 1
        ax[i].imshow(scale, cmap="viridis")
        ax[i].set_xticks(np.arange(n_channels_use), labels=use_channels)
        ax[i].set_yticks(
            np.arange(n_tiles * n_rounds), labels=[f"T{t}, R{r}" for t, r in np.ndindex(n_tiles, n_rounds)], fontsize=8
        )
        ax[i].set_title("tile scale factors $Q_{t,r,c}$" if i == 1 else "relative scale factors $Q_{t,r,c}/V_{rc}$")
        ax[i].set_xlabel("Channel")
        ax[i].set_ylabel("Tile x Round")
        # add horizontal red lines at each tile
        for t in range(1, n_tiles):
            ax[i].axhline(t * n_rounds - 0.5, color="red", linestyle="--")
        # add colorbar
        plt.colorbar(ax[i].imshow(scale, cmap="viridis"), ax=ax[i])
    plt.show()

