import numpy as np


def dot_product_score(spot_colours: np.ndarray, bled_codes: np.ndarray, dot_product_weight: float) -> np.ndarray:
    """
    Score each spot to each gene. The maximum score is the assigned gene for said spot. The scores range from 0 to 1.

    Args:
        spot_colours (`(n_spots x n_rounds x n_channels_use) ndarray[float]`): spot colours.
        bled_codes (`(n_genes x n_rounds x n_channels_use) ndarray[float]`): normalised bled codes.
        dot_product_weight (float): how much the dot product score is a pure dot product score. 1 is a simple dot
            product score, 0 gives each round an equal weighting on the score. A value between 0 and 1 is somewhere in
            between the two.

    Returns:
        (`(n_spots x n_genes) ndarray[float]`): score. score such that `score[d, c]` gives a "semi-dot product" between
            `spot_colours` vector `d` with `bled_codes` vector `c`.
    """
    assert spot_colours.ndim == 3
    assert bled_codes.ndim == 3
    if dot_product_weight < 0 or dot_product_weight > 1:
        raise ValueError(f"dot_product_weight must be between 0 and 1, got {dot_product_weight}")

    # L2 normalise the entire spot colour for each spot.
    spot_colours /= np.linalg.norm(spot_colours, axis=(1, 2), keepdims=True)
    bled_codes /= np.linalg.norm(bled_codes, axis=(1, 2), keepdims=True)
    # The spot colour and bled code L2 norms for each round separately.
    spot_colours_norms = np.linalg.norm(spot_colours.copy(), axis=2)
    bled_code_norms = np.linalg.norm(bled_codes.copy(), axis=2)
    # F hat as defined in the docs, has shape (n_spots x n_rounds x n_channels).
    f_hat = spot_colours.copy()
    f_hat /= spot_colours_norms[:, :, np.newaxis]
    # K hat as defined in the docs, has shape (n_genes x n_rounds x n_channels).
    k_hat = bled_codes.copy()
    k_hat /= bled_code_norms[:, :, np.newaxis]

    # FIXME: A potential mistake in the scores computation here as scores are way above 1 for robominnie.
    # Has shape (n_spots x n_genes x n_rounds).
    scores = (spot_colours_norms[:, np.newaxis] * bled_code_norms[np.newaxis]) ** dot_product_weight
    scores *= (f_hat[:, np.newaxis] * k_hat[np.newaxis]).sum(3)
    # Has shape (n_spots x n_genes).
    scores = scores.sum(2)
    del f_hat, k_hat
    scores /= np.sqrt((spot_colours_norms ** (2 * dot_product_weight)).sum(1))[:, np.newaxis]
    del spot_colours_norms
    scores /= np.sqrt((bled_code_norms ** (2 * dot_product_weight)).sum(1))[np.newaxis]
    del bled_code_norms
    scores = scores.clip(0, 1)

    return scores


def gene_prob_score(spot_colours: np.ndarray, bled_codes: np.ndarray, kappa: float = 2) -> np.ndarray:
    """
    Probability model says that for each spot in a particular round, the normalised fluorescence vector follows a
    Von-Mises Fisher distribution with mean equal to the normalised fluorescence for each dye and concentration
    parameter kappa. Then invert this to get prob(dye | fluorescence) and multiply across rounds to get
    prob(gene | spot_colours).

    Args:
        spot_colours (`(n_spots x n_rounds x n_channels_use) ndarray`): spot colours.
        bled_codes (`(n_genes x n_rounds x n_channels_use) ndarray`): normalised bled codes.
        kappa (float, optional), scaling factor for dot product score. Default: 2.

    Returns:
        (`(n_spots x n_genes) ndarray[float]`): gene probabilities.
    """
    n_genes = bled_codes.shape[0]
    n_spots, n_rounds, n_channels_use = spot_colours.shape
    # First, normalise spot_colours so that for each spot s and round r, norm(spot_colours[s, r, :]) = 1
    spot_colours = spot_colours / np.linalg.norm(spot_colours, axis=2)[:, :, None]
    # Do the same for bled_codes
    bled_codes = bled_codes / np.linalg.norm(bled_codes, axis=2)[:, :, None]
    # Flip the sign of a single spot and round if spot_colours[s, r, c] < 0 for the greatest magnitude channel.
    spot_colours_reshaped = spot_colours.copy().reshape((-1, n_channels_use))
    negatives = np.take_along_axis(spot_colours_reshaped, np.argmax(spot_colours_reshaped, axis=1)[:, None], 1) < 0
    spot_colours[negatives.reshape((n_spots, n_rounds))] *= -1
    # At this point, reshape spot_colours to be [n_spots, n_rounds * n_channels_use] and bled_codes to be
    # [n_genes, n_rounds * n_channels_use]
    spot_colours = spot_colours.reshape((n_spots, -1))
    bled_codes = bled_codes.reshape((n_genes, -1))
    # Now we can compute the dot products of each spot with each gene, producing a matrix of shape [n_spots, n_genes]
    dot_product = spot_colours @ bled_codes.T
    probability = np.exp(kappa * dot_product)
    # Now normalise so that each row sums to 1
    probability = np.nan_to_num(probability / np.sum(probability, axis=1)[:, None])

    return probability
