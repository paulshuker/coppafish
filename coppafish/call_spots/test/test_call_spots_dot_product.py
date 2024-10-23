import numpy as np

from coppafish.call_spots import dot_product


def test_dot_product():
    n_spots, n_rounds, n_channels_use, n_genes = 1, 2, 2, 1
    spot_colours = np.zeros((n_spots, n_rounds, n_channels_use), np.float32)
    spot_colours[0, 0, 1] = np.sqrt(0.1)
    spot_colours[0, 1, 1] = np.sqrt(0.9)
    bled_codes = np.zeros((n_genes, n_rounds, n_channels_use))
    bled_codes[0, 0, 1] = np.sqrt(0.5)
    bled_codes[0, 1, 1] = np.sqrt(0.5)

    scores = dot_product.dot_product_score(spot_colours, bled_codes, 0)
    assert type(scores) is np.ndarray
    assert scores.shape == (n_spots, n_genes)
    assert np.isclose(scores, 1)
    assert (scores >= 0).all()
    assert (scores <= 1).all()
    scores = dot_product.dot_product_score(spot_colours, bled_codes, 1)
    assert type(scores) is np.ndarray
    assert scores.shape == (n_spots, n_genes)
    assert np.isclose(scores, 0.8944, atol=1e-3)
    assert (scores >= 0).all()
    assert (scores <= 1).all()
    lower_bound = scores.item()
    for dot_product_weight in [(10 - i - 1) / 10 for i in range(9)]:
        scores = dot_product.dot_product_score(spot_colours, bled_codes, dot_product_weight)
        assert type(scores) is np.ndarray
        assert scores.shape == (n_spots, n_genes)
        assert (scores < 1).all() and (scores > lower_bound).all()
        assert (scores >= 0).all()
        assert (scores <= 1).all()
        lower_bound = scores.item()


def test_gene_prob_score():
    # Test that the gene probabilities are different when kappa is varied
    rng = np.random.RandomState(0)
    n_spots = 11
    n_rounds = 3
    n_channels_use = 4
    n_genes = 5
    # Colours range from -1 to 1
    spot_colours = (rng.rand(n_spots, n_rounds, n_channels_use) - 0.5) * 2
    bled_codes = rng.rand(n_genes, n_rounds, n_channels_use)
    kappa_option = 1
    spot_colours_clone = spot_colours.copy()
    bled_codes_clone = bled_codes.copy()
    probabilities_1 = dot_product.gene_prob_score(spot_colours, bled_codes)
    assert np.allclose(spot_colours, spot_colours_clone), "Function changed spot_colours input"
    assert np.allclose(bled_codes, bled_codes_clone), "Function changed bled_codes input"
    probabilities_2 = dot_product.gene_prob_score(spot_colours, bled_codes, kappa_option)
    assert np.allclose(spot_colours, spot_colours_clone), "Function changed spot_colours input"
    assert np.allclose(bled_codes, bled_codes_clone), "Function changed bled_codes input"
    assert isinstance(probabilities_1, np.ndarray), "Expected ndarray as output"
    assert isinstance(probabilities_2, np.ndarray), "Expected ndarray as output"
    assert probabilities_1.shape == probabilities_2.shape == (n_spots, n_genes), "Expected shape (n_spots, n_genes)"
    assert not np.allclose(probabilities_1, probabilities_2), "Gene probabilities should change as kappa varies"


if __name__ == "__main__":
    test_dot_product()
