import numpy as np


def get_gene_scatter_positions(
    n_genes: int,
    n_columns: int = 3,
    yx_spacing: tuple[float, float] = (0.05, 0.1),
    top_left_corner: tuple[float, float] = (0.8, 0.0),
) -> np.ndarray:
    """
    Compute the gene positions for the gene legend plot based on the number of genes.

    Args:
        - n_genes (int): the number of genes in the legend.
        - n_columns (int, optional): the number of gene columns to have. Genes are placed across a row from left to
            right before moving to the next row below.
        - yx_spacing (tuple of two floats, optional): the y and x spacing between the gene positions.
        - top_left_corner (tuple of two floats, optional): the top left corner to start placing gene positions.

    Returns:
        `(n_genes x 2) ndarray[float32]` yx_positions: the y and x positions of each gene.
    """
    assert type(n_genes) is int
    assert type(n_columns) is int
    assert type(yx_spacing) is tuple

    yx_positions = np.zeros((n_genes, 2), np.float32)
    row = 0
    for g in range(n_genes):
        column = g % n_columns
        yx_positions[g] = [top_left_corner[0] - row * yx_spacing[0], top_left_corner[1] + column * yx_spacing[1]]
        if column == (n_columns - 1):
            row += 1
    return yx_positions
