from typing import Tuple

import numpy as np
import torch

from ..call_spots import dot_product
from ..utils import system


class PixelScoreSolver:
    DTYPE = np.float32
    DTYPE_T = torch.float32
    NO_GENE_ASSIGNMENT: int = -32_768

    def __init__(self) -> None:
        """
        Initialise the PixelScoreSolver class. Used to compute OMP pixel scores.
        """
        pass

    def solve(
        self,
        pixel_colours: np.ndarray[DTYPE],
        bled_codes: np.ndarray[DTYPE],
        background_codes: np.ndarray[DTYPE],
        maximum_iterations: int,
        dot_product_threshold: float,
        minimum_intensity: float,
        alpha: float,
        beta: float,
        return_all_scores: bool = False,
        return_all_weights: bool = False,
        return_all_residuals: bool = False,
        force_cpu: bool = True,
    ) -> (
        np.ndarray[DTYPE]
        | Tuple[np.ndarray[DTYPE], np.ndarray[DTYPE]]
        | Tuple[np.ndarray[DTYPE], np.ndarray[DTYPE], np.ndarray[DTYPE]]
        | Tuple[np.ndarray[DTYPE], np.ndarray[DTYPE], np.ndarray[DTYPE], np.ndarray[DTYPE]]
    ):
        """
        Compute OMP pixel scores on all pixel colours from the same tile. At each iteration of OMP, the next best gene
        assignment is found from the residual spot colours. A pixel is stopped iterating on if gene assignment fails.
        See function `get_next_gene_assignments` below for details on the stopping criteria and gene scoring. Pixels
        that do not fail are weighted and a new pixel score is added to the final pixel scores. Pixels that are gene
        assigned are then fitted with the additional gene to find updated pixel scores. See function
        `get_gene_pixel_scores` for details on the pixel score computation.

        Args:
            pixel_colours (`(n_pixels x n_rounds_use x n_channels_use) ndarray[float]`): pixel intensity in each
                sequencing round and channel.
            bled_codes (`(n_genes x n_rounds_use x n_channels_use) ndarray[float32]`): every gene bled code.
            background_codes (`(n_channels_use x n_rounds_use x n_channels_use) tensor[float]`): the background bled
                codes. These are simply uniform brightness in one channel for all rounds. background_codes[0] is the
                first code, background_codes[1] is the second code, etc.
            maximum_iterations (int): the maximum number of gene assignments allowed for one pixel.
            dot_product_threshold (float): a gene must have a dot product score above this value on the residual spot
                colour to be assigned the gene. If more than one gene is above this threshold, the top score is used.
            minimum_intensity (float): a pixel's residual intensity must be above minimum_intensity to pass gene
                assignment.
            alpha (float): the alpha parameter. Used to compute the error variance after each iteration.
            beta (float): the beta parameter. Used to compute the error variance after each iteration.
            return_all_scores (bool, optional): return all gene round dot product scores on each iteration. Default:
                false.
            return_all_weights (bool, optional): return all gene bled code weights for every gene that was assigned.
                Default: false.
            return_all_residuals (bool, optional): return all residual colours used to compute the final pixel scores.
                Default: false.
            force_cpu (bool, optional): only use the CPU to solve. Default: true.

        Returns:
            Tuple (tensor if only one tensor is returned) containing the following:
                - (`(n_pixels x n_genes) ndarray[float32]`): pixel_scores. Each gene's final pixel score for every
                    pixel.
                - (`((n_iterations + 1) x n_pixels x n_genes_all) ndarray[float32]`): dp_scores. The dot product
                    score for every gene on each iteration. This even includes the iteration that did not assign any new
                    genes so you can see what the final gene scores were before stopping. Only returned if
                    return_dp_scores is true.
                - (`(n_pixels x n_genes) ndarray[float32]`): gene_weights. The gene weights given to each gene on all
                    pixels on their final iteration. For genes that were not assigned on a pixel, nan is placed. Only
                    returned if return_all_weights is true.
                - (`(n_pixels x n_genes x n_rounds_use x n_channels_use) ndarray[float32]`): final_residuals. For every
                    gene, this is the residual colour that is scored against the gene's bled code to find the final
                    pixel scores. In the OMP method documentation, this is denoted by epsilon ^ 2 * tilde{R} with i
                    being the final iteration. For genes that are not assigned to a pixel, nan is placed. Only returned
                    if return_all_residuals is true.

        Notes:
            - All computations are run with 32-bit float precision.
            - The boolean flags are only used in OMP subplots to gather additional OMP insight.
        """
        n_pixels, n_rounds_use, n_channels_use = pixel_colours.shape
        n_rounds_channels_use = n_rounds_use * n_channels_use
        n_genes = bled_codes.shape[0]
        assert type(pixel_colours) is np.ndarray
        assert type(bled_codes) is np.ndarray
        assert type(background_codes) is np.ndarray
        assert type(maximum_iterations) is int
        assert type(dot_product_threshold) is float
        assert type(minimum_intensity) is float
        assert type(alpha) is float
        assert type(beta) is float
        assert type(return_all_scores) is bool
        assert type(return_all_weights) is bool
        assert type(return_all_residuals) is bool
        assert type(force_cpu) is bool
        assert maximum_iterations > 0
        assert dot_product_threshold >= 0
        assert minimum_intensity >= 0
        assert pixel_colours.ndim == 3
        assert bled_codes.ndim == 3
        assert background_codes.ndim == 3
        assert pixel_colours.size > 0, "pixel_colours cannot be empty"
        assert bled_codes.size > 0, "bled_codes cannot be empty"
        assert background_codes.size > 0, "background_codes cannot be empty"
        assert bled_codes.shape == (n_genes, n_rounds_use, n_channels_use)
        assert background_codes.shape == (n_channels_use, n_rounds_use, n_channels_use)

        dp_scores = []
        bled_codes_torch = torch.tensor(bled_codes, dtype=self.DTYPE_T)
        background_codes_torch = torch.tensor(background_codes, dtype=self.DTYPE_T)
        all_bled_codes = torch.concat((bled_codes_torch, background_codes_torch), dim=0)
        # Bled codes and background codes must be L2 normalised.
        assert torch.isclose(torch.linalg.matrix_norm(all_bled_codes), torch.ones(1).float()).all()

        device = system.get_device(force_cpu)

        pixel_scores = torch.zeros((n_pixels, n_genes), dtype=self.DTYPE_T)
        colours = torch.from_numpy(pixel_colours).to(dtype=self.DTYPE_T)
        # Remember the residual colour between iterations.
        residual_colours = colours.detach().clone()
        # Remember what pixels still need iterating on.
        pixels_to_continue = torch.ones(n_pixels, dtype=bool)
        # Remember the gene selections made for each pixel. NO_GENE_ASSIGNMENT for no gene selection made.
        genes_selected = torch.full((n_pixels, maximum_iterations), self.NO_GENE_ASSIGNMENT, dtype=torch.int32)
        bg_gene_indices = torch.linspace(n_genes, n_genes + n_channels_use - 1, n_channels_use, dtype=torch.int32)
        bg_gene_indices = bg_gene_indices[np.newaxis].repeat_interleave(n_pixels, dim=0)

        if return_all_weights:
            # Remember the gene weightings given to each pixel.
            all_weights = torch.full_like(pixel_scores, torch.nan, dtype=self.DTYPE_T)
        if return_all_residuals:
            all_residuals = torch.full((n_pixels, n_genes, n_rounds_use, n_channels_use), torch.nan, dtype=self.DTYPE_T)

        # Move tensors to the right device.
        pixel_scores = pixel_scores.to(device)
        colours = colours.to(device)
        residual_colours = residual_colours.to(device)
        pixels_to_continue = pixels_to_continue.to(device)
        genes_selected = genes_selected.to(device)
        bled_codes_torch = bled_codes_torch.to(device)
        all_bled_codes = all_bled_codes.to(device)
        bg_gene_indices = bg_gene_indices.to(device)

        for iteration in range(maximum_iterations):
            # Find the next best gene for pixels that have not reached a stopping criteria yet.
            fail_gene_indices = torch.cat((genes_selected[:, :iteration], bg_gene_indices), 1)
            fail_gene_indices = fail_gene_indices[pixels_to_continue]
            gene_assigment_results = self.get_next_gene_assignments(
                residual_colours,
                all_bled_codes,
                fail_gene_indices,
                dot_product_threshold,
                minimum_intensity,
                return_all_scores=return_all_scores,
            )
            del fail_gene_indices
            genes_selected[pixels_to_continue, iteration] = gene_assigment_results[0]
            if return_all_scores:
                dp_score = torch.zeros((n_pixels, n_genes + n_channels_use), dtype=self.DTYPE_T)
                dp_score[pixels_to_continue] = gene_assigment_results[1].cpu()
                dp_scores.append(dp_score)

            # Update what pixels to continue iterating on.
            pixels_to_continue = genes_selected[:, iteration] != self.NO_GENE_ASSIGNMENT
            if pixels_to_continue.sum() == 0:
                break
            del gene_assigment_results

            # On the pixels still being iterated on, update the gene weights and hence the residual colours for the
            # next iteration.
            latest_gene_selections = genes_selected[pixels_to_continue, : iteration + 1]
            # Has shape (n_pixels_continue, iteration + 1, n_rounds_use, n_channels_use).
            bled_codes_to_continue = bled_codes_torch[latest_gene_selections]
            residual_colours = self.get_next_gene_weights(
                colours[pixels_to_continue].reshape((-1, n_rounds_channels_use))[:, :, np.newaxis],
                bled_codes_to_continue.reshape((-1, iteration + 1, n_rounds_channels_use)).swapaxes(1, 2),
                alpha,
                beta,
            )
            iteration_weights = residual_colours[2]
            if return_all_weights:
                all_weights[pixels_to_continue, latest_gene_selections] = iteration_weights.cpu()
            epsilon_squared = residual_colours[1]
            epsilon_squared = epsilon_squared.reshape((-1, n_rounds_use, n_channels_use))
            residual_colours = residual_colours[0]
            residual_colours = residual_colours.reshape((-1, n_rounds_use, n_channels_use))
            residual_colours *= epsilon_squared
            del epsilon_squared

            # Using the new gene weights, update the OMP pixel scores.
            pixel_score_result = self.get_gene_pixel_scores(
                colours[pixels_to_continue],
                bled_codes_to_continue,
                iteration_weights,
                alpha,
                beta,
                return_all_residuals,
            )
            new_pixel_scores = pixel_score_result[0]
            if return_all_residuals:
                new_residuals = pixel_score_result[1]
                for j in range(iteration + 1):
                    all_residuals[pixels_to_continue, latest_gene_selections[:, j]] = new_residuals[:, j]
                del new_residuals
            del bled_codes_to_continue, iteration_weights, pixel_score_result
            for j in range(iteration + 1):
                pixel_scores[pixels_to_continue, latest_gene_selections[:, j]] = new_pixel_scores[:, j]
            del latest_gene_selections, new_pixel_scores

        result = (pixel_scores.cpu().numpy(),)
        if return_all_scores:
            result += (np.array([score.cpu().numpy() for score in dp_scores]),)
        if return_all_weights:
            result += (all_weights.cpu().numpy(),)
        if return_all_residuals:
            result += (all_residuals.cpu().numpy(),)
        if len(result) == 1:
            result = result[0]

        return result

    def create_background_bled_codes(self, n_rounds_use: int, n_channels_use: int) -> np.ndarray:
        """
        Create the background bled codes that are used during OMP pixel score computing.

        Args:
            n_rounds_use (int): the number of sequencing rounds.
            n_channels_use (int): the number of sequencing channels.
        """
        bg_bled_codes = np.eye(n_channels_use)[:, None, :].repeat(n_rounds_use, axis=1)
        # Normalise the codes the same way as gene bled codes.
        bg_bled_codes /= np.linalg.norm(bg_bled_codes, axis=(1, 2))
        return bg_bled_codes

    def get_next_gene_assignments(
        self,
        residual_colours: torch.Tensor,
        all_bled_codes: torch.Tensor,
        fail_gene_indices: torch.Tensor,
        dot_product_threshold: float,
        minimum_intensity: float,
        return_all_scores: bool = False,
    ) -> Tuple[torch.Tensor] | Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the next best gene assignment for each residual colour. Each gene is scored to each pixel using a dot
        product scoring where each round has an equal contribution. A pixel fails gene assignment if one or more of the
        conditions is met:

        - The top gene dot product score is below the dot_product_threshold.
        - The next best gene is in the fail_gene_indices list.
        - The intensity of the colour is below the minimum intensity.

        The reason for each of these conditions is:

        - to cut out unconfident gene reads.
        - to not doubly assign a gene and avoid assigning background genes.
        - to cut out dim colour.

        respectively.

        Args:
            residual_colours (`(n_pixels x n_rounds_use x n_channels_use) tensor[float32]`): residual pixel colour. Each
                round/channel pair has been multiplied by a weighting (denoted by epsilon in documentation) such that
                highly uncertain round/channel pairs have a very low contribution to the next scores.
            all_bled_codes (`(n_genes_all x n_rounds_use x n_channels_use) tensor[float32]`): gene bled codes and
                background genes appended.
            fail_gene_indices (`(n_pixels x n_genes_fail) tensor[int32]`): if the next gene assignment for a pixel is
                included on the list of fail gene indices, consider gene assignment a fail.
            dot_product_threshold (float): a gene can only be assigned if the dot product score is above this threshold.
            minimum_intensity (float): a colour's intensity must be above minimum_intensity to pass gene assignment.
                The intensity is defined as min_r (max_c abs(residual_colour)).
            return_all_scores (bool, optional): return the dot product scores for every gene. Default: false.

        Returns:
            Tuple containing:
                - `(n_pixels) tensor[int32]`: next_best_genes. The next best gene assignment for each pixel. A value of
                    -32_768 is placed for pixels that failed to find a next best gene.
                - `(n_pixels x n_genes_all) tensor[float32]`: all_gene_scores. Every genes' round dot product score.
                    This includes genes that are in fail_gene_indices. Only returned if return_scores is set to true.
        """
        assert type(residual_colours) is torch.Tensor
        assert type(all_bled_codes) is torch.Tensor
        assert type(fail_gene_indices) is torch.Tensor
        assert type(dot_product_threshold) is float
        assert type(minimum_intensity) is float
        assert residual_colours.ndim == 3
        assert all_bled_codes.ndim == 3
        assert fail_gene_indices.ndim == 2
        assert residual_colours.shape[0] > 0, "Require at least one pixel"
        assert residual_colours.shape[1] > 0, "Require at least one round/channel"
        assert residual_colours.shape[1:] == all_bled_codes.shape[1:]
        assert all_bled_codes.shape[0] > 0, "Require at least one bled code"
        assert fail_gene_indices.shape[0] == residual_colours.shape[0]
        assert (fail_gene_indices >= 0).all() and (fail_gene_indices < all_bled_codes.shape[0]).all()
        assert dot_product_threshold >= 0
        assert minimum_intensity >= 0

        intensity_is_low = residual_colours.clone().abs().max(2).values.min(1).values < minimum_intensity

        all_gene_scores = dot_product.dot_product_score(
            residual_colours[np.newaxis], all_bled_codes[np.newaxis, np.newaxis]
        )[0]

        next_best_gene_scores, next_best_genes = torch.max(all_gene_scores, dim=1)
        next_best_genes = next_best_genes.int()

        # A pixel only passes if the highest scoring gene is above the dot product threshold.
        pixels_passed = (all_gene_scores > dot_product_threshold).any(1)

        # A best gene in the fail_gene_indices means assignment failed.
        in_fail_gene_indices = (fail_gene_indices == next_best_genes[:, np.newaxis]).any(1)
        pixels_passed = pixels_passed & (~in_fail_gene_indices)

        # An intensity below the minimum_intensity means assignment failed.
        pixels_passed = pixels_passed & (~intensity_is_low)

        next_best_genes[~pixels_passed] = self.NO_GENE_ASSIGNMENT
        next_best_gene_scores[~pixels_passed] = torch.nan

        output = (next_best_genes,)

        if return_all_scores:
            output += (all_gene_scores,)

        return output

    def get_next_gene_weights(
        self,
        pixel_colours: torch.Tensor,
        bled_codes: torch.Tensor,
        alpha: float,
        beta: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        For each pixel, compute a weight for every gene by least squares. These weighted bled codes are then subtracted
        off the pixel colour to get the minimised residual colour for each pixel.

        Args:
            pixel_colours (`(n_pixels x n_rounds_channels_use x 1) tensor[float32]`): each pixel's colour.
            bled_codes (`(n_pixels x n_rounds_channels_use x n_genes_added) tensor[float32]`): the bled code for each
                added gene for each pixel.
            alpha (float): the alpha parameter.
            beta (float): the beta parameter.

        Returns:
            Tuple containing:
                - (`(n_pixels x n_rounds_channels_use) tensor[float32]`): residuals. The residual colour after
                    subtracting the assigned, weighted gene bled codes.
                - (`(n_pixels x n_rounds_channels_use) tensor[float32]`): epsilon_squared. The weighting given to every
                    round/channel during scoring. Weightings below 1 are given when the round/channel already had been
                    strongly assigned to by a bled code. This is due to a higher variance.
                - (`(n_pixels x n_genes_added) tensor[float32]`): gene_weights. The weight given to every gene bled
                    code.
        """
        assert type(pixel_colours) is torch.Tensor
        assert type(bled_codes) is torch.Tensor
        n_rounds_channels_use = pixel_colours.shape[1]
        assert pixel_colours.ndim == 3
        assert bled_codes.ndim == 3
        assert pixel_colours.shape[0] == bled_codes.shape[0]
        assert pixel_colours.shape[1] == bled_codes.shape[1] == n_rounds_channels_use
        assert pixel_colours.shape[2] == 1
        assert bled_codes.shape[0] > 0, "Require at least one pixel to run on"
        assert bled_codes.shape[1] > 0, "Require at least one round and channel"
        assert bled_codes.shape[2] > 0, "Require at least one gene assigned"

        # Compute least squares for gene weights of every gene on the total spot colour.
        # First parameter has shape (n_pixels, n_rounds_channels_use, n_genes_added).
        # Second parameter has shape (n_pixels, n_rounds_channels_use, 1).
        # Therefore, the result has shape (n_pixels, n_genes_added, 1).
        weights = torch.linalg.lstsq(bled_codes, pixel_colours)[0]
        # Squeeze weights to (n_pixels, n_genes_added).
        weights = weights[:, :, 0]

        epsilon_squared = self.get_uncertainty_weights(weights[np.newaxis], bled_codes[np.newaxis], alpha, beta)[0]

        # From the new weights, find the residual spot colours.
        pixel_residuals = pixel_colours[..., 0] - (weights[:, np.newaxis] * bled_codes).sum(2)

        return (pixel_residuals, epsilon_squared, weights)

    def get_gene_pixel_scores(
        self,
        pixel_colours: torch.Tensor,
        bled_codes: torch.Tensor,
        weights: torch.Tensor,
        alpha: float,
        beta: float,
        return_residuals: bool = False,
    ) -> Tuple[torch.Tensor] | Tuple[torch.Tensor, torch.Tensor]:
        """
        For each gene assignment in a pixel, compute its pixel score. For each gene, a residual colour is computed by
        subtracting all other assigned genes. Then, the pixel score for said gene is the dot product with this residual
        and the genes bled code.

        Args:
            pixel_colours (`(n_pixels x n_rounds_use x n_channels_use) tensor[float32]`): the pixel colours.
            bled_codes (`(n_pixels x n_genes_assigned x n_rounds_use x n_channels_use) tensor[float32]`): the bled codes
                for every assigned gene. Their L2 norm over rounds and channels is always one.
            weights (`(n_pixels x n_genes_assigned) tensor[float32]`): the computed weight given to each bled code to
                best match the pixel colour.
            alpha (float): the alpha parameter.
            beta (float): the beta parameter.
            return_residuals (bool, optional): return the residuals used to compute the pixel scores for each gene.
                Default: false.

        Returns tuple containing:
            - (`(n_pixels x n_genes_assigned) tensor[float32]`): gene_pixel_scores. The gene pixel scores for every
                given pixel.
            - (`(n_pixels x n_genes_assigned x n_rounds_use x n_channels_use) tensor[float32]`): residuals. The
                residuals used to compute the pixel scores. Denoted by epsilon ^ 2 * tilde{R} in the OMP method
                documentation. Only given if return_residuals is true.
        """
        assert type(pixel_colours) is torch.Tensor
        assert type(bled_codes) is torch.Tensor
        assert type(weights) is torch.Tensor
        assert type(alpha) is float
        assert type(beta) is float
        assert type(return_residuals) is bool
        assert pixel_colours.ndim == 3
        assert bled_codes.ndim == 4
        assert weights.ndim == 2
        assert pixel_colours.shape == bled_codes.shape[:1] + bled_codes.shape[2:]
        assert weights.shape[:2] == bled_codes.shape[:2]

        n_pixels, n_rounds_use, n_channels_use = pixel_colours.shape
        n_genes_assigned = bled_codes.shape[1]

        # Has shape (n_pixels, n_genes_assigned, n_rounds_use, n_channels_use).
        weighted_bled_codes = bled_codes * weights[:, :, np.newaxis, np.newaxis]

        # bled_codes_sums_except_one[:, g] is the sum of weighted bled codes except gene g's weighted bled code.
        # It has shape (n_pixels, n_genes_assigned, n_rounds_use, n_channels_use).
        bled_codes_sum_except_one = weighted_bled_codes.sum(1, keepdim=True).repeat_interleave(n_genes_assigned, 1)
        bled_codes_sum_except_one -= weighted_bled_codes
        # Change its shape to (n_genes_assigned, n_pixels, n_rounds_use, n_channels_use).
        bled_codes_sum_except_one = bled_codes_sum_except_one.swapaxes(0, 1)
        del weighted_bled_codes

        # colour_residuals has shape (n_genes_assigned, n_pixels, n_rounds_use, n_channels_use).
        # colour_residuals[g] is the pixel colour minus all weighted bled codes except the one for gene g.
        #
        # Denoted as $\tilde{R}$ in the docs.
        colour_residuals = pixel_colours.detach().clone()[np.newaxis] - bled_codes_sum_except_one
        del bled_codes_sum_except_one

        # bled_codes_except_one[g] is every bled code except the bled code for gene g.
        # It has shape (n_genes_assigned, n_pixels, n_genes_assigned - 1, n_rounds_use, n_channels_use).
        # This will be needed to calculate the uncertainty weighting for each gene assignment individually.
        # See Step 3 in OMP method documentation for details.
        bled_codes_except_one = bled_codes.detach().clone()[np.newaxis].repeat_interleave(n_genes_assigned, 0)
        bled_codes_except_one = bled_codes_except_one[:, :, :-1]
        for g in range(n_genes_assigned):
            bled_codes_except_one[g] = torch.cat((bled_codes[:, :g], bled_codes[:, (g + 1) :]), dim=1)
        # Flatten to shape (n_genes_assigned, n_pixels, n_genes_assigned - 1, n_rounds_channels_use).
        bled_codes_except_one = bled_codes_except_one.reshape(
            (n_genes_assigned, n_pixels, n_genes_assigned - 1, n_rounds_use * n_channels_use)
        )
        # Swap dimensions to shape (n_genes_assigned, n_pixels, n_rounds_channels_use, n_genes_assigned - 1).
        bled_codes_except_one = bled_codes_except_one.swapaxes(2, 3)

        # Similarly, weights_except_one[g] is every weight except the weight for gene g.
        # It has shape (n_genes_assigned, n_pixels, n_genes_assigned - 1).
        weights_except_one = weights.detach().clone()[np.newaxis].repeat_interleave(n_genes_assigned, 0)
        weights_except_one = weights_except_one[:, :, :-1]
        for g in range(n_genes_assigned):
            weights_except_one[g] = torch.cat((weights[:, :g], weights[:, (g + 1) :]), dim=1)

        # epsilon_squared has shape (n_genes_assigned, n_pixels, n_rounds_channels_use).
        epsilon_squared = self.get_uncertainty_weights(weights_except_one, bled_codes_except_one, alpha, beta)
        del bled_codes_except_one, weights_except_one
        # Expand to shape (n_genes_assigned, n_pixels, n_rounds_use, n_channels_use).
        epsilon_squared = epsilon_squared.reshape((n_genes_assigned, n_pixels, n_rounds_use, n_channels_use))

        colour_residuals *= epsilon_squared
        del epsilon_squared

        pixel_scores = dot_product.dot_product_score(colour_residuals, bled_codes.swapaxes(0, 1)[:, :, np.newaxis])[
            :, :, 0
        ]

        # Change pixel_scores shape to (n_pixels x n_genes_assigned).
        pixel_scores = pixel_scores.swapaxes(0, 1)

        # Set pixel scores to be negative if their gene's weight is negative.
        pixel_scores *= torch.sign(weights)

        result = (pixel_scores,)

        if return_residuals:
            colour_residuals = colour_residuals.swapaxes(0, 1)
            result += (colour_residuals,)

        return result

    def get_uncertainty_weights(
        self, gene_weights: torch.Tensor, bled_codes: torch.Tensor, alpha: float, beta: float
    ) -> torch.Tensor:
        """
        Compute the weights given to each round/channel pair. A round/channel pair has a lower weight if it has high
        bled code brightness in said round/channel and alpha is > 0.

        Args:
            gene_weights (`(n_batches x n_pixels x n_genes_assigned) tensor[float32]`): the weight found for each bled code.
            bled_codes (`(n_batches x n_pixels x n_rounds_channels_use x n_genes_assigned) tensor[float32]`): the assigned bled codes.
            alpha (float): how much the error scales based on the weighted bled code brightness in the round/channel
                pair.
            beta (float): the square root of the constant error uncertainty that is there even if the brightness is
                zero.

        Returns:
            (`(n_batches x n_pixels x n_rounds_channels_use) tensor[float32]`): epsilon_squared. The weighting given to
                each pixel's round/channel pair. Weightings are lower for more uncertain brightnesses so the have a
                lower contribution to further gene scores.

        Notes:
            - If n_batches is 1 for one of the tensors, then it is repeated for the maximum batch count.
            - See the OMP method documentation for more detail on the uncertainty calculation.
        """
        assert type(gene_weights) is torch.Tensor
        assert type(bled_codes) is torch.Tensor
        assert gene_weights.ndim == 3
        assert bled_codes.ndim == 4
        assert gene_weights.shape[:2] == bled_codes.shape[:2]
        assert gene_weights.shape[2] == bled_codes.shape[3]
        assert type(alpha) is float
        assert type(beta) is float
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        if beta <= 0:
            raise ValueError(f"beta must be > 0, got {beta}")

        n_rounds_channels_use = bled_codes.shape[2]

        # Has shape (n_batches, n_pixels, n_rounds_channels_use).
        sigma_squared = beta**2 + alpha * (torch.square(gene_weights[:, :, np.newaxis] * bled_codes)).sum(-1)
        sigma_squared = torch.reciprocal(sigma_squared)

        # Computing epsilon squared like in the documentation.
        epsilon_squared = n_rounds_channels_use * sigma_squared / sigma_squared.sum(-1, keepdim=True)

        return epsilon_squared
