# Originally created by Max Shinn, August 2023
# Refactored and expanded by Paul Shuker, September 2023 - present
import os
import csv
import bz2
import json
import time
import pickle
import pandas
import shutil
import napari
import warnings
import dask.array
import scipy.stats
import numpy as np
import math as maths
from tqdm import tqdm
import numpy.typing as npt
from typing import Dict, List, Any, Tuple, Optional

from ..omp import scores as omp_scores
from .. import utils
from ..pipeline import run


DEFAULT_IMAGE_DTYPE = float
DEFAULT_INSTANCE_FILENAME = "robominnie.pkl"
USE_INSTANCE_GENE_CODES = None
# Y, X, Z giant image padding. Useful for extra space when rotating/translating tiles when "un-stitching"
IMAGE_PADDING = [200, 200, 0]


class RoboMinnie:
    """
    RoboMinnie
    ==========
    Coppafish integration suite

    Provides:
    ---------
    1. Modular, customisable synthetic data generation for coppafish
    2. Coppafish raw ``.npy`` file generation for full pipeline run
    3. Coppafish scoring using ground-truth synthetic spot data

    Usage:
    ------
    Create new RoboMinnie instance for each integration test. Call functions for data generation (see ``robominnie.py``
    functions for options). Call ``save_raw_data``, then ``run_coppafish``. Use ``compare_spots_omp`` or
    ``compare_ref_spots`` to evaluate spot results.
    """

    def __init__(
        self,
        n_channels: int = 7,
        n_tiles_x: int = 1,
        n_tiles_y: int = 1,
        n_rounds: int = 7,
        n_planes: int = 4,
        n_tile_yx: Tuple[int, int] = (2048, 2048),
        include_anchor: bool = True,
        include_presequence: bool = True,
        include_dapi: bool = True,
        anchor_channel: int = 1,
        tile_overlap: float = 0.15,
        image_minimum: float = 0,
        image_dtype: Any = None,
        brightness_scale_factor: npt.NDArray[np.float_] = None,
        seed: int = 0,
    ) -> None:
        """
        Create a new RoboMinnie instance. Used to manipulate, create and save synthetic data in a modular, customisable
        way. Synthetic data is saved as raw .npy files and includes everything needed for coppafish to run a full
        pipeline.

        Args:
            n_channels (int, optional): number of sequencing channels. Default: `7`.
            n_tiles_x (int, optional): number of tiles along the x axis. Default: `1`.
            n_tiles_y (int, optional): number of tiles along the y axis. Default: `1`.
            n_rounds (int, optional): number of sequencing rounds. Default: `7`.
            n_planes (int, optional): number of z planes. Default: `4`.
            n_tile_yx (`tuple[int, int]`, optional): number of pixels in each tile in the y and x directions
                respectively. Default: `(2048, 2048)`.
            include_anchor (bool, optional): whether to include the anchor round. Default: true.
            include_presequence (bool, optional): whether to include the pre-sequence round. Default: true.
            include_dapi (bool, optional): whether to include a DAPI image. The DAPI image will be saved in all rounds
                (sequence, presequence, anchor rounds) if they are set to be included. Uses a single channel, stains
                the cell nuclei so they light up to find cell locations. The DAPI channel is set to `0`, before the
                sequencing channels. Default: true.
            anchor_channel (int, optional): the anchor channel, cannot be the same as the dapi channel (0). Default:
                `1`.
            tile_overlap (float, optional): amount of tile overlap, as a fraction of tile length. Default: `0.15`.
            image_minimum (float, optional): minimum pixel value possible. Images will be shifted to be at least
                `image_minimum` before saving the raw image files. Default: `0`.
            image_dtype (any, optional): datatype of images. Default: float.
            brightness_scale_factor (`(n_tiles x (n_rounds + 2) x n_channels) ndarray[float]`): a brightness scale
                factor, applied to each image before saving as raw `.npy` images for a coppafish run. Round 0 is the
                preseq, rounds (1 to self.n_rounds) is sequence, self.n_rounds + 1 is anchor round. Default: All ones.
            seed (int, optional): seed used throughout the generation of random data, specify integer value for
                reproducible output. if none, seed is randomly picked. Default: `0`.

        Notes:
            - RAM usage will scale with tile count.
            - We are assuming that `n_channels` is the number of dyes used.
        """
        self.n_channels = n_channels
        self.n_tiles_x = n_tiles_x
        self.n_tiles_y = n_tiles_y
        self.n_tiles = self.n_tiles_x * self.n_tiles_y
        self.n_rounds = n_rounds
        self.n_planes = n_planes
        self.n_tile_yx = n_tile_yx
        self.n_spots = 0
        self.bleed_matrix = None
        # Has shape n_spots x 3 (y,x,z). These positions are on the giant image that contains all tiles on it.
        self.true_spot_positions_pixels = np.zeros((0, 3), dtype=np.float64)
        self.true_spot_identities = np.zeros((0), dtype=str)  # Has shape n_spots, saves every spots gene
        self.include_anchor = include_anchor
        self.include_presequence = include_presequence
        self.include_dapi = include_dapi
        self.anchor_channel = anchor_channel
        self.tile_overlap = tile_overlap
        self.image_minimum = image_minimum
        # DAPI channel should be appended to the start of the sequencing images array
        self.dapi_channel = 0
        if image_dtype is None:
            self.image_dtype = DEFAULT_IMAGE_DTYPE
        else:
            self.image_dtype = image_dtype
        self.seed = seed
        expected_brightness_scale_factor_shape = (self.n_tiles, self.n_rounds + 2, self.n_channels + 1)
        if brightness_scale_factor is None:
            self.brightness_scale_factor = np.ones(expected_brightness_scale_factor_shape)
        if brightness_scale_factor is not None:
            self.brightness_scale_factor = brightness_scale_factor

        assert (
            self.brightness_scale_factor.shape == expected_brightness_scale_factor_shape
        ), f"Unexpected brightness_scale_factor shape, expected {expected_brightness_scale_factor_shape}"
        # n_yx is the calculated giant image size required to fit all the tiles in, with the given tile overlap
        self.n_yxz = (
            IMAGE_PADDING[0]
            + self.n_tile_yx[0] * self.n_tiles_y
            - round(tile_overlap * self.n_tile_yx[0] * (self.n_tiles_y - 1)),
            IMAGE_PADDING[1]
            + self.n_tile_yx[1] * self.n_tiles_x
            - round(tile_overlap * self.n_tile_yx[1] * (self.n_tiles_x - 1)),
            IMAGE_PADDING[2] + self.n_planes,
        )
        self.instructions = []  # Keep track of the functions called inside RoboMinnie, in order
        self.instructions.append(utils.base.get_function_name())
        assert self.n_channels > 0, "Require at least one channel"
        assert self.n_rounds > 0, "Require at least one round"
        assert self.n_planes > 0, "Require at least one z plane"
        assert self.n_tiles_x > 0, "Require at least 1 tile in the x direction"
        assert self.n_tiles_y > 0, "Require at least 1 tile in the y direction"
        assert 0 <= self.tile_overlap < 1, f"Require a tile overlap (0,1], got {self.tile_overlap}"
        if self.include_anchor:
            assert self.anchor_channel > 0 and self.anchor_channel <= self.n_channels, (
                f"Anchor channel must be in range {1} to {self.n_channels} (inclusive), but got "
                + f"{self.anchor_channel}"
            )
        if self.include_dapi:
            assert (
                self.dapi_channel != self.anchor_channel
            ), "Cannot have DAPI and anchor channel identical because they are both saved in the same anchor round"
        # Ordering for data matrices is round x channel x Y x X x Z, like the nd2 raw files for coppafish
        # Extra channel is added for DAPI channel support, if needed
        self.shape = (
            self.n_rounds,
            self.n_channels + 1,
            self.n_yxz[0],
            self.n_yxz[1],
            self.n_yxz[2],
        )
        self.anchor_shape = (
            self.n_channels + 1,
            self.n_yxz[0],
            self.n_yxz[1],
            self.n_yxz[2],
        )
        self.presequence_shape = (
            self.n_channels + 1,
            self.n_yxz[0],
            self.n_yxz[1],
            self.n_yxz[2],
        )
        # These are the images we will build throughout RoboMinnie and eventually save, starting with just zeroes.
        # The DAPI channel is contained within the anchor, pre-sequencing and sequencing images. We use one giant
        # image, across all tiles. Then, when saving as a raw files, we "un-stitch" the giant image into each
        # individual tile with the given tile overlap.
        self.image = np.zeros(self.shape, dtype=self.image_dtype)
        self.anchor_image = np.zeros(self.anchor_shape, dtype=self.image_dtype)
        self.presequence_image = np.zeros(self.presequence_shape, dtype=self.image_dtype)
        if self.n_tile_yx[0] != self.n_tile_yx[1]:
            raise NotImplementedError("Coppafish does not support non-square tiles")
        if self.n_planes < 4:
            warnings.warn("Coppafish may break with fewer than four z planes")

    def generate_gene_codes(
        self, n_genes: int = 20, n_rounds: Optional[int] = None, n_channels: Optional[int] = None
    ) -> Dict:
        """
        Generates random gene codes based on reed-solomon principle, using the lowest degree polynomial possible
        relative to the number of genes wanted. Saves codes in self, can be used in function `Add_Spots`. The `i`th
        gene name will be `gene_i`. `ValueError` is raised if all gene codes created are not unique. We assume that
        `n_rounds` is also the number of unique dyes, each dye is labelled between `(0, n_rounds]`.

        Args:
            n_genes (int, optional): number of unique gene codes to generate. Default: 20.
            n_rounds (int, optional): number of sequencing rounds. Default: Use round number saved in `self`.

        Returns:
            Dict (str: str): gene names as keys, gene codes as values.

        Notes:
            See [here](https://en.wikipedia.org/wiki/Reed%E2%80%93Solomon_error_correction) for more details.
        """
        self.instructions.append(utils.base.get_function_name())
        if n_rounds is None:
            n_rounds = self.n_rounds
        if n_channels is None:
            n_channels = self.n_channels

        codes = utils.base.reed_solomon_codes(n_genes, n_rounds, n_channels)
        self.codes = codes
        return codes

    def generate_pink_noise(
        self,
        noise_amplitude: float = 1.5e-3,
        noise_spatial_scale: float = 0.1,
        include_sequence: bool = True,
        include_anchor: bool = True,
        include_presequence: bool = True,
        include_dapi: bool = True,
    ) -> None:
        """
        Superimpose pink noise onto images, if used. The noise is identical on all images because pink noise is a good
        estimation for biological things that fluoresce. See
        [here](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.73.814) for more details. You may expect the
        DAPI image to include pink noise that is not part of the other images because of the distinct nuclei staining.

        Args:
            noise_amplitude (float): The maximum possible noise intensity. Default: `0.0015`.
            noise_spatial_scale (float): Spatial scale of noise. Scales with image size. Default: `0.1`.
            include_sequence (bool, optional): Include sequencing images. Default: true.
            include_anchor (bool, optional): Include anchor image. Default: true.
            include_presequence (bool, optional): Include pre-sequencing images. Default: true.
            include_dapi (bool, optional): Include DAPI image. Default: true.
        """
        # TODO: Add the ability to square the pink noise, gives sharper peaks which is meant to be more realistic
        self.instructions.append(utils.base.get_function_name())
        print(f"Generating pink noise...")

        # True spatial scale should be maintained regardless of the image size, so we scale it as such.
        true_noise_spatial_scale = noise_spatial_scale * np.asarray([*self.n_yxz[:2], 10 * self.n_yxz[2]])
        # Generate pink noise
        pink_spectrum = 1 / (
            1
            + np.linspace(0, true_noise_spatial_scale[0], self.n_yxz[0])[:, None, None] ** 2
            + np.linspace(0, true_noise_spatial_scale[1], self.n_yxz[1])[None, :, None] ** 2
            + np.linspace(0, true_noise_spatial_scale[2], self.n_yxz[2])[None, None, :] ** 2
        )
        rng = np.random.RandomState(self.seed)

        pink_sampled_spectrum = pink_spectrum * np.fft.fftshift(scipy.fft.fftn(rng.randn(*self.n_yxz)))
        pink_noise = np.abs(scipy.fft.ifftn(np.fft.ifftshift(pink_sampled_spectrum)))
        pink_noise = (pink_noise - np.mean(pink_noise)) * noise_amplitude / np.std(pink_noise)

        for r in range(self.n_rounds):
            for c in range(self.n_channels):
                if include_sequence:
                    self.image[r, c + 1 + self.dapi_channel] += pink_noise
            if include_dapi and self.include_dapi:
                self.image[r, self.dapi_channel] += pink_noise
        if include_presequence and self.include_presequence:
            for c in range(self.n_channels):
                self.presequence_image[(c + self.dapi_channel + 1)] += pink_noise
        if include_anchor and self.include_anchor:
            self.anchor_image[self.anchor_channel] += pink_noise
        if include_dapi and self.include_dapi:
            self.anchor_image[self.dapi_channel] += pink_noise
            self.presequence_image[self.dapi_channel] += pink_noise

    def generate_random_noise(
        self,
        noise_std: float,
        noise_mean_amplitude: float = 0,
        noise_type: str = "normal",
        include_anchor: bool = True,
        include_presequence: bool = True,
        include_dapi: bool = True,
    ) -> None:
        """
        Superimpose random, white noise onto every pixel individually. Good for modelling random noise from the camera.

        Args:
            noise_mean_amplitude (float, optional): Mean amplitude of random noise. Default: 0.
            noise_std (float): Standard deviation/width of random noise.
            noise_type (str('normal', 'uniform', or 'poisson'), optional): Type of random noise to apply. Default:
                'normal'.
            include_anchor (bool, optional): Whether to apply random noise to anchor image. Default: true.
            include_presequence (bool, optional): Whether to apply random noise to presequence rounds. Default: true.
            include_dapi (bool, optional): Whether to apply random noise to DAPI image. Default: true.
        """
        self.instructions.append(utils.base.get_function_name())
        print("Generating random noise")

        assert noise_std > 0, f"Noise standard deviation must be > 0, got {noise_std}"

        def _generate_noise(
            _rng: np.random.Generator, _noise_type: str, _noise_mean_amplitude: float, _noise_std: float, _size: tuple
        ) -> npt.NDArray[np.float64]:
            """
            Generate noise based on the specified noise type and parameters.

            Args:
                _rng (Generator): Random number generator.
                _noise_type (str): Type of noise ('normal', 'uniform', or 'poisson').
                _noise_mean_amplitude (float): Mean or center of the noise distribution.
                _noise_std (float): Standard deviation or spread of the noise distribution.
                _size (tuple of int, int, int): Size of the noise array.

            Returns:
                ndarray: Generated noise array.

            Raises:
                ValueError: If an unsupported noise type is provided.
            """
            if _noise_type == "normal":
                return _rng.normal(_noise_mean_amplitude, _noise_std, _size)
            elif _noise_type == "uniform":
                return _rng.uniform(
                    _noise_mean_amplitude - _noise_std / 2, _noise_mean_amplitude + _noise_std / 2, _size
                )
            elif _noise_type == "poisson":
                return _noise_std * (rng.poisson(size=_size) + _noise_mean_amplitude - 1)
            else:
                raise ValueError(f"Unknown noise type: {_noise_type}")

        rng = np.random.RandomState(self.seed)

        sequence_noise = np.zeros(shape=self.shape, dtype=np.float32)
        anchor_noise = np.zeros(shape=self.anchor_shape, dtype=np.float32)
        presequence_noise = np.zeros(shape=self.presequence_shape, dtype=np.float32)

        sequence_noise[:, (self.dapi_channel + 1) :] = _generate_noise(
            rng, noise_type, noise_mean_amplitude, noise_std, self.shape
        )[:, (self.dapi_channel + 1) :]

        if include_anchor and self.include_anchor:
            anchor_noise[self.anchor_channel] = _generate_noise(
                rng, noise_type, noise_mean_amplitude, noise_std, self.anchor_shape
            )[self.anchor_channel]
            np.add(self.anchor_image, anchor_noise, out=self.anchor_image)
        if include_presequence and self.include_presequence:
            presequence_noise[(self.dapi_channel + 1) :] = _generate_noise(
                rng, noise_type, noise_mean_amplitude, noise_std, self.presequence_shape
            )[(self.dapi_channel + 1) :]
        if include_dapi and self.include_dapi:
            sequence_noise[:, self.dapi_channel] = _generate_noise(
                rng, noise_type, noise_mean_amplitude, noise_std, self.shape
            )[:, self.dapi_channel]
            anchor_noise[self.dapi_channel] = _generate_noise(
                rng, noise_type, noise_mean_amplitude, noise_std, self.anchor_shape
            )[self.dapi_channel]
            presequence_noise[self.dapi_channel] = _generate_noise(
                rng, noise_type, noise_mean_amplitude, noise_std, self.presequence_shape
            )[self.dapi_channel]
        del anchor_noise

        # Add new random noise to each pixel
        np.add(self.image, sequence_noise, out=self.image)
        np.add(self.presequence_image, presequence_noise, out=self.presequence_image)

    def add_spots(
        self,
        n_spots: Optional[int] = None,
        bleed_matrix: npt.NDArray[np.float_] = None,
        spot_size_pixels: npt.NDArray[np.float_] = None,
        gene_codebook_path: str = USE_INSTANCE_GENE_CODES,
        spot_amplitude: float = 1,
        include_dapi: bool = False,
        spot_size_pixels_dapi: npt.NDArray[np.float_] = None,
        spot_amplitude_dapi: float = 1,
        gene_efficiency: npt.NDArray[np.float_] = None,
        background_offset: npt.NDArray[np.float_] = None,
    ) -> None:
        """
        Superimpose spots onto images in both space and channels (based on the bleed matrix). Also applied to the
        anchor when included. The spots are uniformly, randomly distributed across each image. Spots are never added to
        presequence images. We assume that `n_channels == n_dyes`.

        Args:
            n_spots (int, optional): Number of spots to superimpose. Default: `floor(3% * total_imaging_volume)`.
            bleed_matrix (`n_dyes x n_channels ndarray[float, float]`, optional): The bleed matrix, used to map each
                dye to its pattern as viewed by the camera in each channel. Default: Ones along the diagonals.
            spot_size_pixels (`(3) ndarray[float]`): The spot's standard deviation in directions `x, y, z`
                respectively. Default: `array([1.5, 1.5, 1.5])`.
            gene_codebook_path (str, optional): Path to the gene codebook, saved as a .txt file. Default: use `self`
                gene codes instead, which can be generated by calling `Generate_Gene_Codes`.
            spot_amplitude (float, optional): Peak spot brightness scale factor. Default: `1`.
            include_dapi (bool, optional): Add spots to the DAPI channel in sequencing and anchor rounds, at the same
                positions. Default: false.
            spot_size_pixels_dapi (`(3) ndarray[float]`, optional): Spots' standard deviation when in the
                DAPI image. Default: Same as `spot_size_pixels`.
            spot_amplitude_dapi (float, optional): Peak DAPI spot brightness scale factor. Default: `1`.
            gene_efficiency (`(n_genes x n_rounds) ndarray[float]`): Gene efficiency scale factor (sometimes denoted
                by lambda), each spot is multiplied by this value, can be unique for every sequencing round and anchor
                round. Not applied to the DAPI spots. Default: Ones everywhere.
            background_offset (`(n_spots x n_channels) ndarray[float]`): After `gene_efficiency` is applied, a
                background offset[s,c] is added to spot `s` in channel `c`. Default: Zero offset.
        """
        self.instructions.append(utils.base.get_function_name())

        def _blit(source, target, loc):
            """
            Superimpose given spot image (source) onto a target image (target) at the centred position loc. The
            parameter target is then updated with the final image.

            Args:
                source (n_channels (optional) x spot_size_y x spot_size_x x spot_size_z ndarray): The spot image.
                target (n_channels (optional) x tile_size_y x tile_size_x x tile_size_z ndarray): The tile image.
                loc (channel (optional), y, x, z ndarray): Central spot location.
            """
            source_size = np.asarray(source.shape)
            target_size = np.asarray(target.shape)
            # If we had infinite boundaries, where would we put it?  Assume "loc" is the centre of "target"
            target_loc_tl = loc - source_size // 2
            target_loc_br = target_loc_tl + source_size
            # Compute the index for the source
            source_loc_tl = -np.minimum(0, target_loc_tl)
            source_loc_br = source_size - np.maximum(0, target_loc_br - target_size)
            # Recompute the index for the target
            target_loc_br = np.minimum(target_size, target_loc_tl + source_size)
            target_loc_tl = np.maximum(0, target_loc_tl)
            # Compute slices from positions
            target_slices = [slice(s1, s2) for s1, s2 in zip(target_loc_tl, target_loc_br)]
            source_slices = [slice(s1, s2) for s1, s2 in zip(source_loc_tl, source_loc_br)]
            # Perform the blit
            target[tuple(target_slices)] += source[tuple(source_slices)]

        if bleed_matrix is None:
            bleed_matrix = np.diag(np.ones(self.n_channels))
        if spot_size_pixels is None:
            spot_size_pixels = np.asarray([1.5, 1.5, 1.5])
        assert (
            bleed_matrix.shape[1] == self.n_channels
        ), f"Bleed matrix does not have n_channels={self.n_channels} as expected"
        if gene_codebook_path != USE_INSTANCE_GENE_CODES:
            assert os.path.isfile(gene_codebook_path), f"Gene codebook at {gene_codebook_path} does not exist"
        assert spot_size_pixels.size == 3, "`spot_size_pixels` must be in three dimensions"
        if bleed_matrix.shape[0] != bleed_matrix.shape[1]:
            warnings.warn(f"Given bleed matrix does not have equal channel and dye counts like usual")
        if self.bleed_matrix is None:
            self.bleed_matrix = bleed_matrix
        else:
            assert self.bleed_matrix == bleed_matrix, "All added spots must have the same shared bleed matrix"
        if spot_size_pixels_dapi is None:
            spot_size_pixels_dapi = spot_size_pixels.copy()
        assert spot_size_pixels_dapi.size == 3, "DAPI spot size must be in three dimensions"
        if gene_efficiency is None:
            gene_efficiency = np.ones((len(self.codes), self.n_rounds + self.include_anchor))
        assert gene_efficiency.shape == (len(self.codes), self.n_rounds + self.include_anchor), (
            f"Gene efficiency must have shape `n_genes x n_rounds`=="
            + f"{(len(self.codes), self.n_rounds + self.include_anchor)}, got {gene_efficiency.shape}"
        )
        if n_spots is None:
            n_spots = maths.floor(0.03 * self.n_planes * self.n_tile_yx[0] * self.n_tile_yx[1] * self.n_tiles)
        assert n_spots > 0, f"Expected n_spots > 0, got {n_spots}"
        if background_offset is None:
            background_offset = np.zeros((n_spots, self.n_channels))
        assert background_offset.shape == (
            n_spots,
            self.n_channels,
        ), f"background_offset must have shape {(n_spots, self.n_channels)}, got {background_offset.shape}"

        self.n_spots += n_spots
        if gene_codebook_path != USE_INSTANCE_GENE_CODES:
            # Read in the gene codebook txt file
            _codes = dict()
            with open(gene_codebook_path, "r") as f:
                lines = f.readlines()
                for line in tqdm(lines, desc="Reading genebook", ascii=True, unit="genes"):
                    if not line:
                        # Skip empty lines
                        continue
                    phrases = line.split()
                    gene, code = phrases[0], phrases[1]
                    # Save the gene name as a key, the value is the gene's code
                    _codes[gene] = code
            self.codes = _codes

        values = list(self.codes.values())
        assert len(values) == len(set(values)), f"Duplicate gene code found in dictionary: {self.codes}"

        # Generate random spots
        rng = np.random.RandomState(self.seed)
        # The tile number that the spot is stored on, in the future this could be two tiles at once, with overlap
        # included.
        # Store the spots' global positions relative to the entire giant image
        true_spot_positions_pixels = rng.rand(n_spots, 3) * [*self.n_yxz]
        true_spot_identities = list(rng.choice(list(self.codes.keys()), n_spots))

        codes_list = list(self.codes.items())

        # We assume each spot is a multivariate gaussian with a diagonal covariance,
        # where variance in each dimension is given by the spot size.  We create a spot
        # template image and then iterate through spots.  Each iteration, we add
        # ("blit") the spot onto the image such that the centre of the spot is in the
        # middle.  The size of the spot template is guaranteed to be odd, and is about
        # 1.5 times the standard deviation.  We add it to the appropriate color channels
        # (by transforming through the bleed matrix) and then also add the spot to the
        # anchor.
        ind_size = np.ceil(spot_size_pixels * 1.5).astype(int) * 2 + 1
        indices = np.indices(ind_size) - ind_size[:, None, None, None] // 2
        spot_img = scipy.stats.multivariate_normal([0, 0, 0], np.eye(3) * spot_size_pixels).pdf(
            indices.transpose(1, 2, 3, 0)
        )
        np.multiply(spot_img, spot_amplitude * np.prod(spot_size_pixels) / 3.375, out=spot_img)
        ind_size = np.ceil(spot_size_pixels_dapi * 1.5).astype(int) * 2 + 1
        indices = np.indices(ind_size) - ind_size[:, None, None, None] // 2
        spot_img_dapi = scipy.stats.multivariate_normal([0, 0, 0], np.eye(3) * spot_size_pixels_dapi).pdf(
            indices.transpose(1, 2, 3, 0)
        )
        np.multiply(spot_img_dapi, spot_amplitude_dapi * np.prod(spot_size_pixels_dapi) / 3.375, out=spot_img_dapi)
        s = 0
        for p, ident in tqdm(
            zip(true_spot_positions_pixels, true_spot_identities),
            desc="Superimposing spots",
            ascii=True,
            unit="spots",
            total=n_spots,
        ):
            gene_index = [idx for idx, key in enumerate(codes_list) if key[0] == ident][0]
            p = np.asarray(p).astype(int)
            p_chan = np.round([self.n_channels // 2, p[0], p[1], p[2]]).astype(int)
            for r in range(self.n_rounds):
                dye = int(self.codes[ident][r])
                ge = gene_efficiency[gene_index, r]
                source = (spot_img * ge)[None, :] * bleed_matrix[dye][:, None, None, None]
                for c in range(source.shape[0]):
                    bg_offset = background_offset[s, c]
                    source[c] -= bg_offset

                _blit(source, self.image[r, (self.dapi_channel + 1) :], p_chan)
                if include_dapi and self.include_dapi:
                    _blit(spot_img_dapi, self.image[r, self.dapi_channel], p)
            if self.include_anchor:
                ge = gene_efficiency[gene_index, self.n_rounds]
                bg_offset = background_offset[s, self.anchor_channel]
                _blit(spot_img * ge - bg_offset, self.anchor_image[self.anchor_channel], p)
            if include_dapi and self.include_dapi and self.include_anchor:
                _blit(spot_img_dapi, self.anchor_image[self.dapi_channel], p)
            s += 1

        # Append just in case spots are superimposed multiple times
        assert len(set(true_spot_identities)) == len(
            self.codes
        ), "Some gene codes were never added, consider increasing the number of spots"
        self.true_spot_identities = np.append(self.true_spot_identities, np.asarray(true_spot_identities))
        self.true_spot_positions_pixels = np.append(self.true_spot_positions_pixels, true_spot_positions_pixels, axis=0)

    # Post-Processing function
    def fix_image_minimum(self, minimum: float = 0.0) -> None:
        """
        Ensure all pixels in the images are greater than or equal to given value (minimum). Includes the presequence
        and anchor images, if they exist.

        Args:
            minimum (float, optional): Minimum pixel value allowed. Default: `0`.
        """
        self.instructions.append(utils.base.get_function_name())
        print(f"Fixing image minima")

        minval = self.image[:, (self.dapi_channel + 1) :].min()
        minval = np.min([minval, self.anchor_image[self.anchor_channel].min() if self.include_anchor else np.inf])
        minval = np.min(
            [minval, self.presequence_image[(self.dapi_channel + 1) :].min() if self.include_presequence else np.inf]
        )
        minval = np.min([minval, self.image[:, self.dapi_channel].min() if self.include_dapi else np.inf])
        minval = np.min([minval, self.anchor_image[self.dapi_channel].min() if self.include_dapi else np.inf])
        minval = np.min([minval, self.presequence_image[self.dapi_channel].min() if self.include_dapi else np.inf])
        offset = -(minval + minimum)
        np.add(self.image[:, (not self.include_dapi) :], offset, out=self.image[:, (not self.include_dapi) :])
        if self.include_anchor:
            np.add(
                self.anchor_image[(not self.include_dapi) :], offset, out=self.anchor_image[(not self.include_dapi) :]
            )
        if self.include_presequence:
            np.add(
                self.presequence_image[(not self.include_dapi) :],
                offset,
                out=self.presequence_image[(not self.include_dapi) :],
            )

    # Post-Processing function
    def offset_images_by(
        self, constant: float, include_anchor: bool = True, include_presequence: bool = True, include_dapi: bool = True
    ) -> None:
        """
        Shift every image pixel, in all tiles, by a constant value.

        Args:
            constant (float): Shift value
            include_anchor (bool, optional): Include anchor image in the shift, if exists. Default: true.
            include_presequence (bool, optional): Include preseq images in the shift, if exists. Default: true.
            include_dapi (bool, optional): Offset DAPI image, if exists. Default: true.
        """
        self.instructions.append(utils.base.get_function_name())
        print(f"Shifting image by {constant}")

        np.add(self.image, constant, out=self.image)
        if include_anchor and self.include_anchor:
            np.add(self.anchor_image[:, self.anchor_channel], constant, out=self.anchor_image[:, self.anchor_channel])
        if include_presequence and self.include_presequence:
            np.add(self.presequence_image, constant, out=self.presequence_image)
        if include_dapi and self.include_dapi:
            np.add(self.anchor_image[:, self.dapi_channel], constant, out=self.anchor_image[:, self.dapi_channel])

    # Post-Processing function
    def scale_images_to_type(self, type: np.dtype) -> None:
        """
        Offset and multiply all used images by the same constant to fill the entire data range of given datatype.

        Args:
            type (np.dtype): datatype to scale images to.
        """
        print(f"Scaling images to type {np.dtype(type).name}...")

        type_min = np.iinfo(type).min
        type_max = np.iinfo(type).max
        image_min = self.image.min()
        image_max = self.image.max()
        if self.include_anchor or self.include_dapi:
            image_min = min(self.anchor_image.min(), image_min)
            image_max = max(self.anchor_image.max(), image_max)
        if self.include_presequence:
            image_min = min(self.presequence_image.min(), image_min)
            image_max = max(self.presequence_image.max(), image_max)

        multiplier = (type_max - type_min) / (image_max - image_min)
        offset = type_max - multiplier * image_max
        assert np.isclose(offset, type_min - multiplier * image_min), f"oopps"
        np.multiply(self.image, multiplier, out=self.image)
        np.add(self.image, offset, out=self.image)
        self.image = self.image.astype(type)
        if self.include_presequence:
            np.multiply(self.presequence_image, multiplier, out=self.presequence_image)
            np.add(self.presequence_image, offset, out=self.presequence_image)
            self.presequence_image = self.presequence_image.astype(type)
        if self.include_anchor or self.include_dapi:
            np.multiply(
                self.anchor_image,
                multiplier,
                out=self.anchor_image,
            )
            np.add(
                self.anchor_image,
                offset,
                out=self.anchor_image,
            )
            self.anchor_image = self.anchor_image.astype(type)
        self.image_dtype = type

    def save_raw_images(
        self,
        output_dir: str,
        overwrite: bool = True,
        omp_iterations: int = 2,
        omp_initial_intensity_thresh_percentile: int = 50,
        register_with_dapi: bool = True,
    ) -> None:
        """
        Save known spot positions and codes, raw .npy image files, metadata.json file, gene codebook and ``config.ini``
        file for coppafish pipeline run. Output directory must be empty. After saving, able to call function
        ``run_coppafish`` to run the coppafish pipeline.

        Args:
            output_dir (str): save directory.
            overwrite (bool, optional): overwrite any saved coppafish data inside the directory, delete old
                `notebook.npz` file if there is one and ignore any other files inside the directory. Default: true.
            omp_iterations (int, optional): number of OMP iterations on every pixel. Increasing this may improve gene
                scoring. Default: `2`.
            omp_initial_intensity_thresh_percentile (float, optional): percentile of the absolute intensity of all
                pixels in the mid z-plane of the central tile. Used as a threshold for pixels to decide what to apply
                OMP on. A higher number leads to stricter picking of pixels. Default: `90`.
            register_with_dapi (bool, optional): apply channel registration using the DAPI channel, if available.
                Default: true.
        """
        # Same dtype as ND2s
        self.scale_images_to_type(np.uint16)

        self.instructions.append(utils.base.get_function_name())
        print(f"Saving raw data")

        self.image_tiles, self.tile_origins_yx, self.tile_yxz_pos = self._unstich_image(
            self.image,
            np.asarray([*self.n_tile_yx, self.n_planes]),
            self.tile_overlap,
            self.n_tiles_y,
            self.n_tiles_x,
            update_global_spots=True,
        )
        del self.image
        # Sequence brightness scaling
        for r in range(self.n_rounds):
            for t in range(self.n_tiles):
                for c in range(self.n_channels + 1):
                    assert np.all(
                        self.brightness_scale_factor[t, r + 1, c] > 0
                    ), "A brightness scaling factor < 0 is not allowed"
                    assert np.all(
                        self.brightness_scale_factor[t, r + 1, c] <= 1
                    ), "A brightness scaling factor > 1 will cause an overflow"
                    scaled_image = np.multiply(
                        self.image_tiles[r, t, c],
                        self.brightness_scale_factor[t, r + 1, c],
                        dtype=np.float32,
                        casting="safe",
                    )
                    self.image_tiles[r, t, c] = np.rint(scaled_image).astype(self.image_dtype)
                    del scaled_image
        self.tile_xy_pos = self.tile_yxz_pos[:2]
        self.tilepos_yx_nd2 = self.tile_xy_pos

        if self.include_presequence:
            self.presequence_image_tiles = self._unstich_image(
                self.presequence_image[None],
                np.asarray([*self.n_tile_yx, self.n_planes]),
                self.tile_overlap,
                self.n_tiles_y,
                self.n_tiles_x,
            )[0]
            self.presequence_image_tiles = self.presequence_image_tiles[0]
            del self.presequence_image
            # Presequence brightness scaling
            for t in range(self.n_tiles):
                for c in range(self.n_channels + 1):
                    assert np.all(
                        self.brightness_scale_factor[t, 0, c] > 0
                    ), "A brightness scaling factor < 0 is not allowed"
                    assert np.all(
                        self.brightness_scale_factor[t, 0, c] <= 1
                    ), "A brightness scaling factor > 1 will cause an overflow"
                    scaled_image = np.multiply(
                        self.presequence_image_tiles[t, c],
                        self.brightness_scale_factor[t, 0, c],
                        dtype=np.float32,
                        casting="safe",
                    )
                    self.presequence_image_tiles[t, c] = np.rint(scaled_image).astype(self.image_dtype)
                    del scaled_image
        if self.include_anchor:
            self.anchor_image_tiles = self._unstich_image(
                self.anchor_image[None],
                np.asarray([*self.n_tile_yx, self.n_planes]),
                self.tile_overlap,
                self.n_tiles_y,
                self.n_tiles_x,
            )[0]
            self.anchor_image_tiles = self.anchor_image_tiles[0]
            del self.anchor_image
            # Anchor brightness scaling
            for t in range(self.n_tiles):
                for c in range(self.n_channels + 1):
                    assert np.all(
                        self.brightness_scale_factor[t, self.n_rounds + 1, c] > 0
                    ), "A brightness scaling factor < 0 is not allowed"
                    assert np.all(
                        self.brightness_scale_factor[t, self.n_rounds + 1, c] <= 1
                    ), "A brightness scaling factor > 1 will cause an overflow"
                    scaled_image = np.multiply(
                        self.anchor_image_tiles[t, c],
                        self.brightness_scale_factor[t, self.n_rounds + 1, c],
                        dtype=np.float32,
                        casting="safe",
                    )
                    self.anchor_image_tiles[t, c] = np.rint(scaled_image).astype(self.image_dtype)
                    del scaled_image

        if os.path.isdir(output_dir) and overwrite:
            shutil.rmtree(output_dir)
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        if not overwrite:
            assert len(os.listdir(output_dir)) == 0, f"Output directory at \n\t{output_dir}\n must be empty"

        # Create an output_dir/output_coppafish directory for coppafish pipeline output saved to disk
        self.output = output_dir
        self.coppafish_output = os.path.join(output_dir, "output_coppafish")
        if overwrite:
            if os.path.isdir(self.coppafish_output):
                shutil.rmtree(self.coppafish_output)
        if not os.path.isdir(self.coppafish_output):
            os.mkdir(self.coppafish_output)

        # Create an output_dir/output_coppafish/tiles directory for coppafish extract output
        self.coppafish_tiles = os.path.join(self.coppafish_output, "tiles")
        if not os.path.isdir(self.coppafish_tiles):
            os.mkdir(self.coppafish_tiles)
        # Remove any old tile files in the tile directory, if any, to make sure coppafish runs extract and filter \
        # again
        for filename in os.listdir(self.coppafish_tiles):
            filepath = os.path.join(self.coppafish_tiles, filename)
            if os.path.isfile(filepath):
                os.remove(filepath)

        # Save the known gene names and positions to a csv.
        df = pandas.DataFrame(
            {
                "gene": self.true_spot_identities,
                "z": self.true_spot_positions_pixels[:, 0],
                "y": self.true_spot_positions_pixels[:, 1],
                "x": self.true_spot_positions_pixels[:, 2],
            }
        )
        df.to_csv(os.path.join(output_dir, "gene_locations.csv"))

        metadata = {
            "n_tiles": self.n_tiles,
            "n_rounds": self.n_rounds,
            "n_channels": self.n_channels + 1,
            "tile_sz": self.n_tile_yx[0],
            "pixel_size_xy": 0.26,
            "pixel_size_z": 0.9,
            "tile_centre": [self.n_tile_yx[0] / 2, self.n_tile_yx[1] / 2, self.n_planes / 2],
            "tilepos_yx": self.tile_origins_yx,
            "tilepos_yx_nd2": list(reversed(self.tile_origins_yx)),
            "channel_camera": [1] * (self.n_channels + 1),
            "channel_laser": [1] * (self.n_channels + 1),
            "xy_pos": self.tile_xy_pos,
            "nz": self.n_planes,
        }
        self.metadata_filepath = os.path.join(output_dir, "metadata.json")
        with open(self.metadata_filepath, "w") as f:
            json.dump(metadata, f, indent=4)

        # Save the raw .npy tile files, one round at a time, in separate round directories. We do this because
        # coppafish expects every rounds (including anchor and presequence) in its own directory.
        # Dask saves each tile as a separate .npy file for coppafish to read properly.
        dask_chunks = (1, self.n_channels + 1, *self.n_tile_yx, self.n_planes)
        for r in range(self.n_rounds):
            save_path = os.path.join(output_dir, f"{r}")
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            # Clear the raw .npy directories before dask saving, so old multi-tile data is not left in the
            # directories
            for filename in os.listdir(save_path):
                filepath = os.path.join(save_path, filename)
                if os.path.isfile(filepath):
                    os.remove(filepath)
                else:
                    raise IsADirectoryError(f"Found unexpected directory in {save_path}")
            image_dask = dask.array.from_array(self.image_tiles[r], chunks=dask_chunks)
            dask.array.to_npy_stack(save_path, image_dask)
            del image_dask
        if self.include_anchor:
            self.anchor_directory_name = f"anchor"
            # Save the presequence image in `coppafish_output/anchor/`
            anchor_save_path = os.path.join(output_dir, self.anchor_directory_name)
            if not os.path.isdir(anchor_save_path):
                os.mkdir(anchor_save_path)
            for filename in os.listdir(anchor_save_path):
                filepath = os.path.join(anchor_save_path, filename)
                if os.path.isfile(filepath):
                    os.remove(filepath)
                else:
                    raise IsADirectoryError(f"Unexpected subdirectory in {anchor_save_path}")
            image_dask = dask.array.from_array(self.anchor_image_tiles.astype(self.image_dtype), chunks=dask_chunks)
            dask.array.to_npy_stack(anchor_save_path, image_dask)
            del image_dask
        if self.include_presequence:
            self.presequence_directory_name = f"presequence"
            # Save the presequence image in `coppafish_output/presequence/`
            presequence_save_path = os.path.join(output_dir, self.presequence_directory_name)
            if not os.path.isdir(presequence_save_path):
                os.mkdir(presequence_save_path)
            for filename in os.listdir(presequence_save_path):
                filepath = os.path.join(presequence_save_path, filename)
                if os.path.isfile(filepath):
                    os.remove(filepath)
                else:
                    raise IsADirectoryError(f"Unexpected subdirectory in {presequence_save_path}")
            image_dask = dask.array.from_array(
                self.presequence_image_tiles.astype(self.image_dtype),
                chunks=dask_chunks,
            )
            dask.array.to_npy_stack(presequence_save_path, image_dask)
            del image_dask

        # Save the gene codebook in `output_dir`
        self.codebook_filepath = os.path.join(output_dir, "codebook.txt")
        with open(self.codebook_filepath, "w") as f:
            for gene_name, code in self.codes.items():
                f.write(f"{gene_name} {code}\n")

        # Save the gene colours, used for the coppafish `Viewer`, in `output_dir`
        self.gene_colours_filepath = os.path.join(output_dir, "gene_colours.csv")
        rng = np.random.RandomState(self.seed)
        with open(self.gene_colours_filepath, "w") as f:
            csvwriter = csv.writer(f, delimiter=",")
            napari_symbols = ["cross", "disc", "square", "triangle_up", "hbar", "vbar"]
            mpl_symbols = ["+", ".", "s", "^", "_", "|"]
            # Heading
            csvwriter.writerow(["", "GeneNames", "ColorR", "ColorG", "ColorB", "napari_symbol", "mpl_symbol"])
            for i, gene_name in enumerate(self.codes):
                random_index = rng.randint(len(napari_symbols))
                csvwriter.writerow(
                    [
                        f"{i}",
                        f"{gene_name}",
                        round(rng.rand(), 2),
                        round(rng.rand(), 2),
                        round(rng.rand(), 2),
                        napari_symbols[random_index],
                        mpl_symbols[random_index],
                    ]
                )

        # Save the initial bleed matrix for the config file
        self.initial_bleed_matrix_filepath = os.path.join(output_dir, "bleed_matrix.npy")
        np.save(self.initial_bleed_matrix_filepath, self.bleed_matrix)

        # Add an extra channel and dye for the DAPI
        self.dye_names = map("".join, zip(["dye_"] * (self.n_channels), list(np.arange(self.n_channels).astype(str))))
        self.dye_names = list(self.dye_names)

        is_3d = self.n_planes > 1
        # Box sizes must be even numbers
        max_box_size_z, max_box_size_yx = 12, 300
        box_size_z = min([max_box_size_z, self.n_planes if self.n_planes % 2 == 0 else self.n_planes - 1])
        box_size_yx = min([max_box_size_yx, self.n_tile_yx[0] if self.n_tile_yx[0] % 2 == 0 else self.n_tile_yx[0] - 1])
        if self.n_planes > 18:
            psf_shape = ""
        else:
            yx_shape = min([*self.n_tile_yx, 181])
            z_shape = self.n_planes if self.n_planes % 2 != 0 else self.n_planes - 1
            if yx_shape % 2 == 0:
                yx_shape -= 1
            psf_shape = f"{yx_shape}, {yx_shape}, {z_shape}"

        # Save the config file. z_subvols is moved from the default of 5 based on n_planes.
        config_file_contents = f"""; This config file is auto-generated by RoboMinnie. 
        [file_names]
        notebook_name = notebook.npz
        input_dir = {output_dir}
        output_dir = {self.coppafish_output}
        tile_dir = {self.coppafish_tiles}
        initial_bleed_matrix = {self.initial_bleed_matrix_filepath}
        round = {', '.join([str(i) for i in range(self.n_rounds)])}
        anchor = {self.anchor_directory_name if self.include_anchor else ''}
        pre_seq = {self.presequence_directory_name if self.include_presequence else ''}
        code_book = {self.codebook_filepath}
        raw_extension = .npy
        raw_metadata = {self.metadata_filepath}

        [basic_info]
        is_3d = {is_3d}
        dye_names = {', '.join(self.dye_names)}
        use_rounds = {', '.join([str(i) for i in range(self.n_rounds)])}
        use_z = {', '.join([str(i) for i in range(self.n_planes)])}
        use_tiles = {', '.join(str(i) for i in range(self.n_tiles))}
        anchor_round = {self.n_rounds if self.include_anchor else ''}
        use_channels = {', '.join([str(i) for i in np.arange((self.dapi_channel + 1), (self.n_channels + 1))])}
        anchor_channel = {self.anchor_channel if self.include_anchor else ''}
        dapi_channel = {self.dapi_channel if self.include_dapi else ''}
        
        [filter]
        deconvolve = {True}
        auto_thresh_multiplier = 2
        num_rotations = 0
        psf_isolation_dist = 4
        psf_min_spots = 100
        psf_shape = {psf_shape}

        [find_spots]
        n_spots_warn_fraction = 0
        n_spots_error_fraction = 1

        [stitch]
        expected_overlap = {self.tile_overlap if self.n_tiles > 1 else 0}

        [register]
        subvols = {self.n_planes}, {8}, {8}
        box_size = {box_size_z}, {box_size_yx}, {box_size_yx}
        pearson_r_thresh = 0.25
        round_registration_channel = {self.dapi_channel if (self.include_dapi and register_with_dapi) else ''}
        icp_min_spots = 10

        [omp]
        max_genes = {omp_iterations}
        shape_isolation_distance_yx = 8
        pixel_max_percentile = 5
        """
        # Remove any large spaces in the config contents
        config_file_contents = config_file_contents.replace("  ", "")

        self.config_filepath = os.path.join(output_dir, "robominnie.ini")
        with open(self.config_filepath, "w") as f:
            f.write(config_file_contents)

        # Save the instructions run so far (every function call)
        self.instruction_filepath = os.path.join(output_dir, "instructions.txt")
        with open(self.instruction_filepath, "w") as f:
            for instruction in self.instructions:
                f.write(instruction + "\n")

    def run_coppafish(
        self,
        time_pipeline: bool = True,
        save_ref_spots_data: bool = True,
    ):
        """
        Run RoboMinnie instance on the entire coppafish pipeline.

        Args:
            time_pipeline (bool, optional): print the time taken to run the coppafish pipeline and OMP. Default: true.
            include_stitch (bool, optional): run up to at least stitch. Default: true.
            include_omp (bool, optional): run up to and including coppafish OMP stage. Default: true.
            profile_omp (bool, optional): profile coppafish OMP stage using a default Python profiler called
                `cprofile`. Dumps the results into a text file in output_coppafish. Default: false.
            save_ref_spots_data (bool, optional): if true, will save ref_spots data, which is used for comparing
                ref_spots results to the true robominnie spots. Default: false to reduce RoboMinnie's memory usage.
                Default: true.

        Returns:
            Notebook: final notebook.
        """
        self.instructions.append(utils.base.get_function_name())
        print(f"Running coppafish")

        config_filepath = self.config_filepath
        n_planes = self.n_planes
        n_tiles = self.n_tiles

        # Run the non-parallel pipeline code
        if time_pipeline:
            start_time = time.time()
        nb = run.run_pipeline(config_filepath)
        # Keep the stitch information to convert local tiles coordinates into global coordinates when comparing
        # to true spots
        self.stitch_tile_origins = nb.stitch.tile_origin

        assert nb.has_page("stitch"), f"Stitch not found in notebook at {config_filepath}"
        run.run_reference_spots(nb, overwrite_ref_spots=False)

        # Keep reference spot information to compare to true spots, if wanted
        assert nb.has_page("ref_spots"), f"Reference spots not found in notebook at {config_filepath}"

        if save_ref_spots_data:
            self.ref_spots_scores = nb.ref_spots.score
            self.ref_spots_local_positions_yxz = nb.ref_spots.local_yxz
            self.ref_spots_intensities = nb.ref_spots.intensity
            self.ref_spots_gene_indices = nb.ref_spots.gene_no
            self.ref_spots_tile = nb.ref_spots.tile

        if time_pipeline:
            end_time = time.time()
            print(
                f"Coppafish pipeline run: {round((end_time - start_time)/60, 1)}mins\n"
                + f"{round((end_time - start_time)//(n_planes * n_tiles), 1)}s per z plane per tile."
            )

        assert nb.has_page("omp"), f"OMP not found in notebook at {config_filepath}"
        # Keep the OMP spot intensities, assigned gene, assigned tile number and the spot positions in the class
        # instance
        self.omp_spot_intensities = np.ones_like(nb.omp.gene_no)
        self.omp_spot_scores = omp_scores.omp_scores_int_to_float(nb.omp.scores)
        self.omp_gene_numbers = nb.omp.gene_no
        self.omp_tile_number = nb.omp.tile
        self.omp_spot_local_positions = nb.omp.local_yxz  # yxz position of each gene found
        assert (
            self.omp_gene_numbers.shape[0] == self.omp_spot_local_positions.shape[0]
        ), "Mismatch in spot count in omp.gene_numbers and omp.local_positions"
        self.omp_spot_count = self.omp_gene_numbers.shape[0]

        if self.omp_spot_count == 0:
            warnings.warn("Copppafish OMP found zero spots")

        return nb

    def compare_spots(
        self,
        spot_types: str = "ref",
        score_threshold: float = 0,
        intensity_threshold: float = 0,
        location_threshold: float = 2,
    ) -> Tuple[int, int, int, int]:
        """
        Compare spot positions and gene codes from coppafish results to the known spot locations. If the spots are
        close enough and the true spot has not been already assigned to a reference spot, then they are considered the
        same spot in both coppafish output and synthetic data. If two or more spots are close enough to a true spot,
        then the closest one is chosen. If equidistant, then take the spot with the correct gene code. If not
        applicable, then just take the spot with the lowest index (effectively choose one of them at random, but
        consistent way). Will save the results in ``self`` (overwrites earlier comparison calls), so can call the
        ``overall_score`` function afterwards without inputting parameters.

        Args:
            spot_types (str, optional): Coppafish spot type to compare to. Either `'ref'` or `'omp'`.
            score_threshold (float, optional): Spot score threshold, any spots below this intensity are ignored. Only
                relevant to reference spots. Default: `0`.
            intensity_threshold (float, optional): Reference spot intensity threshold. Default: `0`.
            location_threshold (float, optional): Maximum distance, in pixels, two spots can be apart to be considered
                the same spot. Default: `4`.

        Returns:
            `tuple` (true_positives: `list` of `int`, wrong_positives: `list` of `int`,
                false_positives: `list` of `int`, false_negatives: `list` of `int`): The number of spots assigned to
                true positive, wrong positive, false positive and false negative as a list for each tile index, where a
                wrong positive is a spot assigned to the wrong gene, but found in the location of a true spot.
        """
        assert (
            "run_coppafish" in self.instructions
        ), "`run_coppafish` must be called before comparing reference spots to ground truth spots"

        self.instructions.append(utils.base.get_function_name())
        print(f"Comparing reference spots to known spots")

        assert (
            score_threshold >= 0 and score_threshold <= 1
        ), f"Intensity threshold must be (0,1), got {score_threshold}"
        assert location_threshold >= 0, f"Location threshold must be >= 0, got {location_threshold}"

        location_threshold_squared = location_threshold**2

        print(f"Stitch tile origins: {self.stitch_tile_origins}")
        print(f"True tile origins:   {self.tile_yxz_pos}")

        if spot_types.lower() == "ref":
            coppafish_spot_positions_yxz = self.ref_spots_local_positions_yxz.astype(np.float64)
            coppafish_spots_gene_indices = self.ref_spots_gene_indices
            coppafish_spots_intensities = self.ref_spots_intensities
            coppafish_spots_scores = self.ref_spots_scores
            coppafish_spots_tiles = self.ref_spots_tile
        elif spot_types.lower() == "omp":
            coppafish_spot_positions_yxz = self.omp_spot_local_positions.astype(np.float64)
            coppafish_spots_gene_indices = self.omp_gene_numbers
            coppafish_spots_intensities = self.omp_spot_intensities
            coppafish_spots_scores = self.omp_spot_scores
            coppafish_spots_tiles = self.omp_tile_number

        # Convert local spot positions into coordinates using coppafish-calculated global tile positions
        np.add(
            coppafish_spot_positions_yxz,
            self.stitch_tile_origins[coppafish_spots_tiles],
            out=coppafish_spot_positions_yxz,
        )
        # Now convert coppafish's global coords into robominnie's idea of global coordinates by correcting for the
        # padding
        np.add(
            coppafish_spot_positions_yxz,
            [[IMAGE_PADDING[0] // 2, IMAGE_PADDING[1] // 2, IMAGE_PADDING[2] // 2]],
            out=coppafish_spot_positions_yxz,
        )

        true_positives, wrong_positives, false_positives, false_negatives = [], [], [], []

        # NOTE: self.true_spot_positions_pixels and ref_spots_positions_yxz has the form yxz
        for t in range(self.n_tiles):
            # Eliminate any reference spots below the thresholds
            indices = np.logical_and(
                coppafish_spots_intensities >= intensity_threshold, coppafish_spots_scores > score_threshold
            )
            indices = np.logical_and(indices, coppafish_spots_tiles == t)
            coppafish_spots_gene_indices_t = coppafish_spots_gene_indices[indices]
            coppafish_spot_positions_yxz_t = coppafish_spot_positions_yxz[indices]

            TPs, WPs, FPs, FNs = utils.errors.compare_spots(
                coppafish_spot_positions_yxz_t,
                coppafish_spots_gene_indices_t,
                self.true_spot_positions_pixels[self.true_spot_tile_numbers == t],
                self.true_spot_identities[self.true_spot_tile_numbers == t],
                location_threshold_squared,
                self.codes,
                f"Checking {spot_types.lower()} spots, t={t}",
            )
            true_positives.append(TPs)
            wrong_positives.append(WPs)
            false_positives.append(FPs)
            false_negatives.append(FNs)
        # Save results in `self` (overwrites)
        self.true_positives = true_positives
        self.wrong_positives = wrong_positives
        self.false_positives = false_positives
        self.false_negatives = false_negatives
        return (true_positives, wrong_positives, false_positives, false_negatives)

    def overall_score(
        self,
        true_positives: int = None,
        wrong_positives: int = None,
        false_positives: int = None,
        false_negatives: int = None,
    ) -> float:
        """
        Overall score from a spot-to-spot comparison, such as `Compare_OMP_Spots`.

        Args:
            true_positives  (`list` of `int`, optional): True positives spot count.  Default: value stored in `self`.
            wrong_positives (`list` of `int`, optional): Wrong positives spot count. Default: value stored in `self`.
            false_positives (`list` of `int`, optional): False positives spot count. Default: value stored in `self`.
            false_negatives (`list` of `int`, optional): False negatives spot count. Default: value stored in `self`.

        Returns:
            float: Overall score.
        """
        if true_positives == None:
            true_positives = self.true_positives
        if wrong_positives == None:
            wrong_positives = self.wrong_positives
        if false_positives == None:
            false_positives = self.false_positives
        if false_negatives == None:
            false_negatives = self.false_negatives
        TPs = 0
        WPs = 0
        FPs = 0
        FNs = 0
        for t in range(self.n_tiles):
            TPs += true_positives[t]
            WPs += wrong_positives[t]
            FPs += false_positives[t]
            FNs += false_negatives[t]
        return TPs / (TPs + WPs + FPs + FNs)

    # Debugging Function:
    def view_images(self, tiles: List[int] = None):
        """
        View all images in `napari` for tile index `t`, including a presequence, anchor and DAPI images, if they exist.

        Args:
            tiles (`list` of `int`, optional): tile indices. If an empty list, will display all tiles, including the
                additional padding pixels. Default: shows all tiles without the padding.
        """
        if tiles is None:
            tiles = list(np.arange(self.n_tiles))

        print(f"Viewing images")
        viewer = napari.Viewer(title=f"RoboMinnie")
        if tiles is not None and len(tiles) == 0:
            for c in range(self.n_channels):
                for r in range(self.n_rounds):
                    # z index must be the first axis for napari to view
                    viewer.add_image(
                        self.image[r, c].transpose([2, 0, 1]),
                        name=f"seq, r={r}, c={c}",
                        visible=False,
                    )
                if self.include_presequence:
                    viewer.add_image(
                        self.presequence_image[c].transpose([2, 0, 1]),
                        name=f"preseq, c={c}",
                        visible=False,
                    )
            if self.include_anchor:
                viewer.add_image(
                    self.anchor_image[self.anchor_channel].transpose([2, 0, 1]),
                    name=f"anchor, c={self.anchor_channel}",
                    visible=False,
                )
            if self.include_dapi:
                viewer.add_image(
                    self.anchor_image[self.dapi_channel].transpose([2, 0, 1]),
                    name=f"dapi, r=anchor",
                    visible=False,
                )
                viewer.add_image(
                    self.presequence_image[self.dapi_channel].transpose([2, 0, 1]),
                    name=f"dapi, r=preseq",
                    visible=False,
                )
                for r in range(self.n_rounds):
                    viewer.add_image(
                        self.image[r, self.dapi_channel].transpose([2, 0, 1]),
                        name=f"dapi, r={r}",
                        visible=False,
                    )

        # TODO: Complete this to add all tile images, including the presequence and sequence images
        if self.include_anchor:
            anchor_image_tiles = self._unstich_image(
                self.anchor_image[None],
                np.asarray([*self.n_tile_yx, self.n_planes]),
                self.tile_overlap,
                self.n_tiles_y,
                self.n_tiles_x,
            )[0][0]
            for t in tiles:
                viewer.add_image(
                    anchor_image_tiles[t, self.anchor_channel].transpose([2, 0, 1]),
                    name=f"anchor, t={t}, c={self.anchor_channel}",
                    visible=False,
                )
        napari.run()

    def _unstich_image(
        self,
        image: npt.NDArray[np.float_],
        tile_size_yxz: npt.NDArray[np.int_],
        tile_overlap: float,
        n_tiles_y: int,
        n_tiles_x: int,
        update_global_spots: bool = False,
    ) -> Tuple[npt.NDArray[np.float_], List[List[int]], List[List[float]]]:
        """
        Separate image into multiple tiles with a tile overlap.

        Args:
            image ((`n_rounds x n_channels x image_y x image_x x image_z`) `ndarray[float]`): giant image to separate
                into tiles.
            tile_size_yxz (`3` `ndarray[int]`): tile sizes.
            tile_overlap (float): overlap of tiles in the x and y directions, given in fraction of tile length.
            n_tiles_y (int): Number of tiles in the y direction.
            n_tiles_x (int): Number of tiles in the x direction.
            update_global_spots (bool, optional): Update ``self.true_spot_identities``, `self.true_spot_tile_numbers`
                and ``self.true_spot_positions_pixels`` to account for overlapping spots (by saving overlapping spots
                twice) and removing true spot positions that are not visible to coppafish.

        Returns:
            - (`n_rounds x n_tiles x n_channels x tile_size_yxz[0] x tile_size_yxz[1] x tile_size_yxz[2]`)
                `ndarray[float]`: tile images, copy of `image`.
            - (`list` of `list` of `int`): list of tile indices of form `[y_index, x_index]`.
            - (`list` of `list` of `float`): list of tile bottom-right corner positions, starting from `[0,0,0]`,
                in the form `[y,x,z]`.
        """
        # TODO: Add support for affine-transformed tile images
        new_true_spot_positions_pixels = np.zeros((0, 3), dtype=float)  # `n_true_spots x 3`
        new_true_spot_identities = np.empty((0), dtype=str)  # `n_true_spots`
        new_true_spot_tile_numbers = np.zeros(0)  # Tile indices for each spot

        tile_images = np.zeros((image.shape[0], 0, image.shape[1], *tile_size_yxz), self.image_dtype)
        tile_indices = []
        tile_positions_yxz = []
        t = 0
        for x in range(n_tiles_x):
            for y in range(n_tiles_y):
                index_start_yxz = np.asarray([IMAGE_PADDING[0] // 2, IMAGE_PADDING[1] // 2, IMAGE_PADDING[2] // 2], int)
                np.add(
                    index_start_yxz,
                    [
                        tile_size_yxz[0] * y - round(tile_size_yxz[0] * tile_overlap * y),
                        tile_size_yxz[1] * x - round(tile_size_yxz[1] * tile_overlap * x),
                        0,
                    ],
                    out=index_start_yxz,
                )
                index_end_yxz = index_start_yxz + tile_size_yxz
                new_tile = image[
                    :,
                    None,
                    :,
                    index_start_yxz[0] : index_end_yxz[0],
                    index_start_yxz[1] : index_end_yxz[1],
                    index_start_yxz[2] : index_end_yxz[2],
                ]
                tile_images = np.append(tile_images, new_tile, axis=1)
                tile_indices.append([y, x])
                tile_positions_yxz.append([y * self.n_tile_yx[0], x * self.n_tile_yx[1], 0.0])
                indices = []
                for s in range(self.true_spot_positions_pixels.shape[0]):
                    indices.append(
                        self.true_spot_positions_pixels[s, 0] >= index_start_yxz[0]
                        and self.true_spot_positions_pixels[s, 1] >= index_start_yxz[1]
                        and self.true_spot_positions_pixels[s, 2] >= index_start_yxz[2]
                        and self.true_spot_positions_pixels[s, 0] < index_end_yxz[0]
                        and self.true_spot_positions_pixels[s, 1] < index_end_yxz[1]
                        and self.true_spot_positions_pixels[s, 2] < index_end_yxz[2]
                    )
                new_true_spot_positions_pixels = np.append(
                    new_true_spot_positions_pixels,
                    self.true_spot_positions_pixels[indices],
                    axis=0,
                )
                new_true_spot_identities = np.append(
                    new_true_spot_identities,
                    self.true_spot_identities[indices],
                    axis=0,
                )
                new_true_spot_tile_numbers = np.append(new_true_spot_tile_numbers, [t] * np.sum(indices), axis=0)
                t += 1
        if update_global_spots:
            self.true_spot_positions_pixels = new_true_spot_positions_pixels
            self.true_spot_identities = new_true_spot_identities
            self.true_spot_tile_numbers = new_true_spot_tile_numbers
        return tile_images, tile_indices, tile_positions_yxz

    def save(self, output_dir: str, filename: str = None, overwrite: bool = True, compress: bool = False) -> None:
        """
        Save `RoboMinnie` instance using the amazing tool pickle inside output_dir directory.

        Args:
            output_dir (str): output directory.
            filename (str, optional): Name of the pickled `RoboMinnie` object. Default: 'robominnie.pkl'
            overwrite (bool, optional): Overwrite any robominnie saved instance. Default: true.
            compress (bool, optional): Compress pickle binary file using bzip2 compression in the default python
                package `bz2`. Default: False.
        """
        self.instructions.append(utils.base.get_function_name())
        print("Saving RoboMinnie instance")
        instance_output_dir = output_dir
        if filename == None:
            instance_filename = DEFAULT_INSTANCE_FILENAME
        else:
            instance_filename = filename
        if not os.path.isdir(instance_output_dir):
            os.mkdir(instance_output_dir)

        instance_filepath = os.path.join(instance_output_dir, instance_filename)
        if not overwrite:
            assert not os.path.isfile(instance_filepath), f"RoboMinnie instance already saved as {instance_filepath}"

        if not compress:
            with open(instance_filepath, "wb") as f:
                pickle.dump(self, f)
        else:
            with bz2.open(instance_filepath, "wb", compresslevel=9) as f:
                pickle.dump(self, f)

    def load(self, input_dir: str, filename: str = None, overwrite_self: bool = True, compressed: bool = False):
        """
        Load `RoboMinnie` instance using the handy pickled information saved inside input_dir.

        Args:
            input_dir (str): The directory where the RoboMinnie data is stored.
            filename (str, optional): Name of the pickle RoboMinnie object. Default: 'robominnie.pkl'.
            overwrite_self (bool, optional): If true, become the RoboMinnie instance loaded from disk.
            compressed (bool, optional): If True, try decompress pickle binary file assuming a bzip2 compression.
                Default: False.

        Returns:
            Loaded `RoboMinnie` class.
        """
        self.instructions.append(utils.base.get_function_name())
        print("Loading RoboMinnie instance")
        instance_input_dir = input_dir
        if filename == None:
            instance_filename = DEFAULT_INSTANCE_FILENAME
        else:
            instance_filename = filename
        instance_filepath = os.path.join(instance_input_dir, instance_filename)

        assert os.path.isfile(instance_filepath), f"RoboMinnie instance not found at {instance_filepath}"

        if not compressed:
            with open(instance_filepath, "rb") as f:
                instance: RoboMinnie = pickle.load(f)
        else:
            with bz2.open(instance_filepath, "rb") as f:
                instance: RoboMinnie = pickle.load(f)

        if overwrite_self:
            self = instance
        return instance
