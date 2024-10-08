import torch
import numpy as np
from typing import List, Union, Optional, Tuple

from .. import utils, find_spots, log
from ..filter import base as filter_base
from ..setup import NotebookPage


def psf_pad(psf: np.ndarray, image_shape: Union[np.ndarray, List[int]]) -> np.ndarray:
    """
    Pads psf with zeros so has same dimensions as image

    Args:
        psf: `float [y_shape x x_shape (x z_shape)]`.
            Point Spread Function with same shape as small image about each spot.
        image_shape: `int [psf.ndim]`.
            Number of pixels in `[y, x, (z)]` direction of padded image.

    Returns:
        `float [image_shape[0] x image_shape[1] (x image_shape[2])]`.
        Array same size as image with psf centered on middle pixel.
    """
    # must pad with ceil first so that ifftshift puts central pixel to (0,0,0).
    pre_pad = np.ceil((np.array(image_shape) - np.array(psf.shape)) / 2).astype(int)
    post_pad = np.floor((np.array(image_shape) - np.array(psf.shape)) / 2).astype(int)
    return np.pad(psf, [(pre_pad[i], post_pad[i]) for i in range(len(pre_pad))])


def get_psf_spots(
    nbp_file: NotebookPage,
    nbp_basic: NotebookPage,
    nbp_extract: NotebookPage,
    round: int,
    use_tiles: List[int],
    channel: int,
    use_z: List[int],
    radius_xy: int,
    radius_z: int,
    min_spots: int,
    intensity_thresh: Optional[float],
    intensity_auto_param: float,
    isolation_dist: float,
    shape: List[int],
    max_spots: Optional[int] = None,
) -> Tuple[np.ndarray, float, List[int]]:
    """
    Finds spot_shapes about spots found in raw data, average of these then used for psf.

    Args:
        nbp_file: `file_names` notebook page.
        nbp_basic: `basic_info` notebook page.
        nbp_extract: `extract` notebook page.
        round: Reference round to get spots from to determine psf.
            This should be the anchor round (last round) if using.
        use_tiles: `int [n_use_tiles]`.
            tiff tile indices used in experiment.
        channel: Reference channel to get spots from to determine psf.
        use_z: `int [n_z]`. Z-planes used in the experiment.
        radius_xy: Radius of dilation structuring element in xy plane (approximately spot radius).
        radius_z: Radius of dilation structuring element in z direction (approximately spot radius)
        min_spots: Minimum number of spots required to determine average shape from. Typical: 300
        intensity_thresh: Spots are local maxima in image with `pixel value > intensity_thresh`.
            if `intensity_thresh = None`, will automatically compute it from mid z-plane of first tile.
        intensity_auto_param: If `intensity_thresh = None` so is automatically computed, it is done using this.
        isolation_dist: Spots are isolated if nearest neighbour is further away than this.
        shape: `int [y_diameter, x_diameter, z_diameter]`. Desired size of image about each spot.
        max_spots (int, optional): maximum number of psf spots to use. Default: no limit.

    Returns:
        - `spot_images` - `int [n_spots x y_diameter x x_diameter x z_diameter]`.
            `spot_images[s]` is the small image surrounding spot `s`.
        - `intensity_thresh` - `float`. Only different from input if input was `None`.
        - `tiles_used` - `int [n_tiles_used]`. Tiles the spots were found on.
    """
    n_spots = 0
    spot_images = np.zeros((0, shape[0], shape[1], shape[2]), dtype=np.float32)
    tiles_used = []
    while n_spots < min_spots:
        if nbp_file.raw_extension == "jobs":
            t = filter_base.central_tile(nbp_basic.tilepos_yx_nd2, use_tiles)
            # choose tile closest to centre
            im = utils.tiles_io._load_image(nbp_file.tile_unfiltered[t][round][channel], nbp_extract.file_type)
        else:
            t = filter_base.central_tile(nbp_basic.tilepos_yx, use_tiles)  # choose tile closet to centre
            im = utils.tiles_io._load_image(nbp_file.tile_unfiltered[t][round][channel], nbp_extract.file_type)
        # zyx -> yxz
        im = im.transpose((1, 2, 0))
        mid_z = np.ceil(im.shape[2] / 2).astype(int)
        median_im = np.median(im[:, :, mid_z])
        if intensity_thresh is None:
            intensity_thresh = median_im + np.median(np.abs(im[:, :, mid_z] - median_im)) * intensity_auto_param
        elif intensity_thresh <= median_im or intensity_thresh >= utils.tiles_io.get_pixel_max():
            log.error(
                utils.errors.OutOfBoundsError(
                    "intensity_thresh", intensity_thresh, median_im, utils.tiles_io.get_pixel_max()
                )
            )
        spot_yxz, _ = find_spots.detect_spots(
            torch.asarray(im.astype(np.float32)), intensity_thresh, radius_xy, radius_z, True
        )
        spot_yxz = spot_yxz.numpy()
        # check fall off in intensity not too large
        not_single_pixel = find_spots.check_neighbour_intensity(im, spot_yxz, median_im)
        isolated = find_spots.get_isolated_points(
            spot_yxz * [1, 1, nbp_basic.pixel_size_z / nbp_basic.pixel_size_xy],
            isolation_dist,
        )
        chosen_spots = np.logical_and(isolated, not_single_pixel)
        if max_spots is not None and np.sum(chosen_spots) > max_spots:
            n_isolated_spots = 0
            for i in range(chosen_spots.shape[0]):
                if n_isolated_spots == max_spots:
                    chosen_spots[i:] = False
                    n_spots = max_spots
                    break
                if chosen_spots[i]:
                    n_isolated_spots += 1
        spot_yxz = spot_yxz[chosen_spots, :]
        if n_spots == 0 and np.shape(spot_yxz)[0] < min_spots / 4:
            # raise error on first tile if looks like we are going to use more than 4 tiles
            log.error(
                ValueError(
                    f"\nFirst tile, {t}, only found {np.shape(spot_yxz)[0]} spots."
                    f"\nMaybe consider lowering intensity_thresh from current value of {intensity_thresh}."
                )
            )
        if spot_images.size > 0:
            spot_images = np.append(spot_images, utils.spot_images.get_spot_images(im, spot_yxz, shape), axis=0)
        else:
            spot_images = utils.spot_images.get_spot_images(im, spot_yxz, shape)
        n_spots = np.shape(spot_images)[0]
        use_tiles = np.setdiff1d(use_tiles, t)
        tiles_used.append(t)
        if len(use_tiles) == 0 and n_spots < min_spots:
            log.error(
                ValueError(
                    f"\nRequired min_spots = {min_spots}, but only found {n_spots}.\n"
                    f"Maybe consider lowering intensity_thresh from current value of {intensity_thresh}."
                )
            )
    return spot_images, float(intensity_thresh), tiles_used


def get_psf(spot_images: np.ndarray, annulus_width: float) -> np.ndarray:
    """
    This gets psf, which is average image of spot from individual images of spots.
    It is normalised so min value is 0 and max value is 1.

    Args:
        spot_images: `int [n_spots x y_diameter x x_diameter x z_diameter]`.
            `spot_images[s]` is the small image surrounding spot `s`.
        annulus_width: Within each z-plane, this specifies how big an annulus to use,
            within which we expect all pixel values to be the same.

    Returns:
        `float [y_diameter x x_diameter x z_diameter]`.
            Average small image about a spot. Normalised so min is 0 and max is 1.
    """
    # normalise each z plane of each spot image first so each has median of 0 and max of 1.
    # Found that this works well as taper psf anyway, which gives reduced intensity as move away from centre.
    spot_images = spot_images - np.expand_dims(np.nanmedian(spot_images, axis=[1, 2]), [1, 2])
    spot_images = spot_images / np.expand_dims(np.nanmax(spot_images, axis=(1, 2)), [1, 2])
    psf = utils.spot_images.get_average_spot_image(spot_images, "median", "annulus_2d", annulus_width)
    # normalise psf so min is 0 and max is 1.
    psf = psf - psf.min()
    psf = psf / psf.max()
    return psf


def get_wiener_filter(psf: np.ndarray, image_shape: Union[np.ndarray, List[int]], constant: float) -> np.ndarray:
    """
    This tapers the psf so goes to 0 at edges and then computes wiener filter from it.

    Args:
        psf: `float [y_diameter x x_diameter x z_diameter]`.
            Average small image about a spot. Normalised so min is 0 and max is 1.
        image_shape: `int [n_im_y, n_im_x, n_im_z]`.
            Indicates the shape of the image to be convolved after padding.
        constant: Constant used in wiener filter.

    Returns:
        `complex128 [n_im_y x n_im_x x n_im_z]`. Wiener filter of same size as image.
    """
    # taper psf so smoothly goes to 0 at each edge.
    psf = (
        psf
        * np.hanning(psf.shape[0]).reshape(-1, 1, 1)
        * np.hanning(psf.shape[1]).reshape(1, -1, 1)
        * np.hanning(psf.shape[2]).reshape(1, 1, -1)
    )
    psf = psf_pad(psf, image_shape)
    psf_ft = np.fft.fftn(np.fft.ifftshift(psf))
    return np.conj(psf_ft) / np.real((psf_ft * np.conj(psf_ft) + constant))


def wiener_deconvolve(image: np.ndarray, im_pad_shape: List[int], filter: np.ndarray) -> np.ndarray:
    """
    This pads `image` so goes to median value of `image` at each edge. Then deconvolves using the given Wiener filter.

    Args:
        image: `int [n_im_y x n_im_x x n_im_z]`.
            Image to be deconvolved.
        im_pad_shape: `int [n_pad_y, n_pad_x, n_pad_z]`.
            How much to pad image in `[y, x, z]` directions.
        filter: `complex128 [n_im_y+2*n_pad_y, n_im_x+2*n_pad_x, n_im_z+2*n_pad_z]`.
            Wiener filter to use.

    Returns:
        `(n_im_y x n_im_x x n_im_z) ndarray[float]`: deconvolved image.
    """
    im_av = np.median(image[:, :, 0])
    image = np.pad(
        image,
        [(im_pad_shape[i], im_pad_shape[i]) for i in range(len(im_pad_shape))],
        "linear_ramp",
        end_values=[(im_av, im_av)] * 3,
    )
    im_deconvolved = np.real(np.fft.ifftn(np.fft.fftn(image) * filter))
    im_deconvolved = im_deconvolved[
        im_pad_shape[0] : -im_pad_shape[0], im_pad_shape[1] : -im_pad_shape[1], im_pad_shape[2] : -im_pad_shape[2]
    ]
    return im_deconvolved
