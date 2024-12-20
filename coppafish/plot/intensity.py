import napari
import numpy as np

from ..setup.notebook import Notebook
from ..spot_colours import base


def view_intensity_images(
    nb: Notebook,
    tiles: list[int] | None = None,
    z_planes: int | None = None,
) -> None:
    """
    View the computed intensity for the given tile(s) and the anchor image(s). The intensity is defined as
    min_r(max_c(abs(colours))) where colours have the call spots colour norm factor applied.

    Args:
        nb (Notebook): notebook.
        tiles (list of int, optional): tiles to view. Default: first tile.
        z_planes (int, optional): the number of z planes to view, starting from 0. Default: 20.
    """
    assert nb.has_page("register"), "Register must be complete"
    assert nb.has_page("call_spots"), "Call spots must be complete"

    if tiles is None:
        tiles = nb.basic_info.use_tiles[:1]
    if z_planes is None:
        z_planes = 20
    z_planes = min(z_planes, max(nb.basic_info.use_z))

    tile_shape = (nb.basic_info.tile_sz, nb.basic_info.tile_sz, z_planes)
    yxz = [np.linspace(0, tile_shape[i] - 1, tile_shape[i]) for i in range(3)]
    yxz = np.array(np.meshgrid(*yxz, indexing="ij")).astype(np.int16).reshape((3, -1), order="F").T

    factor = nb.call_spots.colour_norm_factor.astype(np.float32)

    images: list[np.np.ndarray] = []
    names: list[str] = []
    for t in tiles:
        intensities = base.get_spot_colours_new_safe(
            nb.basic_info,
            yxz=yxz,
            image=nb.filter.images,
            flow=nb.register.flow,
            affine=nb.register.icp_correction,
            tile=t,
            use_rounds=nb.basic_info.use_rounds,
            use_channels=nb.basic_info.use_channels,
            out_of_bounds_value=0,
        )
        intensities *= factor[[t]]
        intensities = np.abs(intensities).max(2).min(1)
        intensities = intensities.reshape((nb.basic_info.tile_sz, nb.basic_info.tile_sz, z_planes), order="F")

        images.append(intensities)
        names.append(f"Intensities tile={t}")

    # Add the anchor images as well.
    for t in tiles:
        anchor_image = nb.filter.images[t, nb.basic_info.anchor_round, nb.basic_info.anchor_channel, :, :, :z_planes]
        images.append(anchor_image)
        names.append(f"Anchor tile={t}")

    # y, x, z -> z, y, x for napari.
    for i, image in enumerate(images):
        images[i] = image.swapaxes(1, 2).swapaxes(0, 1)

    viewer = napari.Viewer(title="Coppafish intensities")
    for image, name in zip(images, names):
        viewer.add_image(image, name=name, rgb=False)

    napari.run()
