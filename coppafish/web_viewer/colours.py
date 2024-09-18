import numpy as np


def interpolate_rgb(color1: np.ndarray, color2: np.ndarray, t: np.ndarray[float]) -> np.ndarray[np.float32]:
    """
    Interpolates between two RGB colours.

    Args:
        - color1 (`(3) ndarray`): the start colour.
        - color2 (`(3) ndarray`): the final colour.
        - t (`(...) ndarray[float]`): lerp for each pixel to interpolate, if > 1, set to final colour. If < 0, set to
            start colour.

    Returns:
        `(... x 3) ndarray[float32]` final_colours: interpolated colours.
    """
    assert type(color1) is np.ndarray
    assert type(color2) is np.ndarray
    assert type(t) is np.ndarray
    assert color1.shape == color2.shape == (3,)

    lerp = t.copy().clip(min=0, max=1)[..., np.newaxis].repeat(3, axis=-1)
    final_colours = (1 - lerp) * color1 + lerp * color2

    return final_colours
