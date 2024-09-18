from collections.abc import Iterable
import math as maths
from typing import Union, Optional


class ViewState:
    # If true, use DataShader to faithfully create an image of the gene reads locations.
    # Gene points are not selectable when using datashade.
    # This is used when the Viewer is zoomed far out.
    datashade: bool
    # The number of pixels in the DataShade background image is
    # floor(W / datashade_downsample_factor) by floor(H / datashade_downsample_factor)
    # where W and H is the number of pixels along the x and y axes for the background image.
    datashade_downsample_factor: int
    # If the user's viewing box length is >= minimum_view_yx and < maximum_view_yx, then this ViewState should be used.
    # If set to None, then these conditions are ignored.
    minimum_range_yx: Union[float, None]
    maximum_range_yx: Union[float, None]
    # The user's viewing must be within this bounding box of coordinates for the ViewState to be on.
    # Set to None for no bounding box.
    minimum_yx: tuple[Optional[float], Optional[float]]
    maximum_yx: tuple[Optional[float], Optional[float]]

    def __init__(
        self,
        minimum_range_yx: Union[float, None],
        maximum_range_yx: Union[float, None],
        minimum_yx: tuple[Optional[float], Optional[float]] = (None, None),
        maximum_yx: tuple[Optional[float], Optional[float]] = (None, None),
    ) -> None:
        assert minimum_range_yx is None or type(minimum_range_yx) is float
        assert maximum_range_yx is None or type(maximum_range_yx) is float
        assert type(minimum_yx) is tuple
        assert type(maximum_yx) is tuple
        self.datashade = False
        self.datashade_downsample_factor = 0
        self.minimum_range_yx = minimum_range_yx
        self.maximum_range_yx = maximum_range_yx
        self.minimum_yx = minimum_yx
        self.maximum_yx = maximum_yx

    def is_on(self, view_ranges_yx: Iterable[Iterable[float, float], Iterable[float, float]]) -> bool:
        assert type(view_ranges_yx) is tuple
        assert len(view_ranges_yx) == 2
        assert len(view_ranges_yx[0]) == 2
        assert len(view_ranges_yx[1]) == 2

        valid: int = 1
        if self.minimum_range_yx is not None:
            valid *= all([abs(view_range[1] - view_range[0]) >= self.minimum_range_yx for view_range in view_ranges_yx])
        if self.maximum_range_yx is not None:
            valid *= all([abs(view_range[1] - view_range[0]) < self.maximum_range_yx for view_range in view_ranges_yx])
        for i in range(2):
            if self.minimum_yx[i] is not None:
                valid *= view_ranges_yx[i][0] >= self.minimum_yx[i]
            if self.maximum_yx[i] is not None:
                valid *= view_ranges_yx[i][1] < self.maximum_yx[i]
        return bool(valid)

    def get_image_slices_yx(self, background_image_shape_yx: tuple[int, int]) -> tuple[slice, slice]:
        assert type(background_image_shape_yx) is tuple
        assert len(background_image_shape_yx) == 2

        yx_min = [0, 0]
        yx_max = [background_image_shape_yx[0], background_image_shape_yx[1]]
        for i in range(2):
            if self.minimum_yx[i] is not None:
                yx_min[i] = maths.floor(self.minimum_yx[i])
            if self.maximum_yx[i] is not None:
                yx_max[i] = maths.floor(self.maximum_yx[i])
        return tuple([slice(yx_min[i], yx_max[i], 1) for i in range(2)])
