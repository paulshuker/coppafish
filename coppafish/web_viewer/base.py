from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px

from ..utils import system as utils_system
from ..setup.notebook import Notebook


def view_web(nb: Notebook, debug: bool = False) -> None:
    if debug:
        print(f"Loading notebook data")
    use_z: list[int] = nb.basic_info.use_z
    mid_z: int = nb.basic_info.use_z[len(nb.basic_info.use_z) // 2]
    dapi_image = nb.stitch.dapi_image[use_z]
    del nb
    if debug:
        print(f"Notebook data loaded")

    if debug:
        print(f"Creating dash app")
    app = Dash()
    app.layout = [
        html.H1(children=f"Coppafish {utils_system.get_software_version()}", style={"textAlign": "center"}),
        dcc.Graph(
            id="image-content",
            style={
                # Inline block allows for figures to be overlapped.
                "display": "inline-block",
            },
        ),
        dcc.Graph(
            id="spots-content",
            style={
                # Inline block allows for figures to be overlapped.
                "display": "inline-block",
            },
        ),
        dcc.RangeSlider(min(use_z), max(use_z), 1, id="z-slider", value=[mid_z, mid_z], allowCross=False),
    ]
    if debug:
        print(f"Dash app created")

    @callback(Output("image-content", "figure"), Input("z-slider", "value"))
    def update_z(value: list[int, int]):
        assert value[0] in use_z
        assert value[1] in use_z
        return px.imshow(dapi_image[value[0]], zmin=0, zmax=dapi_image.max())

    if debug:
        print(f"Running dash app")
    app.run(debug=debug)
