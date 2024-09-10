import itertools
from typing import Any, Optional

from dash import Dash, html, dcc, callback, Output, Input, State
import dash_daq as daq
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objs as go

from . import colours
from .. import log
from ..omp import base as omp_base
from ..setup.notebook import Notebook
from ..utils import system as utils_system


def bound_z(z_value: int, use_z: list[int]) -> int:
    assert type(z_value) is int
    assert type(use_z) is list
    z_value = max(z_value, min(use_z))
    z_value = min(z_value, max(use_z))
    return z_value


def view_web(nb: Notebook, gene_marker_file: Optional[str] = None, debug: bool = False) -> None:
    """
    View the notebook's gene calling results by locally hosting a website, powered by dash and plotly.

    Args:
        - nb (Notebook): the notebook to view.
        - gene_marker_file (str-like, optional): file path to the gene marker file csv. The csv must contain three
            headings: gene_name, colour, and symbol. See plotly/dash for valid colours/symbols. If a gene is not found
            in the file, a warning is issued and is not plotted in the viewer. Default: random gene colours/symbols.
        - debug (bool, optional): additional debugging information. Default: false.
    """
    assert type(nb) is Notebook
    assert gene_marker_file is None or type(str(gene_marker_file)) is str
    assert type(debug) is bool
    if not nb.has_page("call_spots"):
        raise ValueError(f"The notebook must have completed call_spots to view web results.")
    if debug:
        print(f"Loading notebook data")
    use_z: list[int] = nb.basic_info.use_z
    mid_z: int = nb.basic_info.use_z[len(nb.basic_info.use_z) // 2]
    methods = ["prob", "anchor"]
    methods_to_name = {"prob": "Probability", "anchor": "Anchor"}
    gene_names: np.ndarray[str] = nb.call_spots.gene_names
    gene_marker_colours = np.full_like(gene_names, "rgba(0, 0, 0, 0)", "<U20")
    gene_marker_symbols = np.full_like(gene_marker_colours, "circle", "<U20")
    if gene_marker_file is not None:
        # TODO: Read the gene marker file from the given file path.
        raise NotImplementedError()
    else:
        n_genes = gene_names.size
        valid_colours = plotly.colors.DEFAULT_PLOTLY_COLORS
        valid_symbols = ("circle", "circle-open", "square", "diamond", "cross", "x", "triangle-up", "pentagon")
        colour_symbol_combinations = list(itertools.product(valid_colours, valid_symbols))
        unique_genes = len(colour_symbol_combinations) > n_genes
        if not unique_genes:
            log.warn(f"Too many genes to assign each a unique colour/symbol combination. Some will be duplicates")
        rng = np.random.RandomState(0)
        for g in range(n_genes):
            i = rng.randint(len(colour_symbol_combinations))
            gene_marker_colours[g] = colour_symbol_combinations[i][0]
            gene_marker_symbols[g] = colour_symbol_combinations[i][1]
            if unique_genes:
                colour_symbol_combinations.pop(i)
        del n_genes, valid_colours, valid_symbols, colour_symbol_combinations, unique_genes
    # Gather the fused dapi image.
    dapi_image = nb.stitch.dapi_image[:]
    colour_max = np.abs(dapi_image).max()
    colour_min = -np.abs(dapi_image).max()
    dapi_image = (dapi_image - colour_min) / (colour_max - colour_min)
    dapi_colour_min = np.array((0, 0, 255))
    dapi_colour_max = np.array((255, 0, 0))
    # The dapi image colours are explicitly set.
    dapi_image = colours.interpolate_rgb(dapi_colour_min, dapi_colour_max, dapi_image)
    # Gather spot data.
    spot_data: dict[str, np.ndarray] = {}
    spot_data["prob/tile"] = nb.ref_spots.tile
    spot_data["prob/yxz"] = nb.ref_spots.local_yxz.astype(np.float32) + nb.stitch.tile_origin[spot_data["prob/tile"]]
    spot_data["prob/gene_no"] = np.argmax(nb.call_spots.gene_probabilities, 1).astype(np.int16)
    spot_data["prob/score"] = nb.call_spots.gene_probabilities.max(1)
    spot_data["prob/intensity"] = nb.call_spots.intensity
    spot_data["anchor/tile"] = spot_data["prob/tile"].copy()
    spot_data["anchor/yxz"] = spot_data["prob/yxz"].copy()
    spot_data["anchor/gene_no"] = nb.call_spots.dot_product_gene_no
    spot_data["anchor/score"] = nb.call_spots.dot_product_gene_score
    spot_data["anchor/intensity"] = nb.call_spots.intensity
    if nb.has_page("omp"):
        methods.append("omp")
        methods_to_name["omp"] = "OMP"
        spot_data["omp/yxz"], spot_data["omp/tile"] = omp_base.get_all_local_yxz(nb.basic_info, nb.omp)
        spot_data["omp/yxz"] = spot_data["omp/yxz"].astype(np.float32) + nb.stitch.tile_origin[spot_data["omp/tile"]]
        spot_data["omp/gene_no"] = omp_base.get_all_gene_no(nb.basic_info, nb.omp)[0]
        spot_data["omp/score"] = omp_base.get_all_scores(nb.basic_info, nb.omp)[0]
        spot_data["omp/intensity"] = np.ones_like(spot_data["omp/tile"], np.float32)
    del nb
    # Current method being shown.
    method = methods[-1]
    # The min and max z planes to show.
    z_planes = [bound_z(mid_z - 1, use_z), bound_z(mid_z + 1, use_z)]
    if debug:
        print(f"Notebook data loaded")

    if debug:
        print(f"Creating dash app")
    app = Dash(__name__)

    # The viewer content is 2d like the napari Viewer. It is the dapi image with spots overlaid.
    app.layout = [
        dcc.Store(
            id="store-range", data={"xaxis": [0, dapi_image.shape[2]], "yaxis": [0, dapi_image.shape[1]]}
        ),  # Store the current viewing range.
        html.H1(
            children=f"Coppafish {utils_system.get_software_version()} Web Viewer",
            style={
                "textAlign": "center",
                "fontFamily": "Helvetica",
                "font-size": "20px",
            },
        ),
        # The gene legend, viewer, and subplot are stored in a single div so they can be side-by-side on the same row.
        html.Div(
            style={"display": "flex", "flex-direction": "row", "max-width": "99vw", "max-height": "75vh"},
            children=[
                html.Div(
                    dcc.Graph(
                        id="gene-legend",
                        style={
                            "width": "19vw",  # 19% of the viewport width
                        },
                    ),
                    style={"flex": 1, "padding": "2px"},
                ),
                html.Div(
                    dcc.Graph(
                        id="viewer",
                        style={
                            "width": "49vw",
                        },
                    ),
                    style={"flex": 1, "padding": "2px"},
                ),
                html.Div(
                    dcc.Graph(
                        id="subplot",
                        style={
                            "width": "29vw",
                        },
                    ),
                    style={"flex": 1, "padding": "2px"},
                ),
            ],
        ),
        # A slider to select the z planes to view.
        html.Label(
            "Z Planes:",
            style={
                "textAlign": "center",  # Center text horizontally
                "width": "100%",  # Ensure the label takes the full width of its container
                "display": "block",  # Make the label a block element
                "fontFamily": "Arial",
            },
        ),
        dcc.RangeSlider(min(use_z), max(use_z), 1, id="z-slider", value=z_planes, allowCross=False),
        # A toggle switch to turn on/off the gene legend.
        html.Div(
            style={"justify-content": "center", "display": "flex", "flex-direction": "row", "fontFamily": "Arial"},
            children=[
                html.Div(
                    style={"align-items": "center", "display": "flex", "flex-direction": "column", "width": "100px"},
                    children=[
                        html.Label("Gene Legend", style={"display": "flex"}),
                        daq.BooleanSwitch(id="gene-legend-toggle", on=True, style={"display": "flex"}),
                    ],
                ),
                html.Div(
                    style={"align-items": "center", "display": "flex", "flex-direction": "column", "width": "100px"},
                    children=[
                        html.Label("Viewer", style={"display": "flex"}),
                        daq.BooleanSwitch(id="viewer-toggle", on=True, style={"display": "flex"}),
                    ],
                ),
                html.Div(
                    style={"align-items": "center", "display": "flex", "flex-direction": "column", "width": "100px"},
                    children=[
                        html.Label("Subplot", style={"display": "flex"}),
                        daq.BooleanSwitch(id="subplot-toggle", on=True, style={"display": "flex"}),
                    ],
                ),
                html.Div(
                    style={"align-items": "center", "display": "flex", "flex-direction": "column", "width": "120px"},
                    children=[
                        html.Label("Method", style={"display": "flex"}),
                        dcc.Dropdown(
                            methods_to_name,
                            method,
                            searchable=False,
                            clearable=False,
                            style={"width": "110px"},
                            id="method-choice",
                        ),
                    ],
                ),
            ],
        ),
        html.Div(id="placeholder", style={"display": "none"}),  # A hidden div used for callbacks with no UI changes.
    ]
    if debug:
        print(f"Dash app created")

    def create_main_figure(method: str, z_bounds: list[int, int], stored_data: dict[str, Any]) -> go.Figure:
        assert type(method) is str
        assert type(methods) is list
        assert type(z_bounds) is list
        assert type(stored_data) is dict
        assert method in methods
        yxz = spot_data[f"{method}/yxz"]
        gene_numbers = spot_data[f"{method}/gene_no"]
        keep = np.logical_and(yxz[:, 2] >= z_bounds[0], yxz[:, 2] <= z_bounds[1])
        yxz = yxz[keep]
        gene_numbers = gene_numbers[keep]
        # ? This might be a performance hit.
        labels = [f"{gene_number}: {gene_names[gene_number]}" for gene_number in gene_numbers]
        figure = go.Figure()
        figure.add_trace(go.Image(z=dapi_image[bound_z(z_bounds[0], use_z)], opacity=0.5, hoverinfo="none"))
        # Use WebGL to accelerate the rendering of spots. Especially good when there are more than 10,000 spots.
        figure.add_trace(
            go.Scattergl(
                name="",  # Empty name to hide text 'trace_1' when hovering with cursor.
                y=yxz[:, 0],
                x=yxz[:, 1],
                mode="markers",
                text=labels,
                marker=dict(
                    size=10,
                    color=gene_marker_colours[gene_numbers].tolist(),
                    symbol=gene_marker_symbols[gene_numbers].tolist(),
                ),
            )
        )
        # Register click event handler
        figure.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis_title="X",
            yaxis_title="Y",
            xaxis=dict(
                range=stored_data["xaxis"],
                zeroline=False,  # Show zero line
                showline=True,  # Show axis line
                showgrid=False,  # Hide grid lines
                linewidth=1,  # Width of axis line
                linecolor="black",  # Color of axis line
            ),
            yaxis=dict(
                range=stored_data["yaxis"],
                zeroline=False,  # Show zero line
                showline=True,  # Show axis line
                showgrid=False,  # Hide grid lines
                linewidth=1,  # Width of axis line
                linecolor="black",  # Color of axis line
            ),
            plot_bgcolor="rgba(0,0,0,0)",  # Transparent plot background
            paper_bgcolor="rgba(0,0,0,0)",  # Transparent paper background
            dragmode="pan",  # Set dragmode to pan
            modebar=dict(remove=["select2d", "lasso2d"]),  # Remove box select and lasso select
        )
        return figure

    @app.callback(
        Output("viewer", "figure"),
        Input("z-slider", "value"),
        State("store-range", "data"),  # Use stored data to preserve view
    )
    def update_z(value: list[int, int], stored_data: Optional[dict[str, Any]]):
        # value is a list of two integers. First one is the minimum z plane to view points for, second one is the
        # maximum z plane to view points for. Both of these values can be outside of use_z in case there are spots
        # registered outside of the usually z planes.
        return create_main_figure(method, value, stored_data)

    @app.callback()
    def update_method(value: str):
        pass

    # Define callback to handle click events
    @app.callback(
        Output("placeholder", "children"),
        Input("viewer", "clickData"),
    )
    def handle_click(clickData: dict[str, Any]):
        if clickData is None:
            return
        print("CLICK")

    @app.callback(
        Output("gene-legend", "style"),
        Input("gene-legend-toggle", "on"),
    )
    def gene_legend_toggle_switch_update(on: bool) -> None:
        if on:
            return {"display": "flex"}
        return {"display": "none"}

    @app.callback(
        Output("subplot", "style"),
        Input("subplot-toggle", "on"),
    )
    def subplot_toggle_switch_update(on: bool) -> None:
        if on:
            return {"display": "flex"}
        return {"display": "none"}

    @app.callback(
        Output("viewer", "style"),
        Input("viewer-toggle", "on"),
    )
    def viewer_toggle_switch_update(on: bool) -> None:
        if on:
            return {"display": "flex"}
        return {"display": "none"}

    @app.callback(
        Output("store-range", "data"),
        Input("viewer", "relayoutData"),
    )
    def store_current_range(relayout_data):
        # Stores the new viewing region when the user pans/zooms.
        if relayout_data and "xaxis.range" in relayout_data and "yaxis.range" in relayout_data:
            return {"xaxis": relayout_data["xaxis.range"], "yaxis": relayout_data["yaxis.range"]}
        return {"xaxis": [0, dapi_image.shape[2]], "yaxis": [0, dapi_image.shape[1]]}

    if debug:
        print(f"Running dash app")
    app.run(debug=debug)
