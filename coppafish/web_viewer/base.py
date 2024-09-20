import base64
import io
import itertools
import math as maths
import os
import PIL
from typing import Any, Optional

import dash
from dash import Dash, html, dcc, Output, Input, State
import dash_daq as daq
import datashader as ds
import datashader.transfer_functions as tf
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objs as go
import zarr

from . import colours, legend
from .view_states import ViewState
from .. import log
from ..spot_colours import base as spot_colours_base
from ..omp import base as omp_base
from ..setup.notebook import Notebook
from ..utils import system as utils_system


def bound_z(z_value: int, use_z: list[int]) -> int:
    assert type(z_value) is int
    assert type(use_z) is list
    z_value = max(z_value, min(use_z))
    z_value = min(z_value, max(use_z))
    return z_value


def _get_index_from_click_data(clickData: Optional[dict[str, Any]]) -> Optional[int]:
    # Returns the index of the marker that was clicked on from a scatter or scattergl plot. Returns None if a
    # marker was not clicked on.
    spot_index = None
    if clickData is not None and len(clickData["points"]) == 1 and "pointNumber" in clickData["points"][0]:
        spot_index = clickData["points"][0]["pointNumber"]
    return spot_index


def view_web(nb_filepath: str, gene_marker_file: Optional[str] = None, debug: bool = False) -> None:
    """
    View the notebook's gene calling results by locally hosting a website, powered by dash and plotly.

    Args:
        - nb_filepath (str): the notebook file path to view.
        - gene_marker_file (str-like, optional): file path to the gene marker file csv. The csv must contain three
            headings: gene_name, colour, and symbol. See plotly/dash for valid colours/symbols. If a gene is not found
            in the file, a warning is issued and is not plotted in the viewer. Default: random gene colours/symbols.
        - debug (bool, optional): additional debugging information. Default: false.
    """
    # TODO: We need a new way of viewing this data. Viewing millions of scatter points is unfeasible like this.
    # Instead, I think try superimpose the points into one image using DataShader, view this image, interact with this
    # one image by simply using the cursor's position with a scipy KDTree.
    if not os.path.isdir(nb_filepath):
        raise ValueError(f"Notebook at {nb_filepath} not found")
    nb = Notebook(nb_filepath)
    assert gene_marker_file is None or type(str(gene_marker_file)) is str
    assert type(debug) is bool
    if not nb.has_page("call_spots"):
        raise ValueError(f"The notebook must have completed call_spots to view web results.")
    use_z: list[int] = nb.basic_info.use_z
    mid_z: int = nb.basic_info.use_z[len(nb.basic_info.use_z) // 2]
    methods = ["prob", "anchor"]
    name_to_methods = {"Probability": "prob", "Anchor": "anchor"}
    gene_names: np.ndarray[str] = nb.call_spots.gene_names
    gene_codes: np.ndarray[int] = nb.call_spots.gene_codes
    gene_marker_colours = np.full_like(gene_names, "rgba(0, 0, 0, 0)", "<U20")
    gene_marker_symbols = np.full_like(gene_marker_colours, "circle", "<U20")
    if gene_marker_file is not None:
        # TODO: Read the gene marker file from the given file path using pandas.
        raise NotImplementedError()
    else:
        n_genes = gene_names.size
        valid_colours = plotly.colors.DEFAULT_PLOTLY_COLORS
        valid_symbols = ("circle", "square", "diamond", "cross", "x", "triangle-up", "pentagon")
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
    print("Gathering background image")
    dapi_image = nb.stitch.dapi_image[:].astype(np.float32)
    colour_max = np.abs(dapi_image).max()
    colour_min = -np.abs(dapi_image).max()
    dapi_image = (dapi_image - colour_min) / (colour_max - colour_min)
    dapi_colour_min = np.array((0, 0, 255), np.float32)
    dapi_colour_max = np.array((255, 0, 0), np.float32)
    # The dapi image colours are explicitly set.
    dapi_image_colours = colours.interpolate_rgb(dapi_colour_min, dapi_colour_max, dapi_image)
    dapi_image_colours = dapi_image_colours.astype(np.float16)
    use_rounds: list[int] = nb.basic_info.use_rounds
    use_channels: list[int] = nb.basic_info.use_channels
    # Gather spot data.
    print("Gathering spot data")
    spot_data: dict[str, np.ndarray] = {}
    spot_data["prob/tile"] = nb.ref_spots.tile
    spot_data["prob/local_yxz"] = nb.ref_spots.local_yxz.astype(np.float32)
    spot_data["prob/yxz"] = spot_data["prob/local_yxz"] + nb.stitch.tile_origin[spot_data["prob/tile"]]
    spot_data["prob/gene_no"] = np.argmax(nb.call_spots.gene_probabilities, 1).astype(np.int16)
    spot_data["prob/score"] = nb.call_spots.gene_probabilities.max(1)
    spot_data["prob/intensity"] = nb.call_spots.intensity
    spot_data["anchor/tile"] = spot_data["prob/tile"].copy()
    spot_data["anchor/local_yxz"] = nb.ref_spots.local_yxz.astype(np.float32)
    spot_data["anchor/yxz"] = spot_data["prob/yxz"].copy()
    spot_data["anchor/gene_no"] = nb.call_spots.dot_product_gene_no
    spot_data["anchor/score"] = nb.call_spots.dot_product_gene_score
    spot_data["anchor/intensity"] = nb.call_spots.intensity
    if nb.has_page("omp"):
        methods.append("omp")
        name_to_methods["OMP"] = "omp"
        spot_data["omp/local_yxz"], spot_data["omp/tile"] = omp_base.get_all_local_yxz(nb.basic_info, nb.omp)
        spot_data["omp/yxz"] = (
            spot_data["omp/local_yxz"].astype(np.float32) + nb.stitch.tile_origin[spot_data["omp/tile"]]
        )
        spot_data["omp/gene_no"] = omp_base.get_all_gene_no(nb.basic_info, nb.omp)[0]
        spot_data["omp/score"] = omp_base.get_all_scores(nb.basic_info, nb.omp)[0]
        spot_data["omp/intensity"] = np.ones_like(spot_data["omp/tile"], np.float32)
        spot_data["omp/colours"] = omp_base.get_all_colours(nb.basic_info, nb.omp)[0]
    del nb
    min_yxz = np.array([9999, 9999, 99], np.float32)
    max_yxz = np.array([0, 0, 0], np.float32)
    max_score = 1.0
    for method in methods:
        method_max_score = spot_data[f"{method}/score"].max()
        if method_max_score > max_score:
            max_score = method_max_score
        method_min_yxz = spot_data[f"{method}/yxz"].min(0)
        method_max_yxz = spot_data[f"{method}/yxz"].max(0)
        min_yxz = min_yxz.clip(max=method_min_yxz)
        max_yxz = max_yxz.clip(min=method_max_yxz)
    view_states: list[ViewState] = []
    # Level 1, most zoomed out view.
    min_range_level_1 = 1_200.0
    new_state = ViewState(min_range_level_1, None)
    new_state.datashade = True
    new_state.datashade_downsample_factor = 3
    view_states.append(new_state)
    # Level 2+, each point plotted in a scatter. The image is chunked to improve performance.
    chunk_size = min_range_level_1
    chunk_overlap = chunk_size // 2
    x = chunk_overlap
    y = chunk_overlap
    while True:
        minimum_yx = [x - chunk_size, y - chunk_size]
        if minimum_yx[0] <= 0:
            minimum_yx[0] = None
        if minimum_yx[1] <= 0:
            minimum_yx[1] = None
        maximum_yx = [x + chunk_size, y + chunk_size]
        if maximum_yx[0] > dapi_image.shape[1]:
            maximum_yx[0] = None
        if maximum_yx[1] > dapi_image.shape[2]:
            maximum_yx[1] = None
        new_state = ViewState(None, chunk_size, tuple(minimum_yx), tuple(maximum_yx))
        view_states.append(new_state)
        x += chunk_size
        if x > dapi_image.shape[2]:
            y += chunk_size
            x = chunk_overlap
        if y > dapi_image.shape[1]:
            break
    view_states: tuple[ViewState] = tuple(view_states)

    # The min and max z planes to show at the start.
    z_planes = [bound_z(mid_z - 1, use_z), bound_z(mid_z + 1, use_z)]
    subplot_options = ("Summary", "Gene Legend", "Colour", "Colour Map")

    print("Building app")
    app = Dash(__name__, title="Coppafish")
    max_marker_size = 30.0

    # The viewer content is 2d like the napari Viewer. It is the background (dapi) image with spots overlaid.
    app.layout = [
        dcc.Store(
            id="store-range", data={"xaxis": [0, dapi_image.shape[2]], "yaxis": [0, dapi_image.shape[1]]}
        ),  # Store the current viewing range.
        dcc.Store(id="store-z-bounds", data=None),
        dcc.Store(id="store-score-bounds", data=None),
        dcc.Store(id="store-method", data=None),  # Store the current selected method.
        dcc.Store(id="store-spot-indices", data=None),  # Store the current viewing spot indices.
        dcc.Store(id="store-selected-spot", data=None),  # Store the selected spot index.
        dcc.Store(id="store-selected-genes", data=None),  # Store the genes that are currently visible.
        dcc.Store(id="store-selected-genes-last", data=None),  # Store the last gene selections, used by the viewer.
        dcc.Store(id="store-selected-view-state", data=None),
        dcc.Store(id="store-all-genes-n-clicks", data=None),  # Store the number of times the button was clicked.
        dcc.Store(id="store-no-genes-n-clicks", data=None),  # Store the number of times the button was clicked.
        dcc.Store(id="store-marker-size", data=None),  # Store the selected gene marker size.
        html.Div(
            children=[
                html.Img(
                    src="https://raw.githubusercontent.com/paulshuker/coppafish/b3dda7925a8fea63c6ccd050fc04ad10a22af9c0/docs/images/logo.svg",
                    style={"width": "30px", "padding": "0 20px"},
                ),
                html.H1(
                    f"Web Viewer",
                    style={
                        "textAlign": "center",
                        "fontFamily": "Helvetica",
                        "font-size": "20px",
                        "padding": "0 0",
                    },
                ),
            ],
            style={"display": "flex", "flex-direction": "row", "width": "100vw", "flexwrap": "wrap"},
        ),
        # The viewer and subplot are stored in a single div so they can be side-by-side on the same row.
        html.Div(
            style={
                "display": "flex",
                "flex-direction": "row",
                "flexwrap": "wrap",
                "width": "96vw",
                "max-width": "96vw",
                "height": "75vh",
                "max-height": "75vh",
            },
            children=[
                html.Div(
                    style={
                        # "flex": 1,
                        "padding": "0",
                        "margin": "0",
                    },
                    children=[
                        dcc.Graph(
                            id="viewer",
                            style={"width": "69%", "max-width": "100%", "height": "100%"},
                            config=dict(
                                modeBarButtonsToRemove=["select2d", "lasso2d", "select2d", "autoscale"],
                                # FIXME: Zooming using scroll wheel does weird shit to the figure's axes.
                                # scrollZoom=True,
                            ),
                        )
                    ],
                ),
                dcc.Graph(
                    id="subplot",
                    style={"display": "flex", "width": "28%", "max-width": "28%"},
                    config={"displayModeBar": False},
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
            style={
                "justify-content": "center",
                "display": "flex",
                "flex-direction": "row",
                "fontFamily": "Arial",
                "flexwrap": "wrap",
            },
            children=[
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
                    style={"align-items": "center", "display": "flex", "flex-direction": "column", "width": "80px"},
                    children=[
                        html.Label("All genes", style={"display": "flex"}),
                        html.Button(
                            id="show-all-genes",
                            style={"display": "flex", "height": "70%", "width": "90%"},
                        ),
                    ],
                ),
                html.Div(
                    style={"align-items": "center", "display": "flex", "flex-direction": "column", "width": "80px"},
                    children=[
                        html.Label("No genes", style={"display": "flex"}),
                        html.Button(
                            id="show-no-genes",
                            style={"display": "flex", "height": "70%", "width": "90%"},
                        ),
                    ],
                ),
                html.Div(
                    style={"align-items": "center", "display": "flex", "flex-direction": "column", "width": "120px"},
                    children=[
                        html.Label("Method", style={"display": "flex"}),
                        dcc.Dropdown(
                            list(name_to_methods.keys()),
                            list(name_to_methods.keys())[-1],
                            searchable=False,
                            clearable=False,
                            style={"width": "110px"},
                            id="method-choice",
                        ),
                    ],
                ),
                html.Div(
                    style={"align-items": "center", "display": "flex", "flex-direction": "column", "width": "160px"},
                    children=[
                        html.Label("Subplot", style={"display": "flex"}),
                        dcc.Dropdown(
                            subplot_options,
                            subplot_options[0],
                            searchable=False,
                            clearable=False,
                            id="subplot-choice",
                            style={"display": "flex", "width": "150px"},
                        ),
                    ],
                ),
                html.Div(
                    style={"align-items": "center", "display": "flex", "flex-direction": "column", "width": "200px"},
                    children=[
                        html.Label("Score", style={"display": "flex"}),
                        html.Div(
                            dcc.RangeSlider(
                                0.0,
                                max_score,
                                marks={0.0: "0.0", 0.25: "0.25", 0.5: "0.5", 0.75: "0.75", 1.0: "1.0"},
                                id="score-slider",
                                value=[max(max_score / 2, 0.4), max_score],
                                allowCross=False,
                            ),
                            style={"display": "block", "width": "100%"},
                        ),
                    ],
                ),
                html.Div(
                    style={"align-items": "center", "display": "flex", "flex-direction": "column", "width": "160px"},
                    children=[
                        html.Label("Marker size", style={"display": "flex"}),
                        html.Div(
                            dcc.Slider(0.5, max_marker_size, value=10.0, marks=None, id="marker-size-slider"),
                            style={"display": "block", "width": "100%"},
                        ),
                    ],
                ),
            ],
        ),
        html.A(
            f"coppafish {utils_system.get_software_version()}",
            href=f"https://github.com/paulshuker/coppafish/tree/{utils_system.get_software_version()}",
            target="_blank",
            style={"textAlign": "left", "width": "100vw"},
        ),
        html.Div(id="placeholder", style={"display": "none"}),  # A hidden div used for callbacks with no UI changes.
    ]

    def create_main_figure(
        method: str,
        view_state: ViewState,
        z_bounds: list[int, int],
        score_bounds: list[float, float],
        gene_marker_size: float,
        stored_view: dict[str, Any],
        selected_genes: Optional[list[bool]],
    ) -> tuple[go.Figure, list[int]]:
        assert type(method) is str
        assert type(methods) is list
        assert type(view_state) is ViewState
        assert method in methods
        assert type(z_bounds) is list
        assert type(score_bounds) is list
        assert type(float(gene_marker_size)) is float
        assert type(stored_view) is dict
        assert type(selected_genes) is list or selected_genes is None
        yxz = spot_data[f"{method}/yxz"]
        gene_numbers = spot_data[f"{method}/gene_no"]
        scores = spot_data[f"{method}/score"]
        keep = np.logical_and(yxz[:, 2] >= z_bounds[0], yxz[:, 2] <= z_bounds[1])
        keep *= np.logical_and(scores >= score_bounds[0], scores < score_bounds[1])
        if selected_genes is not None:
            keep *= (gene_numbers[:, None] == np.nonzero(selected_genes)[0][None]).any(1)
        spot_indices = keep.nonzero()[0].tolist()
        yxz = yxz[keep]
        gene_numbers = gene_numbers[keep]
        labels = [f"{gene_number}: {gene_names[gene_number]}" for gene_number in gene_numbers]
        figure = go.Figure()
        if view_state.datashade and keep.sum() > 0:
            spot_distributions = pd.DataFrame(dict(x=[], y=[], cat=[]))
            for g in range(gene_names.size):
                is_gene_g = gene_numbers == g
                X = yxz[is_gene_g, 1]
                Y = yxz[is_gene_g, 0]
                cat = np.full_like(X, f"{g}")
                new_data = pd.DataFrame(dict(x=X, y=Y, cat=cat))
                spot_distributions = pd.concat([spot_distributions, new_data], ignore_index=True)
            spot_distributions["cat"] = spot_distributions["cat"].astype("category")

            # Projection:
            x_min = view_state.get_image_slices_yx(dapi_image.shape[1:3])[1].start
            x_max = view_state.get_image_slices_yx(dapi_image.shape[1:3])[1].stop
            y_min = view_state.get_image_slices_yx(dapi_image.shape[1:3])[0].start
            y_max = view_state.get_image_slices_yx(dapi_image.shape[1:3])[0].stop
            canvas = ds.Canvas(
                plot_width=dapi_image.shape[2] // view_state.datashade_downsample_factor,
                plot_height=dapi_image.shape[1] // view_state.datashade_downsample_factor,
                x_range=(x_min, x_max),
                y_range=(y_min, y_max),
                x_axis_type="linear",
                y_axis_type="linear",
            )

            # Aggregation of points by category (gene index):
            aggregate = canvas.points(spot_distributions, "x", "y", ds.by("cat", ds.count()))

            # A different colour for each gene.
            colour_key = {
                g: list(map(int, str(gene_marker_colours[g][5:-1]).split(", ")[:3])) for g in range(gene_names.size)
            }
            image = tf.shade(aggregate, name="Default colour mapping", cmap=colour_key)
            image = tf.set_background(image, "white")

            # # Upsample image.
            # image = image.to_pil().resize((dapi_image.shape[2], dapi_image.shape[1]), PIL.Image.NEAREST)

            # Image is converted into a numpy array.
            image = np.array(image)

            # R, G, and B values are extracted from the uint32 values.
            image_rgb = np.zeros(image.shape + (3,), np.uint8)
            image_rgb[..., 0] = (image >> 24) & 0xFF
            image_rgb[..., 1] = (image >> 16) & 0xFF
            image_rgb[..., 2] = (image >> 8) & 0xFF

            # The image is stretched to reach full viewing size.
            figure.add_trace(
                go.Image(
                    z=image_rgb,
                    dx=view_state.datashade_downsample_factor,
                    dy=view_state.datashade_downsample_factor,
                    hoverinfo="none",
                )
            )
        if not view_state.datashade:
            image_slices_yx = view_state.get_image_slices_yx(dapi_image.shape[1:3])
            print(f"{image_slices_yx=}")
            figure.add_trace(
                go.Image(
                    z=dapi_image_colours[bound_z(z_bounds[0], use_z), image_slices_yx[0], image_slices_yx[1]],
                    opacity=0.5,
                    hoverinfo="none",
                    y0=image_slices_yx[0].start,
                    x0=image_slices_yx[1].start,
                )
            )
            # Use WebGL to accelerate the rendering of spots. Especially good when there are > 10,000 spots.
            figure.add_trace(
                go.Scattergl(
                    name="",  # Empty name to hide text 'trace_1' when hovering with cursor.
                    y=yxz[:, 0],
                    x=yxz[:, 1],
                    mode="markers",
                    text=labels,
                    marker=dict(
                        size=gene_marker_size,
                        color=gene_marker_colours[gene_numbers].tolist(),
                        symbol=gene_marker_symbols[gene_numbers].tolist(),
                    ),
                )
            )
            # # Register click event handler
            # figure.update_layout(
            #     dragmode="pan",  # Set dragmode to pan
            #     modebar=dict(
            #         remove=["select", "select2d", "lasso2d", "resetScale2d"]
            #     ),  # Remove box select and lasso select
            # )
        figure.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis_title="X",
            yaxis_title="Y",
            xaxis=dict(
                range=stored_view["xaxis"],
                zeroline=False,  # Show zero line
                showline=True,  # Show axis line
                showgrid=False,  # Hide grid lines
                linewidth=1,  # Width of axis line
                linecolor="black",  # Color of axis line
                autorange=False,
            ),
            yaxis=dict(
                range=stored_view["yaxis"],
                zeroline=False,  # Show zero line
                showline=True,  # Show axis line
                showgrid=False,  # Hide grid lines
                linewidth=1,  # Width of axis line
                linecolor="black",  # Color of axis line
                autorange=False,
            ),
            plot_bgcolor="rgba(0,0,0,0)",  # Transparent plot background
            paper_bgcolor="rgba(0,0,0,0)",  # Transparent paper background
            dragmode="pan",
        )
        return figure, spot_indices

    def create_subplot_figure(
        subplot_choice: str, method: str, selected_spot: int, selected_genes: Optional[list[int]]
    ) -> go.Figure:
        assert type(subplot_choice) is str
        assert subplot_choice in subplot_options
        assert selected_spot is None or type(selected_spot) is int
        assert selected_genes is None or type(selected_genes) is list

        figure = go.Figure()
        figure.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis_title="",
            yaxis_title="",
            xaxis=dict(showticklabels=False, zeroline=False, showline=False, showgrid=False),
            yaxis=dict(showticklabels=False, zeroline=False, showline=False, showgrid=False),
            plot_bgcolor="rgba(0,0,0,0)",  # Transparent plot background
            paper_bgcolor="rgba(0,0,0,0)",  # Transparent paper background
            dragmode=False,
        )
        kwargs = dict(showarrow=False, xref="paper", yref="paper", xanchor="left", yanchor="top")
        if subplot_choice == "Summary":
            font = {"size": 16, "color": "black"}
            y_spacing = 0.1
            y = 1
            figure.add_annotation(
                go.layout.Annotation(x=0, y=y, text="Spot summary", font={"size": 22, "color": "black"}, **kwargs)
            )
            y -= y_spacing
            gene = ""
            score = ""
            intensity = ""
            gene_code = ""
            gene_index = ""
            spot_index = ""
            tile = ""
            if selected_spot is None:
                figure.add_annotation(go.layout.Annotation(x=0, y=y, text=f"Select a spot", font=font, **kwargs))
            else:
                score = str(spot_data[f"{method}/score"][selected_spot])
                intensity = str(spot_data[f"{method}/intensity"][selected_spot])
                gene_index = spot_data[f"{method}/gene_no"][selected_spot]
                gene = str(gene_names[gene_index])
                gene_code = "".join([str(dye_index) for dye_index in gene_codes[gene_index]])
                tile = str(spot_data[f"{method}/tile"][selected_spot])
                spot_index = str(selected_spot)
                figure.add_annotation(go.layout.Annotation(x=0, y=y, text=f"Gene: {gene}", font=font, **kwargs))
                y -= y_spacing
                figure.add_annotation(go.layout.Annotation(x=0, y=y, text=f"Score: {score}", font=font, **kwargs))
                y -= y_spacing
                figure.add_annotation(
                    go.layout.Annotation(x=0, y=y, text=f"Intensity: {intensity}", font=font, **kwargs)
                )
                y -= y_spacing
                figure.add_annotation(
                    go.layout.Annotation(x=0, y=y, text=f"Gene code: {gene_code}", font=font, **kwargs)
                )
                y -= y_spacing
                figure.add_annotation(go.layout.Annotation(x=0, y=y, text=f"Tile: {tile}", font=font, **kwargs))
                y -= y_spacing
                figure.add_annotation(
                    go.layout.Annotation(x=0, y=y, text=f"Gene index: {gene_index}", font=font, **kwargs)
                )
                y -= y_spacing
                figure.add_annotation(
                    go.layout.Annotation(x=0, y=y, text=f"Spot index: {spot_index}", font=font, **kwargs)
                )
        elif subplot_choice == "Gene Legend":
            figure.add_annotation(
                go.layout.Annotation(x=0, y=1, text="Gene legend", font={"size": 25, "color": "black"}, **kwargs)
            )
            yx_positions = legend.get_gene_scatter_positions(
                gene_marker_symbols.size, n_columns=maths.floor(maths.sqrt(gene_marker_symbols.size))
            )
            yx_positions /= np.max(yx_positions, axis=0, keepdims=True)
            opacity = [1.0] * gene_names.size
            if selected_genes is not None:
                opacity = [0.3] * gene_names.size
                for selected_gene in np.nonzero(selected_genes)[0].tolist():
                    opacity[selected_gene] = 1.0
            figure.add_scatter(
                y=yx_positions[:, 0],
                x=yx_positions[:, 1],
                mode="markers+text",
                marker=dict(
                    size=17 * maths.sqrt(gene_names.size),
                    color=gene_marker_colours.tolist(),
                    symbol=gene_marker_symbols.tolist(),
                    opacity=opacity,
                ),
                text=gene_names.tolist(),
                textfont=dict(size=max(14 * maths.sqrt(gene_names.size), 1.0)),
                textposition="bottom center",
                name="",
                hoverinfo="text",
                hovertemplate="<b>%{customdata}</b>",
                customdata=gene_names.tolist(),
            )
        elif subplot_choice == "Colour":
            figure.add_annotation(
                go.layout.Annotation(x=0, y=1, text="Pixel colour", font={"size": 25, "color": "black"}, **kwargs)
            )
            if selected_spot is None:
                return figure
            spot_yxz = spot_data[f"{method}/local_yxz"][selected_spot]
            tile = spot_data[f"{method}/tile"][selected_spot].item()
            colour = spot_data[f"{method}/colours"][selected_spot]
            if np.allclose(colour, 0):
                return figure
            return px.imshow(
                colour.T,
                zmin=-np.abs(colour).max(),
                zmax=np.abs(colour).max(),
                title=f"Spot colour {tuple(spot_yxz)}, {tile=}",
                color_continuous_scale="bluered",
            )
        else:
            raise NotImplementedError(f"Subplot {subplot_choice} is not implemented")
        return figure

    # The viewer's figure is only updated by a single callback function to avoid duplication problems.
    @app.callback(
        Output("viewer", "figure"),
        Output("store-z-bounds", "data"),
        Output("store-score-bounds", "data"),
        Output("store-method", "data"),
        Output("store-spot-indices", "data"),
        Output("store-selected-spot", "data"),
        Output("store-selected-genes-last", "data"),
        Output("store-selected-view-state", "data"),
        Output("store-marker-size", "data"),
        Input("z-slider", "value"),
        Input("score-slider", "value"),
        Input("marker-size-slider", "value"),
        Input("method-choice", "value"),
        Input("viewer", "clickData"),
        Input("store-selected-genes", "data"),
        Input("store-range", "data"),
        State("store-method", "data"),
        State("store-z-bounds", "data"),
        State("store-score-bounds", "data"),
        State("store-selected-genes-last", "data"),
        State("store-selected-view-state", "data"),
        State("store-marker-size", "data"),
    )
    def update_viewer(
        z_bounds: list[int, int],
        score_bounds: list[float, float],
        gene_marker_size: float,
        method_name: str,
        clickData: Optional[dict[str, Any]],
        selected_genes: list[int],
        stored_view: dict[str, Any],
        last_method: str,
        last_z_bounds: list[int, int],
        last_score_bounds: list[float, float],
        last_selected_genes: list[int],
        last_selected_view_state: int,
        last_gene_marker_size: float,
    ):
        method = name_to_methods[method_name]
        new_figure = dash.no_update
        spot_indices = dash.no_update
        spot_index = _get_index_from_click_data(clickData)
        view_ranges_yx = (stored_view["xaxis"], stored_view["yaxis"])
        print(f"{view_ranges_yx=}")
        selected_view_states = np.array([state.is_on(view_ranges_yx) for state in view_states]).nonzero()[0]
        if selected_view_states.size == 0:
            selected_view_state = 0
        else:
            selected_view_state = selected_view_states[selected_view_states.size // 2].item()
        print(f"{selected_view_state=}")
        if (
            method != last_method
            or z_bounds != last_z_bounds
            or score_bounds != last_score_bounds
            or selected_genes != last_selected_genes
            or selected_view_state != last_selected_view_state
            or gene_marker_size != last_gene_marker_size
        ):
            # The main figure is only updated when it must be updated. This is done to avoid performance impacts from
            # too many refreshes when the input has no effect on the viewer, like clicking on a spot.
            print(f"Updating figure")
            spot_index = None
            new_figure, spot_indices = create_main_figure(
                method,
                view_states[selected_view_state],
                z_bounds,
                score_bounds,
                gene_marker_size,
                stored_view,
                selected_genes,
            )
        return (
            new_figure,
            z_bounds,
            score_bounds,
            method,
            spot_indices,
            spot_index,
            selected_genes,
            selected_view_state,
            gene_marker_size,
        )

    @app.callback(
        Output("subplot", "figure"),
        Input("subplot-choice", "value"),
        Input("store-method", "data"),
        Input("store-selected-spot", "data"),
        Input("store-spot-indices", "data"),
        Input("store-selected-genes", "data"),
    )
    def update_subplot(
        subplot_choice: str,
        method: str,
        selected_spot: Optional[int],
        spot_indices: list[int],
        selected_genes: list[int],
    ) -> go.Figure:
        """
        Create a plotly subplot figure.
        """
        if selected_spot is not None:
            selected_spot = spot_indices[selected_spot]
        return create_subplot_figure(subplot_choice, method, selected_spot, selected_genes)

    @app.callback(
        Output("store-selected-genes", "data"),
        Output("subplot", "clickData"),
        Output("store-all-genes-n-clicks", "data"),
        Output("store-no-genes-n-clicks", "data"),
        Input("subplot", "clickData"),
        Input("show-all-genes", "n_clicks"),
        Input("show-no-genes", "n_clicks"),
        State("subplot-choice", "value"),
        State("store-selected-genes", "data"),
        State("store-all-genes-n-clicks", "data"),
        State("store-no-genes-n-clicks", "data"),
    )
    def clicked_subplot_or_button(
        clickData: dict[str, Any],
        all_genes_n_clicks: int,
        no_genes_n_clicks: int,
        subplot: str,
        last_selected_genes: list[bool],
        last_all_genes_n_clicks: int,
        last_no_genes_n_clicks: int,
    ):
        # The subplot viewer was clicked on or a button was pressed related to subplot changes.
        # Whether anything happens depends on the type of subplot displayed.
        selected_genes = last_selected_genes
        if all_genes_n_clicks != last_all_genes_n_clicks:
            # Select all genes button was clicked.
            selected_genes = [True] * gene_names.size
        if no_genes_n_clicks != last_no_genes_n_clicks:
            # Select no genes button was clicked.
            selected_genes = [False] * gene_names.size
        if selected_genes is None:
            # By default, show all genes.
            selected_genes = [True] * gene_names.size
        if subplot == "Summary":
            pass
        elif subplot == "Gene Legend":
            spot_index = _get_index_from_click_data(clickData)
            if spot_index is not None:
                # Toggle the clicked gene.
                selected_genes[spot_index] = not selected_genes[spot_index]
        # Once the click data has been resolved, the click data is forgotten so the same point can clicked again.
        return selected_genes, None, all_genes_n_clicks, no_genes_n_clicks

    @app.callback(
        Output("viewer", "style"),
        Input("viewer-toggle", "on"),
    )
    def toggle_viewer(on: bool) -> None:
        if on:
            return {"display": "flex"}
        return {"display": "none"}

    @app.callback(
        Output("subplot", "style"),
        Input("subplot-toggle", "on"),
    )
    def toggle_subplot(on: bool) -> None:
        if on:
            return {"display": "flex"}
        return {"display": "none"}

    @app.callback(
        Output("store-range", "data"),
        Input("viewer", "relayoutData"),
    )
    def store_current_range(relayout_data):
        # Stores the new viewing region when the user pans/zooms.
        if relayout_data and "xaxis.range[0]" in relayout_data and "yaxis.range[0]" in relayout_data:
            new_data = {
                "xaxis": [relayout_data["xaxis.range[0]"], relayout_data["xaxis.range[1]"]],
                "yaxis": [relayout_data["yaxis.range[0]"], relayout_data["yaxis.range[1]"]],
            }
        else:
            # The viewing width divided by the viewing height.
            starting_ratio = 2.4
            y_range = max_yxz[0] - min_yxz[0]
            # View the data by fitting all the x data in the view.
            new_data = {"xaxis": [min_yxz[1], min_yxz[1] + y_range * starting_ratio], "yaxis": [min_yxz[0], max_yxz[0]]}
        return new_data

    app.run(debug=debug, host="localhost", port="8080")
