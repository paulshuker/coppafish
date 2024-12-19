import itertools
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

from ..find_spots import detect
from ..setup.notebook import Notebook


def view_find_spots(nb: Notebook, debug: bool = False) -> None:
    """
    View the detected spots from the find spots stage on top of the filtered image.

    Args:
        - nb (Notebook): notebook containing `find_spots` page.
        - debug (bool, optional): run the app continuously after it is built. Default: true.
    """
    page_names_required = ("basic_info", "filter", "find_spots")
    for page_name in page_names_required:
        if not nb.has_page(page_name):
            raise ValueError(f"The notebook does not contain required page: {page_name}")
    anchor_round: int = nb.basic_info.anchor_round
    dapi_channel: int = nb.basic_info.dapi_channel
    use_tiles: list[int] = nb.basic_info.use_tiles
    valid_rounds: list[int] = nb.basic_info.use_rounds + [anchor_round]
    valid_channels: list[int] = nb.basic_info.use_channels
    tile = use_tiles[0]
    round = nb.basic_info.anchor_round
    channel = nb.basic_info.anchor_channel

    def r_to_str(r: int) -> str:
        return "anchor" if r == anchor_round else str(r)

    def c_to_str(c: int) -> str:
        return "dapi" if c == dapi_channel else str(c)

    config = nb.find_spots.associated_configs["find_spots"]
    auto_thresholds = nb.find_spots.auto_thresh
    default_auto_thresh_multiplier = float(config["auto_thresh_multiplier"])
    max_auto_thresh_multiplier = default_auto_thresh_multiplier * 5
    default_radius_xy = int(config["radius_xy"])
    max_radius_xy = max(5 * default_radius_xy, 6)
    default_radius_z = int(config["radius_z"])
    max_radius_z = max(5 * default_radius_z, 10)
    max_z_plane = 7
    # Gather a central square from the filtered images no larger than 250x250 pixels.
    max_xy_pixels = 250
    xy_pixels = min(max_xy_pixels, nb.filter.images.shape[3])
    mid_xy = nb.basic_info.tile_sz // 2
    xy_slice = slice(mid_xy - xy_pixels // 2, mid_xy + xy_pixels // 2, 1)
    new_xy_slice = slice(max_xy_pixels // 2 - xy_pixels // 2, max_xy_pixels // 2 + xy_pixels // 2, 1)
    default_z_plane = max_z_plane // 2
    trc_filter_images = np.zeros(
        (len(use_tiles), len(valid_rounds), len(valid_channels), max_xy_pixels, max_xy_pixels, max_z_plane),
        dtype=np.float16,
    )
    central_z = nb.filter.images.shape[5] // 2
    for (t_i, t), (r_i, r), (c_i, c) in itertools.product(
        enumerate(use_tiles), enumerate(valid_rounds), enumerate(valid_channels)
    ):
        for z_i, z in enumerate(range(central_z - max_z_plane // 2, central_z + max_z_plane // 2)):
            if z not in nb.basic_info.use_z:
                continue
            trc_filter_images[t_i, r_i, c_i, new_xy_slice, new_xy_slice, z_i] = nb.filter.images[
                t, r, c, xy_slice, xy_slice, z
            ]
    del nb

    label_style = {
        "textAlign": "center",  # Center text horizontally
        "width": "100%",  # Ensure the label takes the full width of its container
        "display": "block",  # Make the label a block element
        "fontFamily": "Arial",
    }

    app = Dash(__name__, title="Coppapy: Find Spots")
    app.layout = [
        html.Div(
            children=[
                html.Img(
                    src="https://raw.githubusercontent.com/paulshuker/coppafish/b3dda7925a8fea63c6ccd050fc04ad10a22af9c0/docs/images/logo.svg",
                    style={"width": "30px", "padding": "0 5px"},
                ),
                html.H1(
                    "Find Spots",
                    id="title",
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
        dcc.Graph(
            id="view",
            style=dict(display="block", width="99%", height="80vh"),
            config=dict(modeBarButtonsToRemove=["select2d", "lasso2d", "select2d", "autoscale"]),
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Z Plane", style=label_style),
                        dcc.Slider(0, max_z_plane - 1, 1, id="z-plane-slider", value=default_z_plane),
                    ],
                    style=dict(width="15vw"),
                ),
                html.Div(
                    [
                        html.Label("Auto Threshold Multiplier", style=label_style),
                        dcc.Slider(
                            0.0,
                            max_auto_thresh_multiplier,
                            id="auto-thresh-multiplier-slider",
                            value=default_auto_thresh_multiplier,
                        ),
                    ],
                    style=dict(width="15vw"),
                ),
                html.Div(
                    [
                        html.Label("Radius XY", style=label_style),
                        dcc.Slider(1, max_radius_xy, id="radius-xy-slider", value=default_radius_xy),
                    ],
                    style=dict(width="15vw"),
                ),
                html.Div(
                    [
                        html.Label("Radius Z", style=label_style),
                        dcc.Slider(1, max_radius_z, id="radius-z-slider", value=default_radius_z),
                    ],
                    style=dict(width="15vw"),
                ),
                html.Div(
                    [
                        html.Label("Marker Size", style=label_style),
                        dcc.Slider(1, 30, id="marker-size-slider", value=6),
                    ],
                    style=dict(width="15vw"),
                ),
                html.Div(
                    [
                        dcc.Checklist(["Remove Duplicates"], value=["Remove Duplicates"], id="remove-duplicates"),
                    ],
                    style=dict(width="15vw", fontFamily="Arial"),
                ),
            ],
            style={"display": "flex", "flex-direction": "row", "flexwrap": "wrap", "justifyContent": "center"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Tile", style=label_style),
                        dcc.Slider(
                            min(use_tiles),
                            max(use_tiles),
                            None,
                            marks={t: str(t) for t in use_tiles},
                            id="tile-slider",
                            value=tile,
                        ),
                    ],
                    style=dict(width="30vw"),
                ),
                html.Div(
                    [
                        html.Label("Round", style=label_style),
                        dcc.Slider(
                            min(valid_rounds),
                            max(valid_rounds),
                            None,
                            marks={r: r_to_str(r) for r in valid_rounds},
                            id="round-slider",
                            value=round,
                        ),
                    ],
                    style=dict(width="30vw"),
                ),
                html.Div(
                    [
                        html.Label("Channel", style=label_style),
                        dcc.Slider(
                            min(valid_channels),
                            max(valid_channels),
                            None,
                            marks={c: c_to_str(c) for c in valid_channels},
                            id="channel-slider",
                            value=channel,
                        ),
                    ],
                    style=dict(width="30vw"),
                ),
            ],
            style={"display": "flex", "flex-direction": "row", "flexwrap": "wrap", "justifyContent": "center"},
        ),
    ]

    @app.callback(
        Output("view", "figure"),
        Input("tile-slider", "value"),
        Input("round-slider", "value"),
        Input("channel-slider", "value"),
        Input("z-plane-slider", "value"),
        Input("auto-thresh-multiplier-slider", "value"),
        Input("radius-xy-slider", "value"),
        Input("radius-z-slider", "value"),
        Input("marker-size-slider", "value"),
        Input("remove-duplicates", "value"),
    )
    def update_view(
        tile: int,
        round: int,
        channel: int,
        z_plane: int,
        auto_thresh_multiplier: float,
        radius_xy: float,
        radius_z: float,
        marker_size: float,
        remove_duplicates: Optional[list[str]],
    ):
        selected_t_i = use_tiles.index(tile)
        selected_r_i = valid_rounds.index(round)
        selected_c_i = valid_channels.index(channel)
        remove_duplicates: bool = len(remove_duplicates) != 0
        current_auto_thresh = (
            auto_thresh_multiplier * auto_thresholds.item(tile, round, channel) / default_auto_thresh_multiplier
        )
        fig = go.Figure()
        fig.update_layout(
            title=f"Tile: {t}, round: {r_to_str(r)}, channel: {c_to_str(c)}, Threshold: {current_auto_thresh}",
            xaxis_title="X",
            yaxis_title="Y",
            xaxis=dict(scaleanchor="y"),  # Link x-axis and y-axis scales
            yaxis=dict(constrain="domain"),  # Prevent stretching
            autosize=True,
            template="plotly",
            dragmode="pan",
        )
        # Add the 2D image as a heatmap
        z_image = trc_filter_images[selected_t_i, selected_r_i, selected_c_i, :, :, z_plane]
        fig.add_trace(
            go.Heatmap(
                z=z_image, colorscale="Viridis", colorbar=dict(title="Intensity"), showscale=True, hoverinfo="none"
            )
        )
        if current_auto_thresh <= 0 or np.allclose(z_image, 0):
            return fig
        # Detect spots and add them as scatter points.
        local_yxz, _ = detect.detect_spots(
            trc_filter_images[selected_t_i, selected_r_i, selected_c_i],
            current_auto_thresh,
            remove_duplicates=remove_duplicates,
            radius_xy=radius_xy,
            radius_z=radius_z,
        )
        in_z = local_yxz[:, 2] == z_plane
        fig.add_trace(
            go.Scatter(
                x=local_yxz[in_z, 1],
                y=local_yxz[in_z, 0],
                mode="markers",
                marker=dict(size=marker_size, color="red", opacity=0.8),
                hoverinfo="none",
            )
        )
        return fig

    if debug:
        return

    app.run(debug=True)


def n_spots_grid(nb: Notebook, n_spots_thresh: Optional[int] = None):
    """
    Plots a grid indicating the number of spots detected on each tile, round and channel.

    Args:
        nb: *Notebook* containing `find_spots` page.
        n_spots_thresh: tiles/rounds/channels with fewer spots than this will be highlighted.
            If `None`, will use `n_spots_warn_fraction` from config file.
    """
    if n_spots_thresh is None:
        config = nb.get_config()["find_spots"]
        if nb.basic_info.is_3d:
            n_spots_thresh = config["n_spots_warn_fraction"] * config["max_spots_3d"] * nb.basic_info.nz
        else:
            n_spots_thresh = config["n_spots_warn_fraction"] * config["max_spots_2d"]
        n_spots_thresh = int(np.ceil(n_spots_thresh))
    use_tiles = np.asarray(nb.basic_info.use_tiles)
    use_rounds = np.asarray(nb.basic_info.use_rounds)  # don't consider anchor in this analysis
    use_channels = np.asarray(nb.basic_info.use_channels)
    spot_no = nb.find_spots.spot_no[np.ix_(use_tiles, use_rounds, use_channels)]
    spot_no = np.moveaxis(spot_no, 1, 2)  # put rounds last
    spot_no = spot_no.reshape(len(use_tiles), -1).T  # spot_no[n_rounds, t] is now spot_no[t, r=0, c=1]
    n_round_channels = len(use_rounds) * len(use_channels)
    y_labels = np.tile(use_rounds, len(use_channels))
    vmax = spot_no.max()  # clip colorbar at max of imaging rounds/channels because anchor can be a lot higher
    if nb.basic_info.use_anchor:
        anchor_spot_no = nb.find_spots.spot_no[use_tiles, nb.basic_info.anchor_round, nb.basic_info.anchor_channel][
            np.newaxis
        ]
        spot_no = np.append(spot_no, anchor_spot_no, axis=0)
        y_labels = y_labels.astype(str).tolist()
        y_labels += ["Anchor"]
        n_round_channels += 1
    fig, ax = plt.subplots(1, 1, figsize=(np.clip(5 + len(use_tiles) / 2, 3, 18), 12))
    subplot_adjust = [0.12, 1, 0.07, 0.9]
    fig.subplots_adjust(
        left=subplot_adjust[0], right=subplot_adjust[1], bottom=subplot_adjust[2], top=subplot_adjust[3]
    )
    im = ax.imshow(spot_no, aspect="auto", vmax=vmax)
    fig.colorbar(im, ax=ax)
    ax.set_yticks(np.arange(n_round_channels))
    ax.set_yticklabels(y_labels)
    ax.set_xticks(np.arange(len(use_tiles)))
    ax.set_xticklabels(use_tiles)
    ax.set_xlabel("Tile")

    for c in range(len(use_channels)):
        y_ind = 1 - c * 1 / (len(use_channels))
        ax.text(
            -0.1, y_ind, f"Channel {use_channels[c]}", va="top", ha="left", transform=ax.transAxes, rotation="vertical"
        )
    fig.supylabel("Channel/Round", transform=ax.transAxes, x=-0.15)
    plt.xticks(rotation=90)
    low_spots = np.where(spot_no < n_spots_thresh)
    for j in range(len(low_spots[0])):
        rectangle = plt.Rectangle(
            (low_spots[1][j] - 0.5, low_spots[0][j] - 0.5), 1, 1, fill=False, ec="r", linestyle=":", lw=2
        )
        ax.add_patch(rectangle)
    plt.suptitle("Number of Spots Found on each Tile, Round and Channel")
    plt.show()
