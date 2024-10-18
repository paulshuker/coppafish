import os
import tempfile

import matplotlib
import numpy as np
import zarr

from coppafish.plot.results_viewer.base_new import Viewer
from coppafish.setup.notebook_page import NotebookPage


def test_Viewer() -> None:
    # Using a headless backend to support headless unit testings of the Viewer..
    matplotlib.use("Agg")

    rng = np.random.RandomState(0)

    n_tiles = 4
    use_tiles = tuple(range(n_tiles))
    use_z = tuple(range(8))
    tile_sz = 15
    n_genes = 12
    n_ref_spots = 45
    n_rounds_use = 5
    n_channels_use = 9

    nbp_basic = NotebookPage("basic_info")
    nbp_basic.dye_names = tuple([f"Dye_{i}" for i in range(n_channels_use)])
    nbp_basic.tile_sz = tile_sz
    nbp_basic.use_tiles = use_tiles
    nbp_basic.use_z = use_z
    nbp_basic.use_rounds = tuple(range(n_rounds_use))
    nbp_basic.use_channels = tuple(range(n_channels_use))
    nbp_filter = NotebookPage("filter")
    nbp_filter.images = zarr.array(
        200 * rng.rand(n_tiles, n_rounds_use, n_channels_use, tile_sz, tile_sz, len(use_z)).astype(np.float16) - 50
    )
    nbp_register = NotebookPage("register")
    nbp_register.flow = zarr.array(rng.rand(n_tiles, n_rounds_use, 3, tile_sz, tile_sz, len(use_z)).astype(np.float16))
    affine = np.zeros((n_tiles, n_rounds_use, n_channels_use, 4, 3))
    affine[:, :, :, :3] = np.eye(3)
    nbp_register.icp_correction = affine
    nbp_stitch = NotebookPage("stitch")
    tile_origin = np.zeros((n_tiles, 3), np.float32)
    for t in range(n_tiles):
        # All tiles align along the x axis.
        tile_origin[t, 1] = tile_sz * t
        # Random offsets.
        tile_origin[t] += rng.rand()
    nbp_stitch.tile_origin = tile_origin
    dapi_image_shape = (
        (tile_origin.max(0)[[2, 0, 1]] + np.array([len(use_z), tile_sz, tile_sz], int)).astype(int).tolist()
    )
    nbp_stitch.dapi_image = zarr.array(rng.rand(*dapi_image_shape).astype(np.float16))
    nbp_ref_spots = NotebookPage("ref_spots")
    nbp_ref_spots.local_yxz = zarr.array(rng.randint(low=0, high=max(use_z), size=(n_ref_spots, 3), dtype=np.int16))
    nbp_ref_spots.tile = zarr.array(rng.randint(n_tiles, size=n_ref_spots, dtype=np.int16))
    nbp_ref_spots.colours = zarr.array(rng.rand(n_ref_spots, n_rounds_use, n_channels_use).astype(np.float32))
    nbp_call_spots = NotebookPage("call_spots")
    # Using gene names used in the default gene marker file as a test.
    nbp_call_spots.gene_names = np.array(
        ["Snca", "Npy", "Chodl", "unknown", "Reln", "Map2", "Cdh13", "Plcg2", "Pld4", "Vim", "Fam19a1", "unknown_2"]
    )
    nbp_call_spots.colour_norm_factor = rng.rand(n_tiles, n_rounds_use, n_channels_use).astype(np.float32) + 1
    bled_codes = np.ones((n_genes, n_rounds_use, n_channels_use), np.float32)
    bled_codes[0] = 1.5
    bled_codes[9] = 4
    # Normalise bled codes as this is checked by other functions.
    bled_codes /= np.linalg.norm(bled_codes, axis=(1, 2), keepdims=True)
    nbp_call_spots.bled_codes = bled_codes
    nbp_call_spots.gene_probabilities = zarr.array(rng.rand(n_ref_spots, n_genes).astype(np.float32))
    nbp_call_spots.intensity = zarr.array(rng.rand(n_ref_spots) + 0.5).astype(np.float32)
    nbp_call_spots.dot_product_gene_no = zarr.array(rng.randint(n_genes, size=n_ref_spots, dtype=np.int16))
    nbp_call_spots.dot_product_gene_score = zarr.array(rng.rand(n_ref_spots)).astype(np.float16)
    assert nbp_call_spots.gene_names.size == n_genes
    nbp_call_spots.bleed_matrix_raw = rng.rand(n_channels_use, n_channels_use).astype(np.float32)
    nbp_call_spots.bleed_matrix_initial = rng.rand(n_channels_use, n_channels_use).astype(np.float32)
    nbp_call_spots.bleed_matrix = rng.rand(n_channels_use, n_channels_use).astype(np.float32)
    nbp_omp = None

    npy_filepath = os.path.join(tempfile.gettempdir(), "test_array.npy")
    background_array = rng.rand(25, 31)
    np.save(npy_filepath, background_array)
    npz_filepath = os.path.join(tempfile.gettempdir(), "test_array.npz")
    background_array = rng.randint(0, 100, size=(8, 25, 31))
    np.savez_compressed(npz_filepath, background_array)

    # Try different background image valid parameters.
    background_images = []
    background_images.append("dapi")
    background_images.append(None)
    background_images.append(npy_filepath)
    background_images.append(npz_filepath)
    for background_image in background_images:
        viewer = Viewer(
            background_image=background_image,
            gene_legend_order_by="row",
            nbp_basic=nbp_basic,
            nbp_filter=nbp_filter,
            nbp_register=nbp_register,
            nbp_stitch=nbp_stitch,
            nbp_ref_spots=nbp_ref_spots,
            nbp_call_spots=nbp_call_spots,
            nbp_omp=nbp_omp,
            show=False,
        )
        # Test every hotkey.
        for hotkey in viewer.hotkeys:
            viewer.selected_spot = 20
            hotkey.invoke(None)
            viewer.clear_spot_selections()
        viewer.close_all_subplots()
        viewer.close()
    viewer = Viewer(
        background_image="dapi",
        nbp_basic=nbp_basic,
        nbp_filter=nbp_filter,
        nbp_register=nbp_register,
        nbp_stitch=nbp_stitch,
        nbp_ref_spots=nbp_ref_spots,
        nbp_call_spots=nbp_call_spots,
        nbp_omp=nbp_omp,
        show=False,
    )
    # Test every hotkey.
    for hotkey in viewer.hotkeys:
        viewer.selected_spot = 20
        hotkey.invoke(None)
        viewer.close_all_subplots()
        viewer.clear_spot_selections()
        viewer.clear_spot_selections()
    viewer.close()

    n_omp_spots = 85 // n_tiles
    omp_config = {"omp": {"max_genes": 2, "dp_thresh": 0.01, "lambda_d": 0.001}}
    nbp_omp = NotebookPage("omp", omp_config)
    spot_shape = (9, 9, 5)
    mean_spot = rng.rand(*spot_shape).astype(np.float32)
    mean_spot[tuple([dim // 2 for dim in spot_shape])] = 1
    nbp_omp.mean_spot = mean_spot
    temp_zgroup = tempfile.TemporaryDirectory()
    group = zarr.group(store=temp_zgroup.name, zarr_version=2)
    for t in use_tiles:
        subgroup = group.create_group(f"tile_{t}")
        subgroup.create_dataset("local_yxz", shape=(n_omp_spots, 3), dtype=np.int16)
        subgroup.create_dataset("scores", shape=(n_omp_spots), dtype=np.float16)
        subgroup.create_dataset("gene_no", shape=(n_omp_spots), dtype=np.int16)
        subgroup.create_dataset("colours", shape=(n_omp_spots, n_rounds_use, n_channels_use), dtype=np.float16)
        subgroup.local_yxz[:] = np.vstack(
            (
                rng.randint(tile_sz, size=n_omp_spots),
                rng.randint(tile_sz, size=n_omp_spots),
                rng.randint(max(use_z), size=n_omp_spots),
            ),
            dtype=np.int16,
        ).T
        subgroup.scores[:] = rng.rand(n_omp_spots).astype(np.float16)
        subgroup.gene_no[:] = rng.randint(n_genes, size=n_omp_spots).astype(np.int16)
        subgroup.colours[:] = rng.rand(*subgroup.colours.shape).astype(np.float16)
    nbp_omp.results = group
    viewer = Viewer(
        background_image="dapi",
        nbp_basic=nbp_basic,
        nbp_filter=nbp_filter,
        nbp_register=nbp_register,
        nbp_stitch=nbp_stitch,
        nbp_ref_spots=nbp_ref_spots,
        nbp_call_spots=nbp_call_spots,
        nbp_omp=nbp_omp,
        show=False,
    )
    for method in ("prob", "anchor", "omp"):
        viewer.selected_method = method
        viewer.clear_spot_selections()
        viewer.selected_spot = 10
        viewer.clear_spot_selections()
        viewer.clear_spot_selections()
        viewer.selected_spot = 10
        # Check clicking the gene legend.
        for _ in range(10):
            event = type("Event", (object,), {})()
            event.inaxes = viewer.legend.canvas.axes
            event.xdata = rng.rand()
            event.ydata = rng.rand()
            button = type("Button", (object,), {})()
            button.name = "LEFT" if rng.randint(2) == 0 else "RIGHT"
            event.button = button
            viewer.legend_clicked(event)
        viewer.close_all_subplots()
        # Test every hotkey.
        for hotkey in viewer.hotkeys:
            viewer.selected_spot = 20
            hotkey.invoke(None)
            hotkey.invoke(None)
            viewer.close_all_subplots()
            viewer.clear_spot_selections()
            viewer.selected_spot = 5
            hotkey.invoke(None)
            viewer.clear_spot_selections()
    viewer.close_all_subplots()
    viewer.close()

    temp_zgroup.cleanup()
