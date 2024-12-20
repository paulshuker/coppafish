import importlib.resources as importlib_resources
import os

from .. import log, utils
from ..setup.config import Config
from ..setup.notebook_page import NotebookPage
from .tile_details import get_tile_file_names


def get_file_names(nbp_basic_info: NotebookPage, config_path: str) -> NotebookPage:
    """
    Function to set add `file_names` page to notebook. It requires notebook to be able to access a
    config file containing a `file_names` section and also the notebook to contain a `basic_info` page.

    !!! note
        This will be called every time the notebook is loaded to deal will case when `file_names` section of
        config file changed.

    Args:
        nbp_basic_info (NotebookPage): `basic_info` notebook page.
        config_path (str): file path to the config.

    Returns:
        (NotebookPage): nbp_file. `file_names` notebook page.
    """
    config = Config()
    config.load(config_path, post_check=False)
    config = config["file_names"]
    nbp = NotebookPage("file_names", {config.name: config.to_dict()})
    # Copy some variables that are in config to page.
    nbp.input_dir = config["input_dir"]
    nbp.output_dir = config["output_dir"]
    nbp.extract_dir = os.path.join(config["tile_dir"], "extract")
    nbp.fluorescent_bead_path = config["fluorescent_bead_path"]

    # Remove file extension from round and anchor file names if it is present.
    if config["raw_extension"] == "jobs":
        all_files: list[str] = os.listdir(config["input_dir"])
        all_files.sort()  # Sort files by ascending number
        n_rounds = len(nbp_basic_info.use_rounds)
        # FIXME: What is this magical number 8 here?
        n_tiles = int(len(all_files) / (n_rounds * 8))

        round = []
        for r in range(n_rounds):
            r_files = all_files[n_tiles * r * n_rounds : n_tiles * (r + 1) * n_rounds]
            round.append([file.replace(".nd2", "") for file in r_files])

        config["anchor"] = tuple([file.replace(".nd2", "") for file in all_files[n_tiles * n_rounds * n_rounds :]])
    else:
        if config["round"] is None:
            if config["anchor"] is None:
                log.error(ValueError("Neither imaging rounds nor anchor_round provided"))
            config["round"] = tuple()  # Sometimes the case where just want to run the anchor round.
        config["round"] = tuple([r.replace(config["raw_extension"], "") for r in config["round"]])

        if config["anchor"] is not None:
            config["anchor"] = config["anchor"].replace(config["raw_extension"], "")

    nbp.round = config["round"]
    nbp.anchor = config["anchor"]
    nbp.raw_extension = config["raw_extension"]
    nbp.raw_metadata = config["raw_metadata"]
    nbp.initial_bleed_matrix = config["initial_bleed_matrix"]
    nbp.omp_mean_spot = config["omp_mean_spot"]

    if config["code_book"] is not None:
        config["code_book"] = config["code_book"].replace(".txt", "")
        nbp.code_book = config["code_book"] + ".txt"
    else:
        # If the user has not put their code_book in, default to the one included in this project
        config["code_book"] = os.path.join(os.getcwd(), "coppafish/setup/code_book_73g.txt")

    if config["psf"] is None:
        config["psf"] = str(importlib_resources.files("coppafish.setup").joinpath("default_psf.npz"))
    nbp.psf = config["psf"]

    if config["anchor"] is not None:
        round_files = config["round"] + (config["anchor"],)
    else:
        round_files = config["round"]

    if config["raw_extension"] == "jobs":
        round_files = config["round"] + [config["anchor"]]
        _, tile_names_unfiltered = get_tile_file_names(
            "",
            nbp.extract_dir,
            round_files,
            nbp_basic_info.n_tiles,
            ".zarr",
            nbp_basic_info.n_channels,
            jobs=True,
        )
    else:
        _, tile_names_unfiltered = get_tile_file_names(
            "",
            nbp.extract_dir,
            round_files,
            nbp_basic_info.n_tiles,
            ".zarr",
            nbp_basic_info.n_channels,
        )
    nbp.tile_unfiltered = utils.base.deep_convert(tile_names_unfiltered.tolist())

    return nbp
