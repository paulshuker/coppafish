import os
from typing import Tuple

from .. import log
from ..compatibility import base as utils_version
from ..find_spots import check_spots
from ..pdf.base import BuildPDF
from ..setup import file_names
from ..setup.config import Config
from ..setup.notebook import Notebook
from ..setup.notebook_page import NotebookPage
from ..utils import system as utils_system
from ..utils import tiles_io as utils_tiles_io
from . import (
    basic_info,
    call_reference_spots,
    extract_run,
    filter_run,
    find_spots,
    get_reference_spots,
    omp_torch,
    register,
    stitch,
)


def run_pipeline(config_file: str) -> Notebook:
    """
    Run every step of the pipeline.

    Args:
        - config_file (str): path to config file.

    Returns:
        Notebook: notebook containing all information gathered during the pipeline.
    """
    utils_tiles_io.set_zarr_global_configs()
    nb, nbp_file, config = initialize_notebook(config_file)

    if not nb.has_page("extract"):
        nbp = log.error_catch(extract_run.run_extract, config["extract"], nbp_file, nb.basic_info)
        nb += nbp
    log.error_catch(BuildPDF, nb, nbp_file)

    if not nb.has_page("filter") or not nb.has_page("filter_debug"):
        nbp, nbp_debug = log.error_catch(filter_run.run_filter, config["filter"], nbp_file, nb.basic_info)
        nb += nbp
        nb += nbp_debug
    log.error_catch(BuildPDF, nb, nbp_file)

    if not nb.has_page("find_spots"):
        nbp = log.error_catch(find_spots.find_spots, config["find_spots"], nb.basic_info, nbp_file, nb.filter)
        nb += nbp
    log.error_catch(check_spots.check_n_spots, nb)
    log.error_catch(BuildPDF, nb, nbp_file)

    if not nb.has_page("register") or not nb.has_page("register_debug"):
        nbp, nbp_debug = log.error_catch(
            register.register, nb.basic_info, nbp_file, nb.filter, nb.find_spots, config["register"]
        )
        nb += nbp
        nb += nbp_debug
    log.error_catch(BuildPDF, nb, nbp_file)

    if not nb.has_page("stitch"):
        nbp = log.error_catch(stitch.stitch, config["stitch"], nb.basic_info, nbp_file, nb.filter)
        nb += nbp

    if not nb.has_page("ref_spots"):
        nbp = log.error_catch(
            get_reference_spots.get_reference_spots,
            nbp_basic=nb.basic_info,
            nbp_file=nbp_file,
            nbp_filter=nb.filter,
            nbp_find_spots=nb.find_spots,
            nbp_register=nb.register,
            nbp_stitch=nb.stitch,
        )
        nb += nbp
    if not nb.has_page("call_spots"):
        nbp = log.error_catch(
            call_reference_spots.call_reference_spots, config["call_spots"], nb.ref_spots, nbp_file, nb.basic_info
        )
        nb += nbp
    log.error_catch(BuildPDF, nb, nbp_file)
    if not nb.has_page("omp"):
        nbp = log.error_catch(
            omp_torch.run_omp,
            config["omp"],
            nbp_file,
            nb.basic_info,
            nb.extract,
            nb.filter,
            nb.register,
            nb.stitch,
            nb.call_spots,
        )
        nb += nbp
    log.error_catch(BuildPDF, nb, nbp_file)
    log.info("Pipeline complete", force_email=True, notify=config["notifications"]["notify_on_completion"])
    return nb


def initialize_notebook(config_path: str) -> Tuple[Notebook, NotebookPage, Config]:
    """
    Creates a `Notebook` and adds `basic_info` page before saving.
    If `Notebook` already exists with `basic_info`, it will be returned.
    An error is raised if an existing notebook is not compatible with the current coppafish version.

    Args:
        config_file: Path to config file.

    Returns:
        Tuple containing three items:
            - (Notebook): nb. Notebook containing `basic_info` page.
            - (NotebookPage): nbp_file. `file_names` notebook page.
            - (Config): config. The pipeline's config.
    """
    config = Config()
    config.load(config_path)
    config_file = config["file_names"]
    config_notify = config["notifications"]

    nb_path = os.path.join(config_file["output_dir"], config_file["notebook_name"])
    nb = Notebook(nb_path, config_path, must_exist=False)

    log.base.set_log_config(
        config_notify["minimum_print_severity"],
        os.path.join(config_file["output_dir"], config_notify["log_name"]),
        config_notify["allow_notifications"],
        config_notify["notify_on_crash"],
        config_notify["email_me"],
        config_notify["sender_email"],
        config_notify["sender_email_password"],
    )
    log.info("")
    log.info(
        f" COPPAFISH v{utils_system.get_software_version()} ".center(utils_system.get_terminal_size_xy(-33)[0], "=")
    )
    log.base.log_package_versions()

    # Check the notebook for backwards incompatibilities caused by old data.
    compatible_tracker = utils_version.CompatibilityTracker()
    if not compatible_tracker.notebook_is_compatible(nb):
        raise ValueError("The notebook contains incompatible data. Please see the log above for advice")

    online_version = utils_system.get_remote_software_version()
    if online_version != utils_system.get_software_version():
        log.warn(
            f"You are running v{utils_system.get_software_version()}. The latest online version is v{online_version}"
        )
    if not nb.has_page("basic_info"):
        nbp_basic = basic_info.set_basic_info(config)
        nb += nbp_basic
    nbp_file = file_names.get_file_names(nb.basic_info, config_path)
    return nb, nbp_file, config
