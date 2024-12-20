import os

import nd2
import numpy as np
from tqdm import tqdm

from .. import setup
from ..setup import tile_details

# bioformats ssl certificate error solution:
# https://stackoverflow.com/questions/35569042/ssl-certificate-verify-failed-with-python3


def get_raw_extension(input_dir: str) -> str:
    """
    Looks at input directory and returns the raw data format
    Args:
        input_dir: input_directory from config file containing raw data
    Returns:
        raw_extension: str, either 'nd2', 'npy' or 'jobs'
    """
    # Want to list all files in input directory and all subdirectories. We'll use os.walk
    files = []
    for root, directories, filenames in os.walk(input_dir):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    files.sort()
    # Just need a single npy to confirm this is the format
    if any([directory.endswith("npy") for directory in files]):
        raw_extension = ".npy"
    else:
        # Get the first nd2 file here
        index = min([i for i in range(len(files)) if files[i].endswith("nd2")])

        with nd2.ND2File(os.path.join(input_dir, files[index])) as image:
            if image.sizes["C"] == 28:
                raw_extension = ".nd2"
            else:
                raw_extension = "jobs"
    return raw_extension


def get_metadata(file_path: str, config: dict) -> dict:
    """
    Gets metadata containing information from nd2 data about pixel sizes, position of tiles and numbers of
    tiles/channels/z-planes.

    Args:
        file_path: path to desired nd2 file
        config: config dictionary

    Returns:
        Dictionary containing - n_tiles, n_channels, tile_sz, pixel_size_xy, pixel_size_z, tile_centre, xy_pos, nz,
        tilepos_yx_nd2, tilepos_yx, channel_laser, channel_camera, n_rounds

    """

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"No file with path {file_path}")

    with nd2.ND2File(file_path) as images:
        metadata = {
            "n_tiles": images.sizes["P"],
            "n_channels": images.sizes["C"],
            "tile_sz": images.sizes["X"],
            "pixel_size_xy": images.metadata.channels[0].volume.axesCalibration[0],
            "pixel_size_z": images.metadata.channels[0].volume.axesCalibration[2],
        }
        # Check if data is 3d
        if "Z" in images.sizes:
            # subtract 1 as we always ignore first z plane
            nz = images.sizes["Z"]
            metadata["tile_centre"] = np.array([metadata["tile_sz"], metadata["tile_sz"], nz]) / 2
        else:
            metadata["tile_centre"] = np.array([metadata["tile_sz"], metadata["tile_sz"]]) / 2

        xy_pos = np.array(
            [images.experiment[0].parameters.points[i].stagePositionUm[:2] for i in range(images.sizes["P"])]
        )
        xy_pos = (xy_pos - np.min(xy_pos, 0)) / metadata["pixel_size_xy"]
        metadata["xy_pos"] = xy_pos
        metadata["tilepos_yx_nd2"], metadata["tilepos_yx"] = tile_details.get_tilepos(
            xy_pos=xy_pos, tile_sz=metadata["tile_sz"], expected_overlap=config["stitch"]["expected_overlap"]
        )
        # Now also extract the laser and camera associated with each channel
        desc = images.text_info["description"]
        channel_metadata = desc.split("Plane #")[1:]
        laser = np.zeros(len(channel_metadata), dtype=int)
        camera = np.zeros(len(channel_metadata), dtype=int)
        for i in range(len(channel_metadata)):
            laser[i] = int(
                channel_metadata[i][channel_metadata[i].index("; On") - 3 : channel_metadata[i].index("; On")]
            )
            camera[i] = int(
                channel_metadata[i][channel_metadata[i].index("Name:") + 6 : channel_metadata[i].index("Name:") + 9]
            )
        metadata["channel_laser"] = laser.tolist()
        metadata["channel_camera"] = camera.tolist()
        # Get the entire input directory to list
        metadata["n_rounds"] = len(config["file_names"]["round"])
        metadata["nz"] = nz

    return metadata


def get_jobs_metadata(files: list, input_dir: str, config: dict) -> dict:
    """
    Gets metadata containing information from nd2 data about pixel sizes, position of tiles and numbers of
    tiles/channels/z-planes. This has to be as separate function from above due to the fact that input here is a list
    of directories as rounds are not entirely contained in a single file for jobs data.

    Args:
        files: list of paths to desired nd2 file
        input_dir: Directory to location of files
        config: config dictionary
    Returns:
        Dictionary containing -

        - `xy_pos` - `List [n_tiles x 2]`. xy position of tiles in pixels.
        - `pixel_microns` - `float`. xy pixel size in microns.
        - `pixel_microns_z` - `float`. z pixel size in microns.
        - `sizes` - dict with fov (`t`), channels (`c`), y, x, z-planes (`z`) dimensions.
        - 'channels' - list of colorRGB codes for the channels, this is a unique identifier for each channel
    """
    # Get simple metadata which is constant across tiles from first file
    with nd2.ND2File(os.path.join(input_dir, files[0])) as im:
        metadata = {
            "tile_sz": im.sizes["X"],
            "pixel_size_xy": im.metadata.channels[0].volume.axesCalibration[0],
            "pixel_size_z": im.metadata.channels[0].volume.axesCalibration[2],
        }
        # Check if data is 3d
        if "Z" in im.sizes:
            # subtract 1 as we always ignore first z plane
            nz = im.sizes["Z"]
            metadata["tile_centre"] = np.array([metadata["tile_sz"], metadata["tile_sz"], nz]) / 2
        else:
            # Our microscope setup is always square tiles
            metadata["tile_centre"] = np.array([metadata["tile_sz"], metadata["tile_sz"]]) / 2

    # Now loop through the files to get the more varied data
    xy_pos = []
    laser = []
    camera = []

    # Only want to extract metadata from round 0
    for f_id, f in tqdm(enumerate(files), desc="Reading metadata from all files"):
        with nd2.ND2File(os.path.join(input_dir, f)) as im:
            stage_position = [int(x) for x in im.frame_metadata(0).channels[0].position.stagePositionUm[:2]]
            # We want to append if this stage position is new
            # We also want to break if we have reached the end of the tiles. We expect xy_pos to be the same value for
            # file 0, ..., n_lasers - 1, then the next value for file n_lasers, ..., 2*n_lasers - 1, etc. But when we
            # reach the end of the tiles, we will eventually loop back to tile 0, so we want to break when we reach.
            if stage_position not in xy_pos:
                xy_pos.append(stage_position)
            all_tiles_complete = (stage_position in xy_pos) * (stage_position != xy_pos[-1])
            if all_tiles_complete:
                break
            cal = im.metadata.channels[0].volume.axesCalibration[0]
            # Now also extract the laser and camera associated with each channel
            desc = im.text_info["description"]
            channel_metadata = desc.split("Plane #")[1:]
            # Since channels constant across tiles only need to gauge from tile 1
            if stage_position == xy_pos[0]:
                for i in range(len(channel_metadata)):
                    laser_wavelength = int(
                        channel_metadata[i][channel_metadata[i].index("; On") - 3 : channel_metadata[i].index("; On")]
                    )
                    camera_wavelength = int(
                        channel_metadata[i][
                            channel_metadata[i].index("Name:") + 6 : channel_metadata[i].index("Name:") + 9
                        ]
                    )
                    laser.append(laser_wavelength)
                    camera.append(camera_wavelength)

    # Normalise so that minimum is 0,0
    xy_pos = np.array(xy_pos)
    xy_pos = (xy_pos - np.min(xy_pos, axis=0)) / cal
    metadata["xy_pos"] = xy_pos
    metadata["tilepos_yx_nd2"], metadata["tilepos_yx"] = setup.get_tilepos(
        xy_pos=xy_pos, tile_sz=metadata["tile_sz"], expected_overlap=config["stitch"]["expected_overlap"]
    )
    metadata["n_tiles"] = len(metadata["tilepos_yx_nd2"])
    # get n_channels and channel info
    metadata["channel_laser"], metadata["channel_camera"] = laser, camera
    metadata["n_channels"] = len(laser)
    # Final piece of metadata is n_rounds. Note num_files = num_rounds * num_tiles * num_lasers
    n_files = len(os.listdir(input_dir))
    n_lasers = len(set(laser))

    metadata["n_rounds"] = n_files // (n_lasers * metadata["n_tiles"])
    # TODO find a better solution to fix the number of rounds
    metadata["n_rounds"] -= 1
    metadata["nz"] = nz

    return metadata
