import os
import time
import numpy as np
from tqdm import tqdm
from typing import Tuple

from .. import utils, log


def wait_for_data(data_path: str, wait_time: int, dir: bool = False):
    """
    Waits for wait_time seconds to see if file/directory at data_path becomes available in that time.

    Args:
        data_path: Path to file or directory of interest
        wait_time: Time to wait in seconds for file to become available.
        dir: If True, assumes data_path points to a directory, otherwise assumes points to a file.
    """
    if dir:
        check_data_func = lambda x: os.path.isdir(x)
    else:
        check_data_func = lambda x: os.path.isfile(x)
    if not check_data_func(data_path):
        # wait for file to become available
        if wait_time > 60**2:
            wait_time_print = round(wait_time / 60**2, 1)
            wait_time_unit = "hours"
        else:
            wait_time_print = round(wait_time, 1)
            wait_time_unit = "seconds"
        log.warn(f"\nNo file named\n{data_path}\nexists. Waiting for {wait_time_print} {wait_time_unit}...")
        with tqdm(total=wait_time, position=0) as pbar:
            pbar.set_description(f"Waiting for {data_path}")
            for i in range(wait_time):
                time.sleep(1)
                if check_data_func(data_path):
                    break
                pbar.update(1)
        pbar.close()
        if not check_data_func(data_path):
            log.error(utils.errors.NoFileError(data_path))
        log.info("file found!\nWaiting for file to fully load...")
        # wait for file to stop loading
        old_bytes = 0
        new_bytes = 0.00001
        while new_bytes > old_bytes:
            time.sleep(5)
            old_bytes = new_bytes
            new_bytes = os.path.getsize(data_path)
        log.info("file loaded!")


def get_pixel_length(length_microns: float, pixel_size: float) -> int:
    """
    Converts a length in units of microns into a length in units of pixels

    Args:
        length_microns: Length in units of microns (microns)
        pixel_size: Size of a pixel in microns (microns/pixels)

    Returns:
        Desired length in units of pixels (pixels)

    """
    return int(round(length_microns / pixel_size))


def strip_hack(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds all columns in image where each row is identical and then sets
    this column to the nearest normal column. Basically 'repeat padding'.

    Args:
        image: ```float [n_y x n_x (x n_z)]```
            Image from nd2 file, before filtering (can be after focus stacking) and if 3d, last index must be z.

    Returns:
        - ```image``` - ```float [n_y x n_x (x n_z)]```
            Input array with change_columns set to nearest
        - ```change_columns``` - ```int [n_changed_columns]```
            Indicates which columns have been changed.
    """
    # all rows identical if standard deviation is 0
    if np.ndim(image) == 3:
        # assume each z-plane of 3d image has same bad columns
        # seems to always be the case for our data
        change_columns = np.where(np.std(image[:, :, 0], 0) == 0)[0]
    else:
        change_columns = np.where(np.std(image, 0) == 0)[0]
    good_columns = np.setdiff1d(np.arange(np.shape(image)[1]), change_columns)
    for col in change_columns:
        nearest_good_col = good_columns[np.argmin(np.abs(good_columns - col))]
        image[:, col] = image[:, nearest_good_col]
    return image, change_columns
