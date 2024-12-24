import numbers
import os

import dask.array
import dask.config
import nd2
import numpy as np
import tqdm

from ..setup.notebook_page import NotebookPage
from .raw_reader import RawReader


class JobsReader(RawReader):
    """
    Reader for raw JOBS file format.
    """

    # TODO: Unit test the JOBS reader.
    def read(
        self, nbp_basic: NotebookPage, nbp_file: NotebookPage, tile: int, round: int, channels: list[int]
    ) -> np.ndarray:
        super().read(nbp_basic, nbp_file, tile, round, channels)

        tile_raw = self.get_tile_raw_index(tile, nbp_basic.tilepos_yx_nd2, nbp_basic.tilepos_yx)
        round_dask_array = self._load_as_dask(nbp_basic, nbp_file, round)

        result = tuple()
        # Need the dask config to silence warning.
        with dask.config.set(**{"array.slicing.split_large_chunks": False}):
            for channel in channels:
                result += (np.asarray(round_dask_array[tile_raw, channel, :, :, nbp_basic.use_z]),)
        result = np.array(result)

        return result

    def _load_as_dask(self, nbp_basic: NotebookPage, nbp_file: NotebookPage, round: int) -> dask.array.Array:
        assert type(nbp_basic) is NotebookPage
        assert type(nbp_file) is NotebookPage
        assert isinstance(round, numbers.Number)

        channel_laser = super().get_channel_laser(nbp_basic)
        channel_cam = super().get_channel_laser(nbp_basic)

        n_tiles, tile_sz, nz = nbp_basic.n_tiles, nbp_basic.tile_sz, nbp_basic.nz
        n_lasers, n_cams = max(1, len(np.unique(channel_laser))), max(1, len(np.unique(channel_cam)))

        # TODO: Combine all metadata from the jobs nd2 files.
        # Now deal with the case where files are split by laser.
        if nbp_basic.use_anchor:
            # Always have anchor after imaging rounds.
            round_files = nbp_file.round + [nbp_file.anchor]
        else:
            round_files = nbp_file.round

        round_laser_dask_array = []
        # TODO: These big branches are awfully hard to read and bug prone.
        # It would be ideal to minimise the size of these branches.

        # Deal with non anchor round first as this follows a different format to anchor round.
        if round != nbp_basic.anchor_round:

            for t in tqdm(range(n_tiles), desc="Loading tiles as dask array"):
                # Get all the files of a given tiles (should be 7)
                tile_files = round_files[round][t * n_lasers : (t + 1) * n_lasers]
                tile_dask_array = []

                for f in tile_files:
                    laser_file = os.path.join(nbp_file.input_dir, f + ".nd2")
                    tile_dask_array.append(nd2.load(laser_file))

                tile_da = dask.array.concatenate(tile_dask_array, axis=-1)  # concatenate on the laser axis
                tile_da = dask.array.swapaxes(tile_da, -1, 0)  # we need 'channel', 'z', 'y','x'

                round_laser_dask_array.append(tile_da)

        # If we're dealing with the anchor round, we only need channel 0 (the DAPI) and the anchor.
        # The rest of the array is padded with zeros.
        else:
            anchor_laser_index = channel_laser.index(nbp_basic.channel_laser[nbp_basic.anchor_channel])
            dapi_laser_index = channel_laser.index(nbp_basic.channel_laser[nbp_basic.dapi_channel])

            anchor_files = nbp_file.anchor

            for t in tqdm.tqdm(range(n_tiles), desc="Loading tiles in dask array"):
                # Get all the files of a given tiles (should be 7)
                tile_files = anchor_files[t * n_lasers : (t + 1) * n_lasers]
                tile_dask_array = []
                latest_shape = (nz + 1, tile_sz, tile_sz, n_cams)

                for f_id, f in enumerate(tile_files):
                    if f_id == anchor_laser_index or f_id == dapi_laser_index:
                        laser_file = os.path.join(nbp_file.input_dir, f + ".nd2")
                        new_dask_array = nd2.load(laser_file)
                        latest_shape = new_dask_array.shape
                        tile_dask_array.append(new_dask_array)
                        del new_dask_array
                    else:
                        tile_dask_array.append(dask.array.zeros(latest_shape, dtype=np.uint16))
                        # TODO: find a better fix for nz. Here it is different because of basic_info use_z
                        # Ideally it should have the same shape as the array for dapi.

                tile_da = dask.array.concatenate(tile_dask_array, axis=-1)  # concatenate on the laser axis
                tile_da = dask.array.swapaxes(tile_da, -1, 0)  # we need 'channel', 'z', 'y','x'

                round_laser_dask_array.append(tile_da)

        # Now concatenate dask arrays.
        round_dask_array = dask.array.stack(round_laser_dask_array, axis=0)

        return round_dask_array, None
