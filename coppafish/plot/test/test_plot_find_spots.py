from os import path

import pytest

from coppafish import Notebook
from coppafish.plot import find_spots


@pytest.mark.notebook
def test_view_find_spots() -> None:
    nb_path = path.dirname(path.dirname(path.dirname(__file__)))
    nb_path = path.join(nb_path, "robominnie", "test", ".integration_dir", "output_coppafish", "notebook")
    nb = Notebook(nb_path)
    # We cannot test all the dash app internal functionality like this. But, this can make sure the app can be built.
    find_spots.view_find_spots(nb, debug=True)
