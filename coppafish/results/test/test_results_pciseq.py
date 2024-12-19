from os import path

import pytest

from ...setup.notebook import Notebook
from .. import pciseq


@pytest.mark.notebook
def test_export_to_pciseq() -> None:
    nb_path = path.dirname(path.dirname(path.dirname(__file__)))
    nb_path = path.join(nb_path, "robominnie", "test", ".integration_dir", "output_coppafish", "notebook")

    nb = Notebook(nb_path)

    # Check that export to pciseq does not crash and that a csv file appears for each gene call method.
    for method in ("prob", "anchor", "omp"):
        csv_file_path = pciseq.export_to_pciseq(nb, method)
        assert type(csv_file_path) is str
        assert path.isfile(csv_file_path)
        assert method in csv_file_path
        assert csv_file_path.endswith(".csv")
        csv_file_path = pciseq.export_to_pciseq(nb, method, 0.5)
        assert type(csv_file_path) is str
        assert path.isfile(csv_file_path)
        assert method in csv_file_path
        assert csv_file_path.endswith(".csv")
        csv_file_path = pciseq.export_to_pciseq(nb, method, 0.5, 0.7)
        assert type(csv_file_path) is str
        assert path.isfile(csv_file_path)
        assert method in csv_file_path
        assert csv_file_path.endswith(".csv")
