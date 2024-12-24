from coppafish import Notebook

nb = Notebook("/path/to/notebook")
print(nb.omp.results[f"tile_{tile}"].attrs["minimum_intensity"])
