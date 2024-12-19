# This file allows the program to be called from the command line without
# entering the Python interpreter.  To call, use:
#
#     python3 -m coppafish config.ini

import argparse

from coppafish import Notebook, Viewer, run_pipeline
from coppafish.plot import view_find_spots

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run coppafish pipeline or diagnostics")
    parser.add_argument("filepath", type=str, help="Config file path (notebook file path if --view)")
    parser.add_argument("-v", "--view", action="store_true", help="Flag to view diagnostic plots")
    parser.add_argument("--gene_marker", type=str, help="Gene marker file path when using --view")
    parser.add_argument("-fs", "--find_spots", action="store_true", help="Flag to view find spots Viewer")

    args = parser.parse_args()
    if args.find_spots:
        nb = Notebook(args.filepath, must_exist=True)
        view_find_spots(nb)
    elif args.view:
        nb = Notebook(args.filepath, must_exist=True)
        Viewer(nb, gene_marker_filepath=args.gene_marker)
    elif not args.view and not args.find_spots:
        run_pipeline(args.filepath)
