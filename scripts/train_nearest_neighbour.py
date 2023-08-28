# -*- coding: utf-8 -*-

"""Create a dataset to run the nearest neighbour methods"""

import argparse
import glob
import os
import subprocess


def main():
    """Run the main routine of this script"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        required=True,
        type=str,
        help="path to an experiment root directry",
    )
    parser.add_argument(
        "--smoketest",
        action="store_true",
        help="run smoketest",
    )
    args = parser.parse_args()

    if args.smoketest:
        global_timelimit = "1hour"
    else:
        global_timelimit = "24hours"

    command = [
        "python",
        "scripts/run_solver.py",
        "--result-dir",
        "nearest_neighbour_data",
        "--path",
        args.path,
        "--method",
        "lpr",
        "--tol",
        "2.5e-3",
        "--instances",
        "0:1000000",
        "--global-timelimit",
        global_timelimit,
        "--timelimit",
        "10mins",
        "--n-processes",
        "8",
        "--display-iteration-log",
        "0",
        "--failure",
        "warn",
    ]

    if args.smoketest:
        command += ["--smoketest"]

    subprocess.run(command, check=True)

    # Create symlinks "column_pre_population.npz" and "nearest_neighbour.npz".
    target_filepath = glob.glob(
        f"{args.path}/out/nearest_neighbour_data/*.npz"
    )[0]
    target_relative_filepath = target_filepath[
        target_filepath.index("out/") + 4 :
    ]
    symlink_filepaths = [
        f"{args.path}/out/nearest_neighbour.npz",
        f"{args.path}/out/column_pre_population.npz",
    ]
    for symlink_filepath in symlink_filepaths:
        if os.path.islink(symlink_filepath):
            continue
        os.symlink(
            target_relative_filepath,
            symlink_filepath,
        )


if __name__ == "__main__":
    main()
