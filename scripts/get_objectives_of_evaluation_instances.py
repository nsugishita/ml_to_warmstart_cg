# -*- coding: utf-8 -*-

"""Get the objective values of evaluation instances using CPLEX"""

import argparse
import subprocess


def main():
    """Run the main routine of this script"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="path to an experiment root directry",
    )
    parser.add_argument("--instances", type=str, required=True)
    parser.add_argument(
        "--smoketest",
        action="store_true",
        help="run smoketest",
    )
    args = parser.parse_args()

    if args.smoketest:
        timelimit = "1hour"
    else:
        timelimit = "2hours"

    command = [
        "python",
        "scripts/run_solver.py",
        "--path",
        args.path,
        "--result-dir",
        "evaluation_target_obj",
        "--tol",
        "0.0",
        "--method",
        "cplex",
        "--ph-type",
        "noop",
        "--instances",
        args.instances,
        "--timelimit",
        timelimit,
        "--n-processes",
        "8",
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
