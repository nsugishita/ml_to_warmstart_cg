# -*- coding: utf-8 -*-

"""Train a random forest model"""

import argparse
import glob
import json
import logging
import os

import numpy as np

from sucpy import constants
from sucpy.dual_initialisers.random_forest_initialiser import (
    RandomForestInitialiser,
)
from sucpy.problems.uc import ParametrizedUC
from sucpy.utils import logging_utils


def main():
    """Run the main routine of this script"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, help="path to an experiment root directry"
    )
    parser.add_argument(
        "--smoketest",
        action="store_true",
        help="run smoketest",
    )
    args = parser.parse_args()
    path = args.path

    model_path = f"{path}/out/random_forest.pkl"

    if os.path.exists(model_path):
        print(f"{model_path=} exists")
        return

    log_filepath = f"{path}/out/logs/random_forest.txt"

    logging_utils.initialise(log_filepath)
    logger = logging.getLogger(__name__)

    data_filepath = glob.glob(f"{path}/out/nearest_neighbour_data/*.npz")[0]

    full_data = np.load(data_filepath)

    logger.info(f"full data size: {full_data['demand'].shape[0]}")

    training_budget = 24 * 60 * 60

    mask = full_data["instance_solution_walltime"] < training_budget

    data = {
        "demand": full_data["demand"][mask],
        "y": full_data["y"][mask],
    }

    logger.info(f"used data size: {data['demand'].shape[0]}")

    config = constants.get_default_config()
    with open(f"{path}/experiment_config.json", "r") as f:
        config.update(json.load(f))
    parametrized_problem = ParametrizedUC()
    parametrized_problem.setup(config)

    random_forest = RandomForestInitialiser(
        dim_parameter=parametrized_problem.dim_parameter,
        dim_dual=parametrized_problem.dim_dual,
    )
    parameter = data["demand"]
    dual = data["y"]
    random_forest.fit(parameter=parameter, dual=dual)
    random_forest.save(model_path)


if __name__ == "__main__":
    main()
