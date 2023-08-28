# -*- coding: utf-8 -*-

"""Run a solver on given UC instances"""

import argparse
import json
import logging
import os

import numpy as np
import torch

from sucpy import constants, dual_initialisers, runners
from sucpy.problems.uc import ParametrizedUC
from sucpy.utils import logging_utils, utils


def main():
    """Run the main routine of this script"""
    parser = argparse.ArgumentParser(parents=[constants.argparser()])
    parser.add_argument(
        "--method",
        required=True,
        choices=[
            "lpr",
            "double_sampling_network",
            "single_sampling_network",
            "nearest_neighbour",
            "random_forest",
            "coldstart",
            "cplex",
            "column_pre_population",
        ],
        help="method to be tested",
    )
    parser.add_argument(
        "--tol",
        required=True,
        type=float,
        choices=[5e-1, 1e-2, 5e-3, 4e-3, 2.5e-3, 1e-3, 0],
        help="sub optimality tolerance.",
    )
    parser.add_argument(
        "--path",
        required=True,
        type=str,
        help="path to an experiment root directry",
    )
    parser.add_argument("--instances", type=str, required=True)
    parser.add_argument(
        "--model",
        type=str,
        help="model name to be loaded",
    )
    parser.add_argument(
        "--activation",
        type=str,
        choices=["relu", "tanh"],
        default="tanh",
    )
    parser.add_argument(
        "--shallow",
        action="store_true",
    )
    parser.add_argument(
        "--deep",
        action="store_true",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="config file to be loaded",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="use training instances instead of validation",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="use test instances instead of validation",
    )
    parser.add_argument(
        "--n-processes",
        type=int,
        default=1,
        help=(
            "Run `n-processes` solvers in parallel. Note that "
            "the individual solver still uses 1 process."
        ),
    )
    parser.add_argument(
        "--display-iteration-log",
        type=int,
        default=1,
        help="Display all iteration logs.",
    )
    parser.add_argument("--stepsize-rule", default="adaptive")
    parser.add_argument(
        "--ph-type",
    )
    parser.add_argument(
        "--ph-model-path",
        default="",
    )
    parser.add_argument("--timelimit", type=str)
    parser.add_argument("--iteration-limit", type=int)
    parser.add_argument("--global-timelimit", type=str)
    parser.add_argument("--tuned-lpr", action="store_true")
    parser.add_argument("--suffix")
    parser.add_argument("--result-dir")
    parser.add_argument(
        "--target-objective-file",
        type=str,
        help="path to an npz file containing the target objective",
    )
    parser.add_argument(
        "--failure",
        choices=["pass", "warn", "raise"],
        default="pass",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="overwrite existing file if any",
    )
    parser.add_argument(
        "-c",
        "--check",
        action="store_true",
        help="do not output log or npz file",
    )
    parser.add_argument(
        "--smoketest",
        action="store_true",
        help="run smoketest",
    )
    args = parser.parse_args()

    path = args.path
    method = args.method
    tol = args.tol

    if args.instances is not None:
        instance_ids = utils.parse_ints(args.instances)
    elif args.train:
        instance_ids = range(40)
    elif args.test:
        instance_ids = range(2000000, 2000040)
    else:
        instance_ids = range(1500000, 1500040)

    config = constants.get_default_config()
    if args.tuned_lpr:
        config["cg.lpr_tol"] = 1e-2
        config["cg.lpr_cross_over"] = 0
    with open(f"{path}/experiment_config.json", "r") as f:
        config.update(json.load(f))

    if args.config:
        with open(args.config, "r") as f:
            config.update(**json.load(f))

    config["tol"] = tol

    if args.ph_type:
        ph_type = constants.PHType[args.ph_type.upper().replace("-", "_")]
        config["cg.ph_type"] = ph_type
    ph_name = config["cg.ph_type"].name.lower()

    constants.update_config(config, args)

    parametrized_problem = ParametrizedUC()
    parametrized_problem.setup(config)

    if args.train:
        mode_name = "train"
        parametrized_problem.train()
    elif args.test:
        mode_name = "test"
        parametrized_problem.test()
    else:
        mode_name = "eval"
        parametrized_problem.eval()

    nn_suffix = ""

    if args.model:
        model_path = args.model
        model_name = os.path.basename(model_path)
        model_name = os.path.splitext(model_name)[0].replace("/", "_")
        model_name = f"{args.method}_{model_name}"
    elif args.method == "double_sampling_network":
        nn_suffixes = []
        if (args.activation is not None) and (args.activation != "tanh"):
            nn_suffixes.append(args.activation)
        if args.shallow:
            nn_suffixes.append("shallow")
        if args.deep:
            nn_suffixes.append("deep")

        if nn_suffixes:
            nn_suffix = "_" + "_".join(nn_suffixes)
        else:
            nn_suffix = ""

        model_path = f"{path}/out/double_sampling_network{nn_suffix}.pth"
        model_name = "double_sampling_network"
    else:
        if "network" in args.method:
            model_ext = ".pth"
        elif "random_forest" in args.method:
            model_ext = ".pkl"
        else:
            model_ext = ".npz"
        model_path = f"{path}/out/{args.method}{model_ext}"
        model_name = args.method

    if args.tuned_lpr:
        model_name = "tuned_lpr"

    if args.target_objective_file:
        target_objective_data = np.load(args.target_objective_file)
        config["target_objective"] = {
            k.item(): v.item()
            for k, v in zip(
                target_objective_data["instance_id"],
                target_objective_data["lb"],
            )
        }
    else:
        config["target_objective"] = np.nan

    if args.check:
        result_filepath = log_filepath = None
    else:
        if args.result_dir:
            result_dir = f"{args.result_dir}/"
        else:
            result_dir = ""
        suffix = ""
        if args.suffix:
            suffix += f"_{args.suffix}"
        if nn_suffix:
            suffix += nn_suffix
        result_filepath = (
            f"{path}/out/{result_dir}"
            f"tol_{tol:.1e}_primal_{ph_name}_dual_{model_name}_"
            f"{mode_name}{suffix}.npz"
        )
        log_filepath = (
            f"{path}/out/{result_dir}"
            f"tol_{tol:.1e}_primal_{ph_name}_dual_{model_name}_"
            f"{mode_name}{suffix}.txt"
        )
        if (method == "cplex") and (args.n_processes == 1):
            cplex_logpath = (
                f"{path}/out/{result_dir}"
                f"tol_{tol:.1e}_cplex_log_{mode_name}{suffix}.txt"
            )
        else:
            cplex_logpath = ""

        if os.path.isfile(result_filepath) and (not args.force):
            print(f"result file already exist: {result_filepath}")
            return

    if args.timelimit:
        config["solver.timelimit"] = utils.parse_time_length(
            args.timelimit, default_unit="seconds"
        )

    if args.iteration_limit:
        config["cg.iteration_limit"] = args.iteration_limit
    else:
        config["cg.iteration_limit"] = np.inf

    if args.global_timelimit:
        global_timelimit = utils.parse_time_length(
            args.global_timelimit, default_unit="hours"
        )
        checkpoints = []
        checkpoints.append(np.arange(0, 120, 30))
        checkpoints.append(np.arange(120, 301, 60))
        checkpoints.append(np.arange(0, global_timelimit + 1, 15 * 60))
        checkpoints.append(np.array([global_timelimit + 60 * 60]))
        checkpoints = np.unique(np.concatenate(checkpoints))
    else:
        global_timelimit = np.inf
        checkpoints = np.array([])

    run_data = {
        "mode": mode_name,
        "instance_ids": instance_ids,
        "model_name": model_name,
        "model_path": model_path,
        "ph_type": ph_name,
        "result_filepath": result_filepath,
        "log_filepath": log_filepath,
        "timelimit": config["solver.timelimit"],
        "iteration_limit": config["cg.iteration_limit"],
        "global_timelimit": global_timelimit,
    }

    if (method == "cplex") and cplex_logpath:
        run_data["cplex_logpath"] = cplex_logpath

    logging_utils.initialise(log_filepath, run_data)
    logger = logging.getLogger(__name__)

    if result_filepath and os.path.isfile(result_filepath):
        logger.info(f"overwriting result file: {result_filepath}")

    if method == "lpr":
        dual_initialiser = "lpr"
    elif method in [
        "single_sampling_network",
        "double_sampling_network",
    ]:
        network_kwargs = dict(activation=args.activation)
        if args.shallow:
            network_kwargs["layer_sizes"] = [1000, 1000, 1000]
        if args.deep:
            network_kwargs["layer_sizes"] = [1000, 1000, 1000, 1000, 1000]
        if method == "single_sampling_network":
            Initialiser = dual_initialisers.SingleSamplingNetworkInitialiser
        else:
            Initialiser = dual_initialisers.DoubleSamplingNetworkInitialiser
        dual_initialiser = Initialiser(
            dim_parameter=parametrized_problem.dim_parameter,
            dim_dual=parametrized_problem.dim_dual,
            config=config,
            network_kwargs=network_kwargs,
        )
        dual_initialiser.load_state_dict(torch.load(model_path))
        dual_initialiser.eval()
    elif method == "nearest_neighbour":
        dual_initialiser = dual_initialisers.NearestNeighbourInitialiser(
            config=config
        )
        dual_initialiser.load(model_path)
    elif method == "random_forest":
        dual_initialiser = dual_initialisers.RandomForestInitialiser(
            dim_parameter=parametrized_problem.dim_parameter,
            dim_dual=parametrized_problem.dim_dual,
        )
        dual_initialiser.load(model_path)
    elif method == "coldstart":
        dual_initialiser = "coldstart"
    elif method == "column_pre_population":
        Initialiser = dual_initialisers.ColumnPrePopulationDualInitialiser
        dual_initialiser = Initialiser(config=config)
        dual_initialiser.load(model_path)

    if args.stepsize_rule:
        stepsize_rule = constants.StepsizeRule[
            args.stepsize_rule.upper().replace("-", "_")
        ]
        config["dual_optimizer.stepsize_rule"] = stepsize_rule

    if result_filepath:
        os.makedirs(os.path.dirname(result_filepath), exist_ok=True)

    timer = utils.timer()

    dtypes = {
        "initial_condition": np.int8,
    }

    def save_result(acc):
        if result_filepath:
            for key, dtype in dtypes.items():
                if key in acc:
                    acc[key] = acc[key].astype(dtype)
            np.savez(result_filepath, **acc)

    def callback(acc, result):
        nonlocal checkpoints

        acc.setdefault("instance_solution_walltime", [])
        acc["instance_solution_walltime"].append(timer.walltime)

        instance_id = result["instance_id"]
        all = len(acc["status"])
        success = acc["status"].sum()
        mean_elapse = acc["walltime"].mean()
        mean_n_iterations = acc["n_iterations"].mean()

        test = args.display_iteration_log or (not result["status"])
        test &= args.n_processes > 1

        if test:
            msg = result.get("log", "").strip()
            if msg:
                logger.info(msg)

        header = (
            f"{'instance':>8s}  "
            f"{'all':>4s}  "
            f"{'succ':>4s}  "
            f"{'time':>6s}  "
            f"{'iter':>4s}  "
            f"{'total':>8s}"
        )
        if (len(acc["status"]) - 1) % 10 == 0:
            logger.info(header)
        else:
            logger.info(header)
        logger.info(
            f"{instance_id:8d}  "
            f"{all:4d}  "
            f"{success:4d}  "
            f"{mean_elapse:6.1f}  "
            f"{mean_n_iterations:4.1f}  "
            f"{utils.format_elapse(timer.walltime)}"
        )

        should_save = bool(result_filepath)
        should_save &= (len(checkpoints) == 0) or (
            timer.walltime >= checkpoints[0]
        )
        if should_save:
            if len(checkpoints) > 0:
                checkpoints = checkpoints[checkpoints > timer.walltime]
                solved = 0 if not acc else len(acc["ub"])
                logger.info(
                    f"elapse: {utils.format_elapse(timer.walltime)}  "
                    f"solved: {solved:7d}  "
                    f"data size (MB): {utils.nbytes(acc)/10**6:6.1f}"
                )
            save_result(acc)

    if method != "cplex":
        acc = runners.run_cg(
            instance_ids=instance_ids,
            dual_initialiser=dual_initialiser,
            parametrized_problem=parametrized_problem,
            config=config,
            failure=args.failure,
            n_processes=args.n_processes,
            method="imap",
            callback=callback,
            timelimit=global_timelimit,
            suppress_log=args.n_processes > 1,
            capture_log=True,
        )

        save_result(acc)

    else:
        runners.run_extensive_model(
            instance_ids=instance_ids,
            parametrized_problem=parametrized_problem,
            config=config,
            failure=args.failure,
            n_processes=args.n_processes,
            method="imap",
            callback=callback,
            log_filepath=cplex_logpath,
        )


if __name__ == "__main__":
    main()
