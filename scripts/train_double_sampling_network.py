# -*- coding: utf-8 -*-

"""Train a neural network dual initialiser"""

import argparse
import copy
import functools
import json
import logging
import multiprocessing as mp
import os
import time
import typing

import numpy as np
import torch

from sucpy import constants, runners, solvers
from sucpy.dual_initialisers import DoubleSamplingNetworkInitialiser
from sucpy.problems.uc import ParametrizedUC
from sucpy.solvers.base import LPR
from sucpy.utils import logging_utils, network_utils, utils

n_processes = 8
subprocesses: typing.List[mp.Process] = []
n_normalization = 40
n_evaluations = 10
eval_every = 5 * 60
save_every = 60 * 60
clear_cache_every = 30 * 60


def train(
    worker_index,
    parametrized_problem,
    dual_initialiser,
    step_counter,
    step_counter_lock,
    lr,
    lr_lock,
    config,
    presolve2,
    endtime,
    path,
    result_model_filepath,
):
    """Train a dual_initialiser"""
    starttime = time.perf_counter()
    torch.set_num_threads(1)
    optimizer = torch.optim.Adam(
        dual_initialiser.parameters(),
        lr=lr.value,
    )
    rng = np.random.RandomState(worker_index)
    best_gap = np.inf
    cache = {}
    evaluation_cache = {}
    parametrized_problem.clear_decomposition_cache_()
    validation_index = -1
    next_evaluate = 0
    next_save = 0
    next_cache_clear = next_save + clear_cache_every

    logger = logging.getLogger()

    while time.perf_counter() < endtime:
        with lr_lock:
            network_utils.set_lr(optimizer, lr.value)
        elapse = time.perf_counter() - starttime
        run_one_training_step(
            instance_id=rng.randint(1000000),
            parametrized_problem=parametrized_problem,
            dual_initialiser=dual_initialiser,
            optimizer=optimizer,
            rng=rng,
            config=config,
            elapse=elapse,
            cache=cache,
        )
        with step_counter_lock:
            step_counter.value += 1
        if worker_index > 0:
            continue
        if elapse >= next_cache_clear:
            next_cache_clear += clear_cache_every
            cache.clear()
            evaluation_cache.clear()
        if elapse < next_evaluate:
            continue
        next_evaluate += eval_every
        # Measure performance of the current model.
        validation_index += 1
        lb = evaluate(
            instance_ids=range(1000, 1000 + n_evaluations),
            parametrized_problem=parametrized_problem,
            dual_initialiser=copy.deepcopy(dual_initialiser),
            config=config,
            cache=evaluation_cache,
        )
        ub = presolve2["ub"][:n_evaluations]
        gap = (ub - lb) / ub * 100
        if validation_index % 20 == 0:
            logger.info(
                f"{'it':>5s} "
                f"{'steps':>7s} "
                f"{'elapse':>8s} "
                f"{'loss':>11s}  "
                f"{'lr':>8s}"
            )
        steps = step_counter.value
        logger.info(
            f"{validation_index:5d} "
            f"{steps:7d} "
            f"{utils.format_elapse(elapse):8s} "
            f"{np.mean(gap):11.4f}{'*' if gap.mean() < best_gap else ' ':1s} "
            f"{lr.value:8.1e}"
        )
        if gap.mean() < best_gap:
            best_gap = gap.mean()
            null_step_counter = 0
        else:
            null_step_counter += 1
        if null_step_counter >= 10:
            null_step_counter = 0
            with lr_lock:
                lr.value = lr.value / 1.5

        if elapse > next_save:
            if result_model_filepath:
                torch.save(
                    dual_initialiser.state_dict(), result_model_filepath
                )
            next_save += save_every


def run_one_training_step(
    instance_id,
    parametrized_problem,
    dual_initialiser,
    optimizer,
    rng,
    config,
    check=False,
    elapse=0.0,
    cache={},
):
    """Run one step of training

    Parameters
    ----------
    instance_id : int
    parametrized_problem : ParametrizedUC
    dual_initialiser : DualInitialiser
    optimizer : torch.optim.Optimizer
    config : dict
    check : bool, default False
        If True, do not actually update the network parameters
        but return the gradient norms.
    cache : dict, optional
        If a dict is given, cplex instances are cached on
        this dict.  By passing the same dict on the second call,
        the execusion of `run_one_training_step` can be sped up
        compared with the case when `cahce` is not provided.

    Returns
    -------
    l1 : float, Optional
        Only returned when `check` is True.
    l2 : float, Optional
        Only returned when `check` is True.
    """
    data = parametrized_problem.sample_instance(instance_id, inplace=True)
    y = dual_initialiser(data["parameter"])
    y_clipped = y.detach().squeeze().numpy().clip(0, None)
    if "subproblems" not in cache:
        cache["subproblems"] = [None] * data["n_subproblems"]
    grad = (
        np.array(data["subproblem_constraint_rhs"][-1]) / data["n_subproblems"]
    )
    subproblem_index = rng.choice(data["n_subproblems"])
    subproblem = cache["subproblems"][subproblem_index]
    if subproblem is None:
        subproblem = solvers.Subproblem(subproblem_index, data, config)
        cache["subproblems"][subproblem_index] = subproblem
    else:
        subproblem.set(subproblem_index, data, x_lb=True, x_ub=True)
    subproblem.update_y(y_clipped)
    subproblem.solve()
    subproblem_solution = np.array(subproblem.solution.get_values())
    Ds = data["subproblem_constraint_coefficient_row_border"]
    grad -= Ds[subproblem_index].dot(subproblem_solution)
    grad = torch.Tensor(grad)
    grad[(y < 0) & (grad < 0)] = 0
    loss = -torch.matmul(y, grad)  # Loss is to be minimized.
    # Add penalty for out of the bound y.
    if torch.any(y < 0):
        bound_penalty = torch.nn.functional.smooth_l1_loss(
            y[y < 0], torch.zeros(torch.sum(y < 0))
        )
    else:
        bound_penalty = 0
    whole_loss = loss + bound_penalty
    whole_loss = whole_loss * dual_initialiser.grad_scaler
    optimizer.zero_grad()
    whole_loss.backward()
    if check:
        l1 = torch.nn.utils.clip_grad_norm_(
            dual_initialiser.parameters(), np.inf, 1
        )
        l2 = torch.nn.utils.clip_grad_norm_(
            dual_initialiser.parameters(), np.inf, 2
        )
        return (l1, l2)
    else:
        optimizer.step()


def evaluate(
    instance_ids,
    parametrized_problem,
    dual_initialiser,
    config,
    cache,
):
    """Evaluate the current dual_initialiser

    Returns
    -------
    lb : (n_evaluations,) array
        Resulting lower bound.
    """
    lb = []
    try:
        dual_initialiser.eval()
    except AttributeError:
        pass
    for instance_id in instance_ids:
        data = parametrized_problem.sample_instance(instance_id, inplace=True)
        if dual_initialiser == "lpr":
            _dual_initialiser = LPR(data, config)
        else:
            _dual_initialiser = dual_initialiser
        # Set up subproblems.
        if "subproblems" not in cache:
            cache["subproblems"] = [
                solvers.Subproblem(i, data, config)
                for i in range(data["n_subproblems"])
            ]
        else:
            for i, subproblem in enumerate(cache["subproblems"]):
                subproblem.set(i, data, x_lb=True, x_ub=True)
        with torch.no_grad():
            y = _dual_initialiser.compute_initial_dual(data["parameter"])
        y = y.clip(0, None)
        # Compute the dual value.
        _lb = data["subproblem_constraint_rhs"][-1].dot(y)
        for subproblem in cache["subproblems"]:
            subproblem.update_y(y)
            subproblem.solve()
            _lb += subproblem.solution.get_objective_value()
        np.testing.assert_array_less(0.0, _lb)
        lb.append(_lb)
    lb = np.array(lb)
    return lb


def compute_mean_grad_norm(
    instance_ids, parametrized_problem, dual_initialiser, config
):
    """Compute multiple steps and gather gradient norms"""
    optimizer = torch.optim.SGD(dual_initialiser.parameters(), lr=0.0)
    with utils.timer() as timer:
        target = functools.partial(
            run_one_training_step,
            parametrized_problem=parametrized_problem,
            dual_initialiser=dual_initialiser,
            optimizer=optimizer,
            config=config,
            rng=None,
            check=True,
            elapse=0.0,
        )
        l1 = []
        l2 = []
        with mp.Pool(n_processes) as pool:
            for result in pool.imap(target, instance_ids):
                l1.append(result[0])
                l2.append(result[1])
        l1 = np.array(l1)
        l2 = np.array(l2)
    return dict(l1=l1, l2=l2, walltime=timer.walltime, proctime=timer.proctime)


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
        "--timelimit", type=float, default=24, help="timelimit in hours"
    )
    parser.add_argument(
        "--smoketest",
        action="store_true",
        help="run smoketest",
    )
    parser.add_argument("--suffix")
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing model if any",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write any output nor logs to files",
    )
    args = parser.parse_args()
    path = args.path

    suffixes = []
    if (args.activation is not None) and (args.activation != "tanh"):
        suffixes.append(args.activation)
    if args.shallow:
        suffixes.append("shallow")
    if args.deep:
        suffixes.append("deep")
    if args.suffix:
        suffixes.append(args.suffix)

    if suffixes:
        suffix = "_" + "_".join(suffixes)
    else:
        suffix = ""

    if args.check:
        result_model_filepath = None
        log_filepath = None
    else:
        result_model_filepath = (
            f"{path}/out/double_sampling_network{suffix}.pth"
        )
        log_filepath = f"{path}/out/logs/double_sampling_network{suffix}.txt"

    fileexist = (result_model_filepath is not None) and os.path.isfile(
        result_model_filepath
    )
    if fileexist and (not args.force) and (not args.check):
        print(f"{result_model_filepath} already exists")
        raise SystemExit

    config = constants.get_default_config()
    with open(f"{path}/experiment_config.json") as f:
        config.update(json.load(f))
    config["tol"] = 2.5e-3

    parametrized_problem = ParametrizedUC()
    parametrized_problem.setup(config)
    parametrized_problem.mode = "train"

    np.random.seed(parametrized_problem.n_g + parametrized_problem.n_t)
    torch.manual_seed(parametrized_problem.n_g + parametrized_problem.n_t)

    if args.smoketest:
        timelimit = 60 * 60
    else:
        timelimit = args.timelimit * 60 * 60

    starttime = time.perf_counter()

    network_kwargs = dict(activation=args.activation)
    if args.shallow:
        network_kwargs["layer_sizes"] = [1000, 1000, 1000]
    if args.deep:
        network_kwargs["layer_sizes"] = [1000, 1000, 1000, 1000, 1000]
    dual_initialiser = DoubleSamplingNetworkInitialiser(
        dim_parameter=parametrized_problem.dim_parameter,
        dim_dual=parametrized_problem.dim_dual,
        config=config,
        network_kwargs=network_kwargs,
    )

    dual_initialiser.train()

    presolve1_path = f"{path}/out/cache/train_network_presolve1.npz"
    presolve2_path = f"{path}/out/cache/train_network_presolve2.npz"

    os.makedirs(os.path.dirname(presolve1_path), exist_ok=True)

    # Set up input and output normalizer
    if os.path.isfile(presolve1_path):
        presolve1 = np.load(presolve1_path)
        starttime -= np.sum(presolve1["walltime"])
    else:
        print("preparing presolve1")
        presolve1 = runners.run_lpr(
            instance_ids=range(n_normalization),
            parametrized_problem=parametrized_problem,
            config=config,
            failure="raise",
            n_processes=n_processes,
        )
        np.savez(presolve1_path, **presolve1)
    try:
        param = presolve1["parameter"]
    except KeyError:
        param = np.r_[
            "1,2", presolve1["demand"], presolve1["initial_condition"]
        ]
    param = param[..., : parametrized_problem.dim_parameter]
    dual_initialiser.set_normalizer(
        parameter=param,
        dual=presolve1["y"],
    )
    # Run CG on multiple instances to:
    # - set up data used in evaluation.
    # - set up objective scaler when naive scaling is on.
    if os.path.isfile(presolve2_path):
        presolve2 = dict(np.load(presolve2_path))
        starttime -= np.sum(presolve2["walltime"])
    else:
        print("preparing presolve2")
        presolve2 = runners.run_cg(
            instance_ids=range(1000, 1000 + n_evaluations),
            dual_initialiser="lpr",
            parametrized_problem=parametrized_problem,
            config=config,
            failure="warn",
            n_processes=n_processes,
        )
        np.savez(presolve2_path, **presolve2)
    dual_initialiser.grad_scaler = 1 / np.mean(presolve2["ub"])

    endtime = starttime + timelimit

    # Just in case clear the grad.
    network_utils.zero_grad(dual_initialiser, set_to_none=True)
    dual_initialiser.share_memory()

    lr = mp.Value("f", 1e-4)
    lr_lock = mp.Lock()
    step_counter = mp.Value("i", 0)
    step_counter_lock = mp.Lock()
    config["tol"] = 1e-6  # Evaluate super-gradients in higher precision.

    logging_utils.initialise(log_filepath)
    logger = logging.getLogger()
    logger.info(f"timelimit: {utils.format_elapse(timelimit)}")
    logger.info(
        "presolve time: "
        f"{utils.format_elapse(time.perf_counter() - starttime)}"
    )
    logger.info(f"initial lr: {lr.value:8.1e}")
    logger.info(f"suffix: {suffix[1:]}")
    logger.info(f"result model path: {result_model_filepath}")
    logger.info("network parameters:")
    for key in network_kwargs:
        logger.info(f"  {key}: {network_kwargs[key]}")

    for i in range(n_processes):
        kwargs = {
            "worker_index": i,
            "parametrized_problem": parametrized_problem,
            "dual_initialiser": dual_initialiser,
            "step_counter": step_counter,
            "step_counter_lock": step_counter_lock,
            "lr": lr,
            "lr_lock": lr_lock,
            "config": config,
            "presolve2": presolve2,
            "endtime": endtime,
            "path": args.path,
            "result_model_filepath": result_model_filepath,
        }
        process = mp.Process(target=train, kwargs=kwargs)
        process.start()
        subprocesses.append(process)

    for p in subprocesses:
        p.join()


if __name__ == "__main__":
    with utils.ProcessesContext(subprocesses):
        main()
