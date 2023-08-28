# -*- coding: utf-8 -*-

"""Routines to run solvers on multiple instances"""

import contextlib
import io
import logging
import multiprocessing as mp
import time
from typing import Dict, List, Optional

import numpy as np

from sucpy import constants
from sucpy.solvers.base import LPR, ExtensiveModel
from sucpy.solvers.cg import CG
from sucpy.utils import cplex_utils
from sucpy.utils import utils as utils

logger = logging.getLogger(__name__)


def run_lpr(
    instance_ids,
    parametrized_problem,
    config,
    failure="warn",
    n_processes=1,
    method="imap",
    callback=None,
    timelimit=None,
    suppress_log=False,
    capture_log=False,
):
    """Run LPR and gather results

    This runs LPR sequentially or on `n_processes` processes
    in parallel and gather the result.

    Parameters
    ----------
    instance_ids : list of int
    parametrized_problem : ParametrizedProblem
    config : dict
    failure : {'warn', 'raise', 'pass'}, default 'warn'
        Behaviour on failure.
    n_processes : int, default 1
        If 1 is given (default), instances are
        solved sequentially on the main process.
        Otherwise, this runs multiple processes
        and solves instances in parallel.
    method : {'imap', 'map', 'unordered'}, default 'imap'
    callback : callable, optional
        If callback is given, on each item arrival
        `callback` is with the accumulated result `acc`
        at that point and the new result as
        `callback(acc, result)`.
    timelimit : float, optional
        If positive value is given, timelimit is set.
        All results arrived after the timelimit are
        discarded.  Note that this function does not
        return immediately after the timelimit, but
        this returns when the first result arrives after
        the timelimit (and this last result is not
        returned from this function).
    suppress_log : bool, default False
        If True, do not pass log messages to the root logger.
    capture_log : bool, default False
        Capture the log output and save it on
        `result['log']`.  Note that the captured log
        is not stored in the accumulated result returned
        from this function so one needs to use it
        within `callback`.

    Returns
    -------
    data : dict
        This contains:
        - parametrized_problem_mode
        - instance_id
        - demand
        - initial_condition
        - parameter
        - status
        - obj
        - sol_on
        - out
        - y
        - walltime
        - proctime
    """
    if callback == "progress":
        callback = progress_callback(len(instance_ids))

    acc = {}

    def reducer(i, result):
        accumulated = dict(result)
        accumulated.pop("msg")
        accumulated.pop("log")
        accumulated["demand"] = accumulated["demand"].astype(np.float32)
        accumulated["initial_condition"] = accumulated[
            "initial_condition"
        ].astype(np.int32)
        accumulated["parameter"] = accumulated["parameter"].astype(np.float32)
        accumulated["sol_on"] = np.packbits(accumulated["sol_on"], axis=-1)
        utils.accumulate(
            acc,
            accumulated,
            stacked=accumulated.keys(),
            on_error="warn",
        )
        if callback is not None:
            callback(acc, result)

    _run(
        target=_run_lpr,
        args=instance_ids,
        init=(parametrized_problem, config, suppress_log, capture_log),
        n_processes=n_processes,
        method=method,
        reducer=reducer,
        failure=failure,
        timelimit=timelimit,
    )

    return acc


def run_cg(
    instance_ids,
    dual_initialiser,
    parametrized_problem,
    config,
    failure="warn",
    n_processes=1,
    method="imap",
    callback=None,
    timelimit=None,
    suppress_log=False,
    capture_log=False,
):
    """Run cg in parallel and gather results

    This runs CG sequentially or on `n_processes`
    processes in parallel and gather the result.

    Parameters
    ----------
    instance_ids : list of int
    dual_initialiser : obj
    parametrized_problem : ParametrizedProblem
    config : dict
    failure : {'warn', 'raise', 'pass'}, default 'warn'
        Behaviour on failure.
    n_processes : int, default 1
        If 1 is given (default), instances are
        solved sequentially on the main process.
        Otherwise, this runs multiple processes
        and solves instances in parallel.
    method : {'imap', 'map', 'unordered'}, default 'imap'
    callback : callable, optional
        If callback is given, on each item arrival
        `callback` is with the accumulated result `acc`
        at that point and the new result as
        `callback(acc, result)`.
    timelimit : float, optional
        If positive value is given, timelimit is set.
        All results arrived after the timelimit are
        discarded.  Note that this function does not
        return immediately after the timelimit, but
        this returns when the first result arrives after
        the timelimit (and this last result is not
        returned from this function).
    suppress_log : bool, default False
        If True, do not pass log messages to the root logger.
    capture_log : bool, default False
        Capture the log output and save it on
        `result['log']`.  Note that the captured log
        is not stored in the accumulated result returned
        from this function so one needs to use it
        within `callback`.

    Returns
    -------
    data : dict
        Besides results returned from CG, this contains:
        - parametrized_problem_mode
        - instance_id
        - demand
        - initial_condition
        - parameter
    """
    if callback == "progress":
        callback = progress_callback(len(instance_ids))

    acc = {}

    def reducer(i, result):
        combine_cg_result(
            acc,
            result,
        )
        if callback is not None:
            callback(acc, result)

    _run(
        target=_run_cg,
        args=instance_ids,
        init=(
            dual_initialiser,
            parametrized_problem,
            config,
            suppress_log,
            capture_log,
        ),
        n_processes=n_processes,
        method=method,
        reducer=reducer,
        failure=failure,
        timelimit=timelimit,
    )

    return acc


def run_extensive_model(
    instance_ids,
    parametrized_problem,
    config,
    failure="warn",
    n_processes=1,
    method="imap",
    callback=None,
    timelimit=None,
    suppress_log=False,
    capture_log=False,
    log_filepath=None,
):
    """Solve the extensive model in parallel and gather results

    This solves the extensive model sequentially or
    on `n_processes` processes in parallel and gather the result.

    Parameters
    ----------
    instance_ids : list of int
    parametrized_problem : ParametrizedProblem
    config : dict
    failure : {'warn', 'raise', 'pass'}, default 'warn'
        Behaviour on failure.
    n_processes : int, default 1
        If 1 is given (default), instances are
        solved sequentially on the main process.
        Otherwise, this runs multiple processes
        and solves instances in parallel.
    method : {'imap', 'map', 'unordered'}, default 'imap'
    callback : callable, optional
        If callback is given, on each item arrival
        `callback` is with the accumulated result `acc`
        at that point and the new result as
        `callback(acc, result)`.
    timelimit : float, optional
        If positive value is given, timelimit is set.
        All results arrived after the timelimit are
        discarded.  Note that this function does not
        return immediately after the timelimit, but
        this returns when the first result arrives after
        the timelimit (and this last result is not
        returned from this function).
    suppress_log : bool, default False
        If True, do not pass log messages to the root logger.
    capture_log : bool, default False
        Capture the log output and save it on
        `result['log']`.  Note that the captured log
        is not stored in the accumulated result returned
        from this function so one needs to use it
        within `callback`.
    log_filepath : str, optional
        If given, CPLEX writes log on the file.

    Returns
    -------
    data : dict
        Besides results returned from CG, this contains:
        - parametrized_problem_mode
        - instance_id
        - demand
        - initial_condition
        - parameter
    """
    if callback == "progress":
        callback = progress_callback(len(instance_ids))

    acc = {}

    def reducer(i, result):
        combine_cg_result(
            acc,
            result,
            stacked=[
                "parametrized_problem_mode",
                "instance_id",
                "demand",
                "initial_condition",
                "parametrized_problem_type",
                "n_g",
                "n_t",
                "timelimit",
            ],
            concatenated=[],
        )
        if callback is not None:
            callback(acc, result)

    _run(
        target=_run_extensive_model,
        args=instance_ids,
        init=(
            parametrized_problem,
            config,
            suppress_log,
            capture_log,
            log_filepath,
        ),
        n_processes=n_processes,
        method=method,
        reducer=reducer,
        failure=failure,
        timelimit=timelimit,
    )

    return acc


class progress_callback(object):
    def __init__(self, n=None):
        self.n = n
        self.counter = 0

    def __call__(self, *args, **kwargs):
        self.counter += 1
        now = time.strftime("%H:%M:%S", time.gmtime())
        if self.n is not None:
            logger.info(f"{now}  {self.counter:4d} / {self.n:4d}")
        else:
            logger.info(f"{now}  {self.counter:4d}")


def initialise_target(target, init):
    target.init = init


def _run(
    target,
    args,
    init,
    n_processes,
    method,
    reducer,
    failure,
    timelimit,
):
    assert failure in ["warn", "raise", "pass"]
    if (timelimit is None) or (timelimit <= 0):
        timelimit = float("inf")
    endtime = time.perf_counter() + timelimit

    with contextlib.ExitStack() as stack:
        if n_processes <= 1:
            initialise_target(target, init)
            mapper = map(target, args)
        else:  # n_processes > 1
            if method == "imap":
                pool = mp.Pool(n_processes, initialise_target, (target, init))
                stack.enter_context(pool)
                mapper = pool.imap(target, args)
            elif method == "map":
                pool = mp.Pool(n_processes, initialise_target, (target, init))
                stack.enter_context(pool)
                mapper = pool.map(target, args)
            elif method == "unordered":
                raise NotImplementedError

        for i, result in enumerate(mapper):
            if time.perf_counter() >= endtime:
                logger.info("reached timelimit")
                break
            reducer(i, result)
            if (not result["status"]) and failure:
                msg = f"failed to solve.  instance: {result['instance_id']}"
                if result.get("msg", ""):
                    msg += "\n" + str(result["msg"])
                if failure == "warn":
                    logger.warn(msg)
                elif failure == "raise":
                    logger.warn(msg)
                    raise ValueError(msg)


def _run_lpr(i):
    parametrized_problem, config, suppress_log, capture_log = _run_lpr.init
    data = parametrized_problem.sample_instance(i)
    n_g, n_t = data["n_g"], data["n_t"]
    lpr = LPR(data, config)
    if capture_log:
        string_io = io.StringIO()
        context = cplex_utils.set_stream(lpr, string_io, prefix="elapse")
    else:
        context = cplex_utils.set_stream(lpr, None)
    with context:
        with utils.timer() as timer:
            lpr.solve()
    if capture_log:
        log = string_io.getvalue()
    else:
        log = ""
    try:
        obj = lpr.solution.get_objective_value()
        sol = np.array(lpr.solution.get_values())
        sol_on = (
            sol[n_g * n_t : 2 * n_g * n_t]
            .reshape(n_g, n_t)
            .round()
            .astype(bool)
        )
        dual = lpr.dual_on_linking_constraint
        status = True
        msg = ""
    except cplex_utils.CplexError:
        obj = np.nan
        sol_on = np.full((data["n_g"], data["n_t"]), 0, dtype=bool)
        dual = np.full(2 * data["n_t"], np.nan)
        status = False
        msg = f"failed to solve.  status: {lpr.solution.get_status()}"
    ret = {
        "parametrized_problem_mode": parametrized_problem.mode,
        "instance_id": i,
        "demand": data["demand"],
        "initial_condition": data["initial_condition"],
        "parameter": data["parameter"],
        "status": status,
        "msg": msg,
        "log": log,
        "obj": obj,
        "sol_on": sol_on,
        "y": dual,
        "walltime": timer.walltime,
        "proctime": timer.proctime,
    }
    ret.update(extract_problem_data_to_be_saved(data))
    return ret


def _run_cg(instance_id):
    (
        dual_initialiser,
        parametrized_problem,
        config,
        suppress_log,
        capture_log,
    ) = _run_cg.init

    data = parametrized_problem.sample_instance(instance_id)

    cg = CG(data, dual_initialiser, config)
    string_io = io.StringIO()
    handler = logging.StreamHandler(string_io)
    handler.setLevel(logging.INFO)
    _logger = logging.getLogger("sucpy.solvers")
    _logger.addHandler(handler)
    _propagate = _logger.propagate
    _logger.propagate = not suppress_log
    try:
        ret = cg.run()
    except Exception as e:
        # This is not thread-safe but we shouldn't have this so often.
        logger.warn(e)
        raise
    _logger.removeHandler(handler)
    _logger.propagate = _propagate
    if capture_log:
        ret["log"] = string_io.getvalue()
    return ret


def _run_extensive_model(instance_id):
    """Run ExtensiveModel Sequentially

    Returns
    -------
    result : dict
        This is a dict of items returned from CG with
        the following additional ones:
        - parametrized_problem_mode
        - instance_id
        - demand
        - initial_condition
        - parameter
    """
    init = _run_extensive_model.init
    (
        parametrized_problem,
        config,
        suppress_log,
        capture_log,
        log_filepath,
    ) = init
    data = parametrized_problem.sample_instance(instance_id)
    n_t = parametrized_problem.n_t
    # Set up models.
    model = ExtensiveModel(data, config)
    model.parameters.mip.tolerances.mipgap.set(config["tol"])
    if "solver.timelimit" in config:
        timelimit = config["solver.timelimit"]
        if (timelimit > 0) and np.isfinite(timelimit):
            model.parameters.timelimit.set(config["solver.timelimit"])
    if log_filepath:
        context = cplex_utils.set_stream(model, log_filepath, prefix="all")
    elif capture_log:
        string_io = io.StringIO()
        context = cplex_utils.set_stream(model, string_io, prefix="elapse")
    else:
        context = cplex_utils.set_stream(model, None)
    with context:
        bound_logger = cplex_utils.set_bound_logger(model)
        model.solve()  # Run CPLEX.
    try:
        bound_logger.lb_values.append(model.solution.MIP.get_best_objective())
        bound_logger.lb_times.append(bound_logger.walltime)
    except cplex_utils.CplexError:
        pass
    try:
        bound_logger.ub_values.append(model.solution.get_objective_value())
        bound_logger.ub_times.append(bound_logger.walltime)
    except cplex_utils.CplexError:
        pass
    walltime = bound_logger.walltime
    proctime = bound_logger.proctime
    assert walltime > 0
    assert proctime > 0
    status = model.solution.get_status() in [101, 102]
    code = model.solution.get_status()
    code_name = model.solution.status[code]
    if not status:
        msg = f"failed to solve.  code: {code}  name: {code_name}"
    else:
        msg = ""
    try:
        obj = model.solution.get_objective_value()
    except cplex_utils.CplexError:
        obj = np.inf
    try:
        obj_bound = model.solution.MIP.get_best_objective()
    except cplex_utils.CplexError:
        obj_bound = -np.inf
    try:
        sol = np.array(model.solution.get_values())
    except cplex_utils.CplexError:
        sol = np.full(data["n_vars"], np.nan)
    ret = dict(
        algorithm=constants.Algorithm.EXTENSIVE_MODEL,
        status=status,
        msg=msg,
        obj=obj,
        sol=sol,
        obj_bound=obj_bound,
        y=np.full((2 * n_t,), np.nan, dtype=np.float32),
        n_iterations=0,
        lb=np.array([obj_bound]),
        ub=np.array([obj]),
        walltime=walltime,
        proctime=proctime,
        n_lb_records=len(bound_logger.lb_values),
        iter_lb=np.array(bound_logger.lb_values),
        iter_lb_time=np.array(bound_logger.lb_times),
        n_ub_records=len(bound_logger.ub_values),
        iter_ub=np.array(bound_logger.ub_values),
        iter_ub_time=np.array(bound_logger.ub_times),
        y_list=np.array([]),
        n_timerecords=1,
        timerecord_type=constants.RoutineType.EXTENSIVE_MODEL,
        timerecord_iteration=np.array([-1]),
        timerecord_walltime=np.array([walltime]),
        timerecord_proctime=np.array([proctime]),
        instance_id=instance_id,
        timelimit=config["solver.timelimit"],
    )
    ret.update(extract_problem_data_to_be_saved(data))
    return ret


def combine_cg_result(
    acc,
    val,
    stacked: Optional[List[str]] = None,
    concatenated: Optional[List[str]] = None,
) -> None:
    """Combine the result returned from CG

    This takes two dictionaries.  The data on the second dictionary
    is copied (stacked or concatenated) on the first one.
    If there are additional items besides those returned from CG,
    one need to specify how to combine them by `stacked` or
    `concatenated` arguments.

    If 'sol' or 'column_value' is found in the data, this replaces
    it with 'sol_on' or 'column_on'.

    Parameters
    ----------
    acc : dict
    val : dict
    stacked : list of str, optional
        Additional items in `acc` and `val` which should be stacked.
    concatenated : list of str, optional
        Additional items in `acc` and `val` which should be concatenated.
    """
    if val["algorithm"] == constants.Algorithm.CG:
        return _combine_cg_result(
            acc=acc,
            val=val,
            stacked=stacked,
            concatenated=concatenated,
        )
    elif val["algorithm"] == constants.Algorithm.EXTENSIVE_MODEL:
        return _combine_cplex_result(
            acc=acc,
            val=val,
            stacked=stacked,
            concatenated=concatenated,
        )
    else:
        raise ValueError("unknown algorithm: {val['algorithm']}")


def _combine_cg_result(
    acc,
    val,
    stacked: Optional[List[str]] = None,
    concatenated: Optional[List[str]] = None,
) -> None:
    val = dict(val)
    val.pop("msg", None)
    val.pop("log", None)
    n_t = val["demand"].shape[-1]
    if len(val) > 0:
        if "sol" in val:
            n_g = val["sol"].shape[0] // (4 * n_t)
            val["sol_on"] = (
                val["sol"][n_g * n_t : 2 * n_g * n_t]
                .reshape(n_g, n_t)
                .round()
                .astype(bool)
            )
            val.pop("sol")
        if "iter_sol" in val:
            n_g = val["iter_sol"].shape[-1] // (4 * n_t)
            val["iter_sol_on"] = (
                val["iter_sol"][:, n_g * n_t : 2 * n_g * n_t]
                .reshape(-1, n_g, n_t)
                .round()
                .astype(bool)
            )
            val.pop("iter_sol")
        if ("sol_on" in val) and (val["sol_on"].shape[-1] == n_t):
            val["sol_on"] = np.packbits(val["sol_on"], axis=-1)
        if ("iter_sol_on" in val) and (val["iter_sol_on"].shape[-1] == n_t):
            val["iter_sol_on"] = np.packbits(val["iter_sol_on"], axis=-1)
    stack_items = []
    concat_items = []
    for key in val:
        test = (
            key.startswith("all_")
            or key.startswith("iter_")
            or key.startswith("cuts_")
            or key.startswith("timerecord_")
        )
        if test:
            concat_items.append(key)
        else:
            stack_items.append(key)
    if stacked:
        stack_items.extend(stacked)
    if concatenated:
        concat_items.extend(concatenated)
    utils.accumulate(
        acc,
        val,
        stacked=stack_items,
        concatenated=concat_items,
        on_error="warn",
    )
    astype = {
        np.float32: ["iter_y"],
    }
    for dtype in astype:
        for key in astype[dtype]:
            if key in acc:
                acc[key] = acc[key].astype(dtype)


def _combine_cplex_result(
    acc,
    val,
    stacked: Optional[List[str]] = None,
    concatenated: Optional[List[str]] = None,
) -> None:
    val = dict(val)
    val.pop("msg", None)
    val.pop("log", None)
    n_t = val["y"].shape[-1] // 2
    if len(val) > 0:
        if "sol" in val:
            n_g = val["sol"].shape[0] // (4 * n_t)
            val["sol_on"] = (
                val["sol"][n_g * n_t : 2 * n_g * n_t]
                .reshape(n_g, n_t)
                .round()
                .astype(bool)
            )
            val.pop("sol")
        n_g, n_t = val["sol_on"].shape[-2:]
        if ("sol_on" in val) and (val["sol_on"].shape[-1] == n_t):
            val["sol_on"] = np.packbits(val["sol_on"], axis=-1)
    stack_items = [
        "algorithm",
        "status",
        "obj",
        "sol_on",
        "obj_bound",
        "y",
        "n_iterations",
        "lb",
        "ub",
        "walltime",
        "proctime",
        "y_list",
        "n_lb_records",
        "n_ub_records",
        "n_timerecords",
        "timerecord_type",
        "timerecord_iteration",
        "timerecord_walltime",
        "timerecord_proctime",
    ]
    concat_items = [
        "iter_lb",
        "iter_lb_time",
        "iter_ub",
        "iter_ub_time",
    ]
    if stacked:
        stack_items.extend(stacked)
    if concatenated:
        concat_items.extend(concatenated)
    utils.accumulate(
        acc,
        val,
        stacked=stack_items,
        concatenated=concat_items,
        on_error="warn",
    )
    astype: Dict = {
        np.int8: [],
        np.int16: [],
        np.float16: [],
        np.float32: [],
    }
    for dtype in astype:
        for key in astype[dtype]:
            if key in acc:
                acc[key] = acc[key].astype(dtype)


def extract_problem_data_to_be_saved(data):
    result = dict()
    for key in data:
        if key.startswith("instance_data."):
            result[key[14:]] = data[key]
    return result


if __name__ == "__main__":
    import doctest

    doctest.testmod()
