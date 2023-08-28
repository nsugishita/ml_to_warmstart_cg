# -*- coding: utf-8 -*-

"""solvers routines for the experiment"""

import itertools
import logging
import os

import numpy as np

from sucpy import constants, dual_initialisers, primal_heuristics
from sucpy.solvers.base import ExtensiveModel
from sucpy.solvers.rmp import RegularizedRMP
from sucpy.solvers.subproblems import Subproblems
from sucpy.utils import cplex_utils, utils


class CG(object):
    """Column generation algorithm"""

    def __init__(
        self,
        data,
        dual_initialiser,
        config,
        stop_event=None,
    ):
        """Initialise a CG instance

        This prepare to run column generation algorithm.
        Namely, this defined several CPLEX models.
        One may be interested to measure time spent on run method only.

        Parameters
        ----------
        data : dict
            Data returned from ParametrizedUC.sample_instance.
        dual_initialiser : DualInitialiser
            DualInitialiser instance used to get the dual value
            in the first iteration.
        config : dict
            Configuration of the experiment.
        stop_event : multiprocessing.Event
            An event indicates to stop the solver.
        """
        if stop_event is None:
            stop_event = utils.DummyEvent(set=False)
        # A flag which will be set True after `run`.
        self.used = False
        self.data = data
        self.config = config
        self.stop_event = stop_event

        self.journal = utils.Journal(
            tol=self.config["tol"],
            stop_event=self.stop_event,
        )
        n_subproblems = self.data["n_subproblems"]
        y_nan = np.full(
            (self.data["constraint_subproblem_index"] == -1).sum(), np.nan
        )
        self.journal.register_iteration_items(
            y=dict(default=y_nan, timing=True),
            stepsize=dict(default=np.nan, timing=False),
            solution_polishing=dict(default=False, timing=False),
        )
        if config["cg.save_rmp_size"]:
            self.journal.register_iteration_items(
                rmp_size=dict(
                    default=np.full(n_subproblems, np.nan), timing=False
                ),
            )

        if config["cg.save_subproble_solutions"]:
            n_subproblem_vars = data["subproblem_variable_lb"][0].size
            self.journal.register_iteration_items(
                subproblem_solution=np.full(
                    (n_subproblems, n_subproblem_vars), np.nan
                ),
                subproblem_objective_ub=np.full((n_subproblems,), np.nan),
                subproblem_objective_lb=np.full((n_subproblems,), np.nan),
            )

        if config["cg.save_model_values"]:
            self.journal.register_iteration_items(
                model_value=dict(
                    default=np.full(n_subproblems, np.nan), timing=False
                ),
                function_value=dict(
                    default=np.full(n_subproblems, np.nan), timing=False
                ),
            )

        self.dual_initialiser = dual_initialisers.as_dual_initialiser(
            dual_initialiser=dual_initialiser, data=data, config=config
        )

        self.rmp = RegularizedRMP(
            data=data, config=config, journal=self.journal
        )
        self.subproblems = Subproblems(
            data=data, config=config, stop_event=stop_event
        )
        self.primal_heuristic = primal_heuristics.get(
            ph_type=config["cg.ph_type"],
            data=data,
            config=config,
            journal=self.journal,
        )
        if config["cg.no_progress_action"] == "solution_polishing":
            self.solution_polishing_model = ExtensiveModel(
                data=data, config=config
            )
        else:
            self.solution_polishing_model = None

        # Hooks called in each iteration.
        self.hooks = []

    def run(self):
        """Run column generation

        This runs column generation.  This modifies internal CPLEX models,
        so one should not call this method more than once.

        Parameters
        ----------
        y : (num_linking_constraints,) array or float
            Initial dual variable.

        Returns
        -------
        status : bool
            True if gap is closed.  False otherwise.
        msg : str
            Error message if available.
        obj : float
            Objective value
        sol : (n_variables,) array
        obj_bound : float
            Lower bound of the objective
        n_iterations : int
            Number of iterations.
        lb : (n_iterations,) array
            Lower bound in each iteration.
        ub : (n_iterations,) array
            Upper bound in each iteration.
        """
        # Safe guard the second call.
        if self.used:
            raise ValueError("cannot call `run` method twice")
        self.used = True

        target_objective = self.config["target_objective"]
        if isinstance(target_objective, dict):
            target_objective = target_objective[self.data["instance_id"]]
        if np.isfinite(target_objective):
            self.target_objective = target_objective
        else:
            self.target_objective = np.nan

        if self.config["solver.n_processes"] > 1:
            n = self.config["solver.n_processes"]
            self.rmp.model.parameters.threads.set(n)
            try:
                self.dual_initialiser.parameters.threads.set(n)
            except AttributeError:
                pass
            self.primal_heuristic.n_processes = n
            self.subproblems.n_processes = n
        try:
            self.dual_initialiser.set_cg(self)
        except AttributeError:
            pass

        with self.subproblems:
            return self._run_impl()

    def _run_impl(self):
        logger = logging.getLogger(__name__)
        data = self.data
        config = self.config
        journal = self.journal
        dual_initialiser = self.dual_initialiser
        rmp = self.rmp
        subproblems = self.subproblems
        primal_heuristic = self.primal_heuristic
        run_solution_polishing_flag = False

        link_con_sense = data["subproblem_constraint_sense"][-1]
        y_lb = np.full(link_con_sense.shape, -np.inf)
        y_ub = np.full(link_con_sense.shape, np.inf)
        if "U" in str(link_con_sense.dtype):
            link_con_sense = np.array(
                list("".join(link_con_sense).encode("utf8"))
            )
        np.place(y_lb, link_con_sense == b"G"[0], 0)
        np.place(y_ub, link_con_sense == b"L"[0], 0)

        timelimit = config.get("solver.timelimit", np.inf)
        if timelimit < 0:
            timelimit = np.inf
        iteration_limit = config.get("cg.iteration_limit", np.inf)

        stop_event = self.stop_event
        log_level = logging.INFO

        best_lb = -np.inf
        best_ub = np.inf
        iter_y = []
        no_progress_count = 0  # Counter of its without progress.

        primal_heuristic.set_cg(self)

        journal.start_hook()
        rmp.start_hook()
        primal_heuristic.start_hook()

        # Keep track how long it takes to compute the timelimit
        # of the primal heuristic.
        ph_timelimit_measure_start = journal.walltime

        journal.start(constants.RoutineType.DUAL_INITIALIZATION)
        if getattr(dual_initialiser, "feed_problem_data", False):
            y = dual_initialiser.compute_initial_dual(data)
        else:
            y = dual_initialiser.compute_initial_dual(data.get("parameter"))
        y = y.clip(y_lb, y_ub)
        rmp.initialiser_hook(y)
        primal_heuristic.initialiser_hook(y)

        for iteration_index in itertools.count():
            journal.iteration_start_hook(iteration_index)
            journal.set_iteration_items(stepsize=rmp.stepsize)
            if config["cg.save_rmp_size"]:
                journal.set_iteration_items(
                    rmp_size=rmp.get_n_cuts_by_subproblems()
                )
            rmp.iteration_index = iteration_index
            primal_heuristic.iteration_index = iteration_index
            primal_heuristic.iteration_start_hook(iteration_index)
            lb = -np.inf
            ub = np.inf
            original_stepsize = rmp.stepsize

            if iteration_index >= 15:
                # When prepopulated cuts, due to numerical error
                # some of them makes the algorithm fails to converge.
                # Thus, remove the cuts after enough columns added.
                rmp.remove_cuts_older_than(-1)

            if y.ndim != 1:
                raise ValueError(f"invalid dual shape: {y.shape}")
            y = y.clip(y_lb, y_ub)
            if config["cg.log_iterates"]:
                msg = "y: " + " ".join(map("{:.1f}".format, y))
                logger.info(msg)
            journal.set_iteration_items(y=y)
            iter_y.append(np.array(y))

            patience = config["cg.no_progress_patience"]
            action = config["cg.no_progress_action"]
            test = (no_progress_count >= patience > 0) and (
                action == "terminate"
            )
            test |= run_solution_polishing_flag
            if test:
                return self.make_returned(status=False, msg="no progress")
            elif iteration_index >= iteration_limit:
                return self.make_returned(
                    status=False, msg="reached iteration limit"
                )
            elif journal.walltime >= timelimit > 0:
                return self.make_returned(
                    status=False, msg="reached timelimit."
                )
            elif rmp.stepsize < 1e-12:
                return self.make_returned(
                    status=False, msg="reached stepsize limit"
                )

            # Solve subproblem, generate columns and compute lower bound.
            subproblem_time_start = journal.walltime
            journal.start(constants.RoutineType.SUBPROBLEM)
            evaluation_component_index = np.arange(self.data["n_subproblems"])
            subproblems.evaluate(evaluation_component_index, y)

            if config["cg.save_subproble_solutions"]:
                journal.set_iteration_items(
                    subproblem_solution=subproblems.subproblem_solution,
                    subproblem_objective_ub=subproblems.objective_ub,
                    subproblem_objective_lb=subproblems.objective_lb,
                )

            if config["cg.save_model_values"]:
                buf = np.full(data["n_subproblems"], np.nan)
                buf[
                    subproblems.evaluation_component_index
                ] = subproblems.function_value
                journal.set_iteration_items(function_value=buf)

            if (iteration_index == 0) and (rmp.n_constraints > 0):
                # This is the first iteration and there are constraints
                # (since y may be infeasible).
                lb_available = False
            else:
                # If not all components are evaluated.
                # Note that stop_event may terminate subproblems
                # solver as well, so we need to check how many
                # are actually evaluated.
                lb_available = (
                    len(subproblems.evaluation_component_index)
                    == data["n_subproblems"]
                )

            if not lb_available:
                lb = -np.inf
            else:
                lb = (
                    rmp.problem_data["c_x"].dot(y)
                    + subproblems.function_value.sum()
                    + data["objective_offset"]
                )
                if rmp.problem_data["dim_s"] > 0:
                    lb += rmp.get_s().sum()

            if lb > (1 + 1e-6) * best_lb:
                best_lb = lb
                best_lb_updated = True
            else:
                best_lb_updated = False

            journal.set_lb(lb=lb, data=y)
            subproblem_time = journal.walltime - subproblem_time_start

            if not stop_event.is_set():
                rmp.lb_hook(y, lb)
                if config["dual_optimizer.feed_subproblem_solution"] == 1:
                    rmp.add_cuts(
                        component_index=(
                            subproblems.evaluation_component_index
                        ),
                        evaluation_point=subproblems.evaluation_point,
                        function_value=subproblems.function_value,
                        subgradient_value=subproblems.subgradient_value,
                        subproblem_solution=(subproblems.subproblem_solution),
                    )
                else:
                    rmp.add_cuts(
                        component_index=(
                            subproblems.evaluation_component_index
                        ),
                        evaluation_point=subproblems.evaluation_point,
                        function_value=subproblems.function_value,
                        subgradient_value=subproblems.subgradient_value,
                    )
                primal_heuristic.lb_hook(y, lb)
                if config["cg.log_iterates"]:
                    msg = "sol:\n" + str(
                        np.round(np.array(subproblems.subproblem_solution), 2)
                    )
                    logger.info(msg)

                journal.start(constants.RoutineType.RMP)
                y = rmp.step()
                if config["cg.save_model_values"]:
                    journal.set_iteration_items(
                        model_value=rmp.get_model_value()
                    )

            if config["ph.time_ratio_to"] == "subproblem":
                ph_timelimit_measure = subproblem_time
            elif config["ph.time_ratio_to"] == "all":
                ph_timelimit_measure = (
                    journal.walltime - ph_timelimit_measure_start
                )
            else:
                raise ValueError(
                    f"invalid ph.time_ratio_to={config['ph.time_ratio_to']}"
                )
            ph_time_ratio = config["ph.time_ratio"]
            increment = config["ph.time_ratio_increment"]
            ph_timelimit = (
                ph_time_ratio + increment * (iteration_index / 10)
            ) * ph_timelimit_measure
            if ph_timelimit <= 0:
                ph_timelimit = np.inf
            remaining = max(timelimit - journal.walltime, 0.1)
            ph_timelimit = min(ph_timelimit, remaining)

            if not stop_event.is_set():
                solution_polishing_test = (
                    no_progress_count >= config["cg.no_progress_patience"] > 0
                ) and (config["cg.no_progress_action"] == "solution_polishing")

                if solution_polishing_test:
                    run_solution_polishing_flag = True
                    journal.start(constants.RoutineType.PRIMAL_HEURISTIC)
                    journal.set_iteration_items(solution_polishing=True)
                    solution_polishing_result = run_solution_polishing(
                        lb=journal.bounds_tracker.best_lb,
                        ub=journal.bounds_tracker.best_ub,
                        sol=journal.bounds_tracker.best_ub_data,
                        model=self.solution_polishing_model,
                        timelimit=remaining,
                        config=config,
                    )

                    journal.set_ub(
                        ub=solution_polishing_result["ub"],
                        data=solution_polishing_result["sol"],
                    )

                else:
                    # If we still have a gap, try to improve the
                    # upper bound. Run the primal heuristic.
                    journal.start(constants.RoutineType.PRIMAL_HEURISTIC)
                    ph_result = primal_heuristic.run(timelimit=ph_timelimit)
                    if ph_result.status:
                        journal.set_ub(ub=ph_result.obj, data=ph_result.sol)

            # Start measuring the next iteration.
            ph_timelimit_measure_start = journal.walltime

            ub = journal.get_ub_this_iteration()
            if ub < best_ub:
                best_ub = ub
                best_ub_updated = True
            else:
                best_ub_updated = False

            if (not best_lb_updated) and (not best_ub_updated):
                no_progress_count += 1
            else:
                no_progress_count = 0

            for hook in self.hooks:
                hook()

            if config["cg.log_type"] == "gap":
                log(
                    iteration=iteration_index + 1,
                    walltime=journal.walltime,
                    stepsize=original_stepsize,
                    lb=lb,
                    lb_updated=best_lb_updated,
                    ub=ub,
                    ub_updated=best_ub_updated,
                    gap=journal.get_gap(),
                    rmp_size=rmp.get_n_cuts(),
                    rmp_proximal_term_centre=rmp.proximal_centre,
                    header=iteration_index % 20 == 0,
                    log_type=config["cg.log_type"],
                    log_level=log_level,
                    logger=logger,
                    target_objective=self.target_objective,
                )
            else:
                log(
                    iteration=iteration_index + 1,
                    walltime=journal.walltime,
                    stepsize=original_stepsize,
                    lb=lb,
                    lb_updated=best_lb_updated,
                    ub=ub,
                    ub_updated=best_ub_updated,
                    gap=journal.get_gap(),
                    y=iter_y[-1],
                    header=iteration_index % 20 == 0,
                    log_type=config["cg.log_type"],
                    log_level=log_level,
                    rmp_size=rmp.get_n_cuts(),
                    rmp_proximal_term_centre=rmp.proximal_centre,
                    logger=logger,
                    target_objective=self.target_objective,
                )

            if journal.get_gap() <= config["tol"]:
                return self.make_returned(status=True, msg="")

            if np.isfinite(self.target_objective):
                tol_to_target_objective = config["tol_to_target_objective"]
                best_lb = journal.get_best_lb()
                if (
                    self.target_objective - best_lb
                ) / self.target_objective <= tol_to_target_objective:
                    return self.make_returned(status=True, msg="")

            if stop_event.is_set():
                return self.make_returned(status=False, msg="stop_event")

    def make_returned(self, status, msg):
        """Create a tuple to be returned"""
        logger = logging.getLogger(__name__)
        data = self.data
        config = self.config
        journal = self.journal
        stop_event = journal.stop_event
        dual_initialiser = self.dual_initialiser
        primal_heuristic = self.primal_heuristic

        timelimit = config.get("solver.timelimit", np.inf)
        if timelimit < 0:
            timelimit = np.inf
        iteration_limit = config.get("cg.iteration_limit", np.inf)

        # Notify the end to workers.
        stop_event.set()
        journal.end_hook()
        primal_heuristic.end_hook()

        # Compile results from the problem data and the journal.
        result = dict()
        for key in data:
            if key.startswith("instance_data."):
                result[key[14:]] = data[key]
        result.update(
            dict(
                algorithm=constants.Algorithm.CG,
                hostname=os.uname()[1],
                msg=msg,
                ph_type=config["cg.ph_type"],
                dual_initialiser=(
                    constants.as_dual_initialiser_typecode(dual_initialiser)
                ),
                status=status,
                stepsize=config["dual_optimizer.initial_stepsize"],
                adaptive_proximal_term_update=config[
                    "dual_optimizer.adaptive_proximal_term_update"
                ],
                timelimit=timelimit,
                iteration_limit=iteration_limit,
            )
        )
        self.rmp.dump_data(out=result)
        keys = [
            "dual_optimizer.initial_stepsize",
            "dual_optimizer.stepsize_rule",
            "dual_optimizer.stepsize_diminishing_offset",
            "dual_optimizer.initial_stepsize",
            "dual_optimizer.adaptive_proximal_term_update",
        ]
        for key in keys:
            result["config." + key] = config[key]
        journal.dump_data(out=result)
        for key in result:
            result[key] = np.asarray(result[key])
        result["lb"] = result.pop("best_lb")
        result["y"] = result.pop("best_lb_data")
        result["ub"] = result.pop("best_ub")
        sol = result["sol"] = result.pop("best_ub_data")
        if np.asarray(sol).ndim == 0:
            sol_nan = np.full(len(data["variable_lb"]), np.nan)
            result["sol"] = sol_nan
        elapse = result["walltime"]
        log_msg = (
            f"terminating CG.  "
            f"status: {str(status):5s}  "
            f"elapse: {elapse:6.1f}"
        )
        if msg:
            log_msg = log_msg + f"  msg: {msg}"
        log_level = logging.INFO
        logger.log(log_level, log_msg)
        return result


def log(
    iteration=None,
    walltime=None,
    stepsize=None,
    lb=None,
    ub=None,
    gap=None,
    y=None,
    lb_updated=None,
    ub_updated=None,
    header=False,
    log_type="gap",
    logger=None,
    log_level=logging.INFO,
    rmp_size=None,
    rmp_proximal_term_centre=None,
    target_objective=None,
):
    if logger is None:
        logger = logging.getLogger(__name__)
    if log_type == "gap":
        _header = (
            f"{'it':>3s} "
            f"{'elapse':>8s} "
            f"{'lb':>12s}  "
            f"{'ub':>12s}  "
            f"{'gap':>5s} "
            f"{'stepsz':>7s} "
            f"{'cuts':>4s} "
        )
        if header:
            logger.log(log_level, _header)
        else:
            logger.debug(_header)
        msg = ""
        if iteration is not None:
            msg += f"{iteration:3d} "
        else:
            msg += f"{'':3s} "
        if walltime is not None:
            msg += f"{utils.format_elapse(walltime)} "
        else:
            msg += f"{'':8s} "
        if lb is not None:
            symbol = "*" if lb_updated else " "
            if np.isfinite(target_objective):
                lb_error = (target_objective - lb) / target_objective * 100
                msg += f"{lb_error:12.5f}{symbol:1s} "
            else:
                msg += f"{lb:12.5e}{symbol:1s} "
        else:
            msg += f"{'':12s}  "
        if ub is not None:
            symbol = "*" if ub_updated else " "
            if np.isfinite(target_objective):
                ub_error = (ub - target_objective) / target_objective * 100
                msg += f"{ub_error:12.5f}{symbol:1s} "
            else:
                msg += f"{ub:12.5e}{symbol:1s} "
        else:
            msg += f"{'':11s}  "
        if gap is not None:
            msg += f"{gap*100:5.2f} "
        else:
            msg += f"{'':5s} "
        msg += f"{stepsize:7.1e} "
        if rmp_size is not None:
            msg += f"{rmp_size:4d}"
        else:
            msg += f"{'':4s}"
        logger.log(log_level, msg)
    else:
        _header = (
            f"{'it':>3s} "
            f"{'elapse':>8s} "
            f"{'lb':>12s}  "
            f"{'y':>10s} "
            f"{'stepsz':>7s} "
            f"{'cuts':>4s} "
        )
        if header:
            logger.log(log_level, _header)
        else:
            logger.debug(_header)
        msg = ""
        msg += f"{iteration:3d} "
        msg += f"{utils.format_elapse(walltime)} "
        symbol = "*" if lb_updated else " "
        msg += f"{lb:12.5e}{symbol:1s} "
        msg += f"{np.linalg.norm(y, 2):10.4e} "
        msg += f"{stepsize:7.1e} "
        msg += f"{rmp_size:4d} "
        logger.log(log_level, msg)


def run_solution_polishing(lb, ub, sol, model, timelimit, config):
    abort_at = lb / (1 - config["tol"])
    cplex_utils.abort_at(model, abort_at, "incumbent_callback")
    model.parameters.mip.polishafter.time.set(0.0)
    if np.isfinite(timelimit):
        model.parameters.timelimit.set(timelimit)
    if (sol is not None) and np.all(np.isfinite(sol)):
        var = np.arange(len(sol))
        mip_start = [
            list(map(int, var.ravel())),
            list(map(float, sol.ravel())),
        ]
        model.MIP_starts.add(
            mip_start,
            model.MIP_starts.effort_level.auto,
            "my_mip_start",
        )
    model.solve()
    return {
        "status": model.solution.get_status(),
        "ub": model.solution.get_objective_value(),
        "sol": np.array(model.solution.get_values()),
    }


if __name__ == "__main__":
    import doctest

    doctest.testmod()
