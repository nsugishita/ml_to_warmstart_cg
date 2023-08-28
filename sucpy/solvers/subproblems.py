# -*- coding: utf-8 -*-

"""Pool of workers to solve subproblems."""

import abc
import copy
import logging
import multiprocessing as mp
import time

import cplex
import numpy as np

from sucpy.solvers import common
from sucpy.utils import utils

# Flag to emit warning on imcompatibility of the parallel subproblem
# solver and stop_event.
warn_parallel_solver_ignores_stop_event = True


class Subproblem(cplex.Cplex):
    """Subproblem of column generation.

    This is a CPLEX model which holds a subproblem of column generation.
    Passing problem data and subproblem index, this constructs
    the MIP model.  Also there are a few helper methods to facilitate
    the main routine of column generation.
    """

    def __init__(self, subproblem_index, data, config={}):
        """Construct a CPLEX model of a specified subproblem.

        Parameters
        ----------
        data : dict
            Data returned from ParametrizedUC.sample_instance.
        subproblem_index : int
        config : dict, optional
            Configuration of the experiment.
        """
        super().__init__()
        self.parameters.threads.set(1)
        self.set_error_stream(None)
        self.set_log_stream(None)
        self.set_results_stream(None)
        self.set_warning_stream(None)
        if "tol" in config:
            self.parameters.mip.tolerances.mipgap.set(
                config["tol"] * config["cg.subprob_rtol"]
                + config["cg.subprob_atol"]
            )
        self.already_set_up = False
        self.set(subproblem_index=subproblem_index, data=data, all=True)

    def set(
        self,
        subproblem_index,
        data,
        *,
        x_lb=False,
        x_ub=False,
        c=False,
        b=False,
        B=False,
        all=False,
    ):
        """

        Parameters
        ----------
        subproblem_index : int
        data : dict
        x_lb : bool, default False
        x_ub : bool, default False
        c : bool, default False
        b : bool, default False
        B : bool, default False
        all : bool, default False
        """
        if not (x_lb or x_ub or c or b or B or all):
            raise ValueError(
                "at least one of x_lb, x_ub, c, b, B and all must be True"
            )
        self.subproblem_index = subproblem_index
        self.x_lb = data["subproblem_variable_lb"][subproblem_index]
        self.x_ub = data["subproblem_variable_ub"][subproblem_index]
        self.x_type = data["subproblem_variable_type"][subproblem_index]
        _c = data["subproblem_objective_coefficient"][subproblem_index]
        _b = data["subproblem_constraint_rhs"][subproblem_index]
        if self.already_set_up:
            if x_lb or all:
                self.variables.set_lower_bounds(
                    zip(range(self.n_vars), self.x_lb)
                )
            if x_ub or all:
                self.variables.set_upper_bounds(
                    zip(range(self.n_vars), self.x_ub)
                )
            if c or all:
                self.objective.set_linear(zip(range(self.n_vars), _c))
            if b or all:
                self.linear_constraints.set_rhs(zip(range(self.n_cons), _b))
        else:
            self.variables.add(
                obj=_c,
                lb=self.x_lb,
                ub=self.x_ub,
                types=common.to_str(self.x_type),
            )
            con_sense = data["subproblem_constraint_sense"][subproblem_index]
            if "U" in str(con_sense.dtype):
                con_sense = list(con_sense)
            else:
                con_sense = common.to_str(con_sense)
            self.linear_constraints.add(senses=con_sense, rhs=_b)
        if (not self.already_set_up) or B or all:
            _B = data["subproblem_constraint_coefficient_diagonal"][
                subproblem_index
            ].tocoo()
            self.linear_constraints.set_coefficients(
                zip(map(int, _B.row), map(int, _B.col), map(float, _B.data))
            )
        # Cache objects used in update_y.
        self.n_vars = data["n_variables"]
        self.n_cons = data["n_constraints"]
        self.D_s = data["subproblem_constraint_coefficient_row_border"][
            subproblem_index
        ]
        self.c = data["subproblem_objective_coefficient"][subproblem_index]
        # Reuse constraints next time.
        self.already_set_up = True

    def update_y(self, y):
        """Update the subproblem given values of dual variable.

        This updates the cost coefficient of the subproblem
        given values of the dual variable.

        Parameters
        ----------
        y : (n_linking_constraints,) array
        """
        y = np.broadcast_to(y, self.D_s.shape[0])
        c_new = self.c - self.D_s.T.dot(y)
        self.objective.set_linear(zip(range(self.n_vars), c_new))


class Subproblems(abc.ABC):
    """Subproblem runner."""

    def __init__(self, data, config, stop_event):
        """Initialise a Subproblems instance."""
        self.n_subproblems = data["n_subproblems"]
        self.data = data
        self.stop_event = stop_event
        self.evaluation_point = None
        self.function_value = None
        self.subgradient_value = None
        self.result = None
        self.evaluation_count_since_last_refresh = 0
        self.n_processes = 1
        # This will be incremented for each call of __enter__.
        self.stack = 0
        # Objects used in the parallel mode.
        self.manager = None
        self.processes = []
        self.p_conns = []
        self.c_conns = []
        # Set self.subproblems
        self.set_subproblems(data=data, config=config)

    @property
    def addresses_and_n_processes(self):
        """Get the number of processes used to solve subproblems."""
        return self._addresses_and_n_processes

    @addresses_and_n_processes.setter
    def addresses_and_n_processes(self, n):
        """Set the number of processes used to solve subproblems."""

        def replace_code_to_localhost(x):
            x = str(x)
            if x == "129.0.0.1":
                return "localhost"
            else:
                return x

        n = {replace_code_to_localhost(k): v for k, v in n.items()}
        for k, v in n.items():
            if v <= 0:
                raise ValueError(
                    f"n_processes['{k}'] should be positive but found {v}"
                )
        self._addresses_and_n_processes = n

    @property
    def n_processes(self):
        """Get the number of processes used to solve subproblems."""
        return sum(v for v in self._addresses_and_n_processes.values())

    @n_processes.setter
    def n_processes(self, n):
        """Get the number of processes used to solve subproblems."""
        self.addresses_and_n_processes = {"localhost": n}

    @property
    def addresses(self):
        """Get the addresses used to solve subproblems as tuple."""
        return tuple(k for k in self._addresses_and_n_processes.keys())

    def set_subproblems(self, *, subproblems=None, data=None, config=None):
        """Set or update data of subproblems.

        This set or updates data of subproblems.  If there are
        workers running, the data of the workers is also updated.
        """
        if subproblems is not None:
            self.subproblems = subproblems
        else:
            self.data = data
            self.subproblems = [
                Subproblem(i, data, config)
                for i in range(data["n_subproblems"])
            ]
        # Set self.subproblems and update the remote workers if running.
        sent = {
            "type": SIGNAL_SET_SUBPROBLEMS,
            "subproblems": self.subproblems,
        }
        for p_conn in self.p_conns:
            p_conn.send(sent)
        self._join()

    def _join(self):
        """Block until the subproblem workers reply.

        After sending commands (subproblem setters and kill comand),
        by calling this methods one can wait for the workers
        to complete the command.
        If there are not workers, this does not do anything.
        """
        for p_conn in self.p_conns:
            p_conn.recv()

    def start_workers(self):
        """Start running subproblem workers.

        This starts running subproblem workers
        if `n_processes` is larger than one.
        If `n_processes` is equal to one,
        this does not do anything.
        """
        n_total_processes = self.n_processes
        if n_total_processes <= 1:
            return
        assert (
            len(self.processes) == len(self.p_conns) == len(self.c_conns) == 0
        )
        if self.addresses == ("localhost",):
            for i in range(n_total_processes):
                p_conn, c_conn = mp.Pipe()
                kwargs = {
                    "worker_rank": i,
                    "n_subproblems": self.n_subproblems,
                    "subproblems": self.subproblems,
                    "n_workers": n_total_processes,
                    "c_conn": c_conn,
                }
                process = mp.Process(
                    target=_subproblem_pool_worker, kwargs=kwargs
                )
                process.start()
                self.processes.append(process)
                self.p_conns.append(p_conn)
                self.c_conns.append(c_conn)
        else:
            raise NotImplementedError

    def __enter__(self, *args, **kwargs):
        """Start subproblem workers if necessary."""
        self.stack += 1
        if (self.stack == 1) and (self.n_processes > 1):
            self.start_workers()
        return self

    def __exit__(self, *args, **kwargs):
        """Clear subproblem workers."""
        assert self.stack > 0
        self.stack -= 1
        if self.stack == 0:
            self.close()
        if self.manager:
            self.manager.shutdown()

    def close(self):
        """Clear subproblem workers."""
        for p_conn in self.p_conns:
            p_conn.send({"type": SIGNAL_KILL})
        self._join()
        for p in self.processes:
            p.join()
        self.processes.clear()
        self.p_conns.clear()
        self.c_conns.clear()

    def evaluate(self, component_index, y):
        """Solve subproblems and return the results.

        Parameters
        ----------
        component_index : 1d array of int
        y : (dim_dual,) array

        Returns
        -------
        result : list of dict
            List of results returned from `solve`.
        """
        # Refresh CPLEX model if necessary.
        if self.evaluation_count_since_last_refresh > 1000:
            self.set_subproblems(subproblems=copy.deepcopy(self.subproblems))
            self.evaluation_count_since_last_refresh = 0
        else:
            self.evaluation_count_since_last_refresh += 1

        if self.n_processes <= 1:
            self.result = self._run_sequentially(component_index, y)
        else:
            self.result = self._run_in_parallel(component_index, y)

        # Check all the subproblems are solved to optimality.
        for i, s in enumerate(self.result):
            if s["status"] not in [101, 102, 113]:
                # 113: aborted by user
                raise ValueError(
                    f"failed to solve a subproblem.  "
                    f"instance id: {self.data['instance_id']}  "
                    f"subproblem: {i}  "
                    f"status: {s['status']}  "
                )

        # Extract result.
        self.evaluation_component_index = np.array(
            [r["component_index"] for r in self.result]
        )
        self.evaluation_point = y
        self.function_value = np.array(
            [r["objective_bound"] for r in self.result]
        )
        self.objective_lb = np.array(
            [r["objective_bound"] for r in self.result]
        )
        self.objective_ub = np.array(
            [r["best_objective"] for r in self.result]
        )
        self.subproblem_solution = [r["solution"] for r in self.result]
        D = self.data["subproblem_constraint_coefficient_row_border"]
        self.subgradient_value = np.array(
            [
                -D[i].dot(sol)
                for i, sol in zip(component_index, self.subproblem_solution)
            ]
        )
        return self.result

    def _run_sequentially(self, component_index, y):
        result = []
        for s in component_index:
            if self.stop_event.is_set():
                break
            subproblem = self.subproblems[s]
            starttime = time.perf_counter()
            subproblem.update_y(y)
            res = solve(subproblem)
            res["component_index"] = s
            res["walltime"] = time.perf_counter() - starttime
            result.append(res)
        return result

    def _run_in_parallel(self, component_index, y):
        global warn_parallel_solver_ignores_stop_event
        test = (
            not isinstance(self.stop_event, utils.DummyEvent)
            and warn_parallel_solver_ignores_stop_event
        )
        if test:
            logger = logging.getLogger(__name__)
            logger.warning(
                "stop event is not yet supported in "
                "the parallel subproblem solver"
            )
            warn_parallel_solver_ignores_stop_event = False
        if len(component_index) != len(self.subproblems):
            raise NotImplementedError
        for p_conn in self.p_conns:
            p_conn.send({"type": SIGNAL_SOLVE, "y": y})
        ret = []
        for p_conn in self.p_conns:
            ret.extend(p_conn.recv())
        return ret


def get_assignments(ntasks, nworkers, worker_rank):
    """Given the number of tasks and workers, return the assigned tasks."""
    if worker_rank != nworkers - 1:
        a = ntasks // nworkers
        return range(worker_rank * a, (worker_rank + 1) * a)
    else:
        a = ntasks // nworkers
        return range(worker_rank * a, ntasks)


def _subproblem_pool_worker(
    worker_rank,
    n_subproblems,
    subproblems,
    n_workers,
    c_conn,
):
    assignments = get_assignments(n_subproblems, n_workers, worker_rank)
    subproblems = subproblems

    while True:
        try:
            item = c_conn.recv()
        except EOFError:
            return
        if item["type"] == SIGNAL_KILL:
            c_conn.send(None)
            return
        elif item["type"] == SIGNAL_SET_SUBPROBLEMS:
            subproblems = item["subproblems"]
            c_conn.send(None)
        elif item["type"] == SIGNAL_SOLVE:
            y = item["y"]
            result_buffer = []
            for i in assignments:
                try:
                    subproblems[i].update_y(y)
                except IndexError:
                    print(i)
                    raise
                res = solve(subproblems[i])
                res["component_index"] = i
                result_buffer.append(res)
            c_conn.send(result_buffer)


def solve(problem):
    """Solve cplex model.

    Parameters
    ----------
    problem : cplex.Cplex

    Returns
    -------
    result : dict
        This contains the following items:
        - status : int
            `problem.solution.get_status()`
        - objective_bound : float
            `problem.solution.get_objective()`.
        - best_objective : float
            `problem.solution.MIP.get_best_objective()`
            This is set to be None when problem is not solved to optimal.
        - solution : (dim_sol,) array
            `problem.solution.get_values()`.
            This is set to be None when problem is not solved to optimal.
        - solution_pool_n_solutions : int
            Number of solutions found.
            This is set to be 0 when problem is not solved to optimal.
        - solution_pool_values : (solution_pool_n_solutions, dim_sol) array
            Solutions.
            This is set to be an empty when problem is not solved to optimal.
        - solution_pool_objective_values : (solution_pool_n_solutions,) array
            Objective values.
            This is set to be an empty when problem is not solved to optimal.
    """
    problem.set_error_stream(None)
    problem.set_log_stream(None)
    problem.set_results_stream(None)
    problem.set_warning_stream(None)
    problem.solve()
    res = {
        "status": problem.solution.get_status(),
        "objective_bound": None,
        "best_objective": None,
        "solution": None,
        "solution_pool_n_solutions": 0,
        "solution_pool_values": np.array([]),
        "solution_pool_objective_values": np.array([]),
    }
    try:
        res["objective_bound"] = problem.solution.get_objective_value()
        res["best_objective"] = problem.solution.MIP.get_best_objective()
        res["solution"] = np.array(problem.solution.get_values())
        n = res["solution_pool_n_solutions"] = problem.solution.pool.get_num()
        res["solution_pool_values"] = np.array(
            [problem.solution.pool.get_values(i) for i in range(n)]
        )
        res["solution_pool_objective_values"] = np.array(
            [problem.solution.pool.get_objective_value(i) for i in range(n)]
        )
    except cplex.CplexError:
        pass
    return res


# Constants

SIGNAL_KILL = 0
SIGNAL_SET_SUBPROBLEMS = 1
SIGNAL_SOLVE = 2


if __name__ == "__main__":
    import doctest

    doctest.testmod()
