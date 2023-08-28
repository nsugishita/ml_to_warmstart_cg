# -*- coding: utf-8 -*-

"""Regularised piecewise affine model"""

import copy
import logging
import time

import numpy as np
import scipy.sparse

from sucpy import constants
from sucpy.solvers import common
from sucpy.utils import cplex_utils, utils


def RegularizedRMP(data, config, journal):
    """Create a regularized restricted master problem (RMP)"""
    dim_x = (data["constraint_subproblem_index"] == -1).sum()
    linking_constraint_sense = common.to_byte_array(
        data["subproblem_constraint_sense"][-1]
    )
    c_x = data["subproblem_constraint_rhs"][-1]
    x_lb = np.full(dim_x, -np.inf)
    G = b"G"[0]
    L = b"L"[0]
    x_lb[linking_constraint_sense == G] = 0.0
    x_ub = np.full(dim_x, np.inf)
    x_ub[linking_constraint_sense == L] = 0.0

    c = data["subproblem_objective_coefficient"][-1]
    D = data["subproblem_constraint_coefficient_row_border"][-1]
    assert c.size == D.shape[1]
    xlb = data["subproblem_variable_lb"][-1]
    xub = data["subproblem_variable_ub"][-1]
    UNBOUNDED = 0
    BOUNDED_FROM_BELOW = 1
    BOUNDED_FROM_ABOVE = 2
    BOUNDED_FROM_BOTH = 3
    var_bound_type = np.zeros(len(c))
    var_bound_type[np.isfinite(xlb)] += 1
    var_bound_type[np.isfinite(xub)] += 2

    # Construct constructs.
    dim_s = np.sum(var_bound_type == BOUNDED_FROM_BOTH)
    n_constraints = c.size + np.sum(var_bound_type == BOUNDED_FROM_BOTH)
    A_x = scipy.sparse.coo_matrix((n_constraints, dim_x))
    A_x_row = []
    A_x_col = []
    A_x_data = []
    A_s = scipy.sparse.dok_matrix((n_constraints, dim_s))
    constraint_rhs = []
    constraint_sense = ""
    current_row_index = 0
    lub_count = 0
    for d_col_index, d_row_index, d_data in utils.yield_columns(D):
        # raise NotImplementedError  # Not yet tested.
        n_nonzeros = d_row_index.size
        _xlb = xlb[d_col_index]
        _xub = xub[d_col_index]
        _c = c[d_col_index]
        _var_bound_type = var_bound_type[d_col_index]
        if _var_bound_type == BOUNDED_FROM_BOTH:
            # (D_{0j} xlb_{0j})^T y + s_j <= c_{0j} xlb_{0j}
            # (D_{0j} xub_{0j})^T y + s_j <= c_{0j} xub_{0j}
            A_x_row.append(
                np.repeat(
                    [current_row_index, current_row_index + 1], n_nonzeros
                )
            )
            A_x_col.extend([d_row_index, d_row_index])
            A_x_data.extend([d_data * _xlb, d_data * _xub])
            A_s[current_row_index, lub_count] = 1
            A_s[current_row_index + 1, lub_count] = 1
            constraint_rhs.extend([_c * _xlb, _c * _xub])
            constraint_sense += "LL"
            current_row_index += 2
            lub_count += 1
        else:
            # D_{0j}^T y  [ <= | >= | = ]  c_{0j}
            A_x_row.append(np.repeat(current_row_index, n_nonzeros))
            A_x_col.append(d_row_index)
            A_x_data.append(d_data)
            constraint_rhs.append(_c)
            if _var_bound_type == UNBOUNDED:
                constraint_sense.append("E")
            elif _var_bound_type == BOUNDED_FROM_BELOW:
                constraint_sense.append("L")
                c_x -= d_data * xlb[d_col_index]
                raise NotImplementedError  # TODO Adjust offset
                # objective_offset += c[d_col_index] * xlb[d_col_index]
            elif _var_bound_type == BOUNDED_FROM_ABOVE:
                constraint_sense.append("G")
                c_x -= d_data * xub[d_col_index]
                # objective_offset += c[d_col_index] * xub[d_col_index]
            else:
                raise ValueError(
                    f"var {d_col_index} has invalid "
                    f"var_bound_type {_var_bound_type}"
                )
            current_row_index += 1
    if n_constraints > 0:
        A_x.row = np.r_[tuple(A_x_row)].astype(np.int32)
        A_x.col = np.r_[tuple(A_x_col)].astype(np.int32)
        A_x.data = np.r_[tuple(A_x_data)].astype(A_x.data.dtype)
        constraint_rhs = np.r_[constraint_rhs]
        constraint_sense = np.array(list(constraint_sense))
    A_s = A_s.tocoo((n_constraints, dim_s))

    cp_data = {
        "instance_id": data.get("instance_id", -1),
        "objective_sense": "max",
        "dim_x": dim_x,
        "dim_s": dim_s,
        "n_components": data["n_subproblems"],
        "x_lb": x_lb,
        "x_ub": x_ub,
        "c_x": c_x,
        "s_lb": np.full(dim_s, -np.inf),
        "s_ub": np.full(dim_s, np.inf),
        "c_s": np.ones(dim_s),
        "constraint_matrix_x": A_x,
        "constraint_matrix_s": A_s,
        "constraint_rhs": constraint_rhs,
        "constraint_sense": constraint_sense,
        "regularization_scaling": data.get("y_regularization_scaling", 1),
        "primal_border_submatrices": data[
            "subproblem_constraint_coefficient_row_border"
        ],
    }

    return CuttingPlaneMethod(cp_data, config, journal)


class CuttingPlaneMethod(object):
    """Regularized cutting-plane method

    This is a CPLEX model to run the regularized cutting-plane method.
    This solves the following problem:

      min_{x, s}  c_x x + c_s s + sum_{i = 0, ..., m - 1} f_i(x)
      s.t         A_x x + A_s s  [<=, =, >=]  a,
                  x_lb <= x <= x_ub,
                  s_lb <= s <= s_ub.

    `x` and `s` are supposed to be continuous.
    For each `i`, `f_i` is a convex function which is
    approximated by a piece-wise affine model.

    `problem_data` should contain the following items.

    - objective_sense : {"min", "max"}, default "min"
    - n_components : int, `m`, default 1
    - dim_x : int
    - x_lb : (dim_x,) array, `x_lb`, default -inf.
    - x_ub : (dim_x,) array, `x_ub`, default inf.
    - c_x : (dim_x,) array, `c_x`, default 0
    - dim_s : int, default 0, optional
    - s_lb : (dim_s,) array, `s_lb`, optional
    - s_ub : (dim_s,) array, `s_ub`, optional
    - c_s : (dim_s,) array, `c_s`, optional
    - constraint_matrix_x : (dim_x, n_constraints) COO matrix, `A_x`, optional
    - constraint_matrix_s : (dim_x, n_constraints) COO matrix, `A_s`, optional
    - constraint_rhs : (n_constraints,) array, `a`, optional
    - constraint_sense : (n_constraints,) array of {"L", "G", "E"}, optional
    """

    def __init__(self, problem_data, config, journal):
        """Initialise a CuttingPlaneMethod instance

        This creates a cplex model which represents a regularised
        piecewise affine model.
        """
        super().__init__()
        self.problem_data = copy.deepcopy(problem_data)
        self.config = copy.deepcopy(config)
        self.journal = journal
        self.instance_id = problem_data.get("instance_id", -1)

        if self.journal is not None:
            self.journal.register_iteration_items(
                rmp_solution_count=dict(default=0, timing=False),
            )

        keys = [
            "objective_sense",
            "n_components",
            "dim_s",
            "s_lb",
            "s_ub",
            "x_lb",
            "x_ub",
            "x_type",
            "c_x",
            "c_s",
            "constraint_matrix_x",
            "constraint_matrix_s",
            "constraint_rhs",
            "constraint_sense",
        ]

        for key in keys:
            if key in self.problem_data and self.problem_data[key] is None:
                self.problem_data.pop(key)

        self.problem_data.setdefault("objective_sense", "min")
        self.problem_data.setdefault("n_components", 1)
        self.problem_data.setdefault("dim_s", 0)
        dim_x = self.problem_data["dim_x"]
        dim_s = self.problem_data["dim_s"]
        self.problem_data.setdefault("s_lb", np.full(dim_s, -np.inf))
        self.problem_data.setdefault("s_ub", np.full(dim_s, np.inf))
        self.problem_data.setdefault("x_lb", np.full(dim_x, -np.inf))
        self.problem_data.setdefault("x_ub", np.full(dim_x, np.inf))
        self.problem_data.setdefault("x_type", np.full(dim_x, "C"))
        self.problem_data.setdefault("c_x", np.zeros(dim_x))
        self.problem_data.setdefault("c_s", np.zeros(dim_s))
        self.problem_data.setdefault(
            "constraint_matrix_x", scipy.sparse.coo_matrix((0, dim_x))
        )
        self.problem_data.setdefault(
            "constraint_matrix_s", scipy.sparse.coo_matrix((0, dim_s))
        )
        n_constraints = self.problem_data["constraint_matrix_x"].shape[0]
        self.problem_data.setdefault(
            "constraint_rhs", np.full(n_constraints, 0)
        )
        self.problem_data.setdefault(
            "constraint_sense", np.full(n_constraints, "E")
        )

        for key in ["x_lb", "x_ub", "c_x", "s_lb", "s_ub", "c_s"]:
            self.problem_data[key] = np.atleast_1d(
                self.problem_data[key]
            ).astype(float)
        if isinstance(self.problem_data["x_type"], str):
            self.problem_data["x_type"] = np.array(
                [self.problem_data["x_type"]]
            )
        self.problem_data["x_type"] = np.atleast_1d(
            self.problem_data["x_type"]
        )

        self.model = cplex_utils.Cplex()
        self.model.set_error_stream(None)
        self.model.set_log_stream(None)
        self.model.set_results_stream(None)
        self.model.set_warning_stream(None)
        self.model.parameters.threads.set(1)
        method_name = config["dual_optimizer.cplex_method"]
        self.model.parameters.lpmethod.set(
            constants.method_name_to_cplex_qp_method(method_name)
        )
        self.model.parameters.qpmethod.set(
            constants.method_name_to_cplex_qp_method(method_name)
        )
        if self.problem_data["objective_sense"] == "max":
            self.model.objective.set_sense(self.model.objective.sense.maximize)
        elif self.problem_data["objective_sense"] == "min":
            self.model.objective.set_sense(self.model.objective.sense.minimize)
        else:
            raise ValueError(
                "invalid objective_sense: "
                f"{self.problem_data['objective_sense']}"
            )

        self.c_x = self.problem_data["c_x"]
        x_lb = self.problem_data["x_lb"]
        x_ub = self.problem_data["x_ub"]
        x_type = self.problem_data["x_type"]
        c_s = self.problem_data["c_s"]
        s_lb = self.problem_data["s_lb"]
        s_ub = self.problem_data["s_ub"]
        c_r = [1.0] * self.problem_data["n_components"]
        r_lb = [-np.inf] * self.problem_data["n_components"]
        r_ub = [np.inf] * self.problem_data["n_components"]

        if np.all(x_type == "C"):
            # If all the x_type is "C" we do not pass it. If we specify x_type
            # CPLEX always makes the problem type mixed-integer.
            self.cplex_index_x = np.array(
                self.model.variables.add(
                    obj=self.c_x.tolist(),
                    lb=x_lb.tolist(),
                    ub=x_ub.tolist(),
                )
            )
        else:
            self.cplex_index_x = np.array(
                self.model.variables.add(
                    obj=self.c_x.tolist(),
                    lb=x_lb.tolist(),
                    ub=x_ub.tolist(),
                    types="".join(x_type),
                )
            )
        self.cplex_index_s = np.array(
            self.model.variables.add(
                obj=c_s.tolist(),
                lb=s_lb.tolist(),
                ub=s_ub.tolist(),
            )
        )
        self.cplex_index_r = np.array(
            self.model.variables.add(
                obj=c_r,
                lb=r_lb,
                ub=r_ub,
            )
        )
        # Set up constraints
        Ax = self.problem_data["constraint_matrix_x"]
        As = self.problem_data["constraint_matrix_s"]
        if Ax.shape[0] > 0:
            sense = self.problem_data["constraint_sense"]
            if isinstance(sense, (np.ndarray, list)):
                sense = "".join(sense)
            self.cplex_index_constraint = self.model.linear_constraints.add(
                rhs=self.problem_data["constraint_rhs"].tolist(),
                senses=sense,
            )
            row = map(int, np.r_[Ax.row, As.row])
            x_start = self.cplex_index_x[0]
            s_start = self.cplex_index_s[0] if As.shape[0] > 0 else 0
            col = map(
                int,
                np.r_[
                    Ax.col + x_start,
                    As.col + s_start,
                ],
            )
            data = map(float, np.r_[Ax.data, As.data])
            self.model.linear_constraints.set_coefficients(zip(row, col, data))

        self.n_components = self.problem_data["n_components"]
        self.n_constraints = len(self.problem_data["constraint_sense"])
        self.S = range(self.n_components)

        # Pools to store cuts.
        if config["dual_optimizer.check_duplicate"]:
            import uniquelist

            self.cut_unique_list = [
                uniquelist.UniqueArrayList(Ax.shape[1]) for _ in self.S
            ]
        else:
            self.cut_unique_list = None
        # (n_components)-list of 1d array
        self.cut_first_iteration = [np.array([]) for _ in self.S]
        self.cut_iteration = [np.array([]) for _ in self.S]
        # (n_components)-list of (n_cols, dim_dual) array
        self.cut_subgradient_value = [np.array([]) for _ in self.S]
        # (n_components)-list of 1d array
        self.cut_cplex_index = [np.array([], dtype=int) for _ in self.S]
        # (n_components)-list of 1d array
        self.cut_rhs = [np.array([]) for _ in self.S]
        # (n_components)-list of (n_cols, dim_col) array
        self.cut_data = [np.array([]) for _ in self.S]
        self.cut_active = [np.array([], dtype=bool) for _ in self.S]

        self.iteration_index = -1
        self.iter_y = []
        self.best_lb = None

        self.proximal_centre = np.zeros(dim_x)
        self.stepsize = config["dual_optimizer.initial_stepsize"]
        # These will be set/updated in self.solve.
        self._cached_objective_value = None
        self._cached_solution = None
        self._cached_status = None
        self.objective_scaling_type = config.get(
            "dual_optimizer.objective_scaling_type", "pessimistic"
        )

        self.starttime = None
        self.switched_to_full_step = False

    def start_hook(self):
        self.starttime = time.perf_counter()

    def initialiser_hook(self, point):
        """Handle an initialiser output

        This is a hook called after the initialiser returns
        an initial point.  This returns component indices
        which should be evaluated by the oracle.

        Parameters
        ----------
        point : (dim_x,) array

        Returns
        -------
        component_index : 1d array of int
        """
        self.proximal_centre = point

    def lb_hook(self, point, lb):
        """Handle a lower bound

        This is a hook called after evaluation of a lower bound.

        Parameters
        ----------
        point : (dim_x,) array
        lb : float
        """
        if np.isnan(lb):
            raise ValueError

        # Update the proximal centre.
        if self.config["dual_optimizer.adaptive_proximal_term_update"]:
            if self.iteration_index == 0:
                # This is redundant since we set it in initialiser_hook
                # but in case we modify the other hooks we update
                # the regularization centre here.
                self.proximal_centre = point
            elif np.isfinite(lb):
                if self.best_lb is not None:
                    if self.best_lb < lb:  # Improved.
                        self.proximal_centre = point
            else:
                # No LB available.  Update the centre.
                self.proximal_centre = point
        else:
            # Always update the proximal centre.
            self.proximal_centre = point

        # Update the stepsize.
        rule = self.config["dual_optimizer.stepsize_rule"]
        if rule == constants.StepsizeRule.ADAPTIVE:
            # Only update when LB is available.
            if np.isfinite(lb) and (self.best_lb is not None):
                coef = self.config[
                    "dual_optimizer.stepsize_update_coefficient"
                ]
                if self.best_lb < lb:  # Improved.
                    self.stepsize *= coef
                else:
                    self.stepsize /= coef
                minimum = self.config["dual_optimizer.minimum_stepsize"]
                if minimum > 0:
                    self.stepsize = max(self.stepsize, minimum)
                maximum = self.config["dual_optimizer.maximum_stepsize"]
                if maximum > 0:
                    self.stepsize = min(self.stepsize, maximum)
        elif rule == constants.StepsizeRule.DIMINISHING:
            # initial_stepsize * offset / (cycle_number + offset)
            initial = self.config["dual_optimizer.initial_stepsize"]
            offset = self.config["dual_optimizer.stepsize_diminishing_offset"]
            # We use iteration_index + 1 since we want to compute
            # stepsize of the next iteration.
            cycle_number = np.floor((self.iteration_index + 1))
            self.stepsize = initial * offset / (offset + cycle_number)

        if self.best_lb is None:
            self.best_lb = lb
        else:
            self.best_lb = np.maximum(self.best_lb, lb)

    def step(self):
        """Compute the next point to be evaluated by the oracle

        Returns
        -------
        component_index : 1d array of int
        point : (dim_x,) array
        """
        memory = self.config["dual_optimizer.memory_size"]
        if memory > 0:
            self.remove_cuts_older_than(self.iteration_index - memory - 1)

        original_stepsize = self.stepsize

        while True:
            self.solve()
            if self.journal is not None:
                self.journal.iteration_journal.iteration_item_values[
                    "rmp_solution_count"
                ][-1] += 1
            if self.get_status() in [1, 5, 6]:
                # 5 CPX_STAT_OPTIMAL_INFEAS
                # 6 : CPX_STAT_NUM_BEST
                # Solution is available, but not proved optimal, due to
                # numeric difficulties during optimization.
                break

            # We should not arrive here.
            status_name = self.get_status_name()
            msg = (
                f"failed to solve the CP model. "
                f"instance id: {self.instance_id}  "
                f"iteration: {self.iteration_index}  "
                "stepsize: "
                f"{self.stepsize:7.1e}  "
                f"status: {status_name}  "
            )
            logger = logging.getLogger(__name__)
            logger.info(msg)

            # 2 CPX_STAT_UNBOUNDED
            # 3 CPX_STAT_INFEASIBLE
            # 4 CPX_STAT_INForUNBD
            should_decrease_stepsize = (self.get_status() in [2, 3, 4]) and (
                self.stepsize > 1e-6
            )

            if should_decrease_stepsize:
                self.stepsize /= 2
            else:
                raise ValueError(msg)

        # In case we modified stepsize due to numerical
        # difficulties, set the value back.
        if self.stepsize != original_stepsize:
            self.stepsize = original_stepsize

        # Flag active cuts.
        for s, weight in enumerate(self.get_weight()):
            self.cut_active[s] = ~np.isclose(weight, 0)

        return self.get_x()

    def get_status(self):
        """Return status of the previous solve"""
        return self._cached_status

    def get_status_name(self):
        """Return status name of the previous solve"""
        return self.model.solution.status[self._cached_status]

    def get_objective_value(self):
        """Return objective values

        This must be called after calling `self.solve`.

        Returns
        -------
        objective_value : float
        """
        v = self._cached_objective_value
        if v is None:
            raise ValueError(
                f"solution not found  (status: {self.get_status_name()})"
            )
        return v

    def get_x(self):
        """Return values of x

        This must be called after calling `self.solve`.
        This returns values of x.

        Returns
        -------
        x : (dim_x,) array
        """
        sol = self._cached_solution
        if sol is None:
            raise ValueError(
                f"solution not found  (status: {self.get_status_name()})"
            )
        return sol[self.cplex_index_x]

    def get_s(self):
        """Return values of s

        This must be called after calling `self.solve`.
        This returns values of s.

        Returns
        -------
        s : (dim_s,) array
        """
        if self.problem_data["dim_s"] == 0:
            return []
        else:
            dual = self._cached_solution
            if dual is None:
                raise ValueError(
                    f"solution not found  (status: {self.get_status_name()})"
                )
            return dual[self.cplex_index_s]

    def get_model_value(self):
        """Return the model values at the current point

        Returns
        -------
        value : (n_components,) array
        """
        sol = self._cached_solution
        if sol is None:
            return np.full(self.problem_data["n_components"], np.nan)
        return sol[self.cplex_index_r]

    def get_weight(self):
        """Return the weight of each cut

        Returns
        -------
        value : list of (n_cuts[s],)-array
        """
        res = []
        for s in self.S:
            res.append(self._cached_dual_solution[self.cut_cplex_index[s]])
        return res

    def get_weighted_data(self):
        """Return weighted sum of data according to the values of dual values

        Returns
        -------
        value : list of (dim_cut[s],)-array
            This is the values of the convex combination coefficients
            of the columns.  This is the same format at `cut_data`.
        """
        res = []
        for s in self.S:
            dual_s = self._cached_dual_solution[self.cut_cplex_index[s]]
            res.append(np.sum(dual_s[:, None] * self.cut_data[s], axis=0))
        return res

    def dump_data(self, out=None):
        if out is None:
            out = dict()
        if self.config["cg.save_cuts"]:
            cuts = self.get_cuts()
            out["n_cuts"] = len(cuts["subgradient_value"])
            out.update({f"cuts_{k}": v for k, v in cuts.items()})
        return out

    @classmethod
    def concatenate_cuts_list(cls, cuts_list):
        import copy

        res = {}
        for i, cuts in enumerate(cuts_list):
            if i == 0:
                res = copy.deepcopy(cuts)
                continue
            for key in cuts:
                res[key] = np.concatenate([res[key], cuts[key]], axis=0)
        return res

    @classmethod
    def unique_cuts(
        cls,
        *,
        component_index=None,
        evaluation_point=None,
        function_value=None,
        subgradient_value=None,
        rhs=None,
    ):
        """Return unique cuts

        Parameters
        ----------
        component_index : (n_cuts,) array of int
        evaluation_point : (n_cuts, dim_x) array
        function_value : (n_cuts,) array
        subgradient_value : (n_cuts, dim_x) array
        rhs : (n_cuts,) array of float
        """
        assert component_index is not None
        assert subgradient_value is not None
        res = {}
        keys = [
            "component_index",
            "evaluation_point",
            "function_value",
            "subgradient_value",
            "rhs",
        ]
        for key in keys:
            if locals()[key] is not None:
                res[key] = []
        for _component_index in np.unique(component_index):
            selector = np.nonzero(component_index == _component_index)[0]
            # if subgradient_value is not None:
            buf = subgradient_value[selector]
            _, index = np.unique(buf, axis=0, return_index=True)
            for key in res:
                res[key].append(locals()[key][selector][index])
        for key in res:
            res[key] = np.concatenate(res[key], axis=0)
        return res

    def add_cuts(
        self,
        *,
        component_index=None,
        evaluation_point=None,
        function_value=None,
        subgradient_value=None,
        rhs=None,
        subproblem_solution=None,
    ):
        """Add cuts

        Add cuts to the model.  `component_index` and `subgradient_value`
        is required.  Furthermore, (`evaluation_point`, `function_value`)
        or `rhs` must be provided.  `subproblem_solution` is always optional.
        All parameters must be given as keyword arguments.

        Parameters
        ----------
        component_index : (n_cuts,) array of int
        evaluation_point : (n_cuts, dim_x) array
        function_value : (n_cuts,) array
        subgradient_value : (n_cuts, dim_x) array
        rhs : (n_cuts,) array
        subproblem_solution : (n_cuts)-list of 1d array
        """
        all_none = (
            (component_index is None)
            and (evaluation_point is None)
            and (function_value is None)
            and (subgradient_value is None)
            and (rhs is None)
            and (subproblem_solution is None)
        )
        if all_none:
            return np.array([], dtype=bool)
        if (component_index is None) or (len(component_index) == 0):
            return np.array([], dtype=bool)
        assert component_index is not None
        if rhs is None:
            assert evaluation_point is not None
            assert function_value is not None
        if subgradient_value is None:
            assert subproblem_solution is not None
            assert "primal_border_submatrices" in self.problem_data
            D = self.problem_data["primal_border_submatrices"]
            subgradient_value = np.array(
                [
                    -D[i].dot(sol)
                    for i, sol in zip(component_index, subproblem_solution)
                ]
            )
        if evaluation_point is not None:
            evaluation_point = np.atleast_2d(evaluation_point)
            if evaluation_point.shape[0] == 1:
                evaluation_point = np.broadcast_to(
                    evaluation_point,
                    (len(component_index), evaluation_point.shape[1]),
                )

        cut_unique_list = self.cut_unique_list
        cut_first_iteration = self.cut_first_iteration
        cut_iteration = self.cut_iteration
        cut_subgradient_value = self.cut_subgradient_value
        cut_cplex_index = self.cut_cplex_index
        cut_rhs = self.cut_rhs
        cut_data = self.cut_data
        cut_active = self.cut_active

        # If the problem is minimization, the cuts are
        #     r_i >= g^T (x_i - z) + f
        # or
        #     -g^T x_i + r_i >= f - g^T z
        # where z : evaluation_point,  i : component_index,
        # f : function_value and g : subgradient_value.
        # If the problem is maximization, the inequality
        # will be <= instead of >=.

        # Check whether given cuts are already added or not.
        if not self.config["dual_optimizer.check_duplicate"]:
            new = np.full(len(component_index), True)
        else:
            buf = [
                cut_unique_list[ci].push_back(sv)
                for ci, sv in zip(component_index, subgradient_value)
            ]
            pos = np.array([x[0] for x in buf])
            new = np.array([x[1] for x in buf])

            # Update the iteration index of existing cuts.
            dup_component_index = component_index[~new]
            dup_pos = pos[~new]
            for _idx, _pos in zip(dup_component_index, dup_pos):
                try:
                    cut_iteration[_idx][_pos] = self.iteration_index
                except IndexError:
                    # If the same cut is added more than once, the unique
                    # list report the duplicated cuts as not new.
                    # However, since the cut is not in `cut_iteration`,
                    # we may be IndexError.  This does not cause any
                    # issues and we simply ignore it.
                    pass

            # Filter only new cuts.
            new_index = np.nonzero(new)[0]
            component_index = component_index[new_index]
            if len(component_index) == 0:
                return new  # No new cuts.  Exit now.
            if evaluation_point is not None:
                evaluation_point = evaluation_point[new_index]
            if function_value is not None:
                function_value = function_value[new_index]
            if subgradient_value is not None:
                subgradient_value = subgradient_value[new_index]
            if subproblem_solution is not None:
                subproblem_solution = [
                    subproblem_solution[i] for i in new_index
                ]
            if rhs is not None:
                rhs = rhs[new_index]

        n_cuts = len(component_index)
        dim_x = self.problem_data["dim_x"]
        cplex_index_x = self.cplex_index_x
        cplex_index_r = self.cplex_index_r

        if rhs is None:
            rhs = function_value - np.sum(
                subgradient_value * evaluation_point, axis=1
            )
        sense = "G" if self.problem_data["objective_sense"] == "min" else "L"
        cplex_index_new_cuts = np.array(
            self.model.linear_constraints.add(
                rhs=rhs.tolist(), senses=sense * n_cuts
            )
        )

        # (n_cuts, dim_x + 1)
        mat_value = np.concatenate(
            [-subgradient_value, np.ones((n_cuts, 1))],
            axis=1,
        )
        mat_rows = np.repeat(cplex_index_new_cuts[:, None], dim_x + 1, axis=1)
        mat_cols = np.concatenate(
            [
                np.repeat(cplex_index_x[None, :], n_cuts, axis=0),
                cplex_index_r[component_index, None],
            ],
            axis=1,
        )
        self.model.linear_constraints.set_coefficients(
            zip(
                map(int, mat_rows.ravel()),
                map(int, mat_cols.ravel()),
                map(float, mat_value.ravel()),
            )
        )

        iter = zip(
            component_index,
            subgradient_value,
            rhs,
            cplex_index_new_cuts,
        )
        for it in iter:
            _component_index = it[0]
            _subgradient_value = it[1]
            _rhs = it[2]
            _cplex_index = it[3]
            cut_iteration[_component_index] = np.r_[
                cut_iteration[_component_index],
                self.iteration_index,
            ]
            cut_first_iteration[_component_index] = np.r_[
                cut_first_iteration[_component_index],
                self.iteration_index,
            ]
            cut_cplex_index[_component_index] = np.r_[
                cut_cplex_index[_component_index], _cplex_index
            ]
            cut_rhs[_component_index] = np.r_[
                cut_rhs[_component_index],
                _rhs,
            ]
            cut_active[_component_index] = np.r_[
                cut_active[_component_index],
                False,
            ]
            if len(cut_subgradient_value[_component_index]) > 0:
                cut_subgradient_value[_component_index] = np.concatenate(
                    [
                        cut_subgradient_value[_component_index],
                        _subgradient_value[None],
                    ],
                    axis=0,
                )
            else:
                cut_subgradient_value[_component_index] = _subgradient_value[
                    None
                ]
        if subproblem_solution is not None:
            iter = zip(component_index, subproblem_solution)
            for _component_index, _solution in iter:
                if len(cut_data[_component_index]) > 0:
                    cut_data[_component_index] = np.concatenate(
                        [cut_data[_component_index], _solution[None]],
                        axis=0,
                    )
                else:
                    cut_data[_component_index] = _solution[None]
            return new

    def get_cuts(
        self,
        only_active=False,
        only_active_at_last=False,
        iteration=False,
        data=False,
        model_index=False,
    ):
        res = {
            "component_index": [],
            "subgradient_value": [],
            "rhs": [],
            "active": [],
            "added_iteration_first": [],
            "added_iteration_last": [],
            "data": [],
            "model_index": [],
        }
        if only_active_at_last:
            selectors = [
                np.nonzero(~np.isclose(weight, 0))[0]
                for weight in self.get_weight()
            ]
        elif only_active:
            selectors = [np.nonzero(x)[0] for x in self.cut_active]
        else:
            selectors = [slice(None)] * len(self.S)
        for s, selector in enumerate(selectors):
            res["component_index"].append(
                np.full(len(self.cut_iteration[s]), s)[selector]
            )
            res["added_iteration_first"].append(
                self.cut_first_iteration[s][selector]
            )
            res["added_iteration_last"].append(self.cut_iteration[s][selector])
            _data = self.cut_data[s]
            indices = np.arange(len(_data))[selector]
            res["data"].append([_data[i] for i in indices])
            res["subgradient_value"].append(
                self.cut_subgradient_value[s][selector]
            )
            res["rhs"].append(self.cut_rhs[s][selector])
            res["active"].append(self.cut_active[s][selector])
            res["model_index"].append(self.cut_cplex_index[s][selector])

        for key in res:
            res[key] = np.concatenate(res[key], axis=0)
        if not iteration:
            del res["added_iteration_first"]
            del res["added_iteration_last"]
        if not data:
            del res["data"]
        if not model_index:
            del res["model_index"]
        return res

    def remove_cuts_older_than(self, older_than, type="all"):
        """Remove cuts added before the specified iteration

        Note that `older_than` is inclusive, namely cuts
        added on the `older_than` iteration may be removed
        as well.

        Parameters
        ----------
        older_than : int
            Cuts added on and before this iteration will be removed.
        """
        cut_unique_list = self.cut_unique_list
        cut_iteration = self.cut_iteration
        cut_first_iteration = self.cut_first_iteration
        cut_subgradient_value = self.cut_subgradient_value
        cut_cplex_index = self.cut_cplex_index
        cut_rhs = self.cut_rhs
        cut_data = self.cut_data
        cut_active = self.cut_active
        # (n_components)-list of 1d bool array
        removed_flag = [cut_iteration[s] <= older_than for s in self.S]
        # 1d array of int
        removed_cplex_constraint_index = np.concatenate(
            [cut_cplex_index[s][removed_flag[s]] for s in self.S]
        )
        if len(removed_cplex_constraint_index) == 0:
            return
        # Update Cplex model.
        self.model.linear_constraints.delete(
            map(int, removed_cplex_constraint_index)
        )
        # Update unique cut list.
        if self.config["dual_optimizer.check_duplicate"] == 1:
            for s, _flag in enumerate(removed_flag):
                cut_unique_list[s].erase_nonzero(_flag)
        if len(cut_data[0]) > 0:
            for s in self.S:
                # Update cut pools.
                # (n_remaining_cuts,)
                buf = cut_cplex_index[s][~removed_flag[s]]
                # For each remaining cut, check how many preceeding
                # constraints are removed and shift the index by
                # the number of such removal.
                # (n_remaining_cuts,)
                buf -= np.sum(
                    buf[:, None] > removed_cplex_constraint_index, axis=-1
                )
                cut_cplex_index[s] = buf
                cut_iteration[s] = cut_iteration[s][~removed_flag[s]]
                cut_first_iteration[s] = cut_first_iteration[s][
                    ~removed_flag[s]
                ]
                cut_subgradient_value[s] = cut_subgradient_value[s][
                    ~removed_flag[s]
                ]
                cut_rhs[s] = cut_rhs[s][~removed_flag[s]]
                cut_active[s] = cut_active[s][~removed_flag[s]]
                cut_data[s] = cut_data[s][~removed_flag[s]]
        else:
            for s in self.S:
                # Update cut pools.
                # (n_remaining_cuts,)
                buf = cut_cplex_index[s][~removed_flag[s]]
                # For each remaining cut, check how many preceeding
                # constraints are removed and shift the index by
                # the number of such removal.
                # (n_remaining_cuts,)
                buf -= np.sum(
                    buf[:, None] > removed_cplex_constraint_index, axis=-1
                )
                cut_cplex_index[s] = buf
                cut_iteration[s] = cut_iteration[s][~removed_flag[s]]
                cut_first_iteration[s] = cut_first_iteration[s][
                    ~removed_flag[s]
                ]
                cut_subgradient_value[s] = cut_subgradient_value[s][
                    ~removed_flag[s]
                ]
                cut_rhs[s] = cut_rhs[s][~removed_flag[s]]
                cut_active[s] = cut_active[s][~removed_flag[s]]

    def remove_all_cuts(self):
        """Remove all cuts from the model"""
        self.remove_cuts_older_than(float("inf"), type=type)

    def get_n_cuts(self):
        """Return the number of cuts in RMP"""
        return self.get_n_cuts_by_subproblems().sum()

    def get_n_cuts_by_subproblems(self):
        return np.array([len(x) for x in self.cut_cplex_index])

    def update_objective(self):
        """Update the proximal term

        This updates the proximal term in the CPLEX model.
        """
        # In case of no scaling, the objective becomes
        #     d^T x + (1 / 2t) | x - c |^2
        #     = (1 / 2t) x^T x + (d - c / t) x + (1 / 2t) c^T c
        # If the norm is replaced with | x |_D^2 = x^T D x,
        #     d^T x + (1 / 2t) | x - c |_D^2
        #     = (1 / 2t) x^T D x + (d - c D / t) x + (1 / 2t) c^T D c
        centre = self.proximal_centre
        adjusted_ss = self.stepsize
        positive_stepsize = np.isfinite(adjusted_ss) and (adjusted_ss > 0)
        if positive_stepsize:
            regularisation_parameter = 1.0 / adjusted_ss
        else:
            regularisation_parameter = 0.0
        cplex_index_x = self.cplex_index_x
        sign = 1 if self.problem_data["objective_sense"] == "min" else -1
        scaling = self.problem_data.get("regularization_scaling", 1)
        scaling = np.broadcast_to(scaling, (self.problem_data["dim_x"],))
        is_QP = "Q" in self.model.problem_type[self.model.get_problem_type()]
        if is_QP or positive_stepsize:
            # If the problem type is (MI)QP, it is likely that we
            # added regularisation in earlier iterations. If current
            # stepsize is 0, we need to remove the regularisation. So
            # we update the quadratic coefficieven no matter if stepsize is 0
            # or not.

            # Note if the current problem type is not QP and stepsize is 0,
            # we avoid calling set_quadratic_coefficients. This function makes
            # the problem type to QP even if the coefficient is 0.

            # Becareful with the behavior of set_quadratic_coefficients!
            # If one calls set_quadratic_coefficients(i, j, v) with distinct
            # i, j, then the objective will be
            #    obj = [v x_i x_j + v x_j x_i] / 2,
            # or simply
            #    obj = [2.0 v x_i x_j] / 2 = v x_i x_j.
            # If i and j is equal, then
            #    obj = [v x_i x_i] / 2.
            # So, if i and j is different, the coefficient will be untouched,
            # BUT IF I AND J IS EQUAL, THE COEFFICIENT WILL BE DIVIDED BY 2.
            iter = enumerate(map(int, cplex_index_x))
            self.model.objective.set_quadratic_coefficients(
                [
                    (xi, xi, sign * scaling[i] * regularisation_parameter)
                    for i, xi in iter
                ]
            )
        iter = enumerate(
            zip(
                map(int, cplex_index_x),
                centre,
                self.c_x,
            )
        )
        self.model.objective.set_linear(
            [
                (
                    xi,
                    float(
                        d - sign * scaling[i] * c * regularisation_parameter
                    ),
                )
                for i, (xi, c, d) in iter
            ]
        )
        # We don't need to update the offset since we don't use the value
        # of the objective value of the self directly.
        # self.objective.set_offset()

    def solve(self):
        """Solve the model and compute x and r"""
        model = self.model
        # Check whether there are subproblems without cuts
        # and fix r=0 for those.
        no_cuts_subproblems = np.nonzero(
            self.get_n_cuts_by_subproblems() == 0
        )[0]
        if len(no_cuts_subproblems) > 0:
            fixed_r_cplex_index = self.cplex_index_r[no_cuts_subproblems]
            fixed_r_cplex_index = fixed_r_cplex_index.tolist()
            zeros = np.zeros(len(fixed_r_cplex_index))
            model.variables.set_lower_bounds(zip(fixed_r_cplex_index, zeros))
            model.variables.set_upper_bounds(zip(fixed_r_cplex_index, zeros))
            # Scale the remaning part of the objective.
            objective_scaler = self.n_components / (
                self.n_components - len(no_cuts_subproblems)
            )
            if self.objective_scaling_type == "optimistic":
                model.objective.set_linear(
                    list(
                        zip(
                            map(int, self.cplex_index_r),
                            np.repeat(objective_scaler, self.n_components),
                        )
                    )
                )
            elif self.objective_scaling_type == "pessimistic":
                c_x_buf = np.array(self.c_x)
                self.c_x = self.c_x / objective_scaler
            else:
                raise ValueError(
                    "Invalid objective scaling type: "
                    f"{self.objective_scaling_type}"
                )
        else:
            fixed_r_cplex_index = []
        # Solve self and store the solution.
        self.update_objective()
        model.solve()
        self._cached_status = model.solution.get_status()
        try:
            self._cached_objective_value = model.solution.get_objective_value()
        except cplex_utils.CplexError:
            self._cached_objective_value = None
        try:
            self._cached_solution = np.array(model.solution.get_values())
        except cplex_utils.CplexError:
            self._cached_solution = None
        try:
            self._cached_dual_solution = np.array(
                model.solution.get_dual_values()
            )
        except cplex_utils.CplexError:
            self._cached_dual_solution = None
        # Remove the bounds of r if they were modified.
        if len(fixed_r_cplex_index) > 0:
            inf = np.full(len(fixed_r_cplex_index), np.inf)
            model.variables.set_lower_bounds(zip(fixed_r_cplex_index, -inf))
            model.variables.set_upper_bounds(zip(fixed_r_cplex_index, inf))
            if self.objective_scaling_type == "optimistic":
                model.objective.set_linear(
                    list(
                        zip(
                            map(int, self.cplex_index_r),
                            np.ones(self.n_components),
                        )
                    )
                )
            elif self.objective_scaling_type == "pessimistic":
                self.c_x = c_x_buf


if __name__ == "__main__":
    import doctest

    doctest.testmod()
