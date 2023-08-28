# -*- coding: utf-8 -*-

"""Definition of parametrized UC problems

This file contains a routine to define/consruct
parametrized UC problems.
"""

import collections
import copy
import datetime
import enum
import logging
import numbers

import numpy as np
import scipy.sparse as sparse

from sucpy.problems import decomposition
from sucpy.problems.base import get_indices
from sucpy.utils import utils

safeguard = False

logger = logging.getLogger(__name__)


@enum.unique
class ParametrizedProblemMode(enum.IntEnum):
    TRAIN = 0
    EVAL = 1
    TEST = 2


class ParametrizedUC(object):
    """Parametrized UC.

    This is a utility class to sample MIP instances.
    `ParametrizedUC.sample_instance` returns a randomly sampled instance.
    """

    def setup(self, config):
        """Set up a ParametrizedUC instance.

        This initialises a ParametrizedUC instance.
        `config` is a dict to control the construction of
        this instance, which must contain the following items:
        - uc.generator_list : list of str
            Name of generator data files to be used.
        - n_periods : int
            Number of time periods in the planning horizon.

        Parameters
        ----------
        config : dict
        """
        config = copy.deepcopy(config)
        self.n_t = config["n_periods"]
        # ----------------------------------------------
        # Load various data.
        # ----------------------------------------------
        self.df_g = create_generator_list(config)
        self.n_g = self.df_g["n_g"]
        self.default_initial_condition = np.array(
            self.df_g["initial_condition"]
        )
        self.demand_data = load_demand_data(config, return_dates=True)
        if config["uc.demand_ratio_to_capacity"] > 0:  # Scale demand data.
            demand_factor = self.df_g["max_out"].sum()
            for d in self.demand_data.values():
                d["demand"] *= demand_factor
        self.base_demand = np.zeros(self.n_t)  # Set mock data.
        # Objects to compute perturbed demand.
        self.demand_noise_process = np.zeros(self.n_t)
        self.demand_noise_process_L = construct_demand_noise_process_L(config)
        # ----------------------------------------------
        # Formulate the optimization problem.
        # ----------------------------------------------
        self._problem_data = formulate_problem(self.df_g, self.n_t, config)
        # ----------------------------------------------
        # Clean up.
        # ----------------------------------------------
        self.train()  # Set training mode by default.

    def clear_decomposition_cache_(self):
        decomposition.clear_decomposition_cache_(
            self._problem_data, submatrices=False
        )

    @property
    def dim_parameter(self):
        """Return the dimension of the parameter."""
        return self.n_t

    @property
    def dim_dual(self):
        """Return the dimension of the dual variable."""
        return 2 * self.n_t

    @property
    def mode(self):
        """Get the current mode."""
        return self._mode

    @mode.setter
    def mode(self, v):
        """Set the current mode."""
        if v == ParametrizedProblemMode.TRAIN:
            self._mode = v
        elif v == ParametrizedProblemMode.EVAL:
            self._mode = v
        elif v == ParametrizedProblemMode.TEST:
            self._mode = v
        elif v == "train":
            self._mode = 0
        elif v == "eval":
            self._mode = 1
        elif v == "test":
            self._mode = 2
        else:
            raise ValueError(f"invalid mode: {v}")

    def train(self):
        """Set train mode."""
        self.mode = ParametrizedProblemMode.TRAIN

    def eval(self, v=True):
        """Set eval mode."""
        if v:
            self.mode = ParametrizedProblemMode.EVAL
        else:
            self.mode = ParametrizedProblemMode.TRAIN

    def test(self):
        """Set test mode."""
        self.mode = ParametrizedProblemMode.TEST

    def sample_instance(
        self,
        instance_id,
        demand=None,
        initial_condition=None,
        validate=False,
        inplace=False,
    ):
        """Sample an MIP instance and return its information.

        This samples an MIP instance and returns all necessary
        information to construct the model or its decomposition.
        The data is stored in a dict.
        Besides items for general decomposable problems,
        returned data has the following additional items:

        demand : (n_t,) array of float
        initial_condition : (n_g,) array of int
            If initial_condition is non-negative, the corresponding
            generator is on before t = 0.  The absolute value indicates
            how long the generator needs to stay on.
            For example, if it's 2, it is on initially, and needs to be
            on in t = 0, 1 as well.
            If initial_condition is negative, the corresponding
            generator is off before t = 0.  (absolute value - 1) indicates
            how long the generator needs to stay off.
            For example, if it's -1, it means the generator is off at first
            but can be turned up immediately.

        Parameters
        ----------
        instance_id : int
        demand : array of float, optional
        initial_condition : array of int, optional

        Returns
        -------
        problem_data : dict
        """
        rng = np.random.RandomState(instance_id + self.n_g * self.n_t)
        demand_given = demand is not None
        initial_condition_given = initial_condition is not None
        # randomize demand and initial condition.
        while True:
            if not demand_given:
                demand = self._sample_demand(rng)
            if not initial_condition_given:
                initial_condition = np.array(self.default_initial_condition)
            if not validate:
                max_demand_plus_reserve = np.max(demand) * (
                    1 + self._problem_data["uc.spinning_reserve_rate"]
                )
                max_out = np.sum(self.df_g["max_out"])
                scaling = max_out / max_demand_plus_reserve
                if scaling < 1:
                    logger = logging.getLogger(__name__)
                    logger.warning(
                        "infeasiblity is detected. "
                        f"instance id: {instance_id}  "
                        f"max demand: {np.max(demand):.1f}  "
                        f"capacity: {max_out:.1f}  "
                        f"scaling: {scaling:.4f}"
                    )
                    demand *= scaling
                break
            is_feasible, msg = check_feasibility(
                demand, self.df_g, initial_condition
            )
            if is_feasible:
                break
            else:
                logger = logging.getLogger(__name__)
                logger.warning(
                    "infeasiblity is detected. "
                    f"instance id: {instance_id}  msg: {msg}"
                )
        return self.instanciate(
            demand=demand,
            initial_condition=initial_condition,
            instance_id=instance_id,
            inplace=inplace,
        )

    def instanciate(
        self, demand, initial_condition, instance_id=None, inplace=False
    ):
        """Build an MIP instance.

        This instanciate an MIP instance and returns all necessary
        information to construct the model or its decomposition.
        The data is stored in a dict.
        For more information, see the doc on `sample_instance`.

        Returns
        -------
        problem_data : dict
        """
        if not inplace:
            out = {**self._problem_data}
            out["demand"] = out["demand"].copy()
            out["initial_condition"] = out["initial_condition"].copy()
            out["constraint_rhs"] = out["constraint_rhs"].copy()
            out["variable_lb"] = out["variable_lb"].copy()
            out["variable_ub"] = out["variable_ub"].copy()
            out.pop("subproblem_variable_lb", None)
            out.pop("subproblem_variable_ub", None)
            out.pop("subproblem_constraint_rhs", None)
            out.pop("_x_lb_sorted", None)
            out.pop("_x_ub_sorted", None)
            out.pop("_a_sorted", None)
            # self._problem_data = out
        elif not safeguard:
            out = self._problem_data  # This should be writeable.
        elif inplace == "try":
            out = self._problem_data
            try:
                out["demand"].setflags(write=True)
                out["initial_condition"].setflags(write=True)
                out["constraint_rhs"].setflags(write=True)
                out["variable_lb"].setflags(write=True)
                out["variable_ub"].setflags(write=True)
                out["_x_lb_sorted"].setflags(write=True)
                out["_x_ub_sorted"].setflags(write=True)
                out["_a_sorted"].setflags(write=True)
            except ValueError:
                # Cannot set writeable flag.
                # This may happen when this is
                # a forked process.
                return self.instanciate(
                    demand=demand,
                    initial_condition=initial_condition,
                    instance_id=instance_id,
                    inplace=False,
                )
        else:
            out = self._problem_data
            out["demand"].setflags(write=True)
            out["initial_condition"].setflags(write=True)
            out["constraint_rhs"].setflags(write=True)
            out["variable_lb"].setflags(write=True)
            out["variable_ub"].setflags(write=True)
            out["_x_lb_sorted"].setflags(write=True)
            out["_x_ub_sorted"].setflags(write=True)
            out["_a_sorted"].setflags(write=True)
        set_demand_and_initial_condition_(out, demand, initial_condition)
        out["instance_id"] = instance_id
        out["parametrized_problem_mode"] = self.mode
        out["dim_parameter"] = self.dim_parameter
        out["dim_dual"] = self.dim_dual

        # To be saved besides the solver results.
        out["instance_data.parametrized_problem_type"] = 0
        out["instance_data.instance_id"] = instance_id
        out["instance_data.parametrized_problem_mode"] = self.mode
        out["instance_data.n_g"] = self.n_g
        out["instance_data.n_t"] = self.n_t
        out["instance_data.demand"] = demand
        out["instance_data.initial_condition"] = initial_condition

        out["parameter"] = out["demand"]
        if safeguard:
            utils.make_read_only(out)
        return out

    def _sample_demand(self, rng):
        mode_demand_data = self.demand_data[self.mode]
        choice = rng.choice(len(mode_demand_data["demand"]))
        self.base_demand = demand = mode_demand_data["demand"][choice]
        # Add perturbation.
        # First, sample an independent noise from standard normal distribution.
        noise = rng.randn(self.n_t)
        # Convert the noise into ones which follows a gaussian process.
        noise = np.matmul(self.demand_noise_process_L, noise)
        self.demand_noise_process = noise
        demand = demand * (1 + noise)
        return demand


def enforce_initial_condition(schedule, initial_condition, out=None):
    """Return a modified schedule satisfying the initial condition."""
    schedule = np.asarray(schedule)
    initial_condition = np.asarray(initial_condition)
    if out is schedule:
        pass
    elif out is not None:
        out[:] = np.array(schedule)
        schedule = out
    else:
        schedule = np.array(schedule, dtype=bool)
    n_g, n_t = schedule.shape
    g = np.nonzero(initial_condition > 0)[0]
    if len(g):
        w = initial_condition[g]
        g = np.repeat(g, w)
        t = np.concatenate([np.arange(i) for i in w])
        schedule[g, t] = 1
    g = np.nonzero(initial_condition < -1)[0]
    if len(g):
        w = -initial_condition[g] - 1
        g = np.repeat(g, w)
        t = np.concatenate([np.arange(i) for i in w])
        schedule[g, t] = 0
    return schedule


def enforce_min_up_down(
    schedule,
    initial_condition,
    min_up=None,
    min_down=None,
    df_g=None,
    g=None,
    out=None,
):
    """Modify a schedule to satisfy the min up/down constraints.

    Examples
    --------
    >>> initial_condition = [0, 1, -1, 1, -2]
    >>> schedule = [
    ...     [1, 0, 0, 1, 1, 1, 1, 0],
    ...     [1, 1, 1, 0, 0, 1, 1, 1],
    ...     [1, 1, 1, 0, 0, 0, 1, 1],
    ...     [1, 1, 1, 1, 1, 1, 1, 1],
    ...     [0, 0, 0, 0, 1, 0, 0, 0],
    ... ]
    >>> ret = enforce_min_up_down(schedule, initial_condition, 3, 3)
    >>> ret.schedule * 1
    array([[1, 1, 1, 1, 1, 1, 1, 0],
           [1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 0, 0, 0, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 1, 1, 1, 0]])
    >>> ret.g
    array([0, 0, 0, 1, 1, 1, 4, 4])
    >>> ret.t
    array([1, 2, 3, 3, 4, 5, 5, 6])

    One can check only a subset of generators.

    >>> res = enforce_min_up_down(schedule, initial_condition, 3, 3, g=4)
    >>> res.schedule * 1
    array([[1, 0, 0, 1, 1, 1, 1, 0],
           [1, 1, 1, 0, 0, 1, 1, 1],
           [1, 1, 1, 0, 0, 0, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 1, 1, 1, 0]])
    >>> res = enforce_min_up_down(schedule, initial_condition, 3, 3, g=[1, 4])
    >>> res.schedule * 1
    array([[1, 0, 0, 1, 1, 1, 1, 0],
           [1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 0, 0, 0, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 1, 1, 1, 0]])

    Parameters
    ----------
    schedule : (n_g, n_t) array of bool
    initial_condition : (n_g,) array of int
    min_up : (n_g,) array of int, optional
        `min_up` and `min_down`, or `df_g` must be given.
    min_down : (n_g,) array of int, optional
        `min_up` and `min_down`, or `df_g` must be given.
    df_g : DataFrame, optional
        `min_up` and `min_down`, or `df_g` must be given.
    g : int or 1d array, optional
        If given, only generators of indices in `g` are fixed.
    out : (n_g, n_t) array of bool, optional
        If given, a modified schedule is written on this array.

    Returns
    -------
    schedule : (n_g, n_t) array of bool
    g : array of int
        Indices of the modified elements along the first axis.
    t : array of int
        Indices of the modified elements along the second axis.
    """
    schedule = np.asarray(schedule)
    initial_condition = np.asarray(initial_condition)
    if g is None:
        n_selected_g = schedule.shape[0]
        selected_g = np.arange(n_selected_g)
    else:  # Only fix min up/down for selected generators.
        selected_g = np.atleast_1d(g)
        n_selected_g = selected_g.shape[0]
    if out is schedule:
        pass
    elif out is not None:
        out[:] = np.array(schedule)
        schedule = out
    else:
        schedule = np.array(schedule, dtype=bool)
    n_t = schedule.shape[1]
    if (min_up is None) or (min_down is None):
        if df_g is None:
            raise ValueError("min_up and min_down, or df_g must be given.")
        min_up = df_g["min_up"]
        min_down = df_g["min_down"]
    else:
        min_up = np.broadcast_to(min_up, (n_t,))
        min_down = np.broadcast_to(min_down, (n_t,))
    ret_g = []
    ret_t = []
    for i in range(10):
        buf = len(ret_g)
        # Fix minimum down time (may not be completely)
        down = np.zeros((n_selected_g, n_t))
        # set 1 when the generators turned down.
        down[:, 1:] = schedule[selected_g, :-1] & (~schedule[selected_g, 1:])
        down[:, 0] = (initial_condition[selected_g] >= 0) & (
            ~schedule[selected_g, 0]
        )
        # gi, si, ti, li : (num_nonzeros,)
        # gi, ti : index of gen and time where switching off occurs.
        # Note that gi is the index of generators relative to
        # the given schedule.
        # If schedule is the shape of (n_all_g, n_t), then they agree but
        # otherwise `gi` is not the actual generator index.
        gi, ti = np.nonzero(down)
        if len(gi) > 0:
            li = min_down[selected_g[gi]]  # minimum periods to be down.
            # Now construct indices which extract all cells
            # in schedule[gi, ti] which are supposed to be off.
            gi = np.repeat(gi, li)
            # corresponding time where generators are switched off.
            si = np.repeat(ti, li)
            ti = np.concatenate([np.arange(i, i + j) for i, j in zip(ti, li)])
            selector = ti < n_t
            gi, si, ti = gi[selector], si[selector], ti[selector]
            sgi = selected_g[gi]
            # schedule[gi, ti] must be all 0.
            # schedule[gi, si] is the cell where switching off occured.
            # `fail` selects gi, ti, and si which
            # violates schedule[gi, ti] == 0.
            # We shall set schedule[gi, si:ti] == 1.
            fail = np.nonzero(schedule[sgi, ti] != 0)[0]
            if len(fail) > 0:
                gi = gi[fail]
                sgi = sgi[fail]
                ti = ti[fail]
                si = si[fail]
                li = ti - si + 1
                sgi = np.repeat(sgi, li)
                ti = np.concatenate(
                    [np.arange(i, i + j) for i, j in zip(si, li)]
                )
                schedule[sgi, ti] = 1
                ret_g.append(sgi)
                ret_t.append(ti)

        # Fix minimum up time (may not be completely)
        up = np.zeros((n_selected_g, n_t))
        # set 1 when the generators turned up.
        up[:, 1:] = (~schedule[selected_g, :-1]) & schedule[selected_g, 1:]
        up[:, 0] = (initial_condition[selected_g] < 0) & schedule[
            selected_g, 0
        ]
        # gi, si, ti, li : (num_nonzeros,)
        # gi, ti : index of gen and time where switching on occurs.
        gi, ti = np.nonzero(up)
        if len(gi) > 0:
            li = min_up[selected_g[gi]]  # minimum periods to be down.
            # Now construct indices which extract all cells
            # in schedule[gi, ti] which are supposed to be on.
            gi = np.repeat(gi, li)
            ti = np.concatenate([np.arange(i, i + j) for i, j in zip(ti, li)])
            selector = ti < n_t
            gi, ti = gi[selector], ti[selector]
            sgi = selected_g[gi]
            # schedule[gi, ti] must be all 1.
            fail = np.nonzero(schedule[sgi, ti] != 1)[0]
            if len(fail):
                sgi = sgi[fail]
                ti = ti[fail]
                schedule[sgi, ti] = 1
                ret_g.append(sgi)
                ret_t.append(ti)

        if buf == len(ret_g):
            # If there were no violations, return the current schedule.
            if len(ret_g):
                ret_g = np.concatenate(ret_g)
                ret_t = np.concatenate(ret_t)
            else:
                ret_g = np.array([])
                ret_t = np.array([])
            return MinUpDownCorrections(schedule=schedule, g=ret_g, t=ret_t)

    raise ValueError("failed to fix minimum up and down constraints")


MinUpDownCorrections = collections.namedtuple(
    "MinUpDownCorrections", "schedule,g,t"
)


def assert_initial_condition(initial_condition, schedule) -> None:
    """Assert a given schedule satisfies the initial condition."""
    if isinstance(initial_condition, dict) and (
        "initial_condition" in initial_condition
    ):
        initial_condition = initial_condition["initial_condition"]
    schedule = schedule.round().astype(bool)
    n_g = len(initial_condition)
    msg = []
    for g in range(n_g):
        init = initial_condition[g]
        if init < 0:
            w = -init - 1  # the length it must be off.
            if not np.all(~schedule[g, :w]):
                msg.append(
                    f"generator {g:4d} must be off for {w} periods "
                    "but was on at: "
                    f"{np.nonzero(schedule[g, :w])[0].tolist()}"
                )
        else:
            w = init  # the length it must be on.
            if not np.all(schedule[g, :w]):
                msg.append(
                    f"generator {g:4d} must be on for {w} periods "
                    "but was off at: "
                    f"{np.nonzero(~schedule[g, :w])[0].tolist()}"
                )
    if msg:
        joined_msg = "\n" + "\n".join(msg)
        raise AssertionError(joined_msg)


def assert_min_up_down(df_g, initial_condition, schedule, G=None) -> None:
    """Assert a given schedule satisfies the min up/down constraints."""
    msg = []
    if G is None:
        G = np.arange(df_g["n_g"])
    n_g, n_t = schedule.shape
    down = np.zeros((n_g, n_t))
    # set 1 when the generators turned down.
    down[:, 1:] = schedule[:, :-1] & (~schedule[:, 1:])
    down[:, 0] = (initial_condition[G] >= 0) & (~schedule[:, 0])
    # g, t, w : (num_nonzeros,)
    # g, t : index of gen and time where switching off occurs.
    # Note that g is the index of generators relative to the given schedule.
    # If schedule is the shape of (n_g, n_t), then they agree but
    # otherwise `g` is not the actual generator index.
    g, t = np.nonzero(down)
    if len(g) > 0:
        w = df_g["min_down"][G[g]]  # minimum periods to be down.
        # Now construct indices which extract all cells in schedule[g, t] which
        # are supposed to be off.
        g = np.repeat(g, w)
        t = np.concatenate([np.arange(i, i + j) for i, j in zip(t, w)])
        selector = t < n_t
        g, t = g[selector], t[selector]
        # schedule[g, t] must be all 0.
        # `fail` selects g and t which violates schedule[g, t] == 0.
        fail = np.nonzero(schedule[g, t] != 0)[0]
        if len(fail) > 0:
            msg.append(
                f"minimum down constraints are violated in the following:\n"
                f"{g}\n"
                f"{t}"
            )

    # Fix minimum up time (may not completely)
    up = np.zeros((n_g, n_t))
    # set 1 when the generators turned up.
    up[:, 1:] = (~schedule[:, :-1]) & schedule[:, 1:]
    up[:, 0] = (initial_condition[G] < 0) & schedule[:, 0]
    # g, s, t, w : (num_nonzeros,)
    # g, t : index of gen and time where switching on occurs.
    g, t = np.nonzero(up)
    if len(g) > 0:
        w = df_g["min_up"][G[g]]  # minimum periods to be down.
        # Now construct indices which extract all cells in schedule[g, t] which
        # are supposed to be on.
        g = np.repeat(g, w)
        t = np.concatenate([np.arange(i, i + j) for i, j in zip(t, w)])
        selector = t < n_t
        g, t = g[selector], t[selector]
        # schedule[g, t] must be all 1.
        # `fail` selects g and t which violates schedule[g, t] == 1.
        fail = np.nonzero(schedule[g, t] != 1)[0]
        if len(fail) > 0:
            msg.append(
                f"minimum up constraints are violated in the following:\n"
                f"{g}\n"
                f"{t}"
            )

    if len(msg) > 0:
        joined_msg = "\n" + "\n".join(msg)
        raise AssertionError(joined_msg)


def schedule_to_solution(
    initial_condition, schedule, out=0, stack_by_subproblems: bool = False
):
    """Create a solution given a schedule.

    Parameters
    ----------
    schedule : (n_g, n_t) array of int or bool
    out : (n_g, n_t) array of float, optional
    stack_by_subproblems : bool, default False
        Return an array of shape (n_g, 4 * n_t) instead of
        an flattened array of shape (4 * n_g * n_t,).

    Returns
    -------
    sol : (n_g, 4 * n_t) or (4 * n_g * n_t,) array
        Solution based on a given schedule.  The shape is specified by
        `stack_by_subproblems`.
    """
    n_g, n_t = schedule.shape
    if stack_by_subproblems:
        sol = np.zeros((n_g, 4 * n_t))
        sol_out = sol[:, 0:n_t]
        sol_on = sol[:, n_t : 2 * n_t]
        sol_up = sol[:, 2 * n_t : 3 * n_t]
        sol_down = sol[:, 3 * n_t : 4 * n_t]
    else:
        sol = np.zeros(4 * n_g * n_t)
        sol_out = sol[0 : n_g * n_t].reshape(n_g, n_t)
        sol_on = sol[n_g * n_t : 2 * n_g * n_t].reshape(n_g, n_t)
        sol_up = sol[2 * n_g * n_t : 3 * n_g * n_t].reshape(n_g, n_t)
        sol_down = sol[3 * n_g * n_t : 4 * n_g * n_t].reshape(n_g, n_t)
    schedule = schedule.round().astype(bool)
    sol_on[:] = schedule.astype(float)
    sol_out[:] = out
    sol_up[:, 1:] = (~schedule[:, :-1]) & (schedule[:, 1:])
    sol_up[:, 0] = (initial_condition < 0) & (schedule[:, 0])
    sol_down[:, 1:] = (schedule[:, :-1]) & (~schedule[:, 1:])
    sol_down[:, 0] = (initial_condition >= 0) & (~schedule[:, 0])
    return sol


def set_demand_and_initial_condition_(
    problem_data, demand, initial_condition, update_decomposition=True
):
    """Update demand inplace.

    This updates demand data and relevant values in
    the optimization problem, namely:
    - demand
    - b
    - subproblem_constraint_rhs
    """
    np.copyto(problem_data["demand"], demand)
    problem_data["constraint_rhs"][
        problem_data["constraint_name_to_index"]["load_balance"]
    ] = demand
    reserve = problem_data["uc.spinning_reserve_rate"] * demand
    problem_data["constraint_rhs"][
        problem_data["constraint_name_to_index"]["spinning_reserve"]
    ] = reserve
    np.copyto(problem_data["initial_condition"], initial_condition)
    buf = initial_condition >= 0
    problem_data["constraint_rhs"][
        problem_data["constraint_name_to_index"]["polynomial"][:, 0]
    ] = buf
    np.copyto(problem_data["variable_lb"], 0)
    np.copyto(problem_data["variable_ub"], 1)
    g = np.nonzero(initial_condition >= 1)[0]
    if len(g) > 0:
        w = initial_condition[g]
        g = np.repeat(g, w)
        t = np.r_[tuple(np.arange(i) for i in w)]
        var_index = problem_data["variable_name_to_index"]["on"][g, t]
        problem_data["variable_ub"][var_index] = problem_data["variable_lb"][
            var_index
        ] = 1
    g = np.nonzero(initial_condition <= -2)[0]
    if len(g) > 0:
        w = -initial_condition[g] - 1
        g = np.repeat(g, w)
        t = np.concatenate([np.arange(i) for i in w])
        var_index = problem_data["variable_name_to_index"]["on"][g, t]
        problem_data["variable_ub"][var_index] = problem_data["variable_lb"][
            var_index
        ] = 0
    if update_decomposition:
        decomposition.set_decomposition_data_(
            problem_data,
            constraint_rhs=True,
            variable_lb=True,
            variable_ub=True,
        )


def create_generator_list(config):
    """Create a generator list.

    This reads raw generator lists, process its data format and
    return it as a DataFrame.

    Parameters
    ----------
    config : dict
        A dict which is passed to ParametrizedUC.__init__.

    Returns
    -------
    df : dict
        A dict with the following items.
        - n_g : int
        - mar_cost : (n_g,) array
        - nol_cost : (n_g,) array
        - st_cost : (n_g,) array
        - min_out : (n_g,) array
        - max_out : (n_g,) array
        - op_rampup : (n_g,) array
        - op_rampdown : (n_g,) array
        - st_rampup : (n_g,) array
        - sh_rampdown : (n_g,) array
        - min_up : (n_g,) array
        - min_down : (n_g,) array
        - notification : (n_g,) array
        - initial_condition : (n_g,) array

    Notes
    -----
    If initial_condition is non-negative, the corresponding
    generator is on before t = 0.  The absolute value indicates
    how long the generator needs to stay on.
    For example, if it's 2, it is on initially, and needs to be
    on in t = 0, 1 as well.
    If initial_condition is negative, the corresponding
    generator is off before t = 0.  (absolute value - 1) indicates
    how long the generator needs to stay off.
    For example, if it's -1, it means the generator is off at first
    but can be turned up immediately.
    """

    if config["generator_data_type"] == "frangioni":
        return _create_generator_list_frangioni(config)
    elif config["generator_data_type"] == "tim":
        return _create_generator_list_tim(config)
    elif config["generator_data_type"] == "miguel":
        return _create_generator_list_miguel(config)
    else:
        raise ValueError


def _create_generator_list_frangioni(config):
    import pandas as pd

    # Decide which files to use.
    filenames = config["generator_list"]

    # Concatenate all data on this `df`.
    df = pd.DataFrame()
    for filename in filenames:
        filepath = f"data/v1/generators/out/{filename}"
        df_new = pd.read_csv(filepath)
        df = pd.concat([df, df_new], ignore_index=True, sort=False)

    # Reindex generators from 0 to len(df) - 1.
    df["generator_index"] = np.arange(len(df))

    # Convert generator data to one which are compatible to the experiment.
    df = df.set_index("generator_index")
    df_out = pd.DataFrame(index=df.index)
    min_out = df["min_out"]
    max_out = df["max_out"]
    min_cost = (
        df["constant_cost"]
        + df["linear_cost"] * min_out
        + df["quadratic_cost"] * (min_out**2)
    )
    max_cost = (
        df["constant_cost"]
        + df["linear_cost"] * max_out
        + df["quadratic_cost"] * (max_out**2)
    )
    df_out["mar_cost"] = (max_cost - min_cost) / (max_out - min_out)
    df_out["nol_cost"] = min_cost - df_out["mar_cost"] * min_out
    df_out["original_cost_constant"] = df["constant_cost"]
    df_out["original_cost_linear"] = df["linear_cost"]
    df_out["original_cost_quadratic"] = df["quadratic_cost"]

    # It seems the description of the data file is not correct,
    # since `hot_and_fuel_cost` seems to be 0 anywhere.
    # Instead, reading values in the following way seems more realistic.
    # - fixed_cost -> cool_and_fuel_cost
    # - succ -> hot_and_fuel_cost
    # - p0 -> fixed_cost
    # df_out["st_cost"] = df["hot_and_fuel_cost"] * 12.0 + df["fixed_cost"]
    df_out["st_cost"] = df["succ"] * 12.0 + df["p0"]
    df_out["min_out"] = df["min_out"]
    df_out["max_out"] = df["max_out"]
    df_out["op_rampup"] = df["ramp_up"]
    df_out["op_rampdown"] = df["ramp_down"]
    # Allow them to ramp up/down at least min_out otherwise
    # they can't switch on/off.
    df_out["st_rampup"] = np.maximum(
        df["ramp_up"].values, df["min_out"].values
    )
    df_out["sh_rampdown"] = np.maximum(
        df["ramp_down"].values, df["min_out"].values
    )
    df_out["min_up"] = df["min_up"]
    df_out["min_down"] = df["min_down"]
    df_out["notification"] = (df["min_up"] * 0.7).astype(int).clip(0)
    if np.any(df["init_status"] == 0):
        raise ValueError("some init_status is 0.")

    df_out["initial_condition"] = 0

    ret = {}
    for col in df_out.columns:
        ret[col] = df_out[col].values
    ret["n_g"] = len(df_out)
    return ret


def _create_generator_list_tim(config):
    import pandas as pd

    df_g = pd.read_csv("data/v1/generators/tim/generators_fc.csv", sep="\t")
    df_g_notif = pd.read_csv(
        "data/v1/generators/tim/notification_time.csv", sep="\t"
    )
    df_g["NTime"] = df_g_notif["NTime"]

    df_g["full_load_cost"] = df_g["c0"] / df_g["PGmin"] + df_g["c"]

    df_g = df_g.sort_values("full_load_cost")

    # proportion of power output that is maximally available for reserve
    # ResRamp;
    # Gname{G} symbolic;  # Generator names
    # Gloc{G};  # Location of Generators
    # Tu {G} >= 0;  # minimum uptime [min]
    # Td {G} >= 0;  # minimum downtime [min]
    # PGmax {G};  # max real power output of generator g [MW]
    # PGmin {G};  # min real power output of generator g [MW]
    # Ru {G};  # ramp up limit [MW/min]
    # Rd {G};  # ramp down limit [MW/min]
    # c {G};  # generator cost coefficient c [$/MWh]
    # c0 {G};   # generator constant cost coefficient c0 [$/h]
    # cS {G};   # startup cost [$]
    # cC;  # carbon emission cost [$/tCO2e]
    # CO2 {G};  # CO2 emissions [tCO2e/MWh]
    # Rsu {G} default max(dt*Ru[g],PGmin[g]);# ramp startup limit [MW]
    # Rsd {G} default max(dt*Rd[g],PGmin[g]);# ramp shutdown limit [MW]
    # RCap{G} default PGmax[g]*ResRamp;  # reserve capacity of generator [MW]
    # NTime {G};  # startup notification time [h]

    # - n_g
    # - c                     : mar_cost
    # - c0                    : nol_cost
    # - cS                    : st_cost
    # - PGmin                 : min_out
    # - PGmax                 : max_out
    # - Ru * 60               : op_rampup
    # - Rd * 60               : op_rampdown
    # - max(Ru * 60, min_out) : st_rampup
    # - max(Rd * 60, min_out) : sh_rampdown
    # - Tu * 60               : min_up
    # - Td * 60               : min_down
    # - NTime                 : notification
    # - none                  : initial_condition

    df_g = {key: df_g[key].values for key in df_g}
    df_g_notif = {key: df_g_notif[key].values for key in df_g_notif}

    df_g["cumulative_capacity"] = np.cumsum(df_g["PGmax"])

    initial_condition = np.zeros(len(df_g["c"]), dtype=int)
    initial_condition[
        df_g["cumulative_capacity"] > 0.5 * df_g["PGmax"].sum()
    ] = -1

    ret = {
        "n_g": len(df_g["c"]),
        "mar_cost": df_g["c"],
        "nol_cost": df_g["c0"],
        "st_cost": df_g["cS"],
        "min_out": df_g["PGmin"],
        "max_out": df_g["PGmax"],
        "op_rampup": df_g["Ru"] * 60.0,
        "op_rampdown": df_g["Rd"] * 60.0,
        "st_rampup": np.maximum(df_g["Ru"] * 60.0, df_g["PGmin"]),
        "sh_rampdown": np.maximum(df_g["Rd"] * 60.0, df_g["PGmin"]),
        "min_up": np.round(df_g["Tu"] / 60).astype(int),
        "min_down": np.round(df_g["Td"] / 60).astype(int),
        "notification": df_g_notif["NTime"],
        "initial_condition": initial_condition,
    }

    return ret


def _create_generator_list_miguel(config):
    import pandas as pd

    df = pd.read_csv(
        "data/v1/generators/miguel/data.csv", sep=" +", engine="python"
    )
    lst = [
        "st_cost",
        "mar_cost",
        "min_out",
        "max_out",
        "st_rampup",
        "sh_rampdown",
        "op_rampup",
        "op_rampdown",
    ]
    df[lst] = df[lst].astype(float)
    rng = np.random.RandomState(0)
    n_g = len(df)
    df["min_out"] *= 1 + 0.2 * rng.rand(n_g) - 0.1
    df["max_out"] *= 1 + 0.2 * rng.rand(n_g) - 0.1
    df["st_rampup"] *= 1 + 0.2 * rng.rand(n_g) - 0.1
    df["sh_rampdown"] *= 1 + 0.2 * rng.rand(n_g) - 0.1
    df["op_rampup"] *= 1 + 0.2 * rng.rand(n_g) - 0.1
    df["op_rampdown"] *= 1 + 0.2 * rng.rand(n_g) - 0.1
    df["nol_cost"] = 0.0
    df["notification"] = 0
    df["initial_condition"] = 0
    """
    - n_g : int
    - mar_cost : (n_g,) array
    - nol_cost : (n_g,) array
    - st_cost : (n_g,) array
    - min_out : (n_g,) array
    - max_out : (n_g,) array
    - op_rampup : (n_g,) array
    - op_rampdown : (n_g,) array
    - st_rampup : (n_g,) array
    - sh_rampdown : (n_g,) array
    - min_up : (n_g,) array
    - min_down : (n_g,) array
    - notification : (n_g,) array
    - initial_condition : (n_g,) array
    """
    out = dict(df)
    for key in out:
        out[key] = out[key].values
    out["n_g"] = len(df)
    return out


def load_demand_data(config, return_dates=False):
    """Load demand data and return as a DataFrame.

    This loads demand data, process the content and
    return the result as a DataFrame.

    Returns
    -------
    data : dict
        This is a mapping of mode -> item, where item
        is a dict with `dates` (if `return_dates` is
        True) and `demand`.
    """
    import pandas as pd

    n_t = config["n_periods"]
    demand_data_path = "data/v1/demand/out/demand.csv"
    df = pd.read_csv(demand_data_path, parse_dates=["SETTLEMENT_DATE"])

    # ----------------------------------------------
    # Tidy up data
    # ----------------------------------------------
    df["datetime"] = df["SETTLEMENT_DATE"] + pd.to_timedelta(
        (df["SETTLEMENT_PERIOD"] - 1) // 2, unit="h"
    )
    df["wind_load_factor"] = (
        df["EMBEDDED_WIND_GENERATION"] / df["EMBEDDED_WIND_CAPACITY"]
    )
    df = df.rename(columns={"ND": "demand"})
    df = (
        df[["demand", "wind_load_factor"]]
        .groupby(df.datetime)
        .mean()
        .reset_index()
    )

    # ----------------------------------------------
    # Normalize data
    # ----------------------------------------------
    if config["uc.demand_ratio_to_capacity"] > 0:
        # Normalize demand data so that typical daily peak demand is 0.5.
        # Extract daily peak demand data.
        daily_peak = df["demand"].groupby(df["datetime"].dt.date).agg(max)
        # Compute the mode.  If there are an even number of data,
        # take mean of modes.
        typical_max = np.mean(daily_peak.mode())
        # Normalize demand so that the ratio of peak demand to
        # the generator capacity on typical days agrees with
        # `demand_ratio_to_capacity`.
        df["demand"] = (
            df["demand"] / typical_max * config["uc.demand_ratio_to_capacity"]
        )

    # ----------------------------------------------
    # Extract dates
    # ----------------------------------------------
    # List all valid dates to sample demand data.
    valid_dates = df["datetime"].dt.date.unique()
    invalid_dates = pd.read_csv(
        "data/v1/demand/out/invalid_dates.csv", parse_dates=["date"]
    )["date"].dt.date
    invalid_dates = pd.to_datetime(invalid_dates)

    tabu_dates = []  # Store all invalid dates in this list.
    for i in range(int(np.ceil(n_t / 24))):
        # Rule out Dates with invalid data and preceesing periods.
        tabu_dates += list(
            (invalid_dates - datetime.timedelta(days=i)).dt.date
        )

    for i in range(int(np.ceil(n_t / 24)) - 1):
        # Rule out the end of the year if we need more than 1 day.
        tabu_dates += list(
            (
                invalid_dates
                + pd.tseries.offsets.YearEnd()
                - datetime.timedelta(days=i)
            ).dt.date
        )

    tabu_dates = np.sort(np.unique(tabu_dates))
    valid_dates = valid_dates[~np.isin(valid_dates, tabu_dates)]

    training_years = [2017]
    eval_years = [2018]
    test_years = [2019]
    dates = {
        ParametrizedProblemMode.TRAIN: np.array(
            [i for i in valid_dates if i.year in training_years]
        ),
        ParametrizedProblemMode.EVAL: np.array(
            [i for i in valid_dates if i.year in eval_years]
        ),
        ParametrizedProblemMode.TEST: np.array(
            [i for i in valid_dates if i.year in test_years]
        ),
    }

    # Create a returned dict.  It is a mapping of mode -> {'dates', 'demand'}.
    ret = {
        ParametrizedProblemMode.TRAIN: {
            "dates": np.array(
                [i for i in valid_dates if i.year in training_years]
            ),
        },
        ParametrizedProblemMode.EVAL: {
            "dates": np.array(
                [i for i in valid_dates if i.year in eval_years]
            ),
        },
        ParametrizedProblemMode.TEST: {
            "dates": np.array(
                [i for i in valid_dates if i.year in test_years]
            ),
        },
    }

    for mode in ret.keys():
        dates = ret[mode]["dates"]
        buf = []
        for d in dates:
            date = pd.to_datetime(d)
            selector = df["datetime"].between(
                date, date + datetime.timedelta(hours=n_t - 1)
            )
            buf.append(df["demand"][selector].values)
        buf = np.array(buf)
        ret[mode]["demand"] = buf

    if not return_dates:
        for mode in ret.keys():
            ret[mode].pop("dates")

    return ret


def formulate_problem(df_g, n_t, config, demand=None):
    """Read data files and construct information to formulate MIP instances.

    This construct various objects to define MIP instances,
    such as constraint coefficient and rhs.

    Parameters
    ----------
    df_g : DataFrame
        Data of generators.
    n_t : int
        Number of time periods to be scheduled.
    config : dict
    demand : (n_t,) array, optional

    Returns
    -------
    dict
    """
    n_g = df_g["n_g"]
    n_subproblems = n_g
    G = range(n_subproblems)
    T = range(n_t)
    if demand is None:
        demand = np.zeros(n_t)  # Create mock demand data.
    else:
        demand = np.asarray(demand)
    initial_condition = np.asarray(df_g["initial_condition"])  # (n_g,)
    mar_cost = np.asarray(df_g["mar_cost"])  # (n_g,)
    nol_cost = np.asarray(df_g["nol_cost"])  # (n_g,)
    st_cost = np.asarray(df_g["st_cost"])  # (n_g,)
    max_out = np.asarray(df_g["max_out"])  # (n_g,)
    min_out = np.asarray(df_g["min_out"])  # (n_g,)
    min_up = np.asarray(df_g["min_up"])  # (n_g,)
    min_down = np.asarray(df_g["min_down"])  # (n_g,)
    op_rampup = np.asarray(df_g["op_rampup"])  # (n_g,)
    st_rampup = np.asarray(df_g["st_rampup"])  # (n_g,)
    op_rampdown = np.asarray(df_g["op_rampdown"])  # (n_g,)
    sh_rampdown = np.asarray(df_g["sh_rampdown"])  # (n_g,)

    # First construct data on the extensive formulation and later decompose it.
    # Define column and row indices.
    vars, n_vars = get_indices(
        p=(n_g, n_t), on=(n_g, n_t), up=(n_g, n_t), down=(n_g, n_t)
    )
    cons, n_cons = get_indices(
        load_balance=(n_t),
        spinning_reserve=(n_t),
        generator_lb=(n_g, n_t),
        generator_ub=(n_g, n_t),
        polynomial=(n_g, n_t),
        switching=(n_g, n_t),
        min_up=(n_g, n_t),
        min_down=(n_g, n_t),
        ramp_up=(n_g, n_t - 1),
        ramp_down=(n_g, n_t - 1),
    )

    # Initialise data.
    variable_lb = np.zeros(n_vars)
    variable_ub = np.ones(n_vars)
    variable_type = np.full(n_vars, b"C"[0])
    c = np.zeros(n_vars)
    A = sparse.dok_matrix((n_cons, n_vars))
    b = np.zeros(n_cons)
    con_sense = np.full(n_cons, "")
    var_subprob_index = np.full(n_vars, 0)
    con_subprob_index = np.full(n_cons, 0)

    variable_type[vars["p"]] = b"C"[0]
    variable_type[vars["on"]] = b"B"[0]
    variable_type[vars["up"]] = b"B"[0]
    variable_type[vars["down"]] = b"B"[0]

    c[vars["p"]] = (mar_cost * max_out)[:, None]  # (n_g, n_t)
    c[vars["on"]] = nol_cost[:, None]  # (n_g, n_t)
    c[vars["up"]] = st_cost[:, None]  # (n_g, n_t)

    # Formulate constraints.
    # load balance
    # sum_g max_out[g] * p[g, t] >= demand[t]
    A[cons["load_balance"], vars["p"]] = max_out[:, None]  # (n_g, n_t)
    b[cons["load_balance"]] = demand  # (n_t,)
    con_sense[cons["load_balance"]] = "G"  # (n_t,)

    # spinning reserve
    # sum_g max_out[g] * (on[g, t] - p[g, t])
    #     >= spinning_reserve_rate * demand[t]
    A[cons["spinning_reserve"], vars["on"]] = max_out[:, None]  # (n_g, n_t)
    A[cons["spinning_reserve"], vars["p"]] = -max_out[:, None]  # (n_g, n_t)
    b[cons["spinning_reserve"]] = (
        config["uc.spinning_reserve_rate"] * demand
    )  # (n_t,)
    con_sense[cons["spinning_reserve"]] = "G"  # (n_t,)

    # generator lower bound
    # max_out[g] * p[g, t] >= min_out[g] * on[g, t]
    # or equivalently
    # min_out[g] * on[g, t] - max_out[g] * p[g, t] <= 0
    A[cons["generator_lb"], vars["on"]] = min_out[:, None]  # (n_g, n_t)
    A[cons["generator_lb"], vars["p"]] = -max_out[:, None]  # (n_g, n_t)
    con_sense[cons["generator_lb"]] = "L"

    # generator upper bound
    # p[g, t] <= on[g, t]
    # or equivalently
    # p[g, t] - on[g, t] <= 0
    # Note that p is a ratio to a generator capacity.
    A[cons["generator_ub"], vars["p"]] = 1  # (n_g, n_t)
    A[cons["generator_ub"], vars["on"]] = -1  # (n_g, n_t)
    con_sense[cons["generator_ub"]] = "L"

    # polynomial
    # on[g, t] - on[g, t-1] == up - down
    # where on[g, -1] is an initial condition of generator g.
    # Equivalently:
    # for t == 0
    #   on[g, t] - up + down == initial_status[g]
    # for t >= 1
    #   on[g, t] - on[g, t-1] - up + down == 0
    # t == 0
    A[cons["polynomial"][:, 0], vars["on"][:, 0]] = 1
    A[cons["polynomial"][:, 0], vars["up"][:, 0]] = -1
    A[cons["polynomial"][:, 0], vars["down"][:, 0]] = 1
    # t >= 1
    A[cons["polynomial"][:, 1:], vars["on"][:, 1:]] = 1
    A[cons["polynomial"][:, 1:], vars["on"][:, :-1]] = -1
    A[cons["polynomial"][:, 1:], vars["up"][:, 1:]] = -1
    A[cons["polynomial"][:, 1:], vars["down"][:, 1:]] = 1
    con_sense[cons["polynomial"]] = "E"

    # switching
    # up[g, t] + down[g, t] <= 1
    A[cons["switching"], vars["up"]] = 1
    A[cons["switching"], vars["down"]] = 1
    b[cons["switching"]] = 1
    con_sense[cons["switching"]] = "L"

    # min up
    # sum_{max(0, t-min_up[g]+1) <= i <= t} up[g, i] <= on[g, t]
    # or equivalently
    # sum_{max(0, t-min_up[g]+1) <= i <= t} up[g, i] - on[g, t] <= 0
    s = np.concatenate(
        [np.arange(i - m + 1, i + 1) for m in min_up for i in T]
    )
    t = np.concatenate([np.repeat(np.arange(n_t), m) for m in min_up])
    g = np.repeat(np.arange(n_g), min_up * n_t)
    selector = s >= 0
    s, t, g = s[selector], t[selector], g[selector]
    A[cons["min_up"][g, t], vars["up"][g, s]] = 1
    A[cons["min_up"], vars["on"]] = -1
    con_sense[cons["min_up"]] = "L"

    # min down
    # sum_{max(0, t-min_down[g]+1) <= i <= t} down[g, i] <= 1 - on[g, t]
    # or equivalently
    # sum_{max(0, t-min_down[g]+1) <= i <= t} down[g, i] + on[g, t] <= 1
    s = np.concatenate(
        [np.arange(i - m + 1, i + 1) for m in min_down for i in T]
    )
    t = np.concatenate([np.repeat(np.arange(n_t), m) for m in min_down])
    g = np.repeat(np.arange(n_g), min_down * n_t)
    selector = s >= 0
    s, t, g = s[selector], t[selector], g[selector]
    A[cons["min_down"][g, t], vars["down"][g, s]] = 1
    A[cons["min_down"], vars["on"]] = 1
    b[cons["min_down"]] = 1
    con_sense[cons["min_down"]] = "L"

    # ramp up
    # max_out[g] * p[g, t] - max_out[g] * p[g, t-1]
    #   <= op_rampup[g] * on[g, t-1] + st_rampup[g] * up[g, t]
    # or equivalently
    # max_out[g] * p[g, t] - max_out[g] * p[g, t-1]
    #   - op_rampup[g] * on[g, t-1] - st_rampup[g] * up[g, t] <= 0
    # (n_g, n_t - 1)
    A[cons["ramp_up"], vars["p"][:, 1:]] = max_out[:, None]
    # (n_g, n_t - 1)
    A[cons["ramp_up"], vars["p"][:, :-1]] = -max_out[:, None]
    # (n_g, n_t - 1)
    A[cons["ramp_up"], vars["on"][:, :-1]] = -op_rampup[:, None]
    # (n_g, n_t - 1)
    A[cons["ramp_up"], vars["up"][:, 1:]] = -st_rampup[:, None]
    con_sense[cons["ramp_up"]] = "L"

    # ramp down
    # max_out[g] * p[g, t-1] - max_out[g] * p[g, t]
    #   <= op_rampdown[g] * on[g, t] + sh_rampdown[g] * down[g, t]
    # or equivalently
    # max_out[g] * p[g, t-1] - max_out[g] * p[g, t]
    #   - op_rampdown[g] * on[g, t] - sh_rampdown[g] * down[g, t] <= 0
    # (n_g, n_t - 1)
    A[cons["ramp_down"], vars["p"][:, :-1]] = max_out[:, None]
    # (n_g, n_t - 1)
    A[cons["ramp_down"], vars["p"][:, 1:]] = -max_out[:, None]
    # (n_g, n_t - 1)
    A[cons["ramp_down"], vars["on"][:, 1:]] = -op_rampdown[:, None]
    # (n_g, n_t - 1)
    A[cons["ramp_down"], vars["down"][:, 1:]] = -sh_rampdown[:, None]
    con_sense[cons["ramp_down"]] = "L"
    A = A.tocoo()

    # Assign subproblem indices on columns and cons.
    var_subprob_index[vars["p"]] = np.arange(n_g)[:, None]  # (n_g, n_t)
    var_subprob_index[vars["on"]] = np.arange(n_g)[:, None]  # (n_g, n_t)
    var_subprob_index[vars["up"]] = np.arange(n_g)[:, None]  # (n_g, n_t)
    var_subprob_index[vars["down"]] = np.arange(n_g)[:, None]  # (n_g, n_t)
    con_subprob_index[cons["load_balance"]] = -1  # (n_t,)
    con_subprob_index[cons["spinning_reserve"]] = -1  # (n_t,)
    # (n_g, n_t)
    con_subprob_index[cons["generator_lb"]] = np.arange(n_g)[:, None]
    # (n_g, n_t)
    con_subprob_index[cons["generator_ub"]] = np.arange(n_g)[:, None]
    # (n_g, n_t)
    con_subprob_index[cons["polynomial"]] = np.arange(n_g)[:, None]
    # (n_g, n_t)
    con_subprob_index[cons["switching"]] = np.arange(n_g)[:, None]
    # (n_g, n_t)
    con_subprob_index[cons["min_up"]] = np.arange(n_g)[:, None]
    # (n_g, n_t)
    con_subprob_index[cons["min_down"]] = np.arange(n_g)[:, None]
    # (n_g, n_t - 1)
    con_subprob_index[cons["ramp_up"]] = np.arange(n_g)[:, None]
    # (n_g, n_t - 1)
    con_subprob_index[cons["ramp_down"]] = np.arange(n_g)[:, None]

    problem_data = {
        # Data required by problem_data_base
        "n_variables": n_vars,
        "n_constraints": n_cons,
        "variable_name_to_index": vars,
        "constraint_name_to_index": cons,
        "objective_sense": "min",
        "objective_offset": 0.0,
        "variable_lb": variable_lb,
        "variable_ub": variable_ub,
        "variable_type": variable_type,
        "objective_coefficient": c,
        "constraint_coefficient": A,
        "constraint_rhs": b,
        "constraint_sense": con_sense,
        # Data required for decomposable problems
        "n_subproblems": n_g,
        "variable_subproblem_index": var_subprob_index,
        "constraint_subproblem_index": con_subprob_index,
        # Data specigic to UC
        "n_g": n_g,
        "n_t": n_t,
        "G": G,
        "T": T,
        "df_g": df_g,
        "demand": np.empty(n_t, dtype=float),
        "initial_condition": np.empty(n_g, dtype=int),
        "uc.spinning_reserve_rate": config["uc.spinning_reserve_rate"],
    }
    set_demand_and_initial_condition_(
        problem_data, demand, initial_condition, update_decomposition=False
    )
    decomposition.set_decomposition_data_(problem_data, all=True)

    return problem_data


def check_feasibility(demand, df_g, initial_condition):
    """Check feasibility of the demand.

    Parameters
    ----------
    demand : (n_t,) or (batch_size, n_g) array
    df_g : DataFrame
    initial_condition : (n_g,) array

    Returns
    -------
    is_feasible : bool
        False if given demand is detected to be infeasible. True otherwise.
    msg : str
        Detailed information.
    """
    # Check feasibility at t = 0, 1 and the peak of the demand.
    _max_out = df_g["max_out"][initial_condition >= -1].sum()  # scalar
    if np.any(1.1 * np.take(demand, 0, axis=-1) > _max_out):
        return False, "cannot satisfy demand at t = 0"
    _max_out += df_g["st_rampup"][initial_condition == -2].sum()  # scalar
    if np.any(1.1 * np.take(demand, 1, axis=-1) > _max_out):
        return False, "cannot satisfy demand at t = 1"
    if np.any(1.1 * np.max(demand, axis=-1) > df_g["max_out"].sum()):
        return False, "cannot satisfy demand at t = {np.argmax(demand)}"
    return True, ""


class SquaredExponentailKernel(object):
    """Squared exponential kernel of a gaussian process."""

    def __init__(self, sigma=1.0, length_scale=1.0):
        """Initialise a SquaredExponentailKernel instance.

        Parameters
        ----------
        sigma : float, default 1.0
        length_scale : float, default 1.0
        """
        self.sigma = sigma
        self.length_scale = length_scale

    def __call__(self, x, y=None):
        """Compute the value of the kernel between two given inputs.

        - If x and y are scalars, returns a scalar.
        - If x and y are 1-dimensional arrays, returns a scalar.
        - If x and y are 2-dimensional arrays, returns a 2-dimensional arrays,
          whose ij-element is equal to kernel(x[i], y[j]).

        If y is omitted, y is set to be x.

        Parameters
        ----------
        x : array or float
        y : array or float, optional

        Returns
        -------
        k : array or float
        """
        if y is None:
            y = x
        length_scale = self.length_scale
        sigma = self.sigma

        if isinstance(x, numbers.Number) or np.asarray(x).size == 1:
            return sigma**2 * np.exp(
                -((x - y) ** 2) / (2 * length_scale**2)
            )

        x = np.asarray(x)
        y = np.asarray(y)

        if x.ndim == 1:
            return sigma**2 * np.exp(
                -np.linalg.norm(x, y, 2) ** 2 / (2 * length_scale**2)
            )
        elif x.ndim >= 3:
            raise ValueError(
                f"invalid dimension of x and y: {x.ndim}, {y.ndim}"
            )
        else:
            x = np.expand_dims(x, 1)  # (batch_size, 1, dim_input)
            norm_squared = ((x - y) ** 2).sum(
                axis=2
            )  # (batch_size, batch_size)
            return sigma**2 * np.exp(-norm_squared / (2 * length_scale**2))


def construct_demand_noise_process_L(config):
    """Return a matrix used to create demand noise process."""
    n_t = config["n_periods"]
    kernel = SquaredExponentailKernel(
        sigma=1.0,
        length_scale=config["uc.demand_noise_process.length_scale"],
    )
    x = np.linspace(0, n_t / 24, n_t)
    K = kernel(x[:, None], x[:, None])
    K[np.diag_indices_from(K)] += config["uc.demand_noise_process.alpha"]
    return np.linalg.cholesky(K) * config["uc.demand_noise_process.scale"]


if __name__ == "__main__":
    import doctest

    doctest.testmod()

# vimquickrun: python %
