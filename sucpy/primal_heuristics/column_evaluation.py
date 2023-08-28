# -*- coding: utf-8 -*-

"""Local search heuristic with economic dispatch used in CG."""

import collections
import logging
import typing
import weakref

import numpy as np

from sucpy.primal_heuristics.base import PHResult, PrimalHeuristicBase
from sucpy.problems import uc
from sucpy.solvers.base import LPR
from sucpy.utils import cplex_utils, utils

logger = logging.getLogger(__name__)


@utils.freeze_attributes
class ColumnEvaluationPrimalHeuristic(PrimalHeuristicBase):
    """Primal heuristic to search feasible solutions in a neighbourhood."""

    def __init__(
        self,
        data: typing.Mapping[str, typing.Any],
        config: typing.Mapping[str, typing.Any],
        journal: typing.Any,
    ) -> None:
        """Initialise a NoOpPrimalHeuristic instance."""
        super().__init__(
            data=data,
            config=config,
            journal=journal,
        )
        self.n_g = len(data["initial_condition"])
        self.n_t = len(data["demand"])
        self.df_g = data["df_g"]
        self.data = data
        self.config = config
        self.column_iteration = [None for i in range(data["n_subproblems"])]
        self.schedule = np.empty((self.n_g, self.n_t), dtype=bool)
        self.compute_deficit_cache = create_compute_deficit_cache(
            data["df_g"], data
        )
        self.stop_event = journal.stop_event
        self.cg = None
        self.lb = None
        self.iteration_index = None

    @property
    def n_processes(self):
        return self.compute_deficit_cache["lpr"].parameters.threads.get()

    @n_processes.setter
    def n_processes(self, n):
        return self.compute_deficit_cache["lpr"].parameters.threads.set(n)

    def set_cg(self, cg):
        """Set a weak reference to CG.

        Parameters
        ----------
        cg : object
        """
        self.cg = weakref.ref(cg)

    def lb_hook(self, y, lb):
        self.lb = lb

    def run(self, *args, **kwargs):
        """Run primal heuristic which is void.

        Returns
        -------
        status : bool
            True if primal heuristic finds a solution. False otherwise.
        msg : str
            Error message if available.
        obj : float
            Objective value.  If it's infeasible, this is set to be inf.
        sol :
            Solution.  If it's infeasible, this is set to be None.
        """
        if (self.lb is None) or (not np.isfinite(self.lb)):
            return PHResult(
                status=False,
                msg="",
                obj=np.nan,
                sol=None,
            )

        cg = self.cg()
        columns = np.array([r["solution"] for r in cg.subproblems.result])
        self.schedule = columns[:, self.n_t : 2 * self.n_t]

        res = recover_feasibility(
            df_g=self.df_g,
            data=self.data,
            schedule=self.schedule,
            compute_deficit_cache=self.compute_deficit_cache,
            stop_event=self.stop_event,
        )

        # if res["status"]:
        #     res2 = switch_off_by_cplex(self.data, res["schedule"])
        #     if res["status"]:
        #         obj1 = res["obj"]
        #     else:
        #         obj1 = np.inf
        #     if res2["status"]:
        #         obj2 = res2["obj"]
        #     else:
        #         obj2 = np.inf
        #   print(
        #       f"{obj1:12.4e}  ->  {obj2:12.4e}  "
        #       f"({res2['walltime']:.2f} sec)"
        #   )
        #     if res2["status"]:
        #         res = res2

        return PHResult(
            status=res.get("status", False),
            msg=res.get("msg", ""),
            obj=res.get("obj", np.nan),
            sol=res.get("sol", None),
        )


def compute_switching_pos(
    initial_condition,
    schedule,
    on=None,
    off=None,
    initially_on=50,
    onpos=None,
    offpos=None,
    toon=None,
    fromon=None,
    tooff=None,
    fromoff=None,
):
    """Compute on off positions.

    Examples
    --------
    >>> initial_condition = [0, 1, -1, 1, -2]
    >>> schedule = [
    ...     [1, 0, 0, 1, 1, 1, 1, 0],
    ...     [1, 1, 1, 0, 0, 1, 1, 1],
    ...     [1, 1, 1, 0, 0, 0, 1, 1],
    ...     [1, 1, 1, 1, 1, 1, 1, 1],
    ...     [0, 0, 0, 0, 0, 0, 0, 0],
    ... ]
    >>> res = compute_switching_pos(initial_condition, schedule)
    >>> print(res['onpos'])
    [[-50   0   0   3   3   3   3   0]
     [-50 -50 -50   0   0   5   5   5]
     [  0   0   0   0   0   0   6   6]
     [-50 -50 -50 -50 -50 -50 -50 -50]
     [  0   0   0   0   0   0   0   0]]
    >>> print(res['offpos'])
    [[ 1  0  0  7  7  7  7  0]
     [ 3  3  3  0  0 50 50 50]
     [ 3  3  3  0  0  0 50 50]
     [50 50 50 50 50 50 50 50]
     [ 0  0  0  0  0  0  0  0]]
    >>> print(res['toon'])
    [[ 0  2  1  0  0  0  0 50]
     [ 0  0  0  2  1  0  0  0]
     [ 0  0  0  3  2  1  0  0]
     [ 0  0  0  0  0  0  0  0]
     [50 50 50 50 50 50 50 50]]
    >>> print(res['fromon'])
    [[50  0  0  0  1  2  3  0]
     [50 50 50  0  0  0  1  2]
     [ 0  1  2  0  0  0  0  1]
     [50 50 50 50 50 50 50 50]
     [ 0  0  0  0  0  0  0  0]]
    >>> print(res['tooff'])
    [[ 1  0  0  4  3  2  1  0]
     [ 3  2  1  0  0 50 50 50]
     [ 3  2  1  0  0  0 50 50]
     [50 50 50 50 50 50 50 50]
     [ 0  0  0  0  0  0  0  0]]
    >>> print(res['fromoff'])
    [[ 0  0  1  0  0  0  0  0]
     [ 0  0  0  0  1  0  0  0]
     [ 0  0  0  0  1  2  0  0]
     [ 0  0  0  0  0  0  0  0]
     [50 50 50 50 50 50 50 50]]

    >>> schedule = [1, 1, 0, 0, 1, 1, 1]
    >>> res = compute_switching_pos(initial_condition=1, schedule=schedule)
    >>> print(res['onpos'])
    [-50 -50   0   0   4   4   4]
    """
    assert initially_on > 0
    initial_condition = np.atleast_1d(initial_condition)
    schedule = np.asarray(schedule)
    batch = schedule.ndim == 2
    schedule = np.atleast_2d(schedule).astype(bool)
    n_g, n_t = schedule.shape
    if on is None:
        on = np.c_[
            (initial_condition <= -1) & (schedule[:, 0]),
            (~schedule[:, :-1]) & (schedule[:, 1:]),
        ]
    else:
        on = np.atleast_2d(on)
    if off is None:
        off = np.c_[
            (initial_condition >= 0) & (~schedule[:, 0]),
            (schedule[:, :-1]) & (~schedule[:, 1:]),
        ]
    else:
        off = np.atleast_2d(off)
    if onpos is None:
        onpos = np.zeros((n_g, n_t), dtype=int)
    else:
        onpos = np.atleast_2d(onpos)
    if offpos is None:
        offpos = np.zeros((n_g, n_t), dtype=int)
    else:
        offpos = np.atleast_2d(offpos)
    if toon is None:
        toon = np.empty((n_g, n_t), dtype=int)
    else:
        toon = np.atleast_2d(toon)
    if fromon is None:
        fromon = np.empty((n_g, n_t), dtype=int)
    else:
        fromon = np.atleast_2d(fromon)
    if tooff is None:
        tooff = np.empty((n_g, n_t), dtype=int)
    else:
        tooff = np.atleast_2d(tooff)
    if fromoff is None:
        fromoff = np.empty((n_g, n_t), dtype=int)
    else:
        fromoff = np.atleast_2d(fromoff)

    _compute_switching_pos_impl(
        initial_condition=initial_condition,
        schedule=schedule,
        on=on,
        off=off,
        initially_on=initially_on,
        onpos=onpos,
        offpos=offpos,
        fromon=fromon,
        tooff=tooff,
    )

    _compute_switching_pos_impl(
        initial_condition=-initial_condition - 1,
        schedule=~schedule,
        on=off,
        off=on,
        initially_on=initially_on,
        onpos=np.empty((n_g, n_t), dtype=int),
        offpos=np.empty((n_g, n_t), dtype=int),
        fromon=fromoff,
        tooff=toon,
    )

    if not batch:
        onpos = onpos.squeeze(0)
        offpos = offpos.squeeze(0)
        toon = toon.squeeze(0)
        fromon = fromon.squeeze(0)
        tooff = tooff.squeeze(0)
        fromoff = fromoff.squeeze(0)

    return {
        "on": on,
        "off": off,
        "onpos": onpos,
        "offpos": offpos,
        "toon": toon,
        "fromon": fromon,
        "tooff": tooff,
        "fromoff": fromoff,
    }


def _compute_switching_pos_impl(
    initial_condition,
    schedule,
    on,
    off,
    initially_on,
    onpos,
    offpos,
    fromon,
    tooff,
):
    n_g, n_t = schedule.shape
    # Compute onpos.
    masked = initially_on + 1
    onpos[:] = -masked
    onpos[initial_condition >= 0, 0] = -initially_on
    _on = _g, _t = on.nonzero()
    onpos[_on] = _t
    onpos[:] = np.maximum.accumulate(onpos, axis=-1)
    onpos[~schedule] = 0
    # Compute offpos.
    offpos[:] = masked
    offpos[schedule[:, -1], -1] = initially_on
    _off = _g, _t = off.nonzero()
    offpos[_off] = _t
    offpos[:, ::-1] = np.minimum.accumulate(offpos[:, ::-1], axis=-1)
    offpos[~schedule] = 0
    # Count the periods to be kept on backward.
    fromon[:] = np.arange(n_t) - onpos
    fromon[:] = fromon.clip(None, initially_on)
    fromon[~schedule] = 0
    # Count the number of periods to the next switching off.
    tooff[:] = offpos - np.arange(n_t)
    tooff[offpos == initially_on] = initially_on
    tooff[~schedule] = 0


class ScheduleManager:
    """UC generator schedule manager."""

    def __init__(
        self,
        schedule,
        initial_condition=None,
        data=None,
        min_out=None,
        max_out=None,
        op_rampup=None,
        op_rampdown=None,
        st_rampup=None,
        sh_rampdown=None,
        df_g=None,
    ):
        """Initialise a ScheduleManager instance."""
        schedule = np.asarray(schedule)
        self.shape = self.n_g, self.n_t = n_g, n_t = schedule.shape
        self.df_g = df_g
        if data is not None:
            self.initial_condition = data["initial_condition"]
        else:
            if initial_condition is None:
                raise ValueError("data is missing")
            self.initial_condition = np.asarray(initial_condition)
        if df_g is not None:
            self.min_out = self.df_g["min_out"]
            self.max_out = self.df_g["max_out"]
            self.st_rampup = self.df_g["st_rampup"]
            self.sh_rampdown = self.df_g["sh_rampdown"]
            self.op_rampup = self.df_g["op_rampup"]
            self.op_rampdown = self.df_g["op_rampdown"]
        else:
            missing = (
                (min_out is None)
                or (max_out is None)
                or (op_rampup is None)
                or (op_rampdown is None)
                or (st_rampup is None)
                or (sh_rampdown is None)
            )
            if missing:
                raise ValueError("df_g is missing")
            self.min_out = np.broadcast_to(min_out, (n_g,))
            self.max_out = np.broadcast_to(max_out, (n_g,))
            self.op_rampup = np.broadcast_to(op_rampup, (n_g,))
            self.op_rampdown = np.broadcast_to(op_rampdown, (n_g,))
            self.st_rampup = np.broadcast_to(st_rampup, (n_g,))
            self.sh_rampdown = np.broadcast_to(sh_rampdown, (n_g,))
        self.extschedule = np.zeros((n_g, n_t + 1), dtype=bool)
        self.extschedule[:, 0] = self.initial_condition >= 0
        self.extschedule[:, 1:] = schedule
        self.schedule = self.extschedule[:, 1:]
        self.on = (~self.extschedule[:, :-1]) & (
            self.extschedule[:, 1:]
        )  # (n_g, n_t)
        self.off = (self.extschedule[:, :-1]) & (~self.extschedule[:, 1:])
        self.onpos = np.zeros((n_g, n_t), dtype=int)
        self.offpos = np.zeros((n_g, n_t), dtype=int)
        self.toon = np.zeros((n_g, n_t), dtype=int)
        self.fromon = np.zeros((n_g, n_t), dtype=int)
        self.tooff = np.zeros((n_g, n_t), dtype=int)
        self.fromoff = np.zeros((n_g, n_t), dtype=int)
        self.rampup_max = np.zeros((n_g, n_t), dtype=float)
        self.rampdown_max = np.zeros((n_g, n_t), dtype=float)
        self.ramp_min_out = np.zeros((n_g, n_t), dtype=float)
        self.ramp_max_out = np.zeros((n_g, n_t), dtype=float)
        self.binding_ramp = np.zeros((n_g, n_t), dtype=int)
        self.update()

    def update(self, g=None):
        """Update internal data.

        This will update the following:
        - onpos
        - offpos
        - toon
        - fromon
        - tooff
        - fromoff
        - rampup_max
        - rampdown_max
        - ramp_min_out
        - ramp_max_out
        - binding_ramp

        Parameters
        ----------
        g : int or slice, optional
            If given, only update specified generators.
        """
        if g is None:
            g = slice(None)
        else:
            g = slice(g, g + 1)
        self.on[g] = (~self.extschedule[g, :-1]) & (self.extschedule[g, 1:])
        self.off[g] = (self.extschedule[g, :-1]) & (~self.extschedule[g, 1:])
        compute_switching_pos(
            initial_condition=self.initial_condition[g],
            schedule=self.schedule[g],
            on=self.on[g],
            off=self.off[g],
            onpos=self.onpos[g],
            offpos=self.offpos[g],
            toon=self.toon[g],
            fromon=self.fromon[g],
            tooff=self.tooff[g],
            fromoff=self.fromoff[g],
        )
        # Update onpos etc.
        min_out = self.min_out
        max_out = self.max_out
        st_rampup = self.st_rampup
        sh_rampdown = self.sh_rampdown
        op_rampup = self.op_rampup
        op_rampdown = self.op_rampdown
        self.rampup_max[g] = (
            self.fromon[g] * op_rampup[g, None] + st_rampup[g, None]
        )
        self.rampup_max[g][~self.schedule[g]] = -1
        self.rampdown_max[g] = (self.tooff[g] - 1) * op_rampdown[
            g, None
        ] + sh_rampdown[g, None]
        self.rampdown_max[g][~self.schedule[g]] = -1
        self.ramp_min_out[g] = 0
        _g, _t = self.schedule[g].nonzero()
        self.ramp_min_out[g][_g, _t] = min_out[_g]
        self.ramp_max_out[g] = np.minimum(
            np.minimum(self.rampup_max[g], self.rampdown_max[g]),
            max_out[g, None],
        )
        # binding : 0 if ramp up is binding, 1 if ramp down is
        # binding and -1 otherwise.
        self.binding_ramp[g] = -1
        self.binding_ramp[g][self.ramp_max_out[g] == self.rampup_max[g]] = 0
        self.binding_ramp[g][self.ramp_max_out[g] == self.rampdown_max[g]] = 1
        self.binding_ramp[g][~self.schedule[g]] = -1


def _fmtsc(schedule, initial_condition=None):
    schedule = np.atleast_2d(schedule).astype(int)
    if initial_condition is not None:
        initial_condition = np.atleast_1d(initial_condition)
        np.testing.assert_equal(len(schedule), len(initial_condition))
        schedule = np.concatenate(
            [(initial_condition >= 0)[:, None], schedule], axis=1
        )
    ret = "\n".join(["".join(map(str, a)) for a in schedule])
    return ret.replace("1", "*").replace("0", ".")


def compute_deficit(df_g, data, schedule, cache=None):
    """Compute deficit of capacity of a given schedule

    This evaluates a given schedule and compute
    deficit of power supply and spinning reserve.
    This is done by fixing on off variable to
    a given schedule and make linear programming
    relaxation.  Note that when a generator gets turned
    up or down, the turn_up and turn_down variables
    are determined uniquely.  Otherwise, they are
    free but minimization forces them to be zeros.

    Parameters
    ----------
    df_g : list of generator data
    data : dict
    schedule : (n_g, n_t) array of bool
    cache : dict, optional
        Use a cache object returned from `create_compute_deficit_cache`.

    Returns
    -------
    res : dict
        This contains the following items.
        - status : status,
        - status_name : status_name,
        - demand : data["demand"],
        - schedule : schedule,
        - obj : obj,
        - sol : lpr_sol[: 4 * n_g * n_t],
        - out : out,
        - deficit : deficit,
        - surplus : surplus,
        - rampup_slack : rampup_slack,
        - rampdown_slack : rampdown_slack,
        - walltime : t.walltime,
        - proctime : t.proctime,
    """
    if cache is None:
        cache = {}
    with utils.timer() as t:
        n_g, n_t = schedule.shape
        if "lpr" in cache:
            lpr = cache["lpr"]
            prev_schedule = cache["schedule"]
            _g, _t = (schedule != prev_schedule).nonzero()
            val = schedule[_g, _t]
            var = n_t * _g + _t + n_g * n_t
            # Fix the schedule.
            if len(_g):
                cplex_utils.fix_variables(
                    lpr, var, val, validate=False, context=False
                )
        else:
            # Use dual this is significantly faster to repeatedly solving
            # economic dispatch.
            cache["lpr"] = lpr = LPR(data, {"cg.lpr_method": "dual_simplex"})
            # Add slack variables.
            load_spinning_con = np.arange(2 * n_t).reshape(2, n_t)
            cache["load_spinning_con"] = load_spinning_con
            penalty = (
                df_g["mar_cost"] + df_g["nol_cost"] + df_g["st_cost"]
            ).max() * 100
            slack_index0 = cplex_utils.add_slacks(
                lpr, load_spinning_con[0], penalty * 10
            )
            slack_index1 = cplex_utils.add_slacks(
                lpr, load_spinning_con[1], penalty
            )
            cache["slack_index"] = np.stack(
                [slack_index0, slack_index1], axis=0
            )
            # Fix the schedule.
            on_var = np.arange(n_g * n_t, 2 * n_g * n_t).reshape(n_g, n_t)
            cplex_utils.fix_variables(
                lpr, on_var, schedule, validate=False, context=False
            )
            cache["n_cons"] = lpr.linear_constraints.get_num()
        cache["schedule"] = np.array(schedule)
        lpr.solve()
        status = lpr.solution.get_status()
        status_name = lpr.solution.status[status]
        if status != lpr.solution.status.optimal:
            error_msg = (
                f"Failed to solve an ED. "
                f"instance_id: {data['instance_id']}  "
                f"status: {lpr.solution.status[lpr.solution.get_status()]}"
            )
            logger.error(error_msg)
        try:
            obj = lpr.solution.get_objective_value()
            lpr_sol = np.array(lpr.solution.get_values())
            lpr_slack = np.array(lpr.solution.get_linear_slacks())
        except cplex_utils.CplexError as e:
            error_msg = (
                f"Failed to solve an ED. "
                f"instance_id: {data['instance_id']}  "
                f"status: {lpr.solution.status[lpr.solution.get_status()]}"
            )
            logger.error(error_msg, exc_info=e)
            raise
        out_var = np.arange(n_g * n_t).reshape(n_g, n_t)
        out = lpr_sol[out_var]
        deficit = lpr_sol[cache["slack_index"]]
        surplus = np.abs(lpr_slack[cache["load_spinning_con"]])
        n_cons = cache["n_cons"]
        rampup_con = np.arange(
            n_cons - 2 * n_g * (n_t - 1), n_cons - n_g * (n_t - 1)
        ).reshape(n_g, n_t - 1)
        rampdown_con = np.arange(n_cons - n_g * (n_t - 1), n_cons).reshape(
            n_g, n_t - 1
        )
        rampup_slack = np.abs(lpr_slack[rampup_con])
        rampdown_slack = np.abs(lpr_slack[rampdown_con])
    return {
        "status": status,
        "status_name": status_name,
        "demand": data["demand"],
        "schedule": schedule,
        "obj": obj,
        "sol": lpr_sol[: 4 * n_g * n_t],
        "out": out,
        "deficit": deficit,
        "surplus": surplus,
        "rampup_slack": rampup_slack,
        "rampdown_slack": rampdown_slack,
        "walltime": t.walltime,
        "proctime": t.proctime,
    }


def create_compute_deficit_cache(df_g, data):
    cache = {}
    n_t = len(data["demand"])
    # Use dual this is significantly faster to repeatedly solving
    # economic dispatch.
    cache["lpr"] = lpr = LPR(data, {"cg.lpr_method": "dual_simplex"})
    # Add slack variables.
    load_spinning_con = np.arange(2 * n_t).reshape(2, n_t)
    cache["load_spinning_con"] = load_spinning_con
    penalty = (
        df_g["mar_cost"] + df_g["nol_cost"] + df_g["st_cost"]
    ).max() * 100
    slack_index0 = cplex_utils.add_slacks(
        lpr, load_spinning_con[0], penalty * 10
    )
    slack_index1 = cplex_utils.add_slacks(lpr, load_spinning_con[1], penalty)
    cache["slack_index"] = np.stack([slack_index0, slack_index1], axis=0)
    cache["n_cons"] = lpr.linear_constraints.get_num()
    cache["schedule"] = -1
    return cache


def increase_spinning_reserve(
    amount,
    schedule,
    initial_condition=None,
    max_out=None,
    priority_list=None,
):
    """Switch on generators and increase capacity by a given amount.

    Examples
    --------
    >>> amount = [1, 0, 3, 1]
    >>> schedule = [
    ...     [0, 0, 0, 1],
    ...     [1, 1, 0, 1],
    ...     [1, 1, 0, 0],
    ...     [1, 1, 0, 0],
    ...     [0, 0, 1, 1],
    ... ]
    >>> initial_condition = [-2, 1, 0, 1, -1]
    >>> ret = increase_spinning_reserve(amount, schedule, initial_condition, 2)
    >>> ret.schedule * 1
    array([[0, 0, 0, 1],
           [1, 1, 1, 1],
           [1, 1, 1, 1],
           [1, 1, 0, 0],
           [1, 1, 1, 1]])
    >>> ret.g
    array([1, 2, 2, 4, 4])
    >>> ret.t
    array([2, 2, 3, 0, 1])

    >>> amount = [0, 0, 3, 0, 0]
    >>> schedule = [
    ...     [0, 0, 0, 0, 1],
    ...     [0, 0, 0, 0, 0],
    ...     [0, 0, 0, 1, 1],
    ...     [1, 1, 1, 0, 0],
    ...     [1, 1, 1, 1, 0],
    ... ]
    >>> initial_condition = [-2, 1, 0, 1, -1]
    >>> ret = increase_spinning_reserve(amount, schedule, initial_condition, 2)
    >>> ret.schedule * 1
    array([[0, 0, 1, 1, 1],
           [0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1],
           [1, 1, 1, 0, 0],
           [1, 1, 1, 1, 0]])
    """
    amount = np.asarray(amount)
    if isinstance(schedule, ScheduleManager):
        sman = schedule
        sman.update()
    else:
        sman = ScheduleManager(
            schedule=schedule,
            initial_condition=initial_condition,
            min_out=0,
            max_out=max_out,
            op_rampup=0,
            op_rampdown=0,
            st_rampup=0,
            sh_rampdown=0,
        )
    n_g, n_t = sman.shape
    schedule = sman.schedule
    if priority_list is None:
        priority_list = np.arange(n_g)
    if priority_list.ndim == 1:
        priority_list = np.broadcast_to(priority_list[:, None], (n_g, n_t))

    modified_g = []
    modified_t = []

    def impl(sman, window, amount):
        """Search a generator schedule to extend.

        Parameters
        ----------
        sman : ScheduleManager
        window : int
            Size of window to look ahead/behind.
        amount : (n_t,) array of float
            Positive amount indicates some generator must be switched on.

        Returns
        -------
        g : array of int
        t : array of int
            Indices of modified generators and time periods.
            `schedule[g, t]` gives the part of the schedule to be modified.
        """
        for t in range(n_t):
            # Search a schedule to extend forward
            # (i.e. delaying switching off).
            if amount[t] <= 0:
                continue
            selector1 = sman.schedule[:, t] == 0
            selector2 = sman.fromoff[:, t] <= window - 1
            joint_selector = selector1 & selector2
            if np.any(joint_selector):
                # Pick up the first one in the priority list.
                g = priority_list[
                    joint_selector[priority_list[:, t]].nonzero()[0][0], t
                ]
                _g = np.repeat(g, sman.fromoff[g, t] + 1)
                _t = np.arange(t - sman.fromoff[g, t], t + 1)
                np.testing.assert_array_compare(np.less, _t, n_t)
                return _g, _t
        for t in range(n_t - 1, -1, -1):
            # Search a schedule to extend backward
            # (i.e. switching on earlier).
            if amount[t] <= 0:
                continue
            selector1 = sman.schedule[:, t] == 0
            selector2 = sman.toon[:, t] <= window
            selector3 = -sman.initial_condition - 1 <= t
            joint_selector = selector1 & selector2 & selector3
            if np.any(joint_selector):
                # Pick up the first one in the priority list.
                g = priority_list[
                    joint_selector[priority_list[:, t]].nonzero()[0][0], t
                ]
                _g = np.repeat(g, sman.toon[g, t])
                _t = np.arange(t, t + sman.toon[g, t])
                np.testing.assert_array_compare(np.less, _t, n_t)
                return _g, _t
        return [], []

    for window in range(1, 6):
        while True:
            _g, _t = impl(sman, window, amount)
            if len(_g) == 0:
                break
            sman.schedule[_g, _t] = 1
            sman.update(_g[0])
            amount[_t] -= sman.max_out[_g]
            modified_g.append(_g)
            modified_t.append(_t)

    if len(modified_g) == 0:
        modified_g = np.array([])
        modified_t = np.array([])
    else:
        modified_g = np.concatenate(modified_g)
        modified_t = np.concatenate(modified_t)
    return ScheduleModification(
        schedule=schedule, g=modified_g, t=modified_t, remaining=amount
    )


ScheduleModification = collections.namedtuple(
    "ScheduleModification", "schedule,g,t,remaining"
)


def increase_capacity(
    amount,
    schedule,
    power_output,
    initial_condition=None,
    max_out=None,
    op_rampup=None,
    op_rampdown=None,
    st_rampup=None,
    sh_rampdown=None,
    priority_list=None,
):
    """Switch on generators to increase power output capacity.

    Examples
    --------
    >>> demand = [2, 2, 4, 4, 6, 4, 2, 2]
    >>> schedule = [
    ...     [0, 0, 0, 0, 0, 0, 0, 0],
    ...     [0, 0, 0, 0, 0, 0, 0, 0],
    ...     [0, 0, 1, 1, 1, 1, 0, 0],
    ...     [1, 1, 1, 1, 1, 1, 1, 1],
    ... ]
    >>> power_out = [
    ...     [0, 0, 0, 0, 0, 0, 0, 0],
    ...     [0, 0, 0, 0, 0, 0, 0, 0],
    ...     [0, 0, 1, 2, 2, 1, 0, 0],
    ...     [2, 2, 2, 2, 2, 2, 2, 2],
    ... ]
    >>> deficit = np.array(demand) - np.array(power_out).sum(0)
    >>> initial_condition = [-1, -1, -1, 0]
    >>> ret = increase_capacity(
    ...     deficit, schedule, power_out, initial_condition, 2, 1
    ... )
    >>> ret.schedule * 1
    array([[0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 1, 1, 1, 0],
           [1, 1, 1, 1, 1, 1, 1, 1]])
    >>> ret.g
    array([2, 2])
    >>> ret.t
    array([6, 1])

    >>> demand = [
    ...      2, 3, 2, 3, 3, 4
    ... ]
    >>> schedule = [
    ...     [0, 0, 0, 0, 0, 0],
    ...     [0, 0, 0, 0, 0, 0],
    ...     [0, 0, 0, 1, 1, 1],
    ...     [1, 1, 1, 1, 1, 1],
    ... ]
    >>> power_out = [
    ...     [0, 0, 0, 0, 0, 0],
    ...     [0, 0, 0, 0, 0, 0],
    ...     [0, 0, 0, 1, 1, 2],
    ...     [2, 2, 2, 2, 2, 2],
    ... ]
    >>> deficit = np.array(demand) - np.array(power_out).sum(0)
    >>> initial_condition = [-1, -1, -1, 0]
    >>> ret = increase_capacity(
    ...     deficit, schedule, power_out, initial_condition, 2, 1
    ... )
    >>> ret.schedule * 1
    array([[0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1]])
    """
    if isinstance(schedule, ScheduleManager):
        sman = schedule
        sman.update()
        n_g, n_t = schedule.shape
        op_rampdown = sman.op_rampdown
        op_rampup = sman.op_rampup
        sh_rampdown = sman.sh_rampdown
        st_rampup = sman.st_rampup
    else:
        initial_condition = np.atleast_1d(initial_condition)
        schedule = np.asarray(schedule)
        n_g, n_t = schedule.shape
        if (
            (op_rampdown is None)
            and (st_rampup is None)
            and (sh_rampdown is None)
        ):
            if op_rampup is None:
                raise ValueError("op_rampup must be given")
            op_rampup = np.broadcast_to(op_rampup, (n_g,))
            op_rampdown = st_rampup = sh_rampdown = op_rampup
        elif (
            (op_rampdown is None)
            or (st_rampup is None)
            or (sh_rampdown is None)
        ):
            raise ValueError(
                "op_rampdown, st_rampup and sh_rampdown must be given"
            )
        sman = ScheduleManager(
            schedule=schedule,
            initial_condition=initial_condition,
            min_out=0,
            max_out=max_out,
            op_rampup=op_rampup,
            op_rampdown=op_rampdown,
            st_rampup=st_rampup,
            sh_rampdown=sh_rampdown,
        )
    if priority_list is None:
        priority_list = np.arange(n_g)
    if priority_list.ndim == 1:
        priority_list = np.broadcast_to(priority_list[:, None], (n_g, n_t))
    schedule = sman.schedule
    power_output = np.array(power_output, dtype=float)
    amount = np.array(amount, dtype=float)

    modified_g = []
    modified_t = []

    def impl(sman, window, amount):
        """Search a generator schedule to extend.

        This searches a part of schedult to be modified.

        This searches two cases. The first case is an active ramp
        constraints, which can be fixed by shifting timing of swiching..
        The second case is where generator is swiched on too late
        (off too early) and extending the schedule can fix
        the power supply shortage.

        Parameters
        ----------
        sman : ScheduleManager
        window : int
            Size of window to look ahead/behind.
        amount : (n_t,) array of float
            Positive amount indicates some generator must be switched on.

        Returns
        -------
        modified_g : array of int
        modified_t : array of int
            Indices of modified generators and time periods.
            `schedule[modified_g, modified_t]` gives the part of
            the schedule to be modified.
        affected_t : array of int
        affected_by : array of float
            A power supply in `amount[affected_t]` can be fixed
            by `affected_by`.
        """
        for t in range(n_t):
            if amount[t] <= 0:
                continue
            # Find a generator whose ramp down constraint is active.
            # Eliminate the case when the power is at maximum.
            selector1 = np.isclose(sman.rampdown_max[:, t], power_output[:, t])
            selector2 = ~np.isclose(sman.max_out, power_output[:, t])
            joint_selector = selector1 & selector2
            if np.any(joint_selector):
                # Pick up the first one in the priority list.
                g = priority_list[
                    joint_selector[priority_list[:, t]].nonzero()[0][0], t
                ]
                pos = sman.offpos[g, t]
                modified_g = [g]
                modified_t = [pos]
                affected_t = np.arange(t, pos + 1)
                affected_by = op_rampdown[g] * np.arange(
                    len(affected_t) - 1, -1, -1
                )
                affected_by[-1] += sh_rampdown[g]
                return modified_g, modified_t, affected_t, affected_by

            # Find a generator which was switched off too early.
            selector1 = sman.schedule[:, t] == 0
            selector2 = sman.fromoff[:, t] <= window - 1
            joint_selector = selector1 & selector2
            if np.any(joint_selector):
                # Pick up the first one in the priority list.
                g = priority_list[
                    joint_selector[priority_list[:, t]].nonzero()[0][0], t
                ]
                modified_g = np.repeat(g, sman.fromoff[g, t] + 1)
                affected_t = modified_t = np.arange(
                    t - sman.fromoff[g, t], t + 1
                )
                affected_by = op_rampdown[g] * np.arange(
                    len(affected_t) - 1, -1, -1
                )
                affected_by[-1] += sh_rampdown[g]
                return modified_g, modified_t, affected_t, affected_by

        for t in range(n_t - 1, -1, -1):
            if amount[t] <= 0:
                continue
            # Find a generator whose ramp up constraint is active.
            selector1 = np.isclose(sman.rampup_max[:, t], power_output[:, t])
            selector2 = ~np.isclose(sman.max_out, power_output[:, t])
            joint_selector = selector1 & selector2
            if np.any(joint_selector):
                # Pick up the first one in the priority list.
                g = priority_list[
                    joint_selector[priority_list[:, t]].nonzero()[0][0], t
                ]
                pos = sman.onpos[g, t] - 1
                modified_g = [g]
                modified_t = [pos]
                affected_t = np.arange(pos, t + 1)
                affected_by = op_rampup[g] * np.arange(len(affected_t))
                affected_by[0] += st_rampup[g]
                return modified_g, modified_t, affected_t, affected_by

            # Find a generator which is switched on too late.
            selector1 = sman.schedule[:, t] == 0
            selector2 = sman.toon[:, t] <= window
            selector3 = -sman.initial_condition - 1 <= t
            joint_selector = selector1 & selector2 & selector3
            if np.any(joint_selector):
                # Pick up the first one in the priority list.
                g = priority_list[
                    joint_selector[priority_list[:, t]].nonzero()[0][0], t
                ]
                modified_g = np.repeat(g, sman.toon[g, t])
                affected_t = modified_t = np.arange(t, t + sman.toon[g, t])
                affected_by = op_rampup[g] * np.arange(len(affected_t))
                affected_by[0] += st_rampup[g]
                return modified_g, modified_t, affected_t, affected_by

        return [], [], [], []

    for window in range(1, 6):
        while True:
            mod_g, mod_t, affected_t, affected_by = impl(sman, window, amount)
            if len(mod_g) == 0:
                break
            # Update schedule.
            sman.schedule[mod_g, mod_t] = 1
            sman.update(mod_g[0])
            # Update amount
            amount[affected_t] -= affected_by
            modified_g.append(mod_g)
            modified_t.append(mod_t)

    if len(modified_g) == 0:
        modified_g = np.array([])
        modified_t = np.array([])
    else:
        modified_g = np.concatenate(modified_g)
        modified_t = np.concatenate(modified_t)
    return ScheduleModification(
        schedule=schedule, g=modified_g, t=modified_t, remaining=amount
    )


def increase_capacity_to_meed_demand(
    demand,
    schedule,
    initial_condition=None,
    max_out=None,
    op_rampup=None,
    op_rampdown=None,
    st_rampup=None,
    sh_rampdown=None,
    priority_list=None,
):
    """Switch on generators to meet demand.

    Examples
    --------
    >>> demand = [2, 2, 4, 4, 6, 4, 2, 2]
    >>> schedule = [
    ...     [0, 0, 0, 0, 0, 0, 0, 0],
    ...     [0, 0, 0, 0, 0, 0, 0, 0],
    ...     [0, 0, 1, 1, 1, 1, 0, 0],
    ...     [1, 1, 1, 1, 1, 1, 1, 1],
    ... ]
    >>> initial_condition = [-1, -1, -1, 0]
    >>> ret = increase_capacity_to_meed_demand(
    ...     demand, schedule, initial_condition, 2, 1
    ... )
    >>> ret.schedule * 1
    array([[0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 1, 1, 1, 0],
           [1, 1, 1, 1, 1, 1, 1, 1]])
    >>> ret.g
    array([2, 2])
    >>> ret.t
    array([6, 1])
    """
    if isinstance(schedule, ScheduleManager):
        sman = schedule
        sman.update()
        n_g, n_t = schedule.shape
    else:
        initial_condition = np.atleast_1d(initial_condition)
        schedule = np.asarray(schedule)
        n_g, n_t = schedule.shape
        if (
            (op_rampdown is None)
            and (st_rampup is None)
            and (sh_rampdown is None)
        ):
            op_rampdown = st_rampup = sh_rampdown = op_rampup
        elif (
            (op_rampdown is None)
            or (st_rampup is None)
            or (sh_rampdown is None)
        ):
            raise ValueError(
                "op_rampdown, st_rampup and sh_rampdown must be given"
            )
        sman = ScheduleManager(
            schedule=schedule,
            initial_condition=initial_condition,
            min_out=0,
            max_out=max_out,
            op_rampup=op_rampup,
            op_rampdown=op_rampdown,
            st_rampup=st_rampup,
            sh_rampdown=sh_rampdown,
        )
    schedule = sman.schedule
    demand = np.asarray(demand)

    power_output = sman.ramp_max_out.clip(0)
    return increase_capacity(
        amount=demand - power_output.sum(0),
        schedule=sman,
        power_output=power_output,
        priority_list=priority_list,
    )


def increase_capacity_with_new_generators(
    amount,
    schedule,
    initial_condition=None,
    max_out=None,
    priority_list=None,
):
    """Switch on new generators and increase capacity by a given amount."""
    amount = np.asarray(amount)
    if isinstance(schedule, ScheduleManager):
        sman = schedule
        sman.update()
        n_g, n_t = schedule.shape
    else:
        sman = ScheduleManager(
            schedule=schedule,
            initial_condition=initial_condition,
            min_out=0,
            max_out=max_out,
            op_rampup=0,
            op_rampdown=0,
            st_rampup=0,
            sh_rampdown=0,
        )
    n_g, n_t = sman.shape
    schedule = sman.schedule
    if priority_list is None:
        priority_list = np.arange(n_g)
    if priority_list.ndim == 1:
        priority_list = np.broadcast_to(priority_list[:, None], (n_g, n_t))

    modified_g = []
    modified_t = []

    # Fix remaining amount by swiching on new generators.
    for t in (amount > 0).nonzero()[0]:
        for g in priority_list[:, t]:
            if np.all(amount[t] <= 0):
                break
            if schedule[g, t] == 1:
                continue
            if t < -sman.initial_condition[g] - 1:
                continue
            schedule[g, t] = 1
            amount[t] -= sman.max_out[g]
            modified_g.append([g])
            modified_t.append([t])

    if len(modified_g) == 0:
        modified_g = np.array([])
        modified_t = np.array([])
    else:
        modified_g = np.concatenate(modified_g)
        modified_t = np.concatenate(modified_t)
    return ScheduleModification(
        schedule=schedule, g=modified_g, t=modified_t, remaining=amount
    )


def recover_feasibility(
    df_g,
    data,
    schedule,
    priority_score=None,
    compute_deficit_cache=None,
    stop_event=None,
    initial_fix=True,
    step_log=False,
):
    with utils.timer() as timer:
        if stop_event is None:
            stop_event = utils.DummyEvent()
        if step_log:
            step_log = {
                "schedule": [],
                "obj": [],
                "out": [],
                "deficit": [],
                "surplus": [],
            }
        else:
            step_log = None
        res = {}
        solved = False
        timer.start(0)
        n_g, n_t = schedule.shape
        if isinstance(schedule, ScheduleManager):
            sman = schedule
            sman.update()
        else:
            schedule = uc.enforce_initial_condition(
                schedule=schedule,
                initial_condition=data["initial_condition"],
            )
            sman = ScheduleManager(df_g=df_g, data=data, schedule=schedule)
        schedule = sman.schedule
        if compute_deficit_cache is None:
            compute_deficit_cache = {}
        demand = data["demand"]
        initial_condition = data["initial_condition"]
        default_score = -df_g["nol_cost"]
        if np.all(default_score == 0):
            default_score = -df_g["mar_cost"]
        default_score -= np.min(default_score)
        if (priority_score is None) or (len(np.unique(priority_score)) == 1):
            priority_score = default_score
        if priority_score.ndim == 1:
            priority_score = np.broadcast_to(
                priority_score[:, None], (n_g, n_t)
            )
        min_step = np.min(np.diff(np.unique(priority_score)))
        tie_breaker = default_score * min_step / (default_score.sum() * 2)
        priority_score = priority_score + tie_breaker[:, None]
        priority_list = np.argsort(-priority_score, axis=0)

        uc.enforce_min_up_down(
            schedule=schedule,
            initial_condition=initial_condition,
            df_g=df_g,
            out=schedule,
        )

        if initial_fix:
            # Satisfy spinning reserve constraints.
            deficit = -(
                (schedule * df_g["max_out"][:, None]).sum(0) - 1.1 * demand
            )
            schedule[:] = increase_spinning_reserve(
                amount=deficit,
                schedule=sman,
                initial_condition=initial_condition,
                priority_list=priority_list,
            ).schedule

            uc.enforce_min_up_down(
                schedule=schedule,
                initial_condition=initial_condition,
                df_g=df_g,
                out=schedule,
            )

        cplex_walltime = cplex_proctime = 0
        for j in range(50):
            if stop_event.is_set():
                break
            timer.start(1)
            with utils.timer() as cplex_timer:
                res = compute_deficit(
                    df_g, data, schedule, cache=compute_deficit_cache
                )
            if step_log:
                step_log["schedule"].append(res["schedule"].copy())
                step_log["obj"].append(res["obj"])
                step_log["out"].append(res["out"].copy())
                step_log["deficit"].append(res["deficit"].copy())
                step_log["surplus"].append(res["surplus"].copy())
            cplex_walltime += cplex_timer.walltime
            cplex_proctime += cplex_timer.proctime
            deficit = res["deficit"]
            if stop_event.is_set():
                break
            elif np.all(deficit <= 0):
                solved = True
                break
            timer.start(2)
            modification = increase_spinning_reserve(
                amount=deficit[1, :],
                schedule=sman,
                priority_list=priority_list,
            )
            if stop_event.is_set():
                break
            if len(modification.g) > 0:
                schedule[:] = modification.schedule
                res["deficit"][1] = modification.remaining
                uc.enforce_min_up_down(
                    schedule=schedule,
                    initial_condition=initial_condition,
                    df_g=df_g,
                    out=schedule,
                )
                continue
            timer.start(3)
            modification = increase_capacity(
                amount=deficit[0, :],
                schedule=sman,
                power_output=res["out"],
                priority_list=priority_list,
            )
            if stop_event.is_set():
                break
            if len(modification.g) > 0:
                schedule[:] = modification.schedule
                res["deficit"][0] = modification.remaining
                uc.enforce_min_up_down(
                    schedule=schedule,
                    initial_condition=initial_condition,
                    df_g=df_g,
                    out=schedule,
                )
                continue
            timer.start(4)
            schedule[:] = increase_capacity_with_new_generators(
                amount=deficit.sum(0),
                schedule=sman,
                priority_list=priority_list,
            ).schedule
            uc.enforce_min_up_down(
                schedule=schedule,
                initial_condition=initial_condition,
                df_g=df_g,
                out=schedule,
            )

        # For a debugging purpose, various helpers may be useful.
        uc.assert_initial_condition(data["initial_condition"], schedule)
        uc.assert_min_up_down(df_g, initial_condition, schedule)
        if not stop_event.is_set():
            timer.start(5)
            with utils.timer() as cplex_timer:
                res = compute_deficit(
                    df_g, data, schedule, cache=compute_deficit_cache
                )
            cplex_walltime += cplex_timer.walltime
            cplex_proctime += cplex_timer.proctime
    res["status"] = solved and len(res) and np.all(res["deficit"] <= 0)
    if res["status"]:
        res["msg"] = ""
    elif stop_event.is_set():
        res["msg"] = "user abort"
    elif np.any(res["deficit"] > 0):
        res["msg"] = "failed to recover feasibility"
        logger.warn("failed to recover feasibility")
    else:
        raise ValueError("invalid state of local search")
    res["walltime"] = timer.walltime
    res["proctime"] = timer.proctime
    res["cplex_walltime"] = cplex_walltime
    res["cplex_proctime"] = cplex_proctime
    rt = np.array(timer.record_type)
    rv = np.array(timer.record_walltime)
    res["walltime_breakdown"] = {
        "init": np.sum(rv[rt == 0]),
        "dispatch": np.sum(rv[rt == 1]),
        "spin": np.sum(rv[rt == 2]),
        "load": np.sum(rv[rt == 3]),
        "new": np.sum(rv[rt == 4]),
        "final": np.sum(rv[rt == 5]),
    }
    res["step_log"] = step_log
    return res


def switch_off_by_cplex(problem_data, schedule):
    """Ask CPLEX to switch off generators on a given feasible schedule.

    Parameters
    ----------
    problem_data : dict
    schedule : 2d array of bool

    Returns
    -------
    res : dict
        This contains the following items.
        - status : bool
            True if success.
        - obj : float
            Objective value of the resulting schedule.
            This is None on failure.
        - sol : 1d array of float
            Improved solution.
            This is None on failure.
        - schedule : 2d array of bool
            Improved schedule.
            This is None on failure.
        - walltime : float
            Elapse of model solution time.
    """
    from sucpy.solvers.base import ExtensiveModel

    n_g, n_t = problem_data["n_g"], problem_data["n_t"]
    model = ExtensiveModel(problem_data, config={})
    np.testing.assert_equal(schedule.shape, (n_g, n_t))
    schedule = np.round(schedule).astype(bool)
    off_variable_index = np.nonzero(schedule.ravel() == 0)[0] + n_g * n_t
    cplex_utils.fix_variables(model, off_variable_index, 0)
    # model.MIP_start.add(
    #     map(int, on_off_variable_index.ravel()),
    #     map(float, schedule.ravel()),
    # )
    with utils.timer() as timer:
        model.solve()
    if cplex_utils.is_mip_optimal(model):
        sol = np.array(model.solution.get_values())
        schedule = (
            np.round(sol[n_g * n_t : 2 * n_g * n_t])
            .reshape(n_g, n_t)
            .astype(bool)
        )
        res = {
            "status": True,
            "obj": model.solution.MIP.get_best_objective(),
            "sol": sol,
            "schedule": schedule,
            "walltime": timer.walltime,
        }
    else:
        res = {
            "status": False,
            "obj": None,
            "sol": None,
            "schedule": None,
            "walltime": timer.walltime,
        }
    return res


def main():
    """Run the main routine of this script."""
    import argparse
    import os

    from sucpy.problems.uc import ParametrizedUC

    parser = argparse.ArgumentParser()
    parser.add_argument("method", type=str, choices=["network", "lpr"])
    parser.add_argument("n_g", type=int)
    args = parser.parse_args()

    n_instances = 40
    method = args.method
    path = f"workspace/v3/{args.n_g}"

    os.makedirs(f"{path}/out/plots", exist_ok=True)
    parametrized_problem = ParametrizedUC()
    parametrized_problem.load(f"{path}/out/parametrized_problem_data.pkl")
    parametrized_problem.use_default_initial_condition = False
    parametrized_problem.validation()
    n_g, n_t = parametrized_problem.n_g, parametrized_problem.n_t
    df_g = parametrized_problem.df_g
    test_data = dict(np.load(f"{path}/out/tol_5e-03/{method}_validation.npz"))
    selector = (
        (test_data["column_iteration"] == 0)
        & (test_data["column_type"] == 1)
        & (test_data["column_rank"] == 0)
    )
    test_data["initial_sol"] = np.unpackbits(
        test_data["column_on"][selector], axis=-1
    ).reshape(n_instances, n_g, n_t)

    first_iteration_selector = np.r_[
        0, np.cumsum(test_data["n_iterations"])[:-1]
    ]
    first_lb = test_data["iter_lb"][first_iteration_selector]

    for i in range(10):
        data = parametrized_problem.sample_instance(1500000 + i)
        schedule = test_data["initial_sol"][i]
        lb = test_data["obj_bound"][i]
        compute_deficit_cache = {}
        compute_deficit_cache = create_compute_deficit_cache(df_g, data)
        with utils.timer() as t:
            res = recover_feasibility(
                df_g,
                data,
                schedule,
                compute_deficit_cache=compute_deficit_cache,
            )
        first_gap = (res["obj"] - first_lb[i]) / res["obj"]
        gap = (res["obj"] - lb) / res["obj"]
        print(
            f"{i:3d} {first_gap * 100:6.2f} {gap * 100:6.2f}  "
            f"time: {t.walltime:5.1f}  cplex: {res['cplex_walltime']:5.1f}"
        )


if __name__ == "__main__":
    import doctest

    fail, _ = doctest.testmod()
    if not fail:
        main()
