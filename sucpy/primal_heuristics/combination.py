# -*- coding: utf-8 -*-

"""Combination of multiple primal heuristics"""

import logging
import time
import typing

import numpy as np

from sucpy.primal_heuristics.base import PHResult, PrimalHeuristicBase
from sucpy.utils import utils

logger = logging.getLogger(__name__)


@utils.freeze_attributes
class CombinedPrimalHeuristic(PrimalHeuristicBase):
    """Combination of multiple primal heuristics."""

    def __init__(
        self,
        classes: typing.List[typing.Any],
        data: typing.Mapping[str, typing.Any],
        config: typing.Mapping[str, typing.Any],
        journal: typing.Optional[typing.Any],
    ):
        """Initialise a CombinedPrimalHeuristic instance."""
        super().__init__(
            data=data,
            config=config,
            journal=journal,
        )
        self.primal_heuristics = [
            cls(
                data=data,
                config=config,
                journal=journal,
            )
            for cls in classes
        ]
        self.lb = -np.inf
        self.tol = config["tol"]

    @property
    def n_processes(self):
        for p in self.primal_heuristics:
            try:
                return p.n_processes
            except AttributeError:
                pass
        return 1

    @n_processes.setter
    def n_processes(self, n):
        for p in self.primal_heuristics:
            try:
                p.n_processes = n
            except AttributeError:
                pass

    @property
    def iteration_index(self):
        return self.primal_heuristics[0].iteration_index

    @iteration_index.setter
    def iteration_index(self, n):
        for p in self.primal_heuristics:
            p.iteration_index = n

    def set_cg(self, cg):
        """Set a weak reference to CG.

        Parameters
        ----------
        cg : object
        """
        for p in self.primal_heuristics:
            try:
                p.set_cg(cg)
            except AttributeError:
                pass

    def start_hook(self):
        """Notify the start of the solver"""
        for p in self.primal_heuristics:
            try:
                p.start_hook()
            except AttributeError:
                pass

    def end_hook(self):
        """Notify the end of the solver"""
        for p in self.primal_heuristics:
            try:
                p.end_hook()
            except AttributeError:
                pass

    def initialiser_hook(self, y):
        for p in self.primal_heuristics:
            try:
                p.initialiser_hook(y)
            except AttributeError:
                pass

    def iteration_start_hook(self, iteration_index):
        for p in self.primal_heuristics:
            try:
                p.iteration_start_hook(iteration_index)
            except AttributeError:
                pass

    def lb_hook(self, y, lb):
        for p in self.primal_heuristics:
            try:
                p.lb_hook(y, lb)
            except AttributeError:
                pass

    def run(self, timelimit):
        """Run primal heuristics.

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

        def gap(res):
            if (not res.status) or (not np.isfinite(self.lb)):
                return np.inf
            else:
                return (res.obj - self.lb) / res.obj

        # Use timelimit for all primal heuristics.
        stop_at = time.perf_counter() + timelimit
        best_obj = np.inf
        best_res = PHResult(
            status=False,
            msg="",
            obj=np.inf,
            sol=np.nan,
        )
        for p in self.primal_heuristics:
            remaining = stop_at - time.perf_counter()
            res = p.run(timelimit=remaining)
            if gap(res) <= self.tol:
                return res
            elif res.status and (res.obj < best_obj):
                best_res = res
                best_obj = res.obj
        return best_res
