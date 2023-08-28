# -*- coding: utf-8 -*-

"""Local search heuristic with economic dispatch used in CG."""

import typing

from .base import PHResult, PrimalHeuristicBase


class NoOpPrimalHeuristic(PrimalHeuristicBase):
    """Primal heuristic which does not do anything."""

    def __init__(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        """Initialise a NoOpPrimalHeuristic instance."""
        super().__init__(*args, **kwargs)

    def set_cg(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        pass

    def lb_hook(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        pass

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
        return PHResult(status=False, msg="", obj=float("inf"), sol=None)
