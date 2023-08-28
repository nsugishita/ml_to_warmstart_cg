# -*- coding: utf-8 -*-

"""Common objects among primal heuristics."""

import abc
import collections
import typing

PHResult = collections.namedtuple("PHResult", "status,msg,obj,sol")


class PrimalHeuristicBase(abc.ABC):
    """Base class for primal heuristics"""

    __slots__ = ()

    def __init__(
        self,
        data: typing.Mapping[str, typing.Any],
        config: typing.Mapping[str, typing.Any],
        journal: typing.Optional[typing.Any],
    ) -> None:
        pass

    @abc.abstractmethod
    def set_cg(self, cg) -> None:
        """Set a weak reference to CG.

        Parameters
        ----------
        cg : object
        """
        pass

    def start_hook(self):
        """Notify the start of the solver"""
        pass

    def initialiser_hook(self, y):
        """Notify the completion of the initialiser"""
        pass

    def iteration_start_hook(self, iteration_index):
        """Notify the start of a new iteration"""
        pass

    def end_hook(self):
        """Notify the end of the solver"""
        pass

    @abc.abstractmethod
    def lb_hook(self, y, lb) -> None:
        """Set/update a lower bound."""
        pass

    @abc.abstractmethod
    def run(self, timelimit=None) -> PHResult:
        pass
