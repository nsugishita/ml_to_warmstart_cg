# -*- coding: utf-8 -*-

"""Primal heuristics."""

import copy
import typing

from sucpy import constants

from .base import PrimalHeuristicBase
from .column_combination import ColumnCombinationHeuristic
from .column_evaluation import ColumnEvaluationPrimalHeuristic
from .combination import CombinedPrimalHeuristic
from .noop_primal_heuristic import NoOpPrimalHeuristic


def get(
    ph_type,
    data: typing.Mapping[str, typing.Any],
    config: typing.Mapping[str, typing.Any],
    journal: typing.Optional[typing.Any],
) -> PrimalHeuristicBase:
    primal_heuristic: PrimalHeuristicBase
    kwargs: typing.Mapping[str, typing.Any] = dict(
        data=data,
        config=config,
        journal=journal,
    )
    if ph_type == constants.PHType.NOOP:
        primal_heuristic = NoOpPrimalHeuristic(**kwargs)
    elif ph_type == constants.PHType.COLUMN_EVALUATION:
        primal_heuristic = ColumnEvaluationPrimalHeuristic(**kwargs)
    elif ph_type == constants.PHType.COLUMN_COMBINATION:
        primal_heuristic = ColumnCombinationHeuristic(**kwargs)
    elif ph_type == constants.PHType.COLUMN_EVALUATION_AND_COLUMN_COMBINATION:
        if config["ph.column_combination.run_after"] == -1:
            config = typing.cast(dict, copy.deepcopy(config))
            config["ph.column_combination.run_after"] = 30
        primal_heuristic = CombinedPrimalHeuristic(
            [ColumnEvaluationPrimalHeuristic, ColumnCombinationHeuristic],
            data=data,
            config=config,
            journal=journal,
        )
    else:
        raise ValueError(f"unknown primal heuristic: {ph_type}")
    return primal_heuristic
