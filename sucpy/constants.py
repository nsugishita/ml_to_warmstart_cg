# -*- coding: utf-8 -*-

"""Definition of constants"""

import collections
import enum

import numpy as np

try:
    import cplex

    has_cplex = True

except ImportError:
    has_cplex = False

from sucpy.utils import utils

if has_cplex:
    m = cplex.Cplex()

    _method_name_to_cplex_lp_method = {
        "auto": m.parameters.lpmethod.values.auto,
        "primal_simplex": m.parameters.lpmethod.values.primal,
        "dual_simplex": m.parameters.lpmethod.values.dual,
        "barrier": m.parameters.lpmethod.values.barrier,
    }

    def method_name_to_cplex_lp_method(v):
        """Convert a method name to a parameter value in CPLEX"""
        return _method_name_to_cplex_lp_method[v]

    _method_name_to_cplex_qp_method = {
        "auto": m.parameters.qpmethod.values.auto,
        "primal_simplex": m.parameters.qpmethod.values.primal,
        "dual_simplex": m.parameters.qpmethod.values.dual,
        "barrier": m.parameters.qpmethod.values.barrier,
    }

    def method_name_to_cplex_qp_method(v):
        """Convert a method name to a parameter value in CPLEX"""
        return _method_name_to_cplex_qp_method[v]

    del m

else:

    def method_name_to_cplex_lp_method(v):
        raise ImportError("cplex")

    def method_name_to_cplex_qp_method(v):
        raise ImportError("cplex")


_method_name_to_gurobi_method = {
    "auto": -1,
    "primal_simplex": 0,
    "dual_simplex": 1,
    "barrier": 2,
}


def method_name_to_gurobi_method(v):
    """Convert a method name to a parameter value in gurobi"""
    return _method_name_to_gurobi_method[v]


class PHType(enum.IntEnum):
    """Values allowed in 'cg.ph_type'"""

    NOOP = 0
    COLUMN_EVALUATION = 1
    COLUMN_COMBINATION = 3
    COLUMN_EVALUATION_AND_COLUMN_COMBINATION = 6
    CPLEX = 14


class DualInitialiserType(enum.IntEnum):
    """Initialiser for dual variable"""

    UNKNOWN = 0
    SINGLE_SAMPLING_NETWORK = 1
    DOUBLE_SAMPLING_NETWORK = 2
    RANDOM_FOREST = 3
    NEAREST_NEIGHBOUR = 4
    COLUMN_PRE_POPULATION = 5
    LPR = 6
    COLDSTART = 7
    CONSTANT = 8
    CPLEX = 9


def as_dual_initialiser_typecode(dual_initialiser):
    given = dual_initialiser.__class__.__name__.lower()
    for key in DualInitialiserType.__members__.keys():
        normalized_key = key.replace("_", "").lower()
        if normalized_key in given:
            return DualInitialiserType[key]
    raise ValueError(dual_initialiser.__class__.__name__)


class StepsizeRule(enum.IntEnum):
    """Stepsize update rules"""

    ADAPTIVE = 0
    CONSTANT = 1
    DIMINISHING = 2


class RoutineType(enum.IntEnum):
    """Routine types in CG"""

    DUAL_INITIALIZATION = 0
    RMP = 1
    SUBPROBLEM = 2
    PRIMAL_HEURISTIC = 3
    EXTENSIVE_MODEL = 10


UBWorkerRecordSOL = collections.namedtuple(
    "UBWorkerRecordSOL",
    [
        "ub",
        "sol",
        "walltime",
        "last_parameter",
        "parameter",
        "parameter_n_fixed",
        "parameter_begin",
        "parameter_end",
        "parameter_status",
        "parameter_ub",
    ],
)


class Algorithm(enum.IntEnum):
    """Algorithm to solve problems"""

    CG = 0
    EXTENSIVE_MODEL = 2
    LBUBWORKER = 3


config_rule = {
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Problem setup.
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    "generator_data_type": {
        "type": "str",
        "choices": ["frangioni", "tim", "miguel"],
        "default": "frangioni",
        "help": ("Source of generator data."),
    },
    "generator_list": {
        "type": "list[str]",
        "help": (
            "file names of generator list used to create "
            "an UC instance. Only used when "
            "`generator_data_type` is `frangioni`."
        ),
    },
    "n_periods": {
        "type": "int",
        "constraint": "positive",
        "help": "Number of time periods in the planning horizon.",
    },
    "n_generators": {
        "type": "int",
        "constraint": "positive",
        "help": "Number of generators.",
    },
    "uc.spinning_reserve_rate": {
        "type": "float",
        "default": 0.1,
        "constraint": "[0, 1)",
        "help": (
            "Ratio of the amount of required spinning reserve "
            "to the demand."
        ),
    },
    "uc.demand_ratio_to_capacity": {
        "type": "float",
        "default": 0.5,
        "constraint": "<= 1",
        "help": (
            "Demand data is scaled by this factor so that "
            "typical daily peak matched to "
            "demand_ratio_to_capacity * total_generator_capacity. "
            "If a nonpositive value is given, demand is not "
            "scaled. "
        ),
    },
    "uc.demand_noise_process.scale": {
        "type": "float",
        "default": 0.05,
        "constraint": "positive",
        "help": ("Size of noise on the demand."),
    },
    "uc.demand_noise_process.length_scale": {
        "type": "float",
        "default": 0.5,
        "constraint": "positive",
        "help": ("Length scale of the noise process of the demand."),
    },
    "uc.demand_noise_process.alpha": {
        "type": "float",
        "default": 1e-10,
        "constraint": "positive",
        "help": (
            "Value added to the diagonal of the kernel "
            "matrix of the noise processof the demand. "
            "This prevents a potential numerical issue, "
            "by ensuring that the calculated values form "
            "a positive definite matrix."
        ),
    },
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Solver config.
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    "tol": {
        "type": "number",
        "constraint": "[0, 1)",
        "help": ("Suboptimality tolerance used in solvers."),
    },
    "target_objective": {
        "default": np.nan,
        "help": (
            "The optimal objective value if known in advance. "
            "This is used when `tol_to_target_objective` is set. "
            "The relative error will be printed in the log messages. "
        ),
    },
    "tol_to_target_objective": {
        "type": "number",
        "constraint": "[-1, 1)",
        "default": -1,
        "argparse": True,
        "help": (
            "The solver is terminated when the solver finds an objective "
            "value close the the target objective value."
        ),
    },
    "cg.lpr_method": {
        "type": "str",
        "choices": ["auto", "primal_simplex", "dual_simplex", "barrier"],
        "default": "barrier",
        "help": ("Cplex lp method to solve LPR as a dual initialiser."),
    },
    "cg.lpr_cross_over": {
        "type": "int",
        "choices": [0, 1],
        "default": 1,
        "argparse": True,
        "help": "If 0, do not execute cross over",
    },
    "cg.lpr_tol": {
        "type": "number",
        "default": 0,
        "help": ("Tolerance of LPR"),
    },
    "cg.lpr.regularization": {
        "type": "number",
        "default": 0.0,
        "constraint": "nonnegative",
        "help": (
            "If a positive value is given, regularization is added "
            "on the dual variables."
        ),
    },
    "nearest_neighbour_initialiser.n_neighbours": {
        "type": "int",
        "default": 3,
        "argparse": True,
        "help": "Number of nearest neighbous to use",
    },
    "column_pre_population_initialiser.n_neighbours": {
        "type": "int",
        "default": 70,
        "argparse": True,
        "help": "Number of nearest neighbous to use",
    },
    "column_pre_population_initialiser.only_active": {
        "type": "int",
        "default": 0,
        "choice": [0, 1],
        "argparse": True,
        "help": "Only return cuts which was active in the last iteration.",
    },
    "column_pre_population_initialiser.populate_rmp": {
        "type": "int",
        "choices": [0, 1],
        "default": 0,
        "argparse": True,
        "help": (
            "Populate columns in the RMP used in the column "
            "generation procedure. "
        ),
    },
    "dual_optimizer.initial_stepsize": {
        "type": "number",
        "default": 1e-3,
        "constraint": "nonnegative",
        "argparse": True,
        "help": "Initial value of the step size used by CG.",
    },
    "dual_optimizer.stepsize_rule": {
        "choices": list(StepsizeRule.__members__.values()),
        "default": StepsizeRule.ADAPTIVE,
        "help": "Stepsize update rule.",
    },
    "dual_optimizer.stepsize_diminishing_offset": {
        "type": "number",
        "default": 1,
        "constraint": "positive",
        "argparse": True,
        "help": (
            "A parameter to compute the diminishing stepsize "
            "when dual_optimizer.stepsize_rule is DIMINISHING. "
            "The stepsize is computed by "
            "`initial_stepsize * offset / (cycle_number + offset)` where "
            "`cycle_number = floor(iteration_index * evaluation_ratio)`. "
            "The stepsize is kept constant within a cycle. "
        ),
    },
    "dual_optimizer.stepsize_update_coefficient": {
        "type": "number",
        "default": 2,
        "constraint": ">= 1",
        "argparse": True,
        "help": (
            "Factor to adjust regularization coefficient. "
            "The timing of update depends on the parameter "
            "dual_optimizer.adaptive_proximal_term_update."
        ),
    },
    "dual_optimizer.feed_subproblem_solution": {
        "type": "number",
        "default": 1,
        "choices": [0, 1],
        "argparse": True,
        "help": (
            "If 1, CG feed subproblem solutions to "
            "the rmp.  Otherwise, only function and subgradient "
            "values are passed"
        ),
    },
    "dual_optimizer.cplex_method": {
        "type": "str",
        "choices": ["auto", "primal_simplex", "dual_simplex", "barrier"],
        "default": "barrier",
        "help": "CPLEX QP method to solve the cutting-plane model.",
    },
    "dual_optimizer.check_duplicate": {
        "type": "int",
        "default": 1,
        "choices": [0, 1],
        "argparse": True,
        "help": (
            "If 1, new cuts are compared against the existing ones "
            "and only new ones are added to the model. "
        ),
    },
    "dual_optimizer.memory_size": {
        "type": "int",
        "default": -1,
        "constraint": ">= -1",
        "argparse": True,
        "help": (
            "Number of iteraion to keep the columns "
            "in RMP.  Non positive value indicates "
            "columns are not dropped. "
        ),
    },
    "dual_optimizer.adaptive_proximal_term_update": {
        "type": "int",
        "default": 1,
        "choices": [0, 1],
        "argparse": True,
        "help": (
            "If 0, the proximal term is update "
            "to the latest evaluation point. "
            "If 1, the proximal term is updated adaptively "
            "based on the progress of LB. "
        ),
    },
    "dual_optimizer.minimum_stepsize": {
        "type": "number",
        "default": 0,
        "argparse": True,
        "help": (
            "Minimal step size. "
            "Value below this parameter will be clipped. "
        ),
    },
    "dual_optimizer.maximum_stepsize": {
        "type": "number",
        "default": 1e0,
        "help": (
            "Minimal step size. "
            "Value above this parameter will be clipped. "
        ),
    },
    "dual_optimizer.objective_scaling_type": {
        "type": "str",
        "choices": ["pessimistic", "optimistic"],
        "default": "pessimistic",
        "help": (
            "Method for objective scaling when there are "
            "components without cuts."
        ),
    },
    "cg.subprob_rtol": {
        "type": "float",
        "default": 0.0,
        "constraint": "[0, 1)",
        "help": (
            "Subproblem suboptimality relative tolerance. "
            "The tolerance will be computed as "
            "tol * cg.subprob_rtol + cg.subprob_atol."
        ),
    },
    "cg.subprob_atol": {
        "type": "float",
        "default": 1e-4,
        "constraint": "[0, 1)",
        "help": (
            "Subproblem suboptimality absolute tolerance. "
            "The tolerance will be computed as "
            "tol * cg.subprob_rtol + cg.subprob_atol."
        ),
    },
    "cg.no_progress_action": {
        "type": "str",
        "default": "none",
        "choices": ["terminate", "solution_polishing", "none"],
        "argparse": True,
        "help": "Behaviour when no progresses are made.",
    },
    "cg.no_progress_patience": {
        "type": "int",
        "default": 4,
        "argparse": True,
    },
    "cg.ph_type": {
        "type": "int;str",
        "choices": list(PHType.__members__.values()),
        "default": PHType.COLUMN_EVALUATION_AND_COLUMN_COMBINATION,
        "help": "Type of primal heuristic used in CG.",
    },
    "cg.ph_run_every": {
        "type": "int",
        "default": 1,
        "constraint": "positive",
        "help": "Frequency to run primal heuristic in CG.",
    },
    "cg.ph_run_on_all_iters": {
        "type": "bool",
        "default": False,
        "help": (
            "If True, run primal heuristics "
            "on every iteration. Notably,  "
            "this forces ph to run after "
            "subproblems closes the gap. "
            "If this is True, cg.ph_run_every "
            "is ignored. "
        ),
    },
    "ph.time_ratio": {
        "type": "number",
        "default": 2.0,
        "argparse": True,
        "help": (
            "Parameter to compute the timelimit of primal heuristics "
            "relative to other CG routines.  The actual ratio is computed "
            "by `time_ratio + time_ratio_increment * (iteration // 10)`."
            "Setting the resulting value to zero puts no timelimit on "
            "primal heuristics. "
        ),
    },
    "ph.time_ratio_increment": {
        "type": "number",
        "default": 1.0,
        "argparse": True,
        "help": (
            "Parameter to compute the timelimit of primal heuristics. "
            "See the help for time_ratio."
        ),
    },
    "ph.time_ratio_to": {
        "choices": ["subproblem", "all"],
        "default": "subproblem",
        "argparse": True,
        "help": "To which routines ph time limit is compared.",
    },
    "ph.column_combination.run_after": {
        "type": "int",
        "default": -1,
        "argparse": True,
        "help": (
            "If positive, skip the first `run_after` iterations. "
            "If -1 (default), `run_after` is set to 0 in "
            "COLUMN_COMBINATION and 10 in "
            "COLUMN_EVALUATION_AND_COLUMN_COMBINATION."
        ),
    },
    "ph.column_combination.history": {
        "type": "int",
        "default": 3,
        "argparse": True,
        "help": ("Number of iterations columns are kept. "),
    },
    "ph.column_combination.reuse_active_columns": {
        "type": "int",
        "default": 0,
        "choices": [0, 1],
        "argparse": True,
        "help": (
            "Reuse columns used in the solution.  That is, "
            "this updates the iteration of columns used in the solution "
            "to the current iteration index so that they will be "
            "available in the next iterations. "
        ),
    },
    "ph.column_combination.freed_ratio": {
        "type": "number",
        "default": 0.00,
        "constraint": "[0, 1)",
        "argparse": True,
        "help": ("Ratio of subproblems to be freed."),
    },
    "ph.column_combination.freed_ratio_scheduler_interval": {
        "type": "number",
        "default": 10,
        "argparse": True,
        "help": (
            "After this many iterations, freed_ratio is "
            "increased by 0.01.  If a nonpositive value "
            "is given, freed ratio is not increased."
        ),
    },
    "solver.timelimit": {
        "type": "number",
        "default": -1,
        "argparse": True,
        "help": (
            "Timelimit to run a solver.  "
            "Nonpositive value set no timelimit. "
            "This is used in CG "
            "and the extensive model."
        ),
    },
    "solver.n_processes": {
        "type": "int",
        "default": 1,
        "constraint": "positive",
        "help": (
            "Number of processes used in the solvers, "
            "CG and extensive model. "
        ),
    },
    "cg.iteration_limit": {
        "type": "number",
        "default": 1000,
        "argparse": True,
        "help": ("Maximum number of iterations in CG."),
    },
    "cg.log_type": {
        "type": "str",
        "choices": ["gap", "lb"],
        "default": "gap",
        "argparse": True,
        "help": "Parameter to control values printed in the log.",
    },
    "cg.log_iterates": {
        "type": "int",
        "choices": [0, 1],
        "default": 0,
        "argparse": True,
        "help": "Log details of iterations",
    },
    "cg.save_rmp_size": {
        "type": "number",
        "default": 0,
        "choices": [0, 1],
        "argparse": True,
        "help": "Save RMP size.",
    },
    "cg.save_subproble_solutions": {
        "type": "number",
        "default": 0,
        "choices": [0, 1],
        "argparse": True,
        "help": "Save subproblem solutions in each iteration.",
    },
    "cg.save_model_values": {
        "type": "number",
        "default": 0,
        "choices": [0, 1],
        "argparse": True,
        "help": "Save RMP model values and actual function values.",
    },
    "cg.save_cuts": {
        "type": "number",
        "default": 1,
        "choices": [0, 1],
        "argparse": True,
        "help": "Save cuts in the RMP.",
    },
    # Parameters to configure validdict.
    "validdictconfig": {"populate_default": True},
}


def default_config() -> utils.validdict:
    """Return a config dict with default values"""
    import logging

    logger = logging.getLogger(__name__)
    logger.warning(
        "`default_config` is deprecated, use `get_default_config` instead"
    )
    return get_default_config()


def get_default_config() -> utils.validdict:
    """Return a config dict with default values

    Examples
    ---------
    >>> import sucpy
    >>> config = sucpy.constants.get_default_config()
    >>> config["solver.timelimit"] = 10
    >>> config["foo.baz.bar"] = 10
    Traceback (most recent call last):
    ...
    KeyError: "assigning a new item 'foo.baz.bar' on a frozen dict"
    """
    return utils.validdict(config_rule)


_parser, _mapping = utils.setup_argparser(config_rule)


def argparser():
    return _parser


def update_config(config, args):
    for dest in _mapping:
        if getattr(args, dest) is not utils.argparse_missing:
            config[_mapping[dest]] = getattr(args, dest)
