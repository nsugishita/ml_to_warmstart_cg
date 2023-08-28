# -*- coding: utf-8 -*-

"""Base components for solvers."""

from typing import Mapping

import cplex
import numpy as np

from sucpy import constants
from sucpy.dual_initialisers.base import DualInitialiser
from sucpy.solvers import common
from sucpy.utils import cplex_utils


class ExtensiveModel(cplex_utils.Cplex):
    """Extensive formulation of a MIP instance.

    This is a CPLEX model which holds an extensive formulation of
    an MIP instance.  Passing problem data, this constructs the MIP model.
    """

    def __init__(self, data: Mapping, config: Mapping) -> None:
        """Construct a CPLEX model of the extensive formulation.

        Parameters
        ----------
        data : dict
            Data returned from ParametrizedUC.sample_instance.
        config : dict
            Configuration of the experiment.
        """
        super().__init__()
        self.parameters.threads.set(config.get("solver.n_processes", 1))
        timelimit = config.get("solver.timelimit", -1)
        if (timelimit > 0) and np.isfinite(timelimit):
            self.parameters.timelimit.set(timelimit)
        self.set_error_stream(None)
        self.set_log_stream(None)
        self.set_results_stream(None)
        self.set_warning_stream(None)
        variable_lb = data["variable_lb"]
        variable_ub = data["variable_ub"]
        variable_type = data["variable_type"]
        if isinstance(variable_type, str):
            variable_type = np.array(list(variable_type))
        objective_coefficient = data["objective_coefficient"]
        C = b"C"[0]
        if np.all(variable_type == C):
            self.variables.add(
                obj=objective_coefficient, lb=variable_lb, ub=variable_ub
            )
        else:
            type_string = common.to_str(variable_type)
            self.variables.add(
                obj=objective_coefficient,
                lb=variable_lb,
                ub=variable_ub,
                types=type_string,
            )
        A = data["constraint_coefficient"].tocoo()
        b = data["constraint_rhs"]
        con_sense = data["constraint_sense"]
        con_sense = common.to_str(con_sense)
        self.linear_constraints.add(senses=con_sense, rhs=b)
        self.linear_constraints.set_coefficients(
            zip(map(int, A.row), map(int, A.col), map(float, A.data))
        )
        self.objective.set_offset(data.get("objective_offset", 0))


class LPR(DualInitialiser, cplex.Cplex):
    """Regularized Linear programming relaxation of an extensive formulation.

    This is a CPLEX model which holds regularized LPR of an MIP instance.
    Passing problem data, this constructs the LP model.
    """

    def __init__(self, data: Mapping, config: Mapping) -> None:
        """Initialise a RegularizedLPR instance.

        Parameters
        ----------
        data : dict
            Data returned from ParametrizedUC.sample_instance.
        config : dict
            Configuration of the experiment.
        """
        super().__init__()
        self.parameters.threads.set(1)
        if "cg.lpr_method" in config:
            method_name = config["cg.lpr_method"]
            lpmethod = constants.method_name_to_cplex_lp_method(method_name)
            qpmethod = constants.method_name_to_cplex_qp_method(method_name)
            self.parameters.lpmethod.set(lpmethod)
            self.parameters.qpmethod.set(qpmethod)
        if not config.get("cg.lpr_cross_over", True):
            # Do not run cross over.
            self.parameters.solutiontype.set(2)
        if config.get("cg.lpr_tol", 0.0) > 0.0:
            tol = config.get("cg.lpr_tol", 0.0)
            self.parameters.barrier.convergetol.set(tol)
        self.set_error_stream(None)
        self.set_log_stream(None)
        self.set_results_stream(None)
        self.set_warning_stream(None)
        variable_lb = np.array(data["variable_lb"])
        variable_ub = np.array(data["variable_ub"])
        variable_type = data["variable_type"]
        B = b"B"[0]
        variable_lb[variable_type == B] = np.maximum(
            variable_lb[variable_type == B], 0
        )
        variable_ub[variable_type == B] = np.minimum(
            variable_ub[variable_type == B], 1
        )
        objective_coefficient = data["objective_coefficient"]
        self.variables.add(
            obj=objective_coefficient, lb=variable_lb, ub=variable_ub
        )
        A = data["constraint_coefficient"].tocoo()
        b = data["constraint_rhs"]
        con_sense = data["constraint_sense"]
        self.linear_constraints.add(senses=con_sense, rhs=b)
        self.linear_constraints.set_coefficients(
            zip(map(int, A.row), map(int, A.col), map(float, A.data))
        )
        self.linking_constraint_index = list(
            map(int, np.nonzero(data["constraint_subproblem_index"] == -1)[0])
        )
        # Add dual variables and its regularization.
        coefficient = config.get("cg.lpr.regularization", 0.0)
        if coefficient > 0:
            n_linking_constraints = len(self.linking_constraint_index)
            self.cplex_dual_index = self.variables.add(
                lb=data["dual_variable_lb"], ub=data["dual_variable_ub"]
            )
            self.objective.set_quadratic_coefficients(
                [(i, i, coefficient) for i in self.cplex_dual_index]
            )
            # TODO Check do we need sign here?
            signs = np.full(n_linking_constraints, -1.0)
            G = b"G"[0]
            signs[data["subproblem_constraint_sense"] == G] = 1.0
            self.linear_constraints.set_coefficients(
                list(
                    zip(
                        self.linking_constraint_index,
                        self.cplex_dual_index,
                        map(float, signs * coefficient),
                    )
                )
            )
        self.objective.set_offset(data.get("objective_offset", 0))

    def load(self, path):
        """Return withou any effect.

        This exists to be compatible with DualInitialiser.
        """
        pass

    @property
    def dual_on_linking_constraint(self):
        """Extract dual on the linking constraints.

        Returns
        -------
        dual : (num_linking_constraints,) array of float
        """
        return np.array(
            self.solution.get_dual_values(self.linking_constraint_index)
        )

    def compute_initial_dual(self, parameter):
        """Output dual for CG.

        This is used within CG.

        Parameters
        ----------
        parameter : (dim_parameter,) array

        Returns
        -------
        dual : (dim_dual,) array
        """
        self.solve()
        return self.dual_on_linking_constraint
