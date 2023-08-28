# -*- coding: utf-8 -*-

"""Local search heuristic with economic dispatch used in CG."""

import doctest
import pickle
import typing
import weakref

import numpy as np

from sucpy.primal_heuristics.base import PHResult, PrimalHeuristicBase
from sucpy.solvers.base import ExtensiveModel
from sucpy.utils import cplex_utils, utils


@utils.freeze_attributes
class ColumnCombinationHeuristic(PrimalHeuristicBase):
    """Primal heuristics of column generation.

    This is a CPLEX model which is used as a primal heuristic
    in column generation.  To be more specific,
    this holds a MIP instance with restricted columns.
    """

    def __init__(
        self,
        data: typing.Mapping[str, typing.Any],
        config: typing.Mapping[str, typing.Any],
        journal: typing.Optional[typing.Any],
    ) -> None:
        """Initialise a ColumnCombinationHeuristic instance.

        Parameters
        ----------
        data : dict
            Data returned from ParametrizedUC.sample_instance.
        config : dict
        journal : utils.Journal
        """
        import uniquelist

        super().__init__(
            data=data,
            config=config,
            journal=journal,
        )
        self.instance_id = data.get("instance_id")
        if self.instance_id is None:
            self.instance_id = 0
        self.config = config
        self.journal = journal
        self.cg = None
        self.n_processes = 1
        # We first construct an ExtensiveModel
        # and pickle it.  In each iteration,
        # we unpickle the model and modify it
        # (e.g. add column choice constraints).
        # This is because unpickling an
        # ExtensiveModel is much quicker than
        # constructing an instance from scratch.
        # See "implementation of ColumnCombinationHeuristic"
        # in sucpy/README.md.
        model = ExtensiveModel(data, config)
        self.model_state = pickle.dumps(model)
        self.iteration_index = None
        self.S = np.arange(data["n_subproblems"])  # Subproblem index.

        self.original_n_vars = data["n_variables"]
        self.x_subprob_index = data["variable_subproblem_index"]
        # index of the variables in the extensive form
        # which are binary.
        integer_variables = np.nonzero(data["variable_type"] == b"B"[0])[0]
        # indices of corresponding subproblems for integer_variables
        subproblem_index = data["variable_subproblem_index"][integer_variables]
        # integer variables grouped by the subproblems
        self.integer_variables_by_sub = [
            integer_variables[subproblem_index == s] for s in self.S
        ]
        self.subproblems_without_integer_variables = np.array(
            [s for s in self.S if self.integer_variables_by_sub[s].size == 0],
            dtype=int,
        )

        # index of integer variables in each subproblem.
        # This is used to extract integer variable values
        # from subproblem solutions.
        # Given a solution x of subproblem `s`, we could do
        # x[self.integer_variables_in_sub[s]] to extract
        # the integer variable values.
        self.integer_variables_in_sub = [
            np.nonzero(data["subproblem_variable_type"][s] == b"B"[0])[0]
            for s in self.S
        ]
        self.column_pool = [
            uniquelist.UniqueArrayList(self.integer_variables_by_sub[s].size)
            for s in self.S
        ]
        self.column_iteration = [np.array([]) for _ in self.S]
        self.column_value = [np.array([]) for _ in self.S]
        # This will be set in set_lower_bound.
        self.lower_bound_callback = None
        # This will be set after self.solve.
        self.last_solution_column = None
        self.last_solution_value = None
        self.cg_miptol = config["tol"]
        self.lb = -np.inf

    def _add_column_choice_constraints(self, model, exclude=[]):
        """Add column choice constraints on a given CPLEX model.

        Parameters
        ----------
        model : cplex.Cplex
        exclude : list of int, default []
            Subproblem indices whose columns are not to be constrained.

        Returns
        -------
        column_choice : list of 1d array
            `column_choice[i]` is an array of indices
            of column choice constraints in `model`.
        """
        _integer_variables_by_sub = list(self.integer_variables_by_sub)
        for s in exclude:
            _integer_variables_by_sub[s] = []
        col = (
            np.concatenate([_integer_variables_by_sub[i] for i in self.S])
            .astype(int)
            .tolist()
        )
        val = [-1.0] * len(col)
        row = model.linear_constraints.add(rhs=[0] * len(col))
        model.linear_constraints.set_coefficients(zip(row, col, val))
        n_integer_variables_sub = [len(s) for s in _integer_variables_by_sub]
        stop = np.cumsum(n_integer_variables_sub)
        start = np.r_[0, stop[:-1]]
        row = [row[a:b] for a, b in zip(start, stop)]
        # (n_subprobs)-list of list which contains indices
        # of column choice constraints.
        column_choice = row

        return column_choice

    def set_cg(self, cg):
        """Set a weak reference to CG.

        Parameters
        ----------
        cg : object
        """
        self.cg = weakref.ref(cg)

    def lb_hook(self, y, lb):
        """Set/update a lower bound."""
        if (lb is None) or (not np.isfinite(lb)):
            lb = -np.inf
        self.lb = max(self.lb, lb)

    def _add_column_to_column_pool(
        self, subproblem_index, column, iteration_index=None
    ):
        if iteration_index is None:
            iteration_index = self.iteration_index
        pos, new = self.column_pool[subproblem_index].push_back(column)
        if not new:
            self.column_iteration[subproblem_index][pos] = iteration_index
        else:
            self.column_iteration[subproblem_index] = np.r_[
                self.column_iteration[subproblem_index], iteration_index
            ]
            if self.column_value[subproblem_index].size > 0:
                self.column_value[subproblem_index] = np.concatenate(
                    [self.column_value[subproblem_index], column[None]], axis=0
                )
            else:
                self.column_value[subproblem_index] = column[None]

    def _remove_old_column_from_column_pool(self, v=None):
        if v is None:
            v = self.config["ph.column_combination.history"]
        for s in self.S:
            removed_flag = (
                np.asarray(self.column_iteration[s])
                <= self.iteration_index - v
            )
            self.column_pool[s].erase_nonzero(removed_flag)
            self.column_iteration[s] = self.column_iteration[s][~removed_flag]
            self.column_value[s] = self.column_value[s][~removed_flag]

    def run(self, timelimit=0):
        if not np.isfinite(self.lb):
            # Do not run while the method is incremental.
            return PHResult(
                status=False,
                msg="skipping an incremental step",
                obj=np.nan,
                sol=None,
            )

        oracle = self.cg().subproblems
        iter = zip(
            oracle.evaluation_component_index, oracle.subproblem_solution
        )
        for subproblem_index, column in iter:
            column = np.round(
                column[self.integer_variables_in_sub[subproblem_index]]
            )
            self._add_column_to_column_pool(subproblem_index, column)

        self._remove_old_column_from_column_pool()

        run_after = self.config["ph.column_combination.run_after"]
        if (run_after > 0) and (self.iteration_index < run_after):
            return PHResult(
                status=False,
                msg=("skipping first {run_after} iterations"),
                obj=np.inf,
                sol=None,
            )

        random_seed = max(
            max(self.instance_id, 0) * 100 + self.iteration_index, 0
        )
        rng = np.random.RandomState(random_seed)

        # Create an ExtensiveModel by unpickling.
        model = pickle.loads(self.model_state)
        model.parameters.threads.set(self.n_processes)

        cplex_utils.set_stream(model, None)
        model.parameters.mip.tolerances.mipgap.set(0.5 * self.cg_miptol)

        def predicate(info):
            if info.incumbent_objective_value_updated:
                n = self.original_n_vars
                self.journal.set_ub(
                    ub=info.incumbent_objective_value,
                    data=np.array(info.cb.get_incumbent_values())[:n],
                )
            return self.journal.stop_event.is_set()

        cplex_utils.abort_if(model, predicate)

        # Add column choice constraints on model.
        freed_ratio = self.config["ph.column_combination.freed_ratio"]
        freed_ratio_scheduler_interval = self.config[
            "ph.column_combination.freed_ratio_scheduler_interval"
        ]
        if freed_ratio_scheduler_interval > 0:
            freed_ratio += 0.01 * (
                self.iteration_index // freed_ratio_scheduler_interval
            )
        if freed_ratio <= 0.0:
            exclude = np.array([], dtype=int)
        else:
            exclude = rng.choice(
                self.S, int(len(self.S) * freed_ratio), replace=False
            )
        exclude = np.r_[exclude, self.subproblems_without_integer_variables]
        constrained_subprobs = np.setdiff1d(self.S, exclude)
        column_choice = self._add_column_choice_constraints(
            model, exclude=exclude
        )

        # Add w.
        for s in constrained_subprobs:
            pool = np.array(self.column_value[s])
            size = len(pool)
            w = model.variables.add(
                lb=[0] * size, ub=[1] * size, types="B" * size
            )
            row = list(map(int, np.tile(column_choice[s], size)))
            col = list(map(int, np.repeat(w, len(pool[0]))))
            val = list(map(float, pool.ravel()))
            model.linear_constraints.set_coefficients(zip(row, col, val))
            # if self.config["ph.column_combination.strict_sos"]:
            #     lin_expr = [cplex.SparsePair(ind=w, val=[1.0] * len(w))]
            #     model.linear_constraints.add(
            #         lin_expr=lin_expr, rhs=[1.0], senses="E"
            #     )
            index = w
            weight = np.arange(len(w)).tolist()
            model.SOS.add(type=model.SOS.type.SOS1, SOS=[index, weight])

        if self.last_solution_column is not None:
            # Try to warmstart the model.
            model.MIP_starts.delete()
            model.MIP_starts.add(
                [self.last_solution_column, self.last_solution_value], 0
            )

        if (timelimit > 0) and np.isfinite(timelimit):
            model.parameters.timelimit.set(timelimit)

        # Run the primal heuristics, namely solve the restricted SUC.
        model.solve()

        if cplex_utils.is_mip_feasible(model):
            obj = model.solution.get_objective_value()
            solution = np.array(model.solution.get_values())[
                : self.original_n_vars
            ]
            # Save solution to warmstart next time.
            self.last_solution_value = solution
            self.last_solution_column = np.arange(
                self.original_n_vars
            ).tolist()

            # Update the iteration index of active columns.
            if self.config["ph.column_combination.reuse_active_columns"]:
                for s in self.S:
                    column = solution[self.x_subprob_index == s]
                    column = column[self.integer_variables_in_sub[s]]
                    self._add_column_to_column_pool(s, column)

            ret = PHResult(status=True, msg="", obj=obj, sol=solution)

        else:
            # Failed to solve (e.g. infeasible, timelimit).  Exit now.
            code = model.solution.get_status()
            name = model.solution.status[code]
            ret = PHResult(
                status=False,
                msg=(
                    "failed to generate a feasible solution.  "
                    f"code: {code}  name: {name}"
                ),
                obj=np.inf,
                sol=None,
            )

        return ret


if __name__ == "__main__":
    doctest.testmod()
