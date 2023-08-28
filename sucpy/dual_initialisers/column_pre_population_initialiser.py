# -*- coding: utf-8 -*-

"""Column pre-population dual initialiser"""

import weakref

import numpy as np

from sucpy.dual_initialisers import DualInitialiser
from sucpy.solvers.rmp import RegularizedRMP


class ColumnPrePopulationDualInitialiser(DualInitialiser):
    """Column pre-population dual initialiser"""

    def __init__(self, config):
        """Initialise a ColumnPrePopulationDualInitialiser instance"""
        super().__init__()
        self.config = config

    def load(self, path):
        """Load data from a disk

        Extended class may override this method
        to load trained parameters or data.

        Parameters
        ----------
        path : str
        """
        self.solution_data = np.load(path)

    def set_cg(self, cg):
        self.cg = weakref.ref(cg)
        self.problem_data = cg.data
        self.rmp = RegularizedRMP(
            data=self.problem_data, config=self.config, journal=None
        )

    def compute_initial_dual(self, problem_data):
        """Compute an initial dual value

        Extended class must overrides this method
        to compute an initial dual value.

        Parameters
        ----------
        parameter : (dim_parameter,) array

        Returns
        -------
        dual : (dim_dual,) array
        """
        rmp = self.rmp

        n_neighbours = self.config[
            "column_pre_population_initialiser.n_neighbours"
        ]
        only_active = self.config[
            "column_pre_population_initialiser.only_active"
        ]
        for i in range(10):
            cuts = get_neighbours_cuts(
                self.problem_data,
                self.solution_data,
                n_neighbours,
                only_active,
            )
            rmp.add_cuts(**cuts)

            if self.config["column_pre_population_initialiser.populate_rmp"]:
                self.cg().rmp.add_cuts(**cuts)

            rmp.solve()
            if rmp.get_status() != 2:
                return rmp.get_x()

            n_neighbours += 1
            rmp = RegularizedRMP(
                data=self.problem_data, config=self.config, journal=None
            )


def get_neighbours_cuts(
    problem_data, solution_data, n_neighbours, only_active
):
    if n_neighbours == 0:
        return {
            "component_index": [],
            "subgradient_value": [],
            "rhs": [],
        }
    else:
        dataset_parameter = np.array(solution_data["demand"])
        np.testing.assert_equal(
            dataset_parameter.shape[1:], problem_data["demand"].shape
        )
        # (n_training_instances,)
        discrepancy = np.sum(
            np.abs(dataset_parameter - problem_data["demand"]), axis=-1
        )
        unique_disc = np.unique(discrepancy)
        if len(unique_disc) == 1:
            step = 1
        else:
            step = min(0.5 * np.min(np.diff(unique_disc)), 1)
        tie_breaker = np.linspace(0, -step, discrepancy.size)
        discrepancy = discrepancy + tie_breaker
        # (n,)
        selector = np.argsort(discrepancy)[:n_neighbours]

        cuts_stop = np.cumsum(solution_data["n_cuts"])
        cuts_start = np.r_[0, cuts_stop[:-1]]

        out = {
            "component_index": [],
            "subgradient_value": [],
            "rhs": [],
            "active": [],
        }

        for index in selector:
            for key in out:
                start = cuts_start[index]
                stop = cuts_stop[index]
                out[key].append(solution_data["cuts_" + key][start:stop])

        for key in out:
            out[key] = np.concatenate(out[key])

        if only_active:
            selector = np.nonzero(out["active"])[0]
            for key in out:
                out[key] = out[key][selector]
        out.pop("active")
        return out
