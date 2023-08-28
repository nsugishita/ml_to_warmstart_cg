# -*- coding: utf-8 -*-

"""Nearest neighbour method to warmstart column generation."""

import numpy as np

from sucpy.dual_initialisers.base import DualInitialiser


class NearestNeighbourInitialiser(DualInitialiser):
    """Nearest neighbour method to warmstart column generation.

    One must prepare data first and feed it via `NearestNeighbour.save` or
    `NearestNeighbour.load`.
    After setting data, one can call this instance with `parameter` array.

    Attributes
    ----------
    demand : (n_data, n_t) array
    demand_std : (n_t,) array
    initial_condition : (n_data, n_g) array
    initial_condition_std : (n_g,) array
    dual : (n_data, dim_dual) array
    """

    def __init__(self, config):
        """Initialise a NearestNeighbour instance."""
        super().__init__()
        self.config = config
        self.demand = None
        self.demand_std = None
        self.initial_condition = None
        self.initial_condition_std = None
        self.demand_and_initial_condition = None
        self.demand_and_initial_condition_std = None
        self.dual = None

    def set_data(self, data, clip_std=1e-2):
        """Extract data from CG result."""
        self.demand = data["demand"]
        self.demand_std = self.demand.std(axis=0)
        self.demand_std = self.demand_std.clip(clip_std)
        self.initial_condition = data["initial_condition"]
        self.initial_condition_std = self.initial_condition.std(axis=0)
        self.initial_condition_std = self.initial_condition_std.clip(clip_std)
        self.demand_and_initial_condition = np.concatenate(
            [self.demand, self.initial_condition],
            axis=-1,
        )
        self.demand_and_initial_condition_std = np.concatenate(
            [self.demand_std, self.initial_condition_std],
            axis=-1,
        )
        self.dual = data["y"]

    def load(self, path):
        """Load data from a disk."""
        if not path.endswith("npz"):
            raise ValueError

        full_data = np.load(path)

        training_budget = 24 * 60 * 60
        mask = full_data["instance_solution_walltime"] < training_budget

        data = {
            "demand": full_data["demand"][mask],
            "initial_condition": full_data["initial_condition"][mask],
            "y": full_data["y"][mask],
        }

        self.set_data(data)

    def _get_nearest_instance_index(self, parameter):
        n = self.config["nearest_neighbour_initialiser.n_neighbours"]
        if parameter.shape[-1] == self.demand.shape[-1]:
            diff = (self.demand - parameter) / self.demand_std
            criteria = np.linalg.norm(diff, ord=2, axis=1)  # (data_size,)
            return np.argsort(criteria)[:n]
        else:
            raise NotImplementedError

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
        selected_index = self._get_nearest_instance_index(parameter)
        return np.mean(self.dual[selected_index], axis=0)
