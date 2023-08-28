# -*- coding: utf-8 -*-

"""Random forest model to warmstart column generation."""

import pickle

from sklearn.ensemble import RandomForestRegressor

from sucpy.dual_initialisers.base import DualInitialiser

n_jobs = 8


class RandomForestInitialiser(DualInitialiser):
    """Random forest method to warmstart column generation.

    One must prepare data first and feed it via `NearestNeighbour.save` or
    `NearestNeighbour.load`.
    After setting data, one can call this instance with `parameter` array.
    """

    def __init__(
        self,
        dim_parameter,
        dim_dual,
    ):
        """Initialise a NearestNeighbour instance."""
        self.dim_parameter = dim_parameter
        self.dim_dual = dim_dual
        self.parameter = None
        self.dual = None
        self.parameter_std = None

    def fit(self, parameter, dual):
        """Fit a model.

        Parameters
        ----------
        data : dict
            dict which contains `parameter` and `dual`.
        """
        self.model = RandomForestRegressor(random_state=0, verbose=0)
        self.model.n_jobs = n_jobs
        self.model.fit(parameter, dual)
        self.model.n_jobs = 1

    def save(self, path):
        if not path.endswith("pkl"):
            raise ValueError
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path):
        if not path.endswith("pkl"):
            raise ValueError
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        self.model.verbose = 0

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
        return self.model.predict(parameter[None, :])[0]
