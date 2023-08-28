# -*- coding: utf-8 -*-

"""Base components of dual initialisers"""

import abc


class DualInitialiser(abc.ABC):
    """Base class of dual initialisers"""

    @abc.abstractmethod
    def load(self, path):
        """Load data from a disk

        Extended class may override this method
        to load trained parameters or data.

        Parameters
        ----------
        path : str
        """
        pass

    @abc.abstractmethod
    def compute_initial_dual(self, parameter):
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
        pass
