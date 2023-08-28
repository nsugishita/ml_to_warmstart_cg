# -*- coding: utf-8 -*-

"""Dual initialisers with a constant output."""

from typing import Mapping

import numpy as np

from sucpy.dual_initialisers.base import DualInitialiser


class ConstantDualInitialiser(DualInitialiser):
    """Dual initialiser which always output a constant value."""

    def __init__(self, data: Mapping, config: Mapping, y) -> None:
        """Initialise a ConstantDualInitialiser instance."""
        n_linking_constraints = (
            data["constraint_subproblem_index"] == -1
        ).sum()
        self.y = np.broadcast_to(y, (n_linking_constraints,))

    def load(self, path):
        """Return withou any effect.

        This exists to be compatible with DualInitialiser.
        """
        pass

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
        return self.y


class ColdstartDualInitialiser(ConstantDualInitialiser, DualInitialiser):
    """Dual initialiser which always output zero."""

    def __init__(self, data: Mapping, config: Mapping) -> None:
        """Initialise a ColdstartDualInitialiser instance."""
        super().__init__(data, config, 0)
