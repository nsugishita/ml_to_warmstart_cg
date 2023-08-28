# -*- coding: utf-8 -*-

"""Dual initialiser getter"""

import numpy as np


def as_dual_initialiser(dual_initialiser, data, config):
    """Return a dual initialiser"""
    from sucpy.solvers.base import LPR

    from .base import DualInitialiser
    from .constant_initialisers import (
        ColdstartDualInitialiser,
        ConstantDualInitialiser,
    )

    if isinstance(dual_initialiser, DualInitialiser):
        return dual_initialiser
    elif isinstance(dual_initialiser, (np.ndarray, int, float)):
        return ConstantDualInitialiser(
            data=data, config=config, y=dual_initialiser
        )
    elif dual_initialiser == "lpr":
        return LPR(data, config)
    elif dual_initialiser == "coldstart":
        return ColdstartDualInitialiser(data, config)
    else:
        raise ValueError(
            f"invalid dual initialiser: {dual_initialiser.__class__}"
        )
