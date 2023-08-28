# -*- coding: utf-8 -*-

"""Function to compute the time to pass checkpoints"""

import numpy as np


def compute_checkpoint(
    n_lb_records, lb, lb_time, n_ub_records, ub, ub_time, tols
):
    """Compute when given tolerances are met

    Returns
    -------
    res : dict
        A dictionary containing the following items.
        - checkpoints_walltime
    """
    iter = zip(
        yield_chunks(n_lb_records, lb=lb, lb_time=lb_time),
        yield_chunks(n_ub_records, ub=ub, ub_time=ub_time),
    )
    res = {
        "checkpoints_tol": [],
        "checkpoints_walltime": [],
    }
    for _lb_chunk, _ub_chunk in iter:
        buf = _compute_checkpoint(
            lb=_lb_chunk["lb"],
            lb_time=_lb_chunk["lb_time"],
            ub=_ub_chunk["ub"],
            ub_time=_ub_chunk["ub_time"],
            tols=tols,
        )
        for key in res:
            res[key].append(buf[key])
    for key in res:
        res[key] = np.stack(res[key])
    return res


def _compute_checkpoint(lb, lb_time, ub, ub_time, tols):
    """Compute when given tolerances are met on a single run

    Returns
    -------
    res : dict
        A dictionary containing the following items.
        - checkpoints_walltime
    """
    gap = _compute_gap(lb, lb_time, ub, ub_time)
    res = {
        "checkpoints_walltime": [],
        "checkpoints_tol": np.asarray(tols),
    }
    for tol in tols:
        solved_indexes = np.nonzero(gap["gap"] < tol)[0]
        if len(solved_indexes):
            first_solved = solved_indexes[0]
            res["checkpoints_walltime"].append(gap["walltime"][first_solved])
        else:
            res["checkpoints_walltime"].append(np.nan)
    return res


def _compute_gap(lb, lb_time, ub, ub_time):
    """Merge lower and upper bound data and compute gaps

    Returns
    -------
    res : dict
        A dictionary containing the following items.
        - walltime
        - lb
        - ub
        - gap
    """
    _time, idx = np.unique(
        np.concatenate([lb_time, ub_time]), return_inverse=True
    )
    _lb = np.full(_time.size, -np.inf)
    _lb_iter = np.full(_time.size, -1, dtype=int)
    _ub = np.full(_time.size, np.inf)
    _ub_iter = np.full(_time.size, -1, dtype=int)
    lb_idx = idx[: len(lb)]
    ub_idx = idx[len(lb) :]
    _lb[lb_idx] = lb
    _lb_iter[lb_idx] = np.arange(lb.size)
    _ub[ub_idx] = ub
    _ub_iter[ub_idx] = np.arange(ub.size)
    _lb = np.maximum.accumulate(_lb)
    _ub = np.minimum.accumulate(_ub)
    _lb_iter = np.maximum.accumulate(_lb_iter)
    _ub_iter = np.maximum.accumulate(_ub_iter)
    _gap = np.empty(_lb.size)
    sel = np.isfinite(_ub)
    _gap[sel] = (_ub - _lb)[sel] / _ub[sel]
    _gap[~sel] = np.inf
    return {
        "walltime": _time,
        "lb": _lb,
        "lb_iter": _lb_iter,
        "ub": _ub,
        "ub_iter": _ub_iter,
        "gap": _gap,
    }


def yield_chunks(n, **kwargs):
    """Given concatenated data, yield each chunk

    Parameters
    ----------
    n : array of int
    **kwargs

    Yields
    ------
    chunk : dict
        Slices of `kwargs`.
    """
    start = 0
    for i in n:
        stop = start + i
        # _args = (a[start:stop] for a in args)
        _kwargs = {key: val[start:stop] for key, val in kwargs.items()}
        yield _kwargs
        start = stop
