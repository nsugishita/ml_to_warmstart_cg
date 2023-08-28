# -*- coding: utf-8 -*-

"""Routines to handle decomposable problem data"""

from typing import Any, MutableMapping, Optional, cast

import numpy as np
import scipy.sparse as sparse

from sucpy.problems import base


def clear_decomposition_cache_(problem_data, submatrices=False, reset=True):
    """Clear all cached data on decomposition.

    Parameters
    ----------
    problem_data
    submatrices : bool, default False
        If True, this clears data on submatrices.
    reset : bool, default True
        If True, call `set_decomposition_data` and
        reset the values.
    """
    # We only clear submatrices data when level >= 2
    subproblem_matrices = [
        "subproblem_constraint_coefficient_diagonal",
        "subproblem_constraint_coefficient_row_border",
        "subproblem_constraint_coefficient_col_border",
    ]
    for key in list(problem_data):
        if key in subproblem_matrices:
            if submatrices:
                problem_data.pop(key)
        elif key.startswith("subproblem_") or key.startswith("_sub"):
            problem_data.pop(key)
    if reset:
        if submatrices:
            set_decomposition_data_(problem_data, all=True)
        else:
            set_decomposition_data_(problem_data, all_vectors=True)


def set_decomposition_data(
    problem_data: MutableMapping,
    *,
    variable_lb: bool = False,
    variable_ub: bool = False,
    variable_type: bool = False,
    objective_coefficient: bool = False,
    constraint_coefficient: bool = False,
    constraint_rhs: bool = False,
    constraint_sense: bool = False,
    all: bool = False,
) -> Optional[MutableMapping]:
    """Set data of decomposed subproblems.

    This takes a problem data and assigns/updates
    data related to the subproblems.
    All arguments except `data` must be passed as
    keyword arguments.
    Note that this does not modify the existing
    data.  For inplace operation, see
    `set_decomposition_data_`.

    Examples
    --------
    >>> import sucpy
    >>> data = sucpy.problems.parse_dense_format('''
    ... min   7.  8.  8.  9.  7.  3.  6.  5.  1.
    ... s.t.  7.  4.  7.  0.  0.  0.  0.  0.  0. <= -3.
    ...       9. -4.  3.  0.  0.  0.  0.  0.  0. <= -4.
    ...       0.  0.  0.  8. -9. -1.  0.  0.  0. <=  3.
    ...       0.  0.  0.  9.  8. -6.  0.  0.  0. <=  5.
    ...      -3. -2. -8. -4.  6.  0. -9. -4.  7. <=  6.
    ...      -2. -7. -2.  7.  0. -8. -9. -2. -9. <=  5.
    ... xlb:  0.  0.  0.  0.  0.  0.  0.  0.  0.
    ... xub: 10. 10. 10. 10. 10. 10. 10. 10. 10.
    ... ''')
    >>> data['n_subproblems'] = 2
    >>> data['variable_subproblem_index'] = np.array([
    ...       0,  0,  0,  1,  1,  1, -1, -1, -1
    ... ])
    >>> data['constraint_subproblem_index'] = np.array([
    ...       0,  0,  1,  1,  -1,  -1
    ... ])
    >>> p = set_decomposition_data(data, all=True)
    >>> print(p['subproblem_constraint_coefficient_row_border'][0].toarray())
    [[-3. -2. -8.]
     [-2. -7. -2.]]
    >>> print(p['subproblem_constraint_coefficient_row_border'][-1].toarray())
    [[-9. -4.  7.]
     [-9. -2. -9.]]
    >>> print(p['subproblem_constraint_coefficient_col_border'][-1].toarray())
    [[-9. -4.  7.]
     [-9. -2. -9.]]
    >>> print(p['subproblem_constraint_coefficient_diagonal'][-1].toarray())
    [[-9. -4.  7.]
     [-9. -2. -9.]]
    >>> print(p['subproblem_constraint_coefficient_diagonal'][1].toarray())
    [[ 8. -9. -1.]
     [ 9.  8. -6.]]
    >>> p['constraint_rhs'] = np.arange(6).astype(float)
    >>> rhs = p['subproblem_constraint_rhs'][1]
    >>> print(rhs)
    [3. 5.]

    >>> set_decomposition_data_(p, constraint_rhs=True)
    >>> print(sucpy.problems.format_densely(p))
    min   7.  8.  8.  9.  7.  3.  6.  5.  1.
    s.t.  7.  4.  7.  0.  0.  0.  0.  0.  0. <= 0.
          9. -4.  3.  0.  0.  0.  0.  0.  0. <= 1.
          0.  0.  0.  8. -9. -1.  0.  0.  0. <= 2.
          0.  0.  0.  9.  8. -6.  0.  0.  0. <= 3.
         -3. -2. -8. -4.  6.  0. -9. -4.  7. <= 4.
         -2. -7. -2.  7.  0. -8. -9. -2. -9. <= 5.
    xlb:  0.  0.  0.  0.  0.  0.  0.  0.  0.
    xub: 10. 10. 10. 10. 10. 10. 10. 10. 10.
    >>> print(rhs)  # values are updated in place.
    [2. 3.]

    >>> p['constraint_rhs'] = -np.arange(6).astype(float)
    >>> q = set_decomposition_data(p, constraint_rhs=True)
    >>> print(sucpy.problems.format_densely(p))
    min   7.  8.  8.  9.  7.  3.  6.  5.  1.
    s.t.  7.  4.  7.  0.  0.  0.  0.  0.  0. <= -0.
          9. -4.  3.  0.  0.  0.  0.  0.  0. <= -1.
          0.  0.  0.  8. -9. -1.  0.  0.  0. <= -2.
          0.  0.  0.  9.  8. -6.  0.  0.  0. <= -3.
         -3. -2. -8. -4.  6.  0. -9. -4.  7. <= -4.
         -2. -7. -2.  7.  0. -8. -9. -2. -9. <= -5.
    xlb:  0.  0.  0.  0.  0.  0.  0.  0.  0.
    xub: 10. 10. 10. 10. 10. 10. 10. 10. 10.
    >>> print(rhs)
    [2. 3.]

    Parameters
    ----------
    data : dict
        A problem data with `variable_subproblem_index` and
        `constraint_subproblem_index`.
    variable_lb : bool, default False
        Update subproblem data related
        to the variable lower bound.
    variable_ub : bool, default False
        Update subproblem data related
        to the variable upper bound.
    variable_type : bool, default False
        Update subproblem data related
        to the variable type.
    objective_coefficient : bool, default False
        Update subproblem data related
        to the objective coefficient.
    constraint_coefficient : bool, default False
        Update subproblem data related
        to the constraint coefficient.
    constraint_rhs : bool, default False
        Update subproblem data related
        to the constraint rhs.
    constraint_sense : bool, default False
        Update subproblem data related
        to the constraint sense.
    all : bool, default False
        Update all subproblem data.
    """
    base.validate_problem_data(problem_data)

    buf = {}
    copied = []
    popped = []
    if variable_lb or all:
        copied.append("subproblem_variable_lb")
        popped.append("_subproblem_variable_lb_concatenated")
    if variable_ub or all:
        copied.append("subproblem_variable_ub")
        popped.append("_subproblem_variable_ub_concatenated")
    if variable_type or all:
        copied.append("subproblem_variable_type")
        popped.append("_subproblem_variable_type_concatenated")
    if objective_coefficient or all:
        copied.append("subproblem_objective_coefficient")
        popped.append("_subproblem_objective_coefficient_concatenated")
    if constraint_rhs or all:
        copied.append("subproblem_constraint_rhs")
        popped.append("_subproblem_constraint_rhs_concatenated")
    if constraint_sense or all:
        copied.append("subproblem_constraint_sense")
        popped.append("_subproblem_constraint_sense_concatenated")
    if constraint_coefficient or all:
        pass
    for k, v in problem_data.items():
        if k in popped:
            continue
        if k not in copied:
            buf[k] = v
            continue
        if isinstance(v, (list, tuple)):
            try:
                buf[k] = [x.copy() for x in v]
            except AttributeError:
                buf[k] = cast(Any, v)
        elif isinstance(v, np.ndarray):
            buf[k] = v.copy()
        else:
            try:
                buf[k] = v.copy()
            except AttributeError:
                buf[k] = v
    set_decomposition_data_(
        problem_data=buf,
        variable_lb=variable_lb,
        variable_ub=variable_ub,
        variable_type=variable_type,
        objective_coefficient=objective_coefficient,
        constraint_coefficient=constraint_coefficient,
        constraint_rhs=constraint_rhs,
        constraint_sense=constraint_sense,
        all=all,
    )
    return buf


def set_decomposition_data_(
    problem_data: MutableMapping,
    *,
    variable_lb: bool = False,
    variable_ub: bool = False,
    variable_type: bool = False,
    objective_coefficient: bool = False,
    constraint_coefficient: bool = False,
    constraint_rhs: bool = False,
    constraint_sense: bool = False,
    all: bool = False,
    all_vectors: bool = False,
) -> None:
    """Set data of the subproblems in place.

    This is an in place version of `set_decomposition_data`.
    Currently, the constraint coefficient is not
    reused.

    Please see the doc of `set_decomposition_data` for
    more detail.
    """
    if (
        (not variable_lb)
        and (not variable_ub)
        and (not variable_type)
        and (not objective_coefficient)
        and (not constraint_coefficient)
        and (not constraint_rhs)
        and (not constraint_sense)
        and (not all)
        and (not all_vectors)
    ):
        raise ValueError("At least one option must be given.")

    if all:
        variable_lb = (
            variable_ub
        ) = (
            variable_type
        ) = (
            objective_coefficient
        ) = constraint_coefficient = constraint_rhs = constraint_sense = True
    if all_vectors:
        variable_lb = (
            variable_ub
        ) = (
            variable_type
        ) = objective_coefficient = constraint_rhs = constraint_sense = True

    p_data = problem_data
    # At least we need those data.
    variable_subproblem_index = p_data["variable_subproblem_index"]
    constraint_subproblem_index = p_data["constraint_subproblem_index"]

    if "n_subproblems" not in p_data:
        if np.any(variable_subproblem_index == -1):
            _n_subprobs1 = np.max(variable_subproblem_index) + 1
        else:
            _n_subprobs1 = np.max(variable_subproblem_index)
        if np.any(constraint_subproblem_index == -1):
            _n_subprobs2 = np.max(constraint_subproblem_index) + 1
        else:
            _n_subprobs2 = np.max(constraint_subproblem_index)
        p_data["n_subproblems"] = max(_n_subprobs1, _n_subprobs2)

    n_subproblems = p_data["n_subproblems"]

    # Set iterators if not found.
    p_data.setdefault("subproblem_index", range(n_subproblems))
    p_data.setdefault(
        "subproblem_index_with_negative_one",
        np.r_[np.arange(n_subproblems), -1],
    )

    if "subproblem_variable_sorter" not in p_data:
        # Create subproblem variables / constraints sorter if not found.
        vsort, csort, cvsort = get_sorter(
            n_subproblems,
            constraint_subproblem_index,
            variable_subproblem_index,
        )
        # Recall sorter permute the variables or constraints
        # so that the linking ones comes first and then
        # subproblems.
        p_data["subproblem_variable_sorter"] = vsort
        p_data["subproblem_constraint_sorter"] = csort
        p_data["subproblem_constraint_variable_sorter"] = cvsort
        # Count each blocks.
        temp = np.r_[
            p_data["variable_subproblem_index"], np.arange(-1, n_subproblems)
        ]
        subproblem_n_variables = np.unique(temp, return_counts=True)[1] - 1
        # Put the number of linking variables at the end to be
        # consistent with others.
        subproblem_n_variables = np.r_[
            subproblem_n_variables[1:], subproblem_n_variables[0]
        ]
        vstop = p_data["subproblem_variable_stop"] = np.cumsum(
            subproblem_n_variables
        )
        vstart = p_data["subproblem_variable_start"] = np.r_[0, vstop[:-1]]
        temp = np.r_[
            p_data["constraint_subproblem_index"], np.arange(-1, n_subproblems)
        ]
        subproblem_n_constraints = np.unique(temp, return_counts=True)[1] - 1
        subproblem_n_constraints = np.r_[
            subproblem_n_constraints[1:], subproblem_n_constraints[0]
        ]
        cstop = p_data["subproblem_constraint_stop"] = np.cumsum(
            subproblem_n_constraints
        )
        cstart = p_data["subproblem_constraint_start"] = np.r_[0, cstop[:-1]]
        temp = np.empty(2 * (n_subproblems + 1), dtype=np.int32)
        temp[::2] = subproblem_n_constraints
        temp[1::2] = subproblem_n_variables
        cvstop = p_data["subproblem_constraint_variable_stop"] = np.cumsum(
            temp
        )
        p_data["subproblem_constraint_variable_start"] = np.r_[0, cvstop[:-1]]
        p_data["subproblem_n_variables"] = subproblem_n_variables
        p_data["subproblem_n_constraints"] = subproblem_n_constraints
        np.testing.assert_equal(
            subproblem_n_variables.sum(), p_data["n_variables"]
        )
        np.testing.assert_equal(
            subproblem_n_variables.shape, (p_data["n_subproblems"] + 1,)
        )
        np.testing.assert_equal(
            subproblem_n_constraints.sum(), p_data["n_constraints"]
        )
        np.testing.assert_equal(
            subproblem_n_constraints.shape, (p_data["n_subproblems"] + 1,)
        )
    else:
        vsort = p_data["subproblem_variable_sorter"]
        vstart = p_data["subproblem_variable_start"]
        vstop = p_data["subproblem_variable_stop"]
        csort = p_data["subproblem_constraint_sorter"]
        cstart = p_data["subproblem_constraint_start"]
        cstop = p_data["subproblem_constraint_stop"]

    if ("subproblem_variable_lb" not in p_data) or (
        "_subproblem_variable_lb_concatenated" not in p_data
    ):
        # Allocate memory.
        p_data["_subproblem_variable_lb_concatenated"] = np.empty_like(
            p_data["variable_lb"]
        )
        # Create views.
        p_data["subproblem_variable_lb"] = [
            p_data["_subproblem_variable_lb_concatenated"][start:stop]
            for start, stop in zip(vstart, vstop)
        ]
        variable_lb = True

    if variable_lb:
        np.take(
            p_data["variable_lb"],
            vsort,
            out=p_data["_subproblem_variable_lb_concatenated"],
        )

    if ("subproblem_variable_ub" not in p_data) or (
        "_subproblem_variable_ub_concatenated" not in p_data
    ):
        p_data["_subproblem_variable_ub_concatenated"] = np.empty_like(
            p_data["variable_ub"]
        )
        p_data["subproblem_variable_ub"] = [
            p_data["_subproblem_variable_ub_concatenated"][start:stop]
            for start, stop in zip(vstart, vstop)
        ]
        variable_ub = True

    if variable_ub:
        np.take(
            p_data["variable_ub"],
            vsort,
            out=p_data["_subproblem_variable_ub_concatenated"],
        )

    if ("subproblem_variable_type" not in p_data) or (
        "_subproblem_variable_type_concatenated" not in p_data
    ):
        p_data["_subproblem_variable_type_concatenated"] = np.empty_like(
            p_data["variable_type"]
        )
        p_data["subproblem_variable_type"] = [
            p_data["_subproblem_variable_type_concatenated"][start:stop]
            for start, stop in zip(vstart, vstop)
        ]
        variable_type = True

    if variable_type:
        np.take(
            p_data["variable_type"],
            vsort,
            out=p_data["_subproblem_variable_type_concatenated"],
        )

    if ("subproblem_objective_coefficient" not in p_data) or (
        "_subproblem_objective_coefficient_concatenated" not in p_data
    ):
        p_data[
            "_subproblem_objective_coefficient_concatenated"
        ] = np.empty_like(p_data["objective_coefficient"])
        p_data["subproblem_objective_coefficient"] = [
            p_data["_subproblem_objective_coefficient_concatenated"][
                start:stop
            ]
            for start, stop in zip(vstart, vstop)
        ]
        objective_coefficient = True

    if objective_coefficient:
        np.take(
            p_data["objective_coefficient"],
            vsort,
            out=p_data["_subproblem_objective_coefficient_concatenated"],
        )

    allocate_rhs = ("subproblem_constraint_rhs" not in p_data) or (
        "_subproblem_constraint_rhs_concatenated" not in p_data
    )
    if allocate_rhs:
        p_data["_subproblem_constraint_rhs_concatenated"] = np.empty_like(
            p_data["constraint_rhs"]
        )
        p_data["subproblem_constraint_rhs"] = [
            p_data["_subproblem_constraint_rhs_concatenated"][start:stop]
            for start, stop in zip(cstart, cstop)
        ]
        constraint_rhs = True

    if constraint_rhs:
        np.take(
            p_data["constraint_rhs"],
            csort,
            out=p_data["_subproblem_constraint_rhs_concatenated"],
        )

    allocate_con_sense = ("subproblem_constraint_sense" not in p_data) or (
        "_subproblem_constraint_sense_concatenated" not in p_data
    )
    if allocate_con_sense:
        p_data["_subproblem_constraint_sense_concatenated"] = np.empty_like(
            p_data["constraint_sense"]
        )
        p_data["subproblem_constraint_sense"] = [
            p_data["_subproblem_constraint_sense_concatenated"][start:stop]
            for start, stop in zip(cstart, cstop)
        ]
        constraint_sense = True

    if constraint_sense:
        np.take(
            p_data["constraint_sense"],
            csort,
            out=p_data["_subproblem_constraint_sense_concatenated"],
        )

    if constraint_coefficient or all:
        buf = slice_block_angular_matrix(
            p_data["constraint_coefficient"],
            n_subproblems,
            constraint_subproblem_index,
            variable_subproblem_index,
        )
        p_data["subproblem_constraint_coefficient_diagonal"] = buf[0]
        p_data["subproblem_constraint_coefficient_row_border"] = buf[1]
        p_data["subproblem_constraint_coefficient_col_border"] = buf[2]

    base.validate_problem_data(p_data, all=True)


class SubprobVector(object):
    """Accessor to a subproblem component."""

    def set(self, v):
        """Set subproblem vectors given an array in the original problem."""
        np.take(v, self.sorter, out=self.sorted)

    def retrieve(self, v=None):
        """Retrieve an array in the original problem."""
        if v is None:
            v = np.empty_like(self.sorted)
            return_v = True
        else:
            return_v = False
        v[self.sorter] = self.sorted
        if return_v:
            return v

    def __getitem__(self, i):
        """Return the i-th subproblem component.

        Parameters
        ----------
        i : int

        Returns
        -------
        v : 1d array
        """
        return self.view[i]

    def __setitem__(self, i, v):
        """Set the i-th subproblem component.

        Parameters
        ----------
        i : int
        v : 1d array
        """
        np.copyto(self.view[i], v)


class SubprobX(SubprobVector):
    """Accessor to subproblem x.

    Examples
    --------
    >>> import sucpy
    >>> data = sucpy.problems.parse_dense_format('''
    ... min   7.  8.  8.  6.  5.  1.  9.  7.  3.
    ... s.t.  7.  4.  7.  0.  0.  0.  0.  0.  0. <= -3.
    ...       9. -4.  3.  0.  0.  0.  0.  0.  0. <= -4.
    ...      -3. -2. -8. -9. -4.  7. -4.  6.  0. <=  6.
    ...      -2. -7. -2. -9. -2. -9.  7.  0. -8. <=  5.
    ...       0.  0.  0.  0.  0.  0.  8. -9. -1. <=  3.
    ...       0.  0.  0.  0.  0.  0.  9.  8. -6. <=  5.
    ... xlb:  0.  0.  0.  0.  0.  0.  0.  0.  0.
    ... xub: 10. 10. 10. 10. 10. 10. 10. 10. 10.
    ... ''')
    >>> data['n_subproblems'] = 2
    >>> data['variable_subproblem_index'] = np.array([
    ...       0,  0,  0, -1, -1, -1,  1,  1,  1
    ... ])
    >>> data['constraint_subproblem_index'] = np.array([
    ...       0,  0, -1, -1,   1,   1
    ... ])
    >>> p = set_decomposition_data(data, all=True)
    >>> subproblem_x = SubprobX(p)
    >>> subproblem_x.set(np.arange(9, dtype=float))
    >>> print(subproblem_x[0])
    [0. 1. 2.]
    >>> print(subproblem_x[1])
    [6. 7. 8.]
    >>> print(subproblem_x[-1])
    [3. 4. 5.]
    >>> subproblem_x[0] = np.array([10., 20., 30.])
    >>> subproblem_x[1] = np.array([1., 2., 3.])
    >>> subproblem_x[-1] = np.array([-1., -2., -3.])
    >>> print(subproblem_x.retrieve())
    [10. 20. 30. -1. -2. -3.  1.  2.  3.]
    """

    def __init__(self, problem_data, dtype=float):
        self.sorter = problem_data["subproblem_variable_sorter"]
        vstart = problem_data["subproblem_variable_start"]
        vstop = problem_data["subproblem_variable_stop"]
        self.sorted = np.empty(problem_data["n_variables"], dtype=dtype)
        self.view = [
            self.sorted[start:stop] for start, stop in zip(vstart, vstop)
        ]


class SubprobY(SubprobVector):
    """Accessor to subproblem y.

    Examples
    --------
    >>> import sucpy
    >>> data = sucpy.problems.parse_dense_format('''
    ... min   7.  8.  8.  6.  5.  1.  9.  7.  3.
    ... s.t.  7.  4.  7.  0.  0.  0.  0.  0.  0. <= -3.
    ...       9. -4.  3.  0.  0.  0.  0.  0.  0. <= -4.
    ...      -3. -2. -8. -9. -4.  7. -4.  6.  0. <=  6.
    ...      -2. -7. -2. -9. -2. -9.  7.  0. -8. <=  5.
    ...       0.  0.  0.  0.  0.  0.  8. -9. -1. <=  3.
    ...       0.  0.  0.  0.  0.  0.  9.  8. -6. <=  5.
    ... xlb:  0.  0.  0.  0.  0.  0.  0.  0.  0.
    ... xub: 10. 10. 10. 10. 10. 10. 10. 10. 10.
    ... ''')
    >>> data['n_subproblems'] = 2
    >>> data['variable_subproblem_index'] = np.array([
    ...       0,  0,  0, -1, -1, -1,  1,  1,  1
    ... ])
    >>> data['constraint_subproblem_index'] = np.array([
    ...       0,  0, -1, -1,   1,   1
    ... ])
    >>> p = set_decomposition_data(data, all=True)
    >>> subproblem_y = SubprobY(p)
    >>> subproblem_y.set(np.arange(6, dtype=float))
    >>> print(subproblem_y[0])
    [0. 1.]
    >>> print(subproblem_y[1])
    [4. 5.]
    >>> print(subproblem_y[-1])
    [2. 3.]
    >>> subproblem_y[0] = np.array([4., 2.])
    >>> subproblem_y[1] = np.array([0., 0.])
    >>> subproblem_y[-1] = np.array([-1., -2.])
    >>> print(subproblem_y.retrieve())
    [ 4.  2. -1. -2.  0.  0.]
    """

    def __init__(self, problem_data):
        self.sorter = problem_data["subproblem_constraint_sorter"]
        vstart = problem_data["subproblem_constraint_start"]
        vstop = problem_data["subproblem_constraint_stop"]
        self.sorted = np.empty(problem_data["n_constraints"])
        self.view = [
            self.sorted[start:stop] for start, stop in zip(vstart, vstop)
        ]


class SubprobYX(object):
    """Accessor to subproblem y and x.

    Examples
    --------
    >>> import sucpy
    >>> data = sucpy.problems.parse_dense_format('''
    ... min   7.  8.  8.  6.  5.  1.  9.  7.  3.
    ... s.t.  7.  4.  7.  0.  0.  0.  0.  0.  0. <= -3.
    ...       9. -4.  3.  0.  0.  0.  0.  0.  0. <= -4.
    ...      -3. -2. -8. -9. -4.  7. -4.  6.  0. <=  6.
    ...      -2. -7. -2. -9. -2. -9.  7.  0. -8. <=  5.
    ...       0.  0.  0.  0.  0.  0.  8. -9. -1. <=  3.
    ...       0.  0.  0.  0.  0.  0.  9.  8. -6. <=  5.
    ... xlb:  0.  0.  0.  0.  0.  0.  0.  0.  0.
    ... xub: 10. 10. 10. 10. 10. 10. 10. 10. 10.
    ... ''')
    >>> data['n_subproblems'] = 2
    >>> data['variable_subproblem_index'] = np.array([
    ...       0,  0,  0, -1, -1, -1,  1,  1,  1
    ... ])
    >>> data['constraint_subproblem_index'] = np.array([
    ...       0,  0, -1, -1,   1,   1
    ... ])
    >>> p = set_decomposition_data(data, all=True)
    >>> subproblem_yx = SubprobYX(p)
    >>> data = np.r_[np.arange(6, dtype=float), -np.arange(9)]
    >>> subproblem_yx.set(yx=data)
    >>> print(subproblem_yx[0])
    [ 0.  1.  0. -1. -2.]
    >>> print(subproblem_yx[1])
    [ 4.  5. -6. -7. -8.]
    >>> print(subproblem_yx[-1])
    [ 2.  3. -3. -4. -5.]
    >>> print(subproblem_yx.y[0])
    [0. 1.]
    >>> print(subproblem_yx.y[1])
    [4. 5.]
    >>> print(subproblem_yx.x[0])
    [ 0. -1. -2.]

    >>> subproblem_yx[0] = np.arange(5, dtype=float)
    >>> subproblem_yx[1] = np.arange(10, 15, dtype=float)
    >>> subproblem_yx[-1] = np.arange(20, 25, dtype=float)
    >>> print(subproblem_yx.retrieve())
    [ 0.  1. 20. 21. 10. 11.  2.  3.  4. 22. 23. 24. 12. 13. 14.]
    """

    def __init__(self, problem_data):
        self.sorter = problem_data["subproblem_constraint_variable_sorter"]
        start = problem_data["subproblem_constraint_variable_start"]
        stop = problem_data["subproblem_constraint_variable_stop"]
        self.n_variables = problem_data["n_variables"]
        self.n_constraints = problem_data["n_constraints"]
        self.sorted = np.empty(
            problem_data["n_constraints"] + problem_data["n_variables"]
        )
        self.view = [
            self.sorted[_start:_stop]
            for _start, _stop in zip(start[::2], stop[1::2])
        ]
        self.y = [
            self.sorted[_start:_stop]
            for _start, _stop in zip(start[::2], stop[::2])
        ]
        self.x = [
            self.sorted[_start:_stop]
            for _start, _stop in zip(start[1::2], stop[1::2])
        ]

    def set(self, *, y=None, x=None, yx=None):
        """Set subproblem vectors given an array in the original problem."""
        if yx is not None:
            np.take(yx, self.sorter, out=self.sorted)
        elif (y is not None) and (x is not None):
            tmp = np.r_[y, x]
            np.take(tmp, self.sorter, out=self.sorted)
        else:
            raise ValueError("yx or x and y are required")

    def retrieve(self, v=None):
        """Retrieve an array in the original problem."""
        if v is None:
            v = np.empty_like(self.sorted)
            return_v = True
        else:
            return_v = False
        v[self.sorter] = self.sorted
        if return_v:
            return v

    def retrieve_y(self, v=None):
        """Retrieve x in the original problem."""
        raise NotImplementedError

    def retrieve_x(self, v=None):
        """Retrieve x in the original problem."""
        raise NotImplementedError

    def __getitem__(self, i):
        """Return the i-th subproblem component.

        Parameters
        ----------
        i : int

        Returns
        -------
        v : 1d array
        """
        return self.view[i]

    def __setitem__(self, i, v):
        """Set the i-th subproblem component.

        Parameters
        ----------
        i : int
        v : 1d array
        """
        np.copyto(self.view[i], v)


def get_subproblem_variable_and_subproblem_constraint(
    n_blocks, row_partition, col_partition
):
    """Return getter of variables and constraints in a subproblem.

    For the format of `row_partition` and `col_partition`,
    see `slice_block_angular_matrix`.

    Parameters
    ----------
    n_blocks : int
        S in the above document.
    row_partition : (N,) array of int
        Each element should be between -1 and
        num_subproblems - 1 (inclusive).
        -1 indicates corresponding rows belong
        to the linking constraint.
    col_partition : (M,) array of int
        Each element should be between -1 and
        num_subproblems -1 (inclusive).
        -1 indicates corresponding rows belong
        to the linking variable.

    Returns
    -------
    get_subproblem_variable : (N,) array of int
    get_subproblem_constraint : (M,) array of int

    Examples
    --------
    >>> row_partition = np.array([0, 0, -1, 1, 1])
    >>> col_partition = np.array([0, 0, -1, 1, 1])
    >>> get_var, _ = get_subproblem_variable_and_subproblem_constraint(
    ...     2, row_partition, col_partition
    ... )
    >>> a = np.array([1, 3, 5, 7, 9])
    >>> print(a[get_var[0]])
    [1 3]
    >>> print(a[get_var[1]])
    [7 9]
    >>> print(a[get_var[-1]])
    [5]
    """
    # 0, 1, ..., n_blocks - 1, -1
    subproblem_index_with_negative_one = list(range(n_blocks)) + [-1]
    # (n_blocks+1) list of tuple.
    get_subproblem_variable = [
        np.nonzero(col_partition == s)[0]
        for s in subproblem_index_with_negative_one
    ]
    # (n_blocks+1) list of tuple.
    get_subproblem_constraint = [
        np.nonzero(row_partition == s)[0]
        for s in subproblem_index_with_negative_one
    ]
    return get_subproblem_variable, get_subproblem_constraint


def get_sorter(n_blocks, row_partition, col_partition):
    """Return sorter to create a block angular structure.

    For the format of `row_partition` and `col_partition`,
    see `slice_block_angular_matrix`.

    Parameters
    ----------
    n_blocks : int
        S in the above document.
    row_partition : (N,) array of int
        Each element should be between -1 and
        num_subproblems - 1 (inclusive).
        -1 indicates corresponding rows belong
        to the linking constraint.
    col_partition : (M,) array of int
        Each element should be between -1 and
        num_subproblems -1 (inclusive).
        -1 indicates corresponding rows belong
        to the linking variable.

    Returns
    -------
    subproblem_variable_sorter : (N,) array of int
    subproblem_constraint_sorter : (M,) array of int
    subproblem_constraint_variable_sorter : (M+N,) array of int

    Examples
    --------
    >>> row_partition = np.array([0, 0, -1, 1, 1])
    >>> col_partition = np.array([0, 0, -1, 1, 1])
    >>> A = np.array([
    ...     [1, 2, 0, 0, 0],
    ...     [3, 4, 0, 0, 0],
    ...     [5, 6, 7, 8, 9],
    ...     [0, 0, 0,10,11],
    ...     [0, 0, 0,12,13],
    ... ])
    >>> var_sorter, con_sorter, con_var_sorter = get_sorter(
    ...     2, row_partition, col_partition
    ... )
    >>> a = np.array([1, 3, 5, 7, 9])
    >>> print(a[var_sorter])
    [1 3 7 9 5]
    >>> print(A[np.ix_(con_sorter, var_sorter)])
    [[ 1  2  0  0  0]
     [ 3  4  0  0  0]
     [ 0  0 10 11  0]
     [ 0  0 12 13  0]
     [ 5  6  8  9  7]]
    >>> a = np.array('c1,c2,c3,c4,c5,v1,v2,v3,v4,v5'.split(','))
    >>> print(a[con_var_sorter])
    ['c1' 'c2' 'v1' 'v2' 'c4' 'c5' 'v4' 'v5' 'c3' 'v3']
    """
    row_partition = np.array(row_partition, dtype=float)
    col_partition = np.array(col_partition, dtype=float)
    row_partition[row_partition < -0.5] = np.max(row_partition) + 1
    col_partition[col_partition < -0.5] = np.max(col_partition) + 1
    row_partition += np.linspace(0.0, 0.1, len(row_partition))
    col_partition += np.linspace(0.5, 0.6, len(col_partition))
    subproblem_row_sorter = np.argsort(row_partition)
    subproblem_col_sorter = np.argsort(col_partition)
    row_col_partition = np.r_[row_partition, col_partition]
    subproblem_row_col_sorter = np.argsort(row_col_partition)
    return (
        subproblem_col_sorter,
        subproblem_row_sorter,
        subproblem_row_col_sorter,
    )


def slice_block_angular_matrix(
    P,
    n_blocks: int,
    row_partition,
    col_partition,
    share_B: bool = False,
    validate: bool = True,
    get_subproblem_variable=None,
    get_subproblem_constraint=None,
):
    """Slice a block angular sparse matrix into submatrices.

    This slices a block angular sparse matrix into submatrices.
    A block angular matrix P has the following shape
    (after appropriate reordering):

           ┌                                      ┐
           │  B_{0}                       E_{0}   │
           │         B_{1}                E_{1}   │
       P = │                ...           ...     │
           │                     B_{S-1}  E_{S-1} │
           │  D_{0}  D_{1}  ...  D_{S-1}  B_{-1}  │
           └                                      ┘

    This accepts an unordered matrix with this form.
    Additionally, this requires `row_partition` and `col_partition`,
    which specifies submatrices elements of P belong to.
    Namely, if row i belongs to one of D_{*}, row_partition[i]
    should be -1.  If row i belongs to B_{x}, row_partition[i]
    should be x.
    Similarly, if column j belongs to one of B_{y},
    col_partition[j] should be y.  And if column j belongs
    to one of E_{*}, col_partition[j] should be -1.
    If P is a constraint matrix of a (mixed-integer) linear programming
    with a decomposable structure, thie partition corresponds
    to a subproblem where variables and constraints live in.

      c.p.      0  0  0   1  1  1           S-1 S-1 S-1  -1 -1 -1      r.p.
           ┌                                                       ┐
           │                                                       │    0
           │      B_{0}                                  E_{0}     │    0
           │                                                       │    0
           │                                                       │
           │                                                       │    1
           │                B_{1}                        E_{1}     │    1
           │                                                       │    1
           │                                                       │
           │                                                       │
       P = │                          ...                          │
           │                                                       │
           │                                                       │
           │                                                       │   S-1
           │                                   B_{S-1}   E_{S-1}   │   S-1
           │                                                       │   S-1
           │                                                       │
           │                                                       │   -1
           │     D_{0}      D_{1}     ...      D_{S-1}   B_{-1}    │   -1
           │                                                       │   -1
           └                                                       ┘

    Given above P, row_partition and col_partition, this slices P
    and returns list of Bs, Ds and Es.  D[i]/E[i] corresponds to
    D_{i}/E_{i} (including i=-1/n_blocks) and B[i] corresponds
    to B_{i}.

    If additional (optional) parameter `share_B` is set to be True,
    this assumes all B_{i} has exactly the same structure
    (so B_{i} == B_{j}).

    Parameters
    ----------
    P : (N,M) sparse matrix
        A matrix to be sliced.
    n_blocks : int
        S in the above document.
    row_partition : (N,) array of int
        Each element should be between -1 and num_subproblems (inclusive).
        -1 and num_subproblems indicate corresponding rows belong
        to the linking constraint.
    col_partition : (M,) array of int
        Each element should be between -1 and num_subproblems (inclusive).
        -1 and num_subproblems indicate corresponding rows belong
        to the linking variable.
    share_B : bool, default False
    validate : bool, default True

    Returns
    -------
    B : (n_blocks+1)-list of sparse matrix
    D : n_blocks-list of sparse matrix
    E : n_blocks-list of sparse matrix

    Examples
    --------
    >>> P = np.array([
    ...     [0, 5, 0, 0, 0],
    ...     [0, 4, 0, 0, 0],
    ...     [1, 2, 3, 0, 9],
    ...     [0, 0, 0, 6, 0],
    ...     [0, 0, 0, 7, 8],
    ... ])
    >>> n_blocks = 2
    >>> row_partition = np.array([0, 0, -1, 1, 1])
    >>> col_partition = np.array([0, 0, -1, 1, 1])
    >>> B, D, E = slice_block_angular_matrix(
    ...     P, n_blocks, row_partition, col_partition
    ... )
    >>> print(B[0].toarray())
    [[0 5]
     [0 4]]
    >>> print(B[1].toarray())
    [[6 0]
     [7 8]]
    >>> print(D[0].toarray())
    [[1 2]]
    >>> print(D[1].toarray())
    [[0 9]]
    >>> print(D[-1].toarray())
    [[3]]

    >>> P = np.array([
    ...     [0, 5, 0,10,11],
    ...     [0, 4, 0, 0, 0],
    ...     [1, 2, 3, 0, 9],
    ...     [0, 0, 0, 6, 0],
    ...     [0, 0, 0, 7, 8],
    ... ])
    >>> n_blocks = 2
    >>> row_partition = np.array([0, 0, -1, 1, 1])
    >>> col_partition = np.array([0, 0, -1, 1, 1])
    >>> B, D, E = slice_block_angular_matrix(
    ...     P, n_blocks, row_partition, col_partition,
    ... )
    Traceback (most recent call last):
     ...
    ValueError: invalid partitions
    item 10 at ((0, 3)) has row partition 0 but column partition 1
    item 11 at ((0, 4)) has row partition 0 but column partition 1

    >> print(E[0].toarray())
    """
    num_rows, num_columns = P.shape
    if row_partition.shape != (num_rows,):
        raise ValueError(
            f"invalid shape of row_partition;  expected {(num_rows,)} "
            f"but found {row_partition.shape}"
        )
    if col_partition.shape != (num_columns,):
        raise ValueError(
            f"invalid shape of col_partition;  expected {(num_columns,)} "
            f"but found {col_partition.shape}"
        )
    row_partition[row_partition == n_blocks] = -1
    col_partition[col_partition == n_blocks] = -1
    range_0_to_minus_one = list(range(n_blocks)) + [-1]
    # full_to_subproblem_constraint: (N,) array
    # full_to_subproblem_constraint[i] is a row index of the i-th element
    # within the submatrix.
    full_to_subproblem_constraint = np.full((num_rows,), -1)
    # full_to_subproblem_variable: (M,) array
    # full_to_subproblem_variable[i] is a column index of
    # the i-th element within the submatrix.
    full_to_subproblem_variable = np.full((num_columns,), -1)
    if (get_subproblem_variable is None) or (
        get_subproblem_constraint is None
    ):
        # get_subproblem_* : (n_blocks+1) list of tuple.
        (
            get_subproblem_variable,
            get_subproblem_constraint,
        ) = get_subproblem_variable_and_subproblem_constraint(
            n_blocks, row_partition, col_partition
        )
    row_filter_sum = [
        get_subproblem_constraint[s].size for s in range_0_to_minus_one
    ]
    column_filter_sum = [
        get_subproblem_variable[s].size for s in range_0_to_minus_one
    ]
    for getter in get_subproblem_constraint:
        full_to_subproblem_constraint[getter] = np.arange(getter.size)
    for getter in get_subproblem_variable:
        full_to_subproblem_variable[getter] = np.arange(getter.size)
    if isinstance(P, np.ndarray):
        P = sparse.coo_matrix(P)
    else:
        P = P.tocoo()
    row = P.row
    column = P.col
    data = P.data

    # (len_data,) array
    # row_submatrix_number[i] = number (subscript of submatrix) of
    # subproblem of a corresponding data.
    row_submatrix_number = row_partition[row]
    # (len_data,) array
    # row_submatrix_index[i] = row index of a corresponding element
    # within a submatrix.
    row_submatrix_index = full_to_subproblem_constraint[row]
    # (len_data,) array
    # column_submatrix_number[i] = number (subscript of submatrix) of
    # subproblem of a corresponding data.
    column_submatrix_number = col_partition[column]
    # (len_data,) array
    # column_submatrix_index[i] = column index of a corresponding
    # element within a submatrix.
    column_submatrix_index = full_to_subproblem_variable[column]

    if validate:
        invalid = np.nonzero(
            (row_submatrix_number >= 0)
            & (column_submatrix_number >= 0)
            & (row_submatrix_number != column_submatrix_number)
        )[0]
        if len(invalid) > 0:
            invalid_row = row[invalid]
            invalid_column = column[invalid]
            invalid_data = data[invalid]
            invalid_row_parition = row_submatrix_number[invalid]
            invalid_column_parition = column_submatrix_number[invalid]
            iter = list(
                zip(
                    invalid_row,
                    invalid_column,
                    invalid_data,
                    invalid_row_parition,
                    invalid_column_parition,
                )
            )
            if len(invalid) > 10:
                suffix = [f"truncated {len(invalid) - 10} items"]
                iter = list(iter)[:10]
            else:
                suffix = []
            msg = "\n".join(
                f"item {d} at ({r, c}) has row partition {rp} "
                f"but column partition {cp}"
                for r, c, d, rp, cp in iter
            )
            if suffix:
                msg += "\n" + "\n".join(suffix)
            raise ValueError(f"invalid partitions\n{msg}")

    B = []
    for s in list(range(n_blocks)) + [-1]:
        _filter = (row_submatrix_number == s) & (column_submatrix_number == s)
        _row = row_submatrix_index[_filter]
        _column = column_submatrix_index[_filter]
        _data = data[_filter]
        _num_rows = row_filter_sum[s]
        _num_columns = column_filter_sum[s]
        B.append(
            sparse.csr_matrix(
                (_data, (_row, _column)), shape=(_num_rows, _num_columns)
            )
        )

    D = []
    for s in list(range(n_blocks)) + [-1]:
        _filter = (row_submatrix_number == -1) & (column_submatrix_number == s)
        _row = row_submatrix_index[_filter]
        _column = column_submatrix_index[_filter]
        _data = data[_filter]
        _num_rows = row_filter_sum[-1]
        _num_columns = column_filter_sum[s]
        D.append(
            sparse.csr_matrix(
                (_data, (_row, _column)), shape=(_num_rows, _num_columns)
            )
        )

    E = []
    for s in list(range(n_blocks)) + [-1]:
        _filter = (row_submatrix_number == s) & (column_submatrix_number == -1)
        _row = row_submatrix_index[_filter]
        _column = column_submatrix_index[_filter]
        _data = data[_filter]
        _num_rows = row_filter_sum[s]
        _num_columns = column_filter_sum[-1]
        E.append(
            sparse.csr_matrix(
                (_data, (_row, _column)), shape=(_num_rows, _num_columns)
            )
        )

    return B, D, E


if __name__ == "__main__":
    import doctest

    doctest.testmod()

# vimquickrun: python %
