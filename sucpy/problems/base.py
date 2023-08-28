# -*- coding: utf-8 -*-

"""Utility functions to handle optimization problem data

This provides utility functions to handle optimization
problem data.

The most standard problem data assumes the following items.

name in program           symbol       description
------------------------  -----------  ----------------------------------------
n_variables                            int
n_constraints                          int
objective_sense           min max      {'min', 'max'}
objective_offset          offset       float
variable_lb               x_lb         (n_vars,) array of float
variable_ub               x_ub         (n_vars,) array of float
variable_type             C B          (n_vars,) array o {'C', 'B'}
objective_coefficient     c            (n_vars,) array of float
constraint_coefficient    A            (n_cons, n_vars) COO
constraint_rhs            b            (n_cons,) array of float
constraint_sense          <= = >=      (n_cons,) array of {'E', 'G', 'L'}
------------------------  -----------  ----------------------------------------

This models the following problem.

{ min, max }  c^T x + offset
s.t.          Ax { <=, =, >= } b
              x_lb <= x <= x_ub,
              x : { C, B }  (continuous or binary).

Optionally the problem data may have the following items.

name in program           description
------------------------  -----------------------------------------------------
variable_name_to_index    dict[str, array] to map a variable name to index
constraint_name_to_index  dict[str, array] to map a constraint name to index
variable_index_to_name    array of str to map a variable index to name
constraint_index_to_name  array of str to map a constraint index to name
------------------------  -----------------------------------------------------

If the problem is decomposable, one can specify the following items to indicate
how the problem is structured.

name in program               description
---------------------------   -------------------------------------------------
n_subproblems                 int of the number of subproblems
variable_subproblem_index     (n_vars,) array of int
                              Index of subproblems of the corresponding
                              variables. -1 indicates the ones which does not
                              corresponds to any subproblems.
constraint_subproblem_index   (n_constraints,) array of int
                              Index of subproblems of the corresponding
                              constraints. -1 indicates the ones which does not
                              corresponds to any subproblems.
---------------------------   -------------------------------------------------

Then, `set_decomposition_data` takes the data and creates the following items.

name in program                               description
--------------------------------------------  ---------------------------------
subproblem_index                              np.arange(n_subprobs)
subproblem_index_with_negative_one            list(range(n_subprobs)) + [-1]
subproblem_variable_lb                        (n_subprobs+1)-list of array
subproblem_variable_ub                        (n_subprobs+1)-list of array
subproblem_variable_type                      (n_subprobs+1)-list of array
subproblem_objective_coefficient              (n_subprobs+1)-list of array
subproblem_constraint_coefficient_diagonal    (n_subprobs+1)-list of COO
subproblem_constraint_coefficient_col_border  (n_subprobs+1)-list of COO
subproblem_constraint_rhs                     (n_subprobs+1)-list of array
subproblem_constraint_sense                   (n_subprobs+1)-list array
subproblem_constraint_coefficient_row_border  (n_subprobs+1)-list of COO
--------------------------------------------  ---------------------------------

`subproblem_variable_lb[i]` is a one-dimensional array which is in the problem
data of the i-th subproblem.  Other objects with prefix `subproblem_` is
in the same format.

This models the problem below.

{min|max} sum_{i in Sbar} sub_c[i]^T sub_x[i] + sub_c[0]^T sub_x[-1]

s.t.                 sub_B[i] sub_x[i] + sub_E[i]  sub_x[-1] = sub_b[i]
                                                               (i in S),
     sum_{i in Sbar} sub_D[i] sub_x[i] + sub_D[-1] sub_x[-1] = sub_d,
     sub_x_lb[i] <= sub_x[i] <= sub_x_ub[i]  (i in Sbar),
     sub_x[i] : {'C', 'B'}  (continuous or binary),

where `sub_x[i]` is a variables in the i-th subproblem,
namely `sub_x[i] = x[variable_subproblem_index == i]`.

Note that

sum_{i in Sbar} sub_c[i]^T sub_x[i]
  = sub_c[-1] sub_x[-1] + sum_{i in S} sub_c[i]^T sub_x[i]

The table below shows the list of names in the program and their corresponding
symbol.

name in program                               symbol
--------------------------------------------  ---------------------------------
subproblem_index                              S
subproblem_index_with_negative_one            Sbar
subproblem_variable_lb                        sub_x_lb
subproblem_variable_ub                        sub_x_ub
subproblem_variable_type
subproblem_objective_coefficient              sub_c
subproblem_constraint_coefficient_diagonal    sub_B
subproblem_constraint_coefficient_col_border  sub_E
subproblem_constraint_rhs                     sub_b, sub_d
subproblem_constraint_sense
subproblem_constraint_coefficient_row_border  sub_D
--------------------------------------------  ---------------------------------

"""

from typing import MutableMapping, Sequence, Tuple, Union

import numpy as np
import scipy.sparse as sparse


def validate_problem_data(
    problem_data, decomposable=False, unknown_ok=True, all=False
):
    """Verify given data is valid problem data

    This verifies given data is valid as problem data or not. If the test fails
    this raises a ValueError.

    Parameters
    ----------
    problem_data : dict
    decomposable : bool, default False
        If True, decomposition data is also required.
    unknown_ok : bool, default True
        If False, unknown items are assumed to be error.
    all : bool, default True
        All problem data are required.
    """
    required_items = []
    optional_items = []

    required_items += [
        "n_variables",
        "n_constraints",
        "objective_sense",
        "objective_offset",
        "variable_lb",
        "variable_ub",
        "variable_type",
        "objective_coefficient",
        "constraint_coefficient",
        "constraint_rhs",
        "constraint_sense",
    ]

    optional_items += [
        "variable_name_to_index",
        "constraint_name_to_index",
        "variable_index_to_name",
        "constraint_index_to_name",
    ]

    _items = required_items if decomposable else optional_items

    _items += [
        "variable_subproblem_index",
        "constraint_subproblem_index",
    ]

    _items = required_items if all else optional_items

    _items += [
        "n_subproblems",
        "subproblem_index",
        "subproblem_index_with_negative_one",
        "subproblem_variable_lb",
        "subproblem_variable_ub",
        "subproblem_variable_type",
        "subproblem_objective_coefficient",
        "subproblem_constraint_coefficient_diagonal",
        "subproblem_constraint_coefficient_col_border",
        "subproblem_constraint_rhs",
        "subproblem_constraint_sense",
        "subproblem_constraint_coefficient_row_border",
    ]

    missing_items = []
    unknown_items = []

    for item in required_items:
        if item not in problem_data:
            missing_items.append(item)

    for item in problem_data:
        if item in required_items:
            continue
        if item in optional_items:
            continue
        unknown_items.append(item)

    test = (len(missing_items) > 0) or (
        (not unknown_ok) and (len(unknown_items) > 0)
    )
    if test:
        msg = ""
        if len(missing_items) > 0:
            msg += f"missing items: {', '.join(missing_items)}  "
        if len(unknown_items) > 0:
            msg += f"unknown items: {', '.join(unknown_items)}  "
        raise ValueError(msg.strip())

    return problem_data


def get_indices(
    **kwargs: Union[int, Sequence[int]]
) -> Tuple[MutableMapping, int]:
    """Index items given their shapes.

    This accepts shapes and assigns create sequential indices.
    This can be used to set indices on variables or constraints.

    Returns
    -------
    indices : dict
        Indices of given items.
    size : int
        The total number of elements.

    Examples
    --------
    >>> cols, size = get_indices(var1=(2, 3), var2=2)
    >>> cols['var1']
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> cols['var2']
    array([6, 7])
    >>> size
    8
    """
    start = 0
    ret = dict()
    for name, shape in kwargs.items():
        index = start + np.arange(np.prod(shape)).reshape(shape)
        ret[name] = index
        start = index.ravel()[-1] + 1
    return ret, start


def parse_dense_format(text):
    """Parse a problem description in the dense format.

    Examples
    --------
    >>> text = '''
    ... min    1.   4.   2.   3.
    ... s.t.   2.   1.   1.   3.   =  5.
    ...        1.  -1.  -1.   2.  >=  1.
    ... xlb:   0.   2.   3.   4.
    ... xub:  inf  inf  inf   8.
    ... '''
    >>> data = parse_dense_format(text)
    >>> print(data['variable_lb'])
    [0. 2. 3. 4.]
    >>> print(format_densely(data))
    min   1.  4.  2.  3.
    s.t.  2.  1.  1.  3.  = 5.
          1. -1. -1.  2. >= 1.
    xlb:  0.  2.  3.  4.
    xub: inf inf inf  8.
    """
    import re

    ret = dict()
    if isinstance(text, str):
        lines = text.strip().split("\n")
    else:
        lines = text
    lines = list(filter(lambda x: len(x), lines))
    pattern = re.compile(r"  +")
    for i in range(len(lines)):
        line = lines[i]
        # Remove extre spaces.
        line = pattern.sub(" ", line.strip())
        # Split by spaves.
        lines[i] = line.split(" ")
    if len(lines) <= 4:
        raise ValueError(
            f"expected at least four lines but found {len(lines)}"
        )
    if lines[0][0] not in ["min", "max"]:
        raise ValueError(
            f"expected min or max but found {lines[0][0]} in line 0"
        )
    if lines[1][0] not in ["s.t."]:
        raise ValueError(
            f"expected 's.t.' keyword but found  {lines[1][0]} in line 1"
        )
    if lines[-2][0] not in ["xlb:"]:
        raise ValueError(
            f"expected 'xlb:' keyword but found  {lines[-2][0]} "
            f"in line {len(lines)-2}"
        )
    if lines[-1][0] not in ["xub:"]:
        raise ValueError(
            f"expected 'xub:' keyword but found  {lines[-2][0]} "
            f"in line {len(lines)-2}"
        )
    n_variables = ret["n_variables"] = len(lines[-1]) - 1
    n_constraints = ret["n_constraints"] = len(lines) - 3
    ret["objective_sense"] = lines[0][0]
    if len(lines[0]) >= n_variables + 2:
        ret["objective_offset"] = float(lines[0][n_variables + 1])
    else:
        ret["objective_offset"] = 0.0
    ret["variable_lb"] = np.array(list(map(float, lines[-2][1:])))
    ret["variable_ub"] = np.array(list(map(float, lines[-1][1:])))
    ret["variable_type"] = np.full(
        n_variables, "C".encode()[0], dtype=np.uint8
    )
    ret["objective_coefficient"] = np.array(
        list(map(float, lines[0][1 : n_variables + 1]))
    )
    A = []
    b = []
    constraint_sense = []
    con_sense_mapping = {
        "=": "E".encode()[0],
        ">=": "G".encode()[0],
        "<=": "L".encode()[0],
    }
    for i in range(n_constraints):
        if i == 0:
            coefs = lines[i + 1][1 : n_variables + 1]
        else:
            coefs = lines[i + 1][0:n_variables]
        A.append(list(map(float, coefs)))
        b.append(float(lines[i + 1][-1]))
        constraint_sense.append(con_sense_mapping[lines[i + 1][-2]])
    ret["constraint_coefficient"] = A = sparse.coo_matrix(np.array(A))
    ret["constraint_rhs"] = b = np.array(b)
    ret["constraint_sense"] = constraint_sense = np.array(constraint_sense)
    return ret


def format_densely(problem_data, **kwargs):
    """format_densely the problem.

    Parameters
    ----------
    problem_data : dict
    kwargs
        Passed to numpy.array2string to format_densely arrays.
    """
    obj_sense = (
        "min" if "min" in str(problem_data["objective_sense"]) else "max"
    )
    cA = np.r_[
        "0,2",
        problem_data["objective_coefficient"],
        problem_data["constraint_coefficient"].toarray(),
        problem_data["variable_lb"],
        problem_data["variable_ub"],
    ]
    cA = (
        np.array2string(cA, **kwargs)
        .replace("[", " ")
        .replace("]", "")
        .split("\n")
    )
    cA = list(map(lambda x: x[2:], cA))
    b = (
        np.array2string(problem_data["constraint_rhs"][:, None], **kwargs)
        .replace("[", " ")
        .replace("]", "")
        .split("\n")
    )
    b = list(map(lambda x: x[2:], b))
    constraint_sense = problem_data["constraint_sense"]
    sense_symbols = {"G": ">=", "E": " =", "L": "<="}
    for key in list(sense_symbols):
        sense_symbols[key.encode()[0]] = sense_symbols[key]
    res = cA
    res = [" " * 5 + r for r in res]
    res[0] = obj_sense + res[0][3:]
    offset = problem_data["objective_offset"]
    if offset != 0:
        res[0] += f" {problem_data['objective_offset']:+.1f}"
    res[1] = "s.t." + res[1][4:]
    for i in range(len(res) - 3):
        res[i + 1] += " " + sense_symbols[constraint_sense[i]] + " " + b[i]
    res[-2] = "xlb: " + res[-2][5:]
    res[-1] = "xub: " + res[-1][5:]
    res = "\n".join(res)
    return res


if __name__ == "__main__":
    import doctest

    doctest.testmod()

# vimquickrun: python %
