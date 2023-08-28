# -*- coding: utf-8 -*-

"""Routines used across solvers"""

import numpy as np


def to_str(a):
    """Create a str

    Examples
    --------
    >>> a = [ 65,  80, 120]
    >>> to_str(a)
    'APx'
    >>> to_str(["A", "P", "x"])
    'APx'
    >>> to_str('APx')
    'APx'

    Parameters
    ----------
    a : str, or array-like of str

    Returns
    -------
    res : str
    """
    if isinstance(a, str):
        return a
    a = np.atleast_1d(a)
    np.testing.assert_equal(a.ndim, 1)
    if "U" in str(a.dtype):
        return "".join(a)
    return bytes(list(a)).decode("utf8")


def to_str_array(a):
    """Create an array of str

    Examples
    --------
    >>> a = [ 65,  80, 120]
    >>> print(to_str_array(a))
    ['A' 'P' 'x']
    >>> print(to_str_array(["A", "P", "x"]))
    ['A' 'P' 'x']
    >>> print(to_str_array('APx'))
    ['A' 'P' 'x']

    Parameters
    ----------
    a : str, or array-like of str

    Returns
    -------
    res : str
    """
    return np.array(list(to_str(a)))


def to_byte_array(a):
    """Create an array of byte

    Examples
    --------
    >>> print(to_byte_array("APx"))
    [ 65  80 120]
    >>> print(to_byte_array(list("APx")))
    [ 65  80 120]
    >>> a = to_byte_array("APx")
    >>> b = to_byte_array(a)
    >>> a is b
    True

    Parameters
    ----------
    a : str, or array-like of str

    """
    if isinstance(a, str):
        a = np.array(list(a))
    else:
        a = np.asarray(a)
    if "U" in str(a.dtype):
        a = np.array(list("".join(a).encode("utf8")))
    return a


if __name__ == "__main__":
    import doctest

    doctest.testmod()
