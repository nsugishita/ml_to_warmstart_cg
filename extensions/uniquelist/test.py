# -*- coding: utf-8 -*-

"""Description of this file"""

import numpy as np
import uniquelist


def main():
    test_int_list()
    test_array_list()


def test_int_list():
    lst = uniquelist.UniqueList()
    np.testing.assert_equal(lst.size(), 0)
    x = lst.push_back(2)
    np.testing.assert_equal(x, (0, True))
    x = lst.push_back(1)
    np.testing.assert_equal(x, (1, True))
    x = lst.push_back(2)
    np.testing.assert_equal(x, (0, False))
    x = lst.push_back(3)
    np.testing.assert_equal(x, (2, True))
    x = lst.push_back(5)
    np.testing.assert_equal(x, (3, True))
    np.testing.assert_equal(lst.size(), 4)
    np.testing.assert_equal(lst.index(2), 0)
    np.testing.assert_equal(lst.index(3), 2)
    np.testing.assert_equal(lst.index(4), -1)
    lst.erase_nonzero([0, 1, 0, 1])
    np.testing.assert_equal(lst.size(), 2)


def test_array_list():
    lst = uniquelist.UniqueArrayList(3)
    np.testing.assert_equal(lst.size(), 0)
    x = lst.push_back([0, 1.5, 2])
    np.testing.assert_equal(x, (0, True))
    x = lst.push_back([2, 1, 2.1])
    np.testing.assert_equal(x, (1, True))
    x = lst.push_back([-1, 0.8, 0])
    np.testing.assert_equal(x, (2, True))
    x = lst.push_back([2, 1, 2.1])
    np.testing.assert_equal(x, (1, False))
    x = lst.push_back([-1, 0.8, -1])
    np.testing.assert_equal(x, (3, True))
    np.testing.assert_equal(lst.size(), 4)
    # lst.erase_nonzero([1, 0, 0, 1])
    lst.erase([0, 3])
    np.testing.assert_equal(lst.size(), 2)
    x = lst.push_back([2, 1, 2.1])
    np.testing.assert_equal(x, (0, False))
    x = lst.push_back([-1, 0.8, -1])
    np.testing.assert_equal(x, (2, True))
    np.testing.assert_equal(lst.size(), 3)


if __name__ == "__main__":
    main()
