from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal
from triangularmap import TMap

from rbnet.util import normalize_non_zero as rbn_normalize_non_zero, TupleTMap
from pyulib import normalize_non_zero as pyulib_normalize_non_zero


class TestUtil(TestCase):
    def test_normalize_non_zero(self):
        # test code equality against https://github.com/robert-lieck/pyulib
        self.assertEqual(rbn_normalize_non_zero.__code__.co_code,
                         pyulib_normalize_non_zero.__code__.co_code)

    def test_tuple_map(self):
        s = TupleTMap.size_from_n(3)
        ttmap = TupleTMap((np.zeros((s, 2)), np.zeros((s, 3))))
        ttmap[0, 2] = [1, 2], [3, 4, 5]
        assert_array_equal(ttmap.arr[0], [[0, 0], [1, 2], [0, 0], [0, 0], [0, 0], [0, 0]])
        assert_array_equal(ttmap.arr[1], [[0, 0, 0], [3, 4, 5], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])

        ttmap = TupleTMap(([0, 1, 2], [3, 4, 5]))
        # for p in ttmap.pretty():
        #     print(p)
        self.assertEqual(ttmap.pretty(), ("""   ╱╲   \n"""
                                          """  ╱ 0╲  \n"""
                                          """ ╱╲  ╱╲ \n"""
                                          """╱ 1╲╱ 2╲""",
                                          """   ╱╲   \n"""
                                          """  ╱ 3╲  \n"""
                                          """ ╱╲  ╱╲ \n"""
                                          """╱ 4╲╱ 5╲"""))