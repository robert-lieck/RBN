from unittest import TestCase

from rbnet.util import normalize_non_zero as rbn_normalize_non_zero
from pyulib import normalize_non_zero as pyulib_normalize_non_zero


class TestUtil(TestCase):
    def test_normalize_non_zero(self):
        # test code equality against https://github.com/robert-lieck/pyulib
        self.assertEqual(rbn_normalize_non_zero.__code__.co_code,
                         pyulib_normalize_non_zero.__code__.co_code)

