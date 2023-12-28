from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal

from rbnet.base import SequentialRBN
from rbnet.pcfg import (AbstractedPCFG, DiscreteNonTermVar, DiscretePrior, DiscreteBinaryNonTerminalTransition,
                        DiscreteTerminalTransition, StaticCell)
from rbnet.util import Prob


class TestPCFG(TestCase):

    binary_grammar_chart = np.array([
        [1 / 2 ** 7, 0, 1 / 2 ** 7],
        [0, 0, 0], [1 / 2 ** 5, 0, 1 / 2 ** 5],
        [0, 0, 0], [0, 0, 0], [1 / 2 ** 3, 0, 1 / 2 ** 3],
        [0, 1 / 2, 0], [0, 1 / 2, 0], [0, 1 / 2, 0], [0, 0, 1 / 2],
    ])

    def test_abstracted_pcfg(self):
        pcfg = AbstractedPCFG(non_terminals="SAB", terminals="ab", start="S", rules=[
            ("S --> A B", 1), ("S --> B A", 1),  # prior + first transition
            ("A --> B A", 1), ("B --> A B", 1),  # non-terminal transitions
            ("A --> a", 1), ("B --> b", 1),      # terminal transition
        ],
                              prob_rep=Prob)
        for n, p in pcfg.named_parameters():
            print(n, p)
        self.assertEqual(pcfg.inside(sequence="aaaa"), 0)
        self.assertEqual(pcfg.inside(sequence="bbbb"), 0)
        self.assertEqual(pcfg.inside(sequence="bbba"), 1 / 2 ** 7)
        self.assertEqual(pcfg.inside(sequence="aaab"), 1 / 2 ** 7)
        # print(pcfg.inside_chart[0].pretty())
        assert_array_equal(pcfg.inside_chart[0].arr.detach(), self.binary_grammar_chart)

    def test_expanded_pcfg(self):
        # same grammar as above but expanded
        zero_var = DiscreteNonTermVar(cardinality=1)
        one_var = DiscreteNonTermVar(cardinality=1)
        prior = DiscretePrior(struc_weights=[0.5, 0.5], prior_weights=[[1.], [1.]])

        zero_non_term_transition = DiscreteBinaryNonTerminalTransition(weights=[[[1.]]], left_idx=1, right_idx=0)
        one_non_term_transition = DiscreteBinaryNonTerminalTransition(weights=[[[1.]]], left_idx=0, right_idx=1)

        zero_term_transition = DiscreteTerminalTransition(weights=[[1.]], term_idx=0)
        one_term_transition = DiscreteTerminalTransition(weights=[[1.]], term_idx=1)

        zero_cell = StaticCell(variable=zero_var,
                               weights=[0.5, 0.5],
                               transitions=[zero_non_term_transition, zero_term_transition])
        one_cell = StaticCell(variable=one_var,
                              weights=[0.5, 0.5],
                              transitions=[one_non_term_transition, one_term_transition])

        rbn = SequentialRBN(cells=[zero_cell, one_cell], prior=prior)

        # aaaa
        self.assertEqual(rbn.inside(sequence=[[0, None],
                                              [0, None],
                                              [0, None],
                                              [0, None]]), 0)
        # bbbb
        self.assertEqual(rbn.inside(sequence=[[None, 0],
                                              [None, 0],
                                              [None, 0],
                                              [None, 0]]), 0)
        # bbba
        self.assertEqual(rbn.inside(sequence=[[None, 0],
                                              [None, 0],
                                              [None, 0],
                                              [0, None]]), 1 / 2 ** 8)
        # aaab
        self.assertEqual(rbn.inside(sequence=[[0, None],
                                              [0, None],
                                              [0, None],
                                              [None, 0]]), 1 / 2 ** 8)
        # print(rbn.inside_chart[0].pretty())
        # print(rbn.inside_chart[1].pretty())
        assert_array_equal(rbn.inside_chart[0].arr[:, 0].detach(), self.binary_grammar_chart[:, 1])
        assert_array_equal(rbn.inside_chart[1].arr[:, 0].detach(), self.binary_grammar_chart[:, 2])
