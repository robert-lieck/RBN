from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from rbnet.base import RBN, SequentialRBN, Transition, Prior, NonTermVar, Cell
from rbnet.pcfg import DiscretePrior, DiscreteBinaryNonTerminalTransition, DiscreteTerminalTransition, StaticCell, \
    DiscreteNonTermVar, AbstractedPCFG
from rbnet.util import Prob


class TestStaticCell(TestCase):

    @staticmethod
    def _get_concrete_class(cls, *args, **kwargs):
        class ConcreteCls(cls):
            pass
        ConcreteCls.__abstractmethods__ = frozenset()
        return ConcreteCls(*args, **kwargs)

    def test_abstract_base_classes(self):
        # RBN
        x = self._get_concrete_class(RBN)
        self.assertRaises(NotImplementedError, lambda: x.inside_chart)
        self.assertRaises(NotImplementedError, lambda: x.terminal_chart)
        self.assertRaises(NotImplementedError, x.inside_schedule)
        self.assertRaises(NotImplementedError, lambda: x.prior)
        self.assertRaises(NotImplementedError, lambda: x.root_location)
        self.assertRaises(NotImplementedError, x.update_inside_chart, None, None, None)
        self.assertIs(x.init_inside(), None)
        # Transition
        x = self._get_concrete_class(Transition)
        self.assertRaises(NotImplementedError, x.inside_marginals, None, None, None)
        # Prior
        x = self._get_concrete_class(Prior)
        self.assertRaises(NotImplementedError, x.marginal_likelihood, None, None)
        # NonTermVar
        x = self._get_concrete_class(NonTermVar)
        self.assertRaises(NotImplementedError, x.get_chart)
        self.assertRaises(NotImplementedError, x.mixture)
        # Cell
        x = self._get_concrete_class(Cell, None)
        self.assertRaises(NotImplementedError, x.transitions)
        self.assertRaises(NotImplementedError, x.inside_mixture, None)

    def test_minimal_grammar(self):
        # a discrete grammar with one variable of cardinality two (two symbols) and uniform transition distributions
        rbn = SequentialRBN(cells=[StaticCell(variable=DiscreteNonTermVar(2),
                                              weights=np.ones(2),
                                              transitions=[
                                                  DiscreteTerminalTransition(weights=np.ones((2, 2)), prob_rep=Prob),
                                                  DiscreteBinaryNonTerminalTransition(weights=np.ones((2, 2, 2)), prob_rep=Prob)
                                              ], prob_rep=Prob)],
                            prior=DiscretePrior(struc_weights=np.ones(1), prior_weights=[np.ones(2)], prob_rep=Prob))
        # parse a random sequence of length N
        N = 5
        marginal_likelihood = rbn.inside(sequence=np.random.randint(0, 2, (N, 1)))
        self.assertNotEqual(marginal_likelihood, 0)

        # check inside probs
        level_insides = np.zeros(N)
        for idx in range(N):
            if idx == 0:
                # prob to terminate * prob for symbol
                level_insides[0] = 0.25
            else:
                # sum over all possible splits times prob to NOT-terminate
                level_insides[idx] = (level_insides[:idx] * np.flip(level_insides[:idx])).sum() * 0.5
            for start in range(0, N - idx):
                end = start + idx + 1
                # print(f"{(start, end)}: {rbnet.inside_chart[0][start, end]}")
                assert_array_equal(rbn.inside_chart[0][start, end].detach(), np.ones(2) * level_insides[idx])

        # check marginal (same as inside of root: we have to sum over two symbols,
        # but prior distribution is 0.5 for both)
        self.assertAlmostEqual(marginal_likelihood, level_insides[-1])

    def test_minimal_multivar_grammar(self):
        # a discrete grammar with two non-terminal variables of cardinality three and four, a terminal variable of
        # cardinality five and uniform transition distributions that which between the two non-terminal variables
        # with equal probability
        rbn = SequentialRBN(cells=[StaticCell(variable=DiscreteNonTermVar(3),
                                              weights=np.ones(3),
                                              transitions=[
                                                  DiscreteTerminalTransition(weights=np.ones((5, 3))),
                                                  DiscreteBinaryNonTerminalTransition(weights=np.ones((3, 3, 3)),
                                                                                      left_idx=0, right_idx=0),
                                                  DiscreteBinaryNonTerminalTransition(weights=np.ones((4, 4, 3)),
                                                                                      left_idx=1, right_idx=1)
                                              ]),
                                   StaticCell(variable=DiscreteNonTermVar(4),
                                              weights=np.ones(3),
                                              transitions=[
                                                  DiscreteTerminalTransition(weights=np.ones((5, 4))),
                                                  DiscreteBinaryNonTerminalTransition(weights=np.ones((4, 4, 4)),
                                                                                      left_idx=1, right_idx=1),
                                                  DiscreteBinaryNonTerminalTransition(weights=np.ones((3, 3, 4)),
                                                                                      left_idx=0, right_idx=0)
                                              ])
                                   ],
                            prior=DiscretePrior(struc_weights=np.ones(2), prior_weights=[np.ones(3), np.ones(4)]))
        # parse a random sequence of length N
        N = 5
        marginal_likelihood = rbn.inside(sequence=np.random.randint(0, 5, (N, 1))).detach().numpy()
        self.assertNotEqual(marginal_likelihood, 0)

        # check inside probs
        level_insides = np.zeros((N, 2))
        for level_idx in range(N):
            if level_idx == 0:
                # prob to terminate * prob for symbol
                level_insides[0, :] = 1 / 3 * 1 / 5
            else:
                # sum over all possible splits times prob to NOT-terminate
                level_insides[level_idx] = (level_insides[:level_idx] * np.flip(level_insides[:level_idx])).sum() / 3
            for start in range(0, N - level_idx):
                end = start + level_idx + 1
                # print(f"{(start, end)}")
                for var_idx, card in enumerate([3, 4]):
                    assert_array_almost_equal(rbn.inside_chart[var_idx][start, end].detach(),
                                              np.ones(card) * level_insides[level_idx, var_idx])

        # check marginal (sum over inside of root, weighted by prior)
        self.assertAlmostEqual(marginal_likelihood, level_insides[-1].sum() / 2)

    def test_counting_grammar(self):
        # a grammar with K symbols that "counts" upwards or downwards
        K = 10
        K_range = np.arange(K)

        # prior deterministically generates 0
        prior_weights = np.zeros(K)
        prior_weights[0] = 1
        prior = DiscretePrior(struc_weights=np.ones(1), prior_weights=[prior_weights], prob_rep=Prob)
        assert_array_equal(prior.structural_distribution.p.detach(), [1])
        self.assertEqual(len(prior.prior_distributions), 1)
        assert_array_equal(prior.prior_distributions[0].p.detach(), [1] + [0] * (K - 1))

        # non-terminal transitions copy parent value to left child and count right child up or down
        non_terminal_weights = np.zeros((K, K, K))
        non_terminal_weights[np.arange(K),
                             np.clip(np.arange(K) + 1, 0, K - 1),  # clip: don't overshoot
                             np.arange(K)] = 1
        non_terminal_weights[np.arange(K),
                             np.clip(np.arange(K) - 1, 0, K - 1),  # clip: don't undershoot
                             np.arange(K)] = 1
        non_terminal_transition = DiscreteBinaryNonTerminalTransition(weights=non_terminal_weights, prob_rep=Prob)
        # check off-diagonals and corners
        assert_array_equal(non_terminal_transition.transition_probabilities.p[K_range[:-1], K_range[1:], K_range[:-1]].detach(),
                           [0.5] * (K - 1))
        assert_array_equal(non_terminal_transition.transition_probabilities.p[K_range[1:], K_range[:-1], K_range[1:]].detach(),
                           [0.5] * (K - 1))
        assert_array_equal(non_terminal_transition.transition_probabilities.p[[0, K - 1], [0, K - 1], [0, K - 1]].detach(),
                           [0.5, 0.5])

        # terminal transition copies last non-terminal
        terminal_weights = np.eye(K)
        terminal_transition = DiscreteTerminalTransition(weights=terminal_weights, prob_rep=Prob)
        # print(terminal_transition.transition_probabilities.T)

        cell = StaticCell(variable=DiscreteNonTermVar(K),
                          weights=np.array([0.4, 0.6]),
                          transitions=[terminal_transition, non_terminal_transition],
                          prob_rep=Prob)
        term_prob, non_term_prob = cell.transition_probabilities.p[:, 0]

        rbn = SequentialRBN(cells=[cell], prior=prior)

        # parse sequence of length N
        terminals = [[0], [1], [2], [3]]
        N = len(terminals)
        marginal_likelihood = rbn.inside(sequence=terminals).detach().numpy()
        self.assertNotEqual(marginal_likelihood, 0)

        # compute non-zero entries of inside prob, assuming there is only one way to generate the given sequence in a
        # left-to-right fashion
        level_insides = np.zeros(N)
        for idx in range(N):
            if idx == 0:
                # prob to terminate
                level_insides[0] = term_prob
            else:
                # inside of left subtree (containing all but the right-most terminal) * inside of the right-most
                # terminal * prob to go up/down * prob to NOT-terminate
                level_insides[idx] = level_insides[idx - 1] * level_insides[0] * 0.5 * non_term_prob

        # check inside probs
        for idx in range(N):
            for start in range(0, N - idx):
                end = start + idx + 1
                insides = rbn.inside_chart[0][start, end].detach().numpy()
                try:
                    self.assertTrue(np.all(np.logical_or(np.isclose(insides, level_insides[idx]), insides == 0)))
                except AssertionError:
                    print(f"(start, end): {(start, end)}")
                    print(f"idx: {idx}")
                    print(f"insides: {insides}")
                    print(f"level_insides: {level_insides}")
                    raise

        # check marginal (same as inside of root: first symbol deterministically generated by prior)
        self.assertAlmostEqual(marginal_likelihood, level_insides[-1])

    def test_letter_pcfg(self):
        # equivalent of counting grammar, but with letters
        pcfg = AbstractedPCFG(non_terminals="ABCDE",
                              terminals="ABCDE",
                              rules=[(f"{lhs} --> {l_child} {r_child}", 1) for lhs, l_child, r_child in
                                     ["AAB", "BBC", "CCD", "DDE", "EEE"]] +
                                    [((lhs, (l_child, r_child)), 1) for lhs, l_child, r_child in
                                     ["AAA", "BBA", "CCB", "DDC", "EED"]] +
                                    [(f"{x} --> {x}", 1) for x in "ABCDE"],
                              start="A",
                              prob_rep=Prob)
        terminals = "ABCDE"
        N = len(terminals)
        term_prob, non_term_prob = pcfg._cells[0].transition_probabilities.p[:, 0]
        marginal_likelihood = pcfg.inside(sequence=terminals).detach().numpy()
        self.assertNotEqual(marginal_likelihood, 0)

        # compute non-zero entries of inside prob, assuming there is only one way to generate the given sequence in a
        # left-to-right fashion
        level_insides = np.zeros(N)
        for idx in range(N):
            if idx == 0:
                # prob to terminate
                level_insides[0] = term_prob
            else:
                # inside of left subtree (containing all but the right-most terminal) * inside of the right-most
                # terminal * prob to go up/down * prob to NOT-terminate
                level_insides[idx] = level_insides[idx - 1] * level_insides[0] * 0.5 * non_term_prob

        # check inside probs
        for idx in range(N):
            for start in range(0, N - idx):
                end = start + idx + 1
                insides = pcfg.inside_chart[0][start, end].detach().numpy()
                try:
                    self.assertTrue(np.all(np.logical_or(np.isclose(insides, level_insides[idx]), insides == 0)))
                except AssertionError:
                    print(f"(start, end): {(start, end)}")
                    print(f"insides: {insides}")
                    print(f"level_insides: {level_insides}")
                    raise

        # check marginal (same as inside of root: first symbol deterministically generated by prior)
        self.assertAlmostEqual(marginal_likelihood, level_insides[-1])

    def test_word_pcfg(self, verbose=False):
        subjects = ["I", "You", "We", "They"]
        verbs = ["run", "drink", "sleep"]
        adverb_non_gradable = ["a-lot", "alone"]
        adverb_gradable = ["fast", "slowly", "quickly"]
        grade = ["very", "veeery", "really"]
        verb_qualifier = ["rarely", "do-not", "never", "always"]
        non_terminals = ["start", "subject", "verb", "gradable_adverb", "non_gradable_adverb", "verb_qualifier", "grade"]
        pcfg = AbstractedPCFG(non_terminals=non_terminals,
                              terminals=subjects + verbs + adverb_non_gradable +
                                        adverb_gradable + grade + verb_qualifier,
                              rules=[("start --> subject verb", 1),
                                     ("verb --> verb_qualifier verb", 1),
                                     ("verb --> verb gradable_adverb", 1),
                                     ("verb --> verb non_gradable_adverb", 1),
                                     ("gradable_adverb --> grade gradable_adverb", 1),
                                     ("grade --> grade grade", 1),
                                     ] + [(f"{n} --> {t}", 1) for n, ts in zip(non_terminals[1:],
                                                                               [subjects, verbs, adverb_gradable,
                                                                                adverb_non_gradable, verb_qualifier,
                                                                                grade]) for t in ts],
                              start="start",
                              prob_rep=Prob)
        # grammatical sentences
        for terminals in [
            "I run",
            "You never run",
            "We run very very slowly",
            "They always run alone",
            "I never sleep really very quickly",
            "You do-not drink very quickly",
        ]:
            marginal_likelihood = pcfg.inside(sequence=terminals.split())
            self.assertGreater(marginal_likelihood, 0)
            if verbose:
                print(pcfg.map_inside_chart().pretty())
                print(marginal_likelihood, terminals)
        # un-grammatical sentences
        for terminals in [
            "I You",
            "run fast"
        ]:
            marginal_likelihood = pcfg.inside(sequence=terminals.split())
            self.assertEqual(marginal_likelihood.detach().numpy(), 0)
            if verbose:
                print(marginal_likelihood, terminals)