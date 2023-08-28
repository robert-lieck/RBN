from typing import Iterable

import numpy as np

from triangularmap import TMap

from rbnet.util import normalize_non_zero
from rbnet.base import Cell, Transition, Prior, NonTermVar, CYKRBN


class PCFG(CYKRBN):

    def __init__(self, cells, prior, terminal_indices, non_terminals):
        self.terminal_indices = terminal_indices
        self._non_terminals = non_terminals
        super().__init__(cells=cells, prior=prior)

    def init_inside(self, sequence):
        sequence = [self.terminal_indices[s] for s in sequence]
        super().init_inside(sequence=sequence)

    def map_inside_chart(self, precision=None):
        new_arr = []
        for probs in self.inside_chart[0]._arr:
            field = []
            for idx, p in enumerate(probs):
                if p > 0:
                    field.append(f"{self._non_terminals[idx]}|{np.format_float_scientific(p, precision=precision)}")
            if len(field) == 1:
                new_arr.append(field[0])
            else:
                new_arr.append(field)
        return TMap(new_arr)


class AbstractedPCFG(PCFG):

    def __init__(self, non_terminals, terminals, rules, start):
        """
        An AbstractedPCFG defines an RBN that has only one non-terminal and one terminal variable
        with the cardinality of the non-terminal and terminal symbols, respectively, of the PCFG.

        :param non_terminals: list or array of non-terminal symbols
        :param terminals: list or array of terminal symbols
        :param rules: iterable of rules-weight tuples with rules provided either as strings of the form ("X --> Y Z", w)
         or ("X --> Y", w) for non-terminal and terminal rules, respectively (symbols have to be strings without
         whitespace for this), or of the form ((X, (Y, Z)), w) or ((X, (Y,)), w) for arbitrary symbols, where w is the
         rule weight.
        :param start: the start symbol
        """
        self.terminals = terminals
        self.non_terminal_indices = {s: i for i, s in enumerate(non_terminals)}
        terminal_indices = {s: i for i, s in enumerate(terminals)}
        non_term_size = len(non_terminals)
        term_size = len(self.terminals)

        # deterministic prior
        prior_weights = np.zeros(non_term_size)
        prior_weights[self.non_terminal_indices[start]] = 1
        prior = DiscretePrior(struc_weights=np.ones(1), prior_weights=[prior_weights])

        # transition probabilities
        non_terminal_weights = np.zeros((non_term_size, non_term_size, non_term_size))
        terminal_weights = np.zeros((term_size, non_term_size))
        for r, w in rules:
            if isinstance(r, str):
                s = r.split()
                if not (len(s) in [3, 4] and s[1] == "-->"):
                    raise ValueError(f"Rules must be of the form 'X --> Y' or 'X --> Y Z', but got '{r}'")
                else:
                    lhs = s[0]
                    rhs = s[2:]
            else:
                lhs, rhs = r
            if len(rhs) == 1:
                # terminal transition
                rhs = rhs[0]
                terminal_weights[terminal_indices[rhs], self.non_terminal_indices[lhs]] = w
            elif len(rhs) == 2:
                # non-terminal transition
                left_child, right_child = rhs
                non_terminal_weights[self.non_terminal_indices[left_child],
                                     self.non_terminal_indices[right_child],
                                     self.non_terminal_indices[lhs]] = w
            else:
                raise ValueError(f"Right-hand side of rules must consist of one (terminal) or two (non-terminal) "
                                 f"symbols; this one has {len(rhs)} symbols: rhs={rhs}, lhs={lhs}, w={w}")

        non_terminal_transition = DiscreteBinaryNonTerminalTransition(weights=non_terminal_weights)
        terminal_transition = DiscreteTerminalTransition(weights=terminal_weights)
        cell = StaticCell(variable=DiscreteNonTermVar(non_term_size),
                          weights=np.array([terminal_weights.sum(), non_terminal_weights.sum()]),
                          transitions=[terminal_transition, non_terminal_transition])
        super().__init__(cells=[cell], prior=prior, non_terminals=non_terminals, terminal_indices=terminal_indices)


class ExpandedPCFG(PCFG):
    # INCOMPLETE / WORK IN PROGRESS / TODO!!

    def __init__(self, non_terminals, terminals, non_term_variables, rules, start):
        """
        An ExpandedPCFG...

        :param non_terminals: list or array of non-terminal symbols
        :param terminals: list or array of terminal symbols
        :param non_term_variables: list of non-terminal variables corresponding to the non-terminal symbols
        :param rules: iterable of rules-weight-transition tuples with rules provided either as strings of the form
         ("X --> Y Z", w, t) or ("X --> Y", w, t) for non-terminal and terminal rules, respectively (symbols have to be
         strings without whitespace for this), or of the form ((X, (Y, Z)), w, t) or ((X, (Y,)), w, t) for arbitrary
         symbols, where w is the rule weight, and t is the transition for the corresponding variables.
        :param start: the start symbol
        """
        self.terminals = terminals
        self.non_terminal_indices = {s: i for i, s in enumerate(non_terminals)}
        terminal_indices = {s: i for i, s in enumerate(terminals)}
        non_term_size = len(non_terminals)
        term_size = len(self.terminals)

        # deterministic prior
        prior_weights = np.zeros(non_term_size)
        prior_weights[self.non_terminal_indices[start]] = 1
        prior = DiscretePrior(struc_weights=np.ones(1), prior_weights=[prior_weights])

        # transition probabilities
        non_terminal_weights = np.zeros((non_term_size, non_term_size, non_term_size))
        terminal_weights = np.zeros((term_size, non_term_size))

        weights = []
        transitions = []
        for r, w in rules:
            if isinstance(r, str):
                s = r.split()
                if not (len(s) in [3, 4] and s[1] == "-->"):
                    raise ValueError(f"Rules must be of the form 'X --> Y' or 'X --> Y Z', but got '{r}'")
                else:
                    lhs = s[0]
                    rhs = s[2:]
            else:
                lhs, rhs = r
            if len(rhs) == 1:
                # terminal transition
                rhs = rhs[0]
                terminal_weights[terminal_indices[rhs], self.non_terminal_indices[lhs]] = w
            elif len(rhs) == 2:
                # non-terminal transition
                left_child, right_child = rhs
                non_terminal_weights[self.non_terminal_indices[left_child],
                self.non_terminal_indices[right_child],
                self.non_terminal_indices[lhs]] = w
            else:
                raise ValueError(f"Right-hand side of rules must consist of one (terminal) or two (non-terminal) "
                                 f"symbols; this one has {len(rhs)} symbols: rhs={rhs}, lhs={lhs}, w={w}")

        non_terminal_transition = DiscreteBinaryNonTerminalTransition(weights=non_terminal_weights)
        terminal_transition = DiscreteTerminalTransition(weights=terminal_weights)
        cell = StaticCell(variable=DiscreteNonTermVar(non_term_size),
                          weights=np.array([terminal_weights.sum(), non_terminal_weights.sum()]),
                          transitions=[terminal_transition, non_terminal_transition])



        prior = DiscretePrior(struc_weights=np.ones(2), prior_weights=[np.ones(3), np.ones(4)])

        cells = []
        for non_term_var, w, t in zip(non_term_variables, weights, transitions):
            cells.append(StaticCell(variable=non_term_var, weights=w, transitions=t))
            cells.append(StaticCell(variable=DiscreteNonTermVar(4),
                                    weights=np.ones(3),
                                    transitions=[
                                        DiscreteTerminalTransition(weights=np.ones((5, 4))),
                                        DiscreteBinaryNonTerminalTransition(weights=np.ones((4, 4, 4)),
                                                                            left_idx=1, right_idx=1),
                                        DiscreteBinaryNonTerminalTransition(weights=np.ones((3, 3, 4)),
                                                                            left_idx=0, right_idx=0)
                                    ]))
        super().__init__(cells=cells, prior=prior, non_terminals=non_terminals, terminal_indices=...)


class DiscreteNonTermVar(NonTermVar):

    def __init__(self, cardinality, chart_type="TMap"):
        """
        A discrete non-terminal variable with cardinality `n`.

        :param cardinality: cardinality
        :param chart_type: type of chart to use ("dict" or "TMap")
        """
        self.cardinality = cardinality
        self.chart_type = chart_type

    def get_chart(self, n, *args, **kwargs):
        """
        Initialise a chart for sequence of length `n`.
        :param n: length of the sequence
        :return: chart
        """
        if self.chart_type == "dict":
            return {}
        elif self.chart_type == "TMap":
            return TMap(np.zeros((TMap.size_from_n(n), self.cardinality)))
        else:
            raise ValueError(f"Unknown chart type '{self.chart_type}'")


class DiscretePrior(Prior):

    """
    A prior distribution over discrete non-terminal variables.
    """

    def __init__(self, struc_weights, prior_weights):
        """
        Initialise prior with structural distribution p(z) and individual prior distributions p(a1), ..., p(an) for
        n non-terminal variables a1, ..., an. The cardinality of z is n.

        :param struc_weights: Numpy array of shape (n,) with weights for the structural distribution
        :param prior_weights: iterable over n Numpy arrays if shapes (K1,), ..., (Kn,) with weights for the prior
         distribution of the n non-terminal variables.
        """
        if isinstance(struc_weights, np.ndarray) and len(struc_weights.shape) == 1:
            if np.any(struc_weights < 0):
                raise ValueError("All weights have to be non-negative")
            self.structural_distributions = struc_weights / np.sum(struc_weights)
        else:
            raise ValueError(f"Expected one-dimensional numpy array with weights, but got: {struc_weights}")
        if len(prior_weights) != len(struc_weights):
            raise ValueError(f"Expected as many distributions as weights, but got: "
                             f"{len(prior_weights)} and {len(struc_weights)}")
        self.prior_distributions = [d / np.sum(d) for d in prior_weights]

    def marginal_likelihood(self, root_location, inside_chart, **kwargs):
        return np.sum(self.structural_distributions *
                      np.array([np.sum(d * c[root_location]) for d, c in zip(self.prior_distributions, inside_chart)]))


class DiscreteBinaryNonTerminalTransition(Transition):

    """
    A binary non-terminal transition for discrete non-terminal variables.
    """

    def __init__(self, weights, left_idx=0, right_idx=0):
        """
        Initialise a non-terminal transition p(a, b | c) for random variables `a`, `b`, `c`. The child variables `b` and
        `c` may be different variables than `a` (if the RBN has multiple non-terminal variables), which is determined by
        the left and right index (the default is to assume index 0, which is the first non-terminal variable – not
        necessarily the same variable as `a`).

        :param weights: Numpy array of shape (K, L, M) with weights proportional to p(a, b | c), where K, L, M are the
         cardinalities of the variables a, b, c, respectively.
        :param left_idx: index of the left child variable
        :param right_idx: index of the right child variable
        """
        if isinstance(weights, np.ndarray) and len(weights.shape) == 3:
            if np.any(weights < 0):
                raise ValueError("All weights have to be non-negative")
            self.transition_probabilities = normalize_non_zero(weights, axis=(0, 1))
        else:
            raise ValueError(f"Expected three-dimensional numpy array with weights, but got: {weights}")
        self.left_idx = left_idx
        self.right_idx = right_idx

    def inside_marginals(self, location, inside_chart, terminal_chart, value=None, **kwargs):
        if value is not None:
            NotImplementedError("Conditional inside probabilities currently not implemented")
        if isinstance(location, tuple) and len(location) == 2:
            start, end = location
            if end - start <= 1:
                # no splitting possible
                return []
            else:
                inside_marginals = []
                for split in range(start + 1, end):
                    left_inside = inside_chart[self.left_idx][start, split]
                    right_inside = inside_chart[self.right_idx][split, end]
                    inside_marginals.append(
                        np.sum(self.transition_probabilities * left_inside[:, None, None] * right_inside[None, :, None],
                               axis=(0, 1))
                    )
                return inside_marginals
        else:
            raise ValueError(f"Expected locations to be (start, end) index, but got: {location}")


class DiscreteTerminalTransition(Transition):
    """
    A binary terminal transition for discrete non-terminal and terminal variables.
    """

    def __init__(self, weights, term_idx=0):
        """
        Initialise a terminal transition p(a | b) for non-terminal variable `b` and terminal variable `a`.

        :param weights: Numpy array of shape (L, M) with weights proportional to p(a | b), where L, M are the
         cardinalities of the variables a, b, respectively.
        :param term_idx: index of the terminal variable
        """
        if isinstance(weights, np.ndarray) and len(weights.shape) == 2:
            if np.any(weights < 0):
                raise ValueError("All weights have to be non-negative")
            self.transition_probabilities = normalize_non_zero(weights, axis=0)
        else:
            raise ValueError(f"Expected two-dimensional numpy array with weights, but got: {weights}")
        self.term_idx = term_idx

    def inside_marginals(self, location, inside_chart, terminal_chart, value=None, **kwargs):
        if value is not None:
            NotImplementedError("Conditional inside probabilities currently not implemented")
        if isinstance(location, tuple) and len(location) == 2:
            start, end = location
            if end - start > 1:
                # no terminal transition possible
                return []
            else:
                return self.transition_probabilities[terminal_chart[start], :][None, :]
        else:
            raise ValueError(f"Expected locations to be (start, end) index, but got: {location}")


class StaticCell(Cell):
    """A cell with a static structural distribution, i.e., the probabilities over the different transitions"""

    def __init__(self, variable, weights, transitions):
        super().__init__(variable=variable)
        if isinstance(weights, np.ndarray) and len(weights.shape) == 1:
            if np.any(weights < 0):
                raise ValueError("All weights have to be non-negative")
            self.transition_probabilities = weights / weights.sum()
        else:
            raise ValueError(f"Expected one-dimensional numpy array with weights, but got: {weights}")
        if len(transitions) != weights.shape[0]:
            raise ValueError(f"Number of transitions and number of weights must be the same, but got: "
                             f"{len(transitions)} and {weights.shape[0]}")
        self._transitions = transitions

    def transitions(self) -> Iterable[Transition]:
        yield from self._transitions

    def inside_mixture(self, inside_marginals):
        # iterate over possible transitions
        s = []
        for pt, i in zip(self.transition_probabilities, inside_marginals):
            # if there are possible transitions
            if len(i) > 0:
                # sum out possible splitting points (first dimension) and weight with transition probability
                s.append(pt * np.array(i).sum(axis=0))
        # sum out transitions and return
        return np.array(s).sum(axis=0)
