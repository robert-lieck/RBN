"""
Discrete RBN
============

This examples goes through all the steps involved in defining a discrete RBN for sequential data.

Discrete RBNs are equivalent to PCFGs and, compared to continuous RBNs, easier to inspect and instructive as a first
step before moving towards continuous variables.

.. note::
    If you are only interested in defining and using a PCFG, see :doc:`/auto_examples/plot_PCFG` for a more
    realistic example using the :class:`~rbnet.pcfg.AbstractedPCFG` class.

"""

# %%
# Abstract PCFG
# -------------
# We start with the simplest non-trivial example of an RBN with a single,
# discrete, binary non-terminal variable. This example could equivalently be written as a PCFG with two
# non-terminal symbols :math:`\{0, 1\}`, two terminal symbols :math:`\{\bar{0}, \bar{1}\}`, and four rules
#
# .. math::
#     0 & \longrightarrow 1 0 \\
#     1 & \longrightarrow 0 1 \\
#     0 & \longrightarrow \bar{0} \\
#     1 & \longrightarrow \bar{1}~.
#
# These rules correspond to the non-zero entries in the transition matrices below (two non-terminal and two terminal
# rules). This is an example where the entire PCFG is `abstracted` into a single RBN variable. The
# :class:`~rbnet.pcfg.AbstractedPCFG` class provides a convenient interface for defining this type of RBN,
# but we will here build one from scratch for demonstration purposes. The second example below shows an alternative
# way (`expansion`) of using a PCFG to define an RBN.
#
# We start by importing some classes for discrete RBNs from the ``pcfg`` submodule.

from rbnet.pcfg import DiscreteNonTermVar, DiscretePrior, DiscreteBinaryNonTerminalTransition, DiscreteTerminalTransition, StaticCell
from rbnet.base import SequentialRBN
import numpy as np

# %%
# Defining Variables and Transitions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We first define a discrete binary non-terminal variable (:class:`~rbnet.pcfg.DiscreteNonTermVar`) and a corresponding
# :class:`~rbnet.pcfg.DiscretePrior` that always generates this one variable with a uniform distribution over its value
non_term_var = DiscreteNonTermVar(cardinality=2)
prior = DiscretePrior(struc_weights=[1.], prior_weights=[[0.5, 0.5]])

# %%
# For the transitions, we use a :class:`~rbnet.pcfg.DiscreteBinaryNonTerminalTransition` ``p(a, b | c)`` were the
# left child ``a`` is always the opposite of the parent ``c`` while the right child ``b`` is always the same; and a
# :class:`~rbnet.pcfg.DiscreteTerminalTransition` that produces a binary observation without changing value
weights = np.zeros((2, 2, 2))  # p(a, b | c)
weights[1, 0, 0] = 1  # p(a=1, b=0 | c=0) = 1
weights[0, 1, 1] = 1  # p(a=0, b=1 | c=1) = 1
non_term_transition = DiscreteBinaryNonTerminalTransition(weights=weights)

weights = np.zeros((2, 2))  # p(a | b)
weights[0, 0] = 1  # p(a=0 | b=0) = 1
weights[1, 1] = 1  # p(a=1 | b=1) = 1
term_transition = DiscreteTerminalTransition(weights=weights)

# %%
# Defining the Cell and RBN
# ^^^^^^^^^^^^^^^^^^^^^^^^^
# We can now create a :class:`~rbnet.pcfg.StaticCell` for the non-terminal variable, which chooses the terminal
# transition 50% of the time, and define our :class:`~rbnet.base.SequentialRBN`
cell = StaticCell(variable=non_term_var,
                  weights=[0.5, 0.5],
                  transitions=[non_term_transition, term_transition])
rbn = SequentialRBN(cells=[cell], prior=prior)

# %%
# Parsing Sequences
# ^^^^^^^^^^^^^^^^^
# It is impossible to generate sequences with all zeros or all ones, because children never have the same value and the
# terminal transition does not change the value. Thus, the marginal likelihood for these sequences, returned by the
# :meth:`~rbnet.base.RBN.inside` method, is always zero

print(rbn.inside(sequence=[[0], [0], [0], [0]]))
print(rbn.inside(sequence=[[1], [1], [1], [1]]))

# %%
# For other sequences, we see that the marginal likelihood is non-zero, and we can also inspect the parse chart, which
# contains the inside probabilities for the values of the non-terminal variable
print(rbn.inside(sequence=[[0], [0], [0], [1]]))
print(rbn.inside_chart[0].pretty())

# %%
# Note how
#
#  - values at the bottom are ``0.5`` because the probability of terminating is ``0.5`` and the transition is
#    deterministic otherwise.
#  - with each level, the values decrease by a factor of ``1/4``, where a factor of ``0.5`` comes from the probability
#    of (not) terminating and another factor of ``0.5`` comes from the inside probability of the left child.
#  - the marginal likelihood is ``1/2`` of the top-level inside probability, because the prior is uniform over values.

# %%
# Using the :class:`~rbnet.pcfg.AbstractedPCFG` Class
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# An equivalent RBN can be defined using the :class:`~rbnet.pcfg.AbstractedPCFG` class (see
# :doc:`/auto_examples/plot_PCFG` for a more realistic example). Note that the prior transition of an RBN provides
# slightly more freedom than a PCFG, because it defines a distribution over the first symbol, whereas a PCFG always
# starts with the start symbol. Therefore, we need combine the prior and the first non-terminal transition into the
# definition for the start symbol (essentially marginalising out the first symbol generated by the prior in the RBN).
# Internally, the :class:`~rbnet.pcfg.AbstractedPCFG` class defines a deterministic prior that generates the start
# symbol. We get identical inside probabilities to the RBN case above (the first value corresponding to the start
# symbol), but the marginal likelihood is a factor of 2 larger because of the deterministic prior.

from rbnet.pcfg import AbstractedPCFG

pcfg = AbstractedPCFG(non_terminals="SAB", terminals="ab", start="S", rules=[
    ("S --> A B", 1), ("S --> B A", 1),  # prior + first transition
    ("A --> B A", 1), ("B --> A B", 1),  # non-terminal transitions
    ("A --> a", 1), ("B --> b", 1)       # terminal transition
])


print(pcfg.inside(sequence="aaaa"))
print(pcfg.inside(sequence="bbbb"))
print(pcfg.inside(sequence="aaab"))
print(pcfg.inside_chart[0].pretty())

# %%
# Expanded PCFG
# -------------
# We will now define an RBN by `expanding` the same PCFG used in the example above. When expanding a PCFG to an
# RBN, each non-terminal symbol becomes a separate non-terminal variable in the RBN. The PCFG thus acts as an outer
# skeleton when being expanded, and we are required to additionally define the domain and transitions for the variables.
# The variables could be discrete or continuous, but for this example, we take the trivial case of single-valued
# discrete variable (i.e. it cannot actually change value) and all the dynamics instead happens on the structural level.
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


print(rbn.inside(sequence=[[0, None],
                           [0, None],
                           [0, None],
                           [0, None]]))
print(rbn.inside(sequence=[[None, 0],
                           [None, 0],
                           [None, 0],
                           [None, 0]]))


print(rbn.inside(sequence=[[0, None],
                           [0, None],
                           [0, None],
                           [None, 0]]))
print(rbn.inside_chart[0].pretty())
print(rbn.inside_chart[1].pretty())
