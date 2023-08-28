from typing import Iterable
from abc import ABC, abstractmethod


class RBN(ABC):
    """Base class for RBNs."""

    @abstractmethod
    def inside_schedule(self, *args, **kwargs):
        r"""
        Iterate through (batches of) non-terminal locations for computing inside probabilities. The iteration order
        has to respect dependencies, i.e., non-terminals :math:`x'` that depend on another non-terminal :math:`x` (in
        the generative direction, i.e. :math:`x` generates :math:`x'`: :math:`x \rightarrow x'`) have to be iterated
        BEFORE the non-terminals they depend on (:math:`x'` has to be visited before :math:`x`).

        :return: location or batch of locations
        """
        raise NotImplementedError

    @abstractmethod
    def root_location(self):
        """
        Return the location of the root variables. This is typically the value returned in the last iteration of
        :meth:`inside_schedule`

        :return: location of the root variables
        """
        raise NotImplementedError

    @abstractmethod
    def non_terminals(self, locations, *args, **kwargs):
        """
        Iterate through all non-terminal variables, yielding tuples of a variable's values at the specified locations
        and its associated cell. If `locations` specifies a batch of locations, the tuple contains the corresponding
        batch of variables' values, but still only the single associated cell.

        Returning ``None`` for the variable's values indicates that should be marginalised out; returning proper values
        indicates that variables should be conditioned on these values.

        :param locations: location or batch of locations of variables in chart
        :yield: (variable's value, cell) or (batch of variables' values, cell)
        """
        raise NotImplementedError

    def init_inside(self, *args, **kwargs):
        """
        Initialise for parsing a new input. This function is called by :meth:`inside` with all provided parameters.
        Derived classes may override it to implement tasks such as initialising charts for the non-terminal variables.
        """
        pass

    @abstractmethod
    def update_inside_chart(self, var_idx, locations, values):
        """
        For the specified variable, update the chart for inside probabilities with given values at given locations.

        :param var_idx: specifies the variable
        :param locations: locations to update
        :param values: values to store in chart
        """
        raise NotImplementedError

    @abstractmethod
    def get_inside_chart(self):
        """
        Return the chart with inside probabilities for all variables.

        :return: inside chart
        """
        raise NotImplementedError

    @abstractmethod
    def get_terminal_chart(self):
        """
        Return the chart with terminal variables.

        :return: terminal chart
        """
        raise NotImplementedError

    @abstractmethod
    def prior(self):
        """
        Return the prior transition (typically an instance of :class:`Prior`), which has to implement
        :meth:`Prior.marginal_likelihood`.

        :return: prior
        """

        raise NotImplementedError

    def inside(self, *args, **kwargs):
        """
        Compute the inside probabilities and return the marginal data likelihood.

        :return: marginal likelihood
        """
        # perform any initialisations
        self.init_inside(*args, **kwargs)
        # get the next batch of non-terminal locations (e.g. level in CYK)
        for non_term_loc in self.inside_schedule():
            # go through all variables and associated cells (which contain the allowed transitions) at these locations
            for non_term_idx, (non_term_value, cell) in enumerate(self.non_terminals(non_term_loc)):
                # go through the possible transitions allowed for this non-terminal and collect inside marginals
                inside_marginals = []
                for transition in cell.transitions():
                    inside_marginals.append(
                        transition.inside_marginals(location=non_term_loc,
                                                    inside_chart=self.get_inside_chart(),
                                                    terminal_chart=self.get_terminal_chart(),
                                                    value=non_term_value)
                    )
                # compute the mixture over inside marginals and update chart for variable(s)
                self.update_inside_chart(var_idx=non_term_idx,
                                         locations=non_term_loc,
                                         values=cell.inside_mixture(inside_marginals))
        # add the prior likelihood for the root location
        return self.prior().marginal_likelihood(root_location=self.root_location(),
                                                inside_chart=self.get_inside_chart())


class Transition(ABC):
    """
    Base class for RBN transitions, which have to implement :meth:`~Transition.inside_marginals`.
    """

    @abstractmethod
    def inside_marginals(self, location, inside_chart, terminal_chart, value=None, **kwargs):
        r"""
        Compute the marginals over inside probabilities

        .. math::
            \widetilde{\beta}_{i:\ldots:k}(x_{i:k})
            =
            {\int\cdots\int}_{\{v_{j:j^\prime}\in\mathcal{X}\}}
            p_{\tau}(v_{i:j_1}, \ldots, v_{j_{\eta-1}:k} \mid x_{i:k})
            \prod_{\{v_{j:j^\prime}\in\mathcal{X}\}}
            \beta(v_{j:j^\prime}),

        for all possible splitting points (also see :ref:`here <general inside probabilities>`). In particular,
        ``location`` specifies the variable's location in the parse chart (the indices :math:`i` and :math:`k` in the
        equation above), from which the possible splitting points follow (:math:`\eta` splitting points :math:`j_1,
        \ldots,j_{\eta-1}` for transitions of arity :math:`\eta`). The marginals should always be returned in an
        array or iterable where the first dimension corresponds to all possible combinations of splitting points,
        even for transitions with arity :math:`\eta\neq2` (i.e. for :math:`\eta=1`, where there are no splits,
        the first dimension should be of size 1 and for :math:`\eta>1` all possible combinations of the
        :math:`\eta-1` splitting points should be listed in a flattened form in the first dimension). Additional
        dimensions, may be used to represent the dependency of the marginal on the variable :math:`x_{i:k}` (e.g. for
        a discrete variable, the second dimension may list the marginal for each possible value :math:`x_{i:k}` can
        take; and for a continuous variable, the marginal may be represented by a set of parameters).

        The output of this function is typically handled by a custom implementation of
        :meth:`Cell.inside_mixture() <rbnet.base.Cell.inside_mixture>`.

        :param location: location of the variable for which to compute the inside marginals
        :param inside_chart: a lookup chart with inside probabilities for other variables
        :param terminal_chart: a lookup chart with values of the terminal variables
        :param value: value of the variable for which to compute the inside marginals or ``None``; if value is
         provided this means that *conditional* inside probabilities should be computed (for the given value);
         otherwise general inside marginals (as a function of the variable) should be computed; also see
         :meth:`RBN.non_terminals`
        :return: array-like or iterable with inside probabilities; if the transition is not possible, this should be
         of length zero or contain only inside probabilities representing zero
        """
        raise NotImplementedError


class Prior(ABC):
    """
    Base class for prior transitions. Prior transitions are similar to normal Transition's, but cannot directly generate
    terminal variables and have no parent variables.
    """

    @abstractmethod
    def marginal_likelihood(self, root_location, inside_chart, **kwargs):
        r"""
        Compute the marginal data likelihood

            .. math::
                p(\mathbf{Y}) = \sum_{x\in\mathcal{X}} w_x \int \beta(x_{0:n}) \ p_P(x_{0:n}) \ dx_{0:n},

        as described in more detail :ref:`here <marginal likelihood>`.

        :param root_location: location of the root variables
        :param inside_chart: chart of inside probabilities
        :return: marginal likelihood
        """
        raise NotImplementedError


class NonTermVar(ABC):
    """Base class for non-terminal variables"""

    @abstractmethod
    def get_chart(self, *args, **kwargs):
        raise NotImplementedError


class Cell(ABC):
    """
    Base class for RBN cells associated to a non-terminal template variable. Cells hold the parse chart for that
    variable, accessible via :meth:`get_chart`, all transitions that are possible from that variable, accessible via
    :meth:`transitions`, and implement computation of the mixture of inside probabilities over that variable in
    :meth:`inside_mixture`.
    """

    def __init__(self, variable):
        """
        A cell for a given non-terminal variable

        :param variable: non-terminal variable
        """
        self.variable = variable

    @abstractmethod
    def transitions(self) -> Iterable[Transition]:
        """
        Iterate through all possible transitions.
        """
        raise NotImplementedError

    @abstractmethod
    def inside_mixture(self, inside_marginals):
        r"""
        For a list of inside marginals :math:`\widetilde{\beta}_{i:\ldots:k}(x_{i:k})`, compute the mixture

        .. math::
            \beta(x_{i:k})
            =
            \sum_{\tau \in \mathcal{T}_x}
            p_S(z_{i:k}=\tau \mid x_{i:k})
            {\sum\cdots\sum}_{i<j_1<\ldots<j_{\eta-1}<k} \ \widetilde{\beta}_{i:\ldots:k}(x_{i:k})

        as described in more detail :ref:`here <general inside probabilities>`.

        The inside marginals are typically computed by a custom implementation of
        :meth:`Transition.inside_marginals() <rbnet.base.Transition.inside_marginals>`

        :param inside_marginals: iterable over inside marginals (as returned by :meth:`Transition.inside_marginals`)
        :return: representation of the inside probability :math:`\beta(x_{i:k})`
        """
        raise NotImplementedError


class CYKRBN(RBN):

    def __init__(self, cells, prior):
        self.cells = cells
        self._prior = prior
        self.n = None
        self.terminals = None
        self.inside_chart = None

    def init_inside(self, sequence):
        self.n = len(sequence)
        self.terminals = sequence
        self.inside_chart = [c.variable.get_chart(len(sequence)) for c in self.cells]

    def inside_schedule(self, *args, **kwargs):
        for span in range(1, self.n + 1):
            for start in range(self.n - span + 1):
                yield start, start + span

    def root_location(self):
        return 0, self.n

    def non_terminals(self, locations, *args, **kwargs):
        for c in self.cells:
            yield None, c

    def update_inside_chart(self, var_idx, locations, values):
        self.inside_chart[var_idx][locations] = values

    def get_inside_chart(self):
        return self.inside_chart

    def get_terminal_chart(self):
        return self.terminals

    def prior(self):
        return self._prior


def main():
    pass


if __name__ == "__main__":
    main()
