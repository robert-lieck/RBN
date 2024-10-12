import torch

from rbnet.base import Prior, NonTermVar, Transition, Cell
from rbnet.util import TupleTMap


class AutoencoderNonTermVar(NonTermVar):
    def __init__(self, dim, chart_type="TMap", *args, **kwargs):
        """
        A point-wise continuous non-terminal variable of dimensionality ``dim``. Distributions over these variables
        can be thought of as Dirac deltas (the limit of infinitely narrow Gaussians), represented by a location
        (a specific variable value) and weight (in the case of mixtures or inside probabilities).

        :param cardinality: cardinality
        :param chart_type: type of chart to use ("dict" or "TMap")
        """
        super().__init__(*args, **kwargs)
        self.dim = dim
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
            return TupleTMap([
                torch.zeros((TupleTMap.size_from_n(n), self.dim)),  # the variable values
                torch.zeros(TupleTMap.size_from_n(n))               # the inside probabilities
            ])
        else:
            raise ValueError(f"Unknown chart type '{self.chart_type}'")

    def mixture(self, components, weights=None, dim=0):
        """
        Approximate a mixture by its weighted average. The new weight is the sum of mixture weights. Mixture weights
        are provided as part of the ``components``; additional weights provided as ``weights`` are multiplied on the
        weights provided in ``components``.

        :param components: array-like with pairs of (values, weights) mixture components along ``dim``
        :param weights: [optional] weights of the mixture components; must be compatible (broadcastable) to weights in
         ``components``
        :param dim: integer or tuple of integers indicating the dimensions of ``components`` along which to sum to
         compute the mixture
        :return: distribution corresponding to the mixture
        """
        if len(components) == 0:
            return torch.zeros(self.dim), torch.zeros(1)
        mix_weights = torch.stack([c[1] for c in components])
        components = torch.stack([c[0] for c in components])
        if not isinstance(dim, tuple):
            dim = (dim,)
        if weights is not None:
            mix_weights = torch.as_tensor(weights) * mix_weights
        return (mix_weights * components).sum(dim=dim), mix_weights.sum(dim=dim)


class AutoencoderTransition(Transition):
    r"""
    An autoencoder transition combining a deterministic binary non-terminal and unary terminal transition. The
    general :meth:`~rbnet.base.Transition.inside_marginals` simplify for autoencoders. First, we operate on point
    estimates (delta distributions), so we assume the following form for the inside distribution

        .. math::
            \beta_{i:k}(x_{i:k}) &:= w_{i:k} \, \delta(x_{i:k}=\bar{x}_{i:k}) \\
            \widetilde{\beta}_{i:j:k}(x_{i:k}) &:= \widetilde{w}_{i:j:k} \, \delta(x_{i:k}=\widetilde{x}_{i:j:k})~,

    where :math:`\bar{x}_{i:k}` and :math:`\widetilde{x}_{i:j:k}` define the location of the delta distributions and
    :math:`w_{i:k}` and :math:`\widetilde{w}_{i:j:k}` their norm.

    For binary non-terminals, we then get

        .. math::
            \widetilde{\beta}_{i:j:k}(x_{i:k})
            &= \int\int p_{N}(x_{i:j}, x_{j:k} \mid x_{i:k}) \beta(x_{i:j}) \beta(x_{j:k}) dx_{i:j} dx_{j:k} \\
            &= p_{N}(\bar{x}_{i:j}, \bar{x}_{j:k} \mid x_{i:k}) \, w_{i:j} \, w_{j:k}

    and for unary terminals

        .. math::
            \widetilde{\beta}_{i:j:k}(x_{i:i+1})
            = p_{T}(y_{i+1} \mid x_{i:i+1})~.

    We now recover the form assumed above by fixing the value of :math:`x_{i:k}` and :math:`x_{i:i+1}` given by a
    deterministic encoder, while the transition probabilities are provided by the stochastic forward model, i.e.,
    the decoder

        .. math::
            \widetilde{x}_{i:j:k} &:= \mbox{non-terminal encoder}(\bar{x}_{i:j}, \bar{x}_{j:k}) \\
            p_{N}(\bar{x}_{i:j}, \bar{x}_{j:k} \mid x_{i:k}) &:= \mbox{non-terminal decoder}(\bar{x}_{i:j}, \bar{x}_{j:k} \mid \widetilde{x}_{i:j:k}) \\
            \widetilde{x}_{i:i+1} &:= \mbox{terminal encoder}(y_{i+1}) \\
            p_{T}(y_{i+1} \mid x_{i:i+1}) &:= \mbox{terminal decoder}(y_{i+1} \mid \widetilde{x}_{i:i+1})~.
    """

    def __init__(self,
                 terminal_encoder, terminal_decoder,
                 non_terminal_encoder, non_terminal_decoder,
                 left_idx=0, right_idx=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.left_idx = left_idx
        self.right_idx = right_idx
        self.terminal_encoder = terminal_encoder
        self.terminal_decoder = terminal_decoder
        self.non_terminal_encoder = non_terminal_encoder
        self.non_terminal_decoder = non_terminal_decoder

    def inside_marginals(self, location, inside_chart, terminal_chart, value=None, **kwargs):
        if value is not None:
            NotImplementedError("Conditional inside probabilities currently not implemented")
        if isinstance(location, tuple) and len(location) == 2:
            start, end = location
            if end - start <= 1:
                # terminal transition
                parent_var = self.terminal_encoder(terminal_chart[start])
                transition_prob = self.terminal_decoder(parent_var, terminal_chart[start])
                return [(parent_var, transition_prob)]
            else:
                inside_marginals = []
                for split in range(start + 1, end):
                    # get inside probabilities (clone to avoid problems with inplace operations below – not sure why)
                    left_var, left_inside = inside_chart[self.left_idx][start, split]
                    right_var, right_inside = inside_chart[self.right_idx][split, end]
                    parent_var = self.non_terminal_encoder(left_var, right_var)
                    transition_prob = self.non_terminal_decoder(parent_var, left_var, right_var)
                    inside_marginals.append((parent_var, transition_prob * left_inside * right_inside))
                return inside_marginals
        else:
            raise ValueError(f"Expected locations to be (start, end) index, but got: {location}")


class AutoencoderCell(Cell):

    def __init__(self, variable, transition, *args, **kwargs):
        """
        :param variable: the :class:`~AutoencoderVariable` for this cell
        :param transition: the :class:`~AutoencoderTransition` for this cell
        :param args: passed on to super().__init__
        :param kwargs: passed on to super().__init__
        """
        super().__init__(variable=variable, *args, **kwargs)
        self._transition = transition

    def transitions(self):
        yield from [self._transition]

    def inside_mixture(self, inside_marginals):
        assert len(inside_marginals) == 1, f"Expected only one element for a single transition, but got {len(inside_marginals)}"
        return self.variable.mixture(components=inside_marginals[0])


class AutoencoderPrior(Prior):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def marginal_likelihood(self, root_location, inside_chart, **kwargs):
        assert len(inside_chart) == 1, f"Expected inside chart with one variable, but got {len(inside_chart)}"
        return inside_chart[0][root_location]
