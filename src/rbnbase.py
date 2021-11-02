#  Copyright (c) 2021 Robert Lieck
from typing import Any
import math
import re
from warnings import warn

import numpy as np
import torch
from torch.nn import Module, Parameter
from torch.distributions import Poisson
from tqdm import tqdm

from pyulib import NestedOutputSingleton, NestedOutputDummy
from util import TMap, MultivariateNormal
from multivariate_normal import ApproximateMixture, PairwiseProduct

NO = NestedOutputDummy


class Categorical(Module):
    """
    Maps categorical distributions into log-space, projected to the 1-plane through the origin.

    Categorical distributions can be modelled as point in (unbounded) Euclidean space by using their log-representation.
    This results in one additional degree of freedom, corresponding to their normalisation constant. Projecting them
    to the 1-plane through the origin (the plane with normal vector (1,...,1)) eliminates this degree of freedom. All
    points then lie on this (N-1)-dimensional plane and can, for instance, be modelled by a mixture of Gaussians.
    """

    @classmethod
    def map(cls, categorical: torch.tensor, dim: int, offset=0., return_offset_categorical=False):
        """
        Map categorical distributions to 1-plane in log-space.
        :param categorical: data points
        :param dim: dimension of categorical distribution (other dimensions are treated as batch dimensions)
        :return: mapped data points
        """
        # assert 0 <= offset < 1, f"offset must be in 0 <= offset < 1 but is {offset}"
        assert 0 <= offset, f"offset must be non-negative but is {offset}"
        if offset > 0:
            categorical = (categorical + offset) / (1 + offset * categorical.shape[dim])
        log_cat = categorical.log()
        norm = log_cat.sum(dim=dim, keepdims=True)
        log_cat -= norm / categorical.shape[dim]
        if return_offset_categorical:
            return log_cat, categorical
        else:
            return log_cat

    @classmethod
    def rmap(cls, log_probs: torch.tensor, dim: int, offset=0.):
        categorical = log_probs.exp()
        categorical /= categorical.sum(dim=dim, keepdims=True)
        if offset > 0:
            categorical = categorical * (1 + offset * categorical.shape[dim]) - offset
        return categorical

    def __init__(self, dim: int = 0, offset=0.):
        # assert 0 <= offset < 1, f"offset must be in 0 <= offset < 1 but is {offset}"
        assert 0 <= offset, f"offset must be non-negative but is {offset}"
        super().__init__()
        self.dim = dim
        # self._log_offset = Parameter(torch.log(offset / (1 - offset)))
        if offset == 0:
            self._log_offset = Parameter(torch.tensor(-np.inf))
        else:
            self._log_offset = Parameter(torch.tensor(np.log(offset)))

    @property
    def offset(self):
        # return self._log_offset.sigmoid()
        return self._log_offset.exp()

    def forward(self, categorical: torch.tensor):
        return self.map(categorical, dim=self.dim, offset=self.offset)


class RBNBase(Module):

    # dummy type hints for automatically variables created
    inside_mean: Any
    inside_cov: Any
    inside_log_coef: Any
    _inside_mean_arr: Any
    _inside_cov_arr: Any
    _inside_log_coef_arr: Any

    outside_mean: Any
    outside_cov: Any
    outside_log_coef: Any
    _outside_mean_arr: Any
    _outside_cov_arr: Any
    _outside_log_coef_arr: Any

    node_mean: Any
    node_cov: Any
    node_log_coef: Any
    _node_mean_arr: Any
    _node_cov_arr: Any
    _node_log_coef_arr: Any

    multi_term_mean: Any
    multi_term_cov: Any
    multi_term_log_coef: Any
    _multi_term_mean_arr: Any
    _multi_term_cov_arr: Any
    _multi_term_log_coef_arr: Any

    _prior_reg: Any
    _term_trans_reg: Any
    _left_non_term_reg: Any
    _right_non_term_reg: Any

    # whether to recompute precision matrices in forward
    recompute_prec = True

    left_split_prec: Any
    _left_split_prec_arr: Any
    right_split_prec: Any
    _right_split_prec_arr: Any

    # for MAP trees: index of maximum split/termination decision
    max_split_indices: Any
    _max_split_indices_arr: Any
    max_split_LL: Any
    _max_split_LL_arr: Any

    def __init__(self, multi_terminal_lambda=None, checks=False,
                 terminal_log_prob=math.log(0.5), scale=None, reg=1e-2,
                 prior_mean=0., prior_sqrt_cov=None, prior_reg=None,
                 term_trans_sqrt_cov=None, term_trans_reg=None,
                 non_term_sqrt_cov=None, non_term_reg=None,
                 left_non_term_sqrt_cov=None, left_non_term_reg=None,
                 right_non_term_sqrt_cov=None, right_non_term_reg=None,
                 log_mixture_weights=(0.,), left_transpositions=(0,), right_transpositions=(0,),
                 log_prior_counts=None, greedy_inside=False, greedy_outside=False,
                 trans_inv_prior=False):
        """
        :param multi_terminal: allow generating multiple terminal/observed variables from one non-terminal/latent variable
        :param checks: enable safety checks (less performant and may give false alarms due to numerical instabilities)
        :param terminal_log_prob: log-probability for terminating (part of the grammar)
        :param scale: global scale that will be used for any unspecified *_sqrt_cov value
        :param reg: global regularisation that will be used for any unspecified *_reg value
        :param prior_mean: mean of prior distribution (part of the grammar)
        :param prior_sqrt_cov: (scalar, 1D, or 2D) square-root of the prior covariance matrix
        :param prior_reg: (scalar, or 1D) regularisation for the prior covariance matrix: the square of this value is added along the diagonal to avoid degenerate distributions
        :param term_trans_sqrt_cov: like `prior_sqrt_cov` but for terminal transition (part of the grammar)
        :param term_trans_reg: like `prior_reg` but for terminal transition
        :param non_term_sqrt_cov: is specified used for shared covariance matrix of left and right transition, otherwise like `prior_sqrt_cov` (part of the grammar)
        :param non_term_reg: like `prior_reg` but for shared non-terminal transition
        :param left_non_term_sqrt_cov: like `prior_sqrt_cov` but for left non-terminal transition (part of the grammar)
        :param left_non_term_reg: like `prior_reg` but for left non-terminal transition
        :param right_non_term_sqrt_cov: like `prior_sqrt_cov` but for right non-terminal transition (part of the grammar)
        :param right_non_term_reg: like `prior_reg` but for left non-terminal transition
        :param log_mixture_weights: weights for the non-terminal transition components
        :param left_transpositions: transpositions of mixture components (rolling along output dimension) for left transition
        :param right_transpositions: transpositions of mixture components (rolling along output dimension) for right transition
        :param trans_inv_prior: (bool; default: False) If true the prior is a uniform mixture over all transpositions (n components for n-dimensional data)
        """
        super().__init__()
        self.multi_terminal = multi_terminal_lambda is not None
        self.checks = checks
        self.couple_non_term = non_term_sqrt_cov is not None
        self.greedy_inside = greedy_inside
        self.greedy_outside = greedy_outside
        self.trans_inv_prior = trans_inv_prior
        if log_prior_counts is None:
            self.categorical = False
        else:
            self.categorical = True
        # observations
        self.observations = None
        # dimensions; will be set when calling forward()
        self.n_obs = None         # the number of observation in the sequence
        self.n_batch = None       # number of batch dimensions (e.g. multiple independent sequences that should be processed in parallel)
        self.n_dim = None         # the dimensionality of the observations (1 for scalar, 12 for pitch classes etc.)
        self.chart_size = None    # the resulting (flattened) chart_size, which is n_obs * (n_obs + 1)) / 2
        self.n_mix = None         # the number of mixture components in non-terminal transitions (i.e. number of transpositions)
        # buffers for storing distribution parameters
        # - will have shape of (self.chart_size, self.n_shape) + (self.n_dim,) * D
        #   where D is 0, 1, 2 for log_coef, mean, and cov, respectively
        for var in ["inside", "outside", "node", "multi_term"]:
            for t in ["mean", "cov", "log_coef"]:
                self._init_tmap(f"{var}_{t}", register=True)
        if not self.recompute_prec:
            self._init_tmap("left_split_prec", register=True)
            self._init_tmap("right_split_prec", register=True)
        self._init_tmap("max_split_indices")
        self._init_tmap("max_split_LL")
        # update flag; indicates whether the outside provabilities are up-to-date or need to be recomputed
        self._outside_uptodate = False
        # marginal likelihood the marginal data log-likelihood (averaged over batch dimension)
        self.marginal_log_likelihood = None
        # defaults for covariances and regularisations
        if prior_sqrt_cov is None:
            prior_sqrt_cov = scale
        if prior_reg is None:
            prior_reg = reg
        if term_trans_sqrt_cov is None:
            term_trans_sqrt_cov = scale
        if term_trans_reg is None:
            term_trans_reg = reg
        if self.couple_non_term:
            # only used if left/right non-terminal transitions ARE coupled
            if non_term_reg is None:
                non_term_reg = reg
        else:
            # only used if left/right non-terminal transitions ARE NOT coupled
            if left_non_term_sqrt_cov is None:
                left_non_term_sqrt_cov = scale
            if left_non_term_reg is None:
                left_non_term_reg = reg
            if right_non_term_sqrt_cov is None:
                right_non_term_sqrt_cov = scale
            if right_non_term_reg is None:
                right_non_term_reg = reg
        # - initialise all parameters
        # - make known the terms computed from them when calling forward()
        # prior
        self._prior_mean = Parameter(torch.tensor(prior_mean).float().to('cpu'))
        self.prior_mean = None
        self._prior_sqrt_cov = Parameter(torch.tensor(prior_sqrt_cov).float().to('cpu'))
        self.register_buffer("_prior_reg", torch.tensor(prior_reg).float().to('cpu'))
        self.prior_cov = None
        # terminal transition
        self.terminal_log_prob = Parameter((terminal_log_prob * torch.ones(1)).float().to('cpu'))
        if not self.categorical:
            self._term_trans_sqrt_cov = Parameter(torch.tensor(term_trans_sqrt_cov).float().to('cpu'))
            self.register_buffer("_term_trans_reg", torch.tensor(term_trans_reg).float().to('cpu'))
            self.term_trans_cov = None
        # non-terminal transition
        if self.couple_non_term:
            # only used if left/right non-terminal transitions ARE coupled
            self._non_term_sqrt_cov = Parameter(torch.tensor(non_term_sqrt_cov).float().to('cpu'))
            self.register_buffer("_non_term_reg", torch.tensor(non_term_reg).float().to('cpu'))
            self.non_term_cov = None
        else:
            # only used if left/right non-terminal transitions ARE NOT coupled
            self._left_non_term_sqrt_cov = Parameter(torch.tensor(left_non_term_sqrt_cov).float().to('cpu'))
            self.register_buffer("_left_non_term_reg", torch.tensor(left_non_term_reg).float().to('cpu'))
            self.left_non_term_cov = None
            self._right_non_term_sqrt_cov = Parameter(torch.tensor(right_non_term_sqrt_cov).float().to('cpu'))
            self.register_buffer("_right_non_term_reg", torch.tensor(right_non_term_reg).float().to('cpu'))
            self.right_non_term_cov = None
        # multi-terminal distribution
        if self.multi_terminal:
            self._log_multi_terminal_lambda = Parameter(torch.tensor(math.log(multi_terminal_lambda)).float().to('cpu'))
            self.multi_terminal_lambda = None
        # prior count parameter for categorical data
        if self.categorical:
            self._log_prior_counts = Parameter(torch.tensor(log_prior_counts).float().to('cpu'))
        # parameters for mixture transitions
        self._log_mixture_weights = torch.tensor(log_mixture_weights).float().to('cpu')
        self._left_transpositions_indices = tuple(left_transpositions)
        self._right_transpositions_indices = tuple(right_transpositions)
        assert self._log_mixture_weights.shape == (len(self._left_transpositions_indices),), (self._log_mixture_weights.shape, (len(self._left_transpositions_indices),))
        assert self._log_mixture_weights.shape == (len(self._right_transpositions_indices),), (self._log_mixture_weights.shape, (len(self._right_transpositions_indices),))
        if len(log_mixture_weights) > 1:
            # if there is more then one mixture/transposition component: make it a learnable parameter
            self._log_mixture_weights = Parameter(self._log_mixture_weights)

    @property
    def prior_counts(self):
        if self.categorical:
            return self._log_prior_counts.exp()
        else:
            return None

    @property
    def log_mixture_weights(self):
        # log mixture weights are normalised to zero L1 norm in log space
        # --> have to normalise to unit L1 norm in probability space
        return self._log_mixture_weights - self._log_mixture_weights.logsumexp(dim=0)

    @property
    def mixture_weights(self):
        return self.log_mixture_weights.exp()

    def print_params(self, print_derived=False, print_type=True, print_gradient=False):
        """
        Print all parameters of the model
        :param print_derived: also print the terms derived from each parameter
        :param print_type: also print the type of each parameter
        :param print_gradient: also print the gradient of each parameter
        """
        for name, param in self.named_parameters():
            print(f"{name} {tuple(param.shape)} {(param.dtype if print_type else '')}")
            print(f"    {param.detach().numpy()}")
            if print_gradient:
                if param.grad is None:
                    print("    [No grad info]")
                else:
                    print(f"    {param.grad.detach().numpy()}")
            if print_derived:
                # try a bunch different naming conventions for the derived terms
                for derived in {
                    name[1:].replace("_sqrt_cov", "_cov"),             # cov from sqrt_cov
                    "left_" + name[1:].replace("_sqrt_cov", "_cov"),   # for coupled left/right
                    "right_" + name[1:].replace("_sqrt_cov", "_cov"),  # for coupled left/right
                    re.sub("^_log_", "", name),                        # for positive parameters stored as _log_*
                    re.sub("^_", "", name),                            # for raw parameters stored as _* with leading underscore
                }:
                    if derived == name or not hasattr(self, derived):
                        continue
                    else:
                        derived_param = getattr(self, derived)
                        if derived_param is None:
                            print(f"    --> {derived} = None")
                        else:
                            print(f"    --> {derived} {tuple(derived_param.shape)}")
                            print(f"        {derived_param.detach().numpy()}")

    def _init_tmap(self, name, register=False):
        """
        Initialise TMap with given name. Optionally register the underlying buffer. The value will be set to a dummy
        tensor torch.zeros(1)
        :param name: name of TMap; map will be available as self.name; underlying array will be self._name_arr
        :param register: whether to register the buffer
        """
        # dummy value
        value = torch.zeros(1)
        # create and register buffer
        if register:
            self.register_buffer(f"_{name}_arr", value)
        # create attribute with TMap wrapping the buffer
        self._update_tmap(name, value)

    def _update_tmap(self, name, value=None):
        """
        Performs two operations (the first one only if value is provided):
        1. assign a new value to the underlying array
            self._<name>_arr.data = value
        2. wrap the array in a TMap and make it available under given <name>
            self.<name> = TMap(self._<name>_arr, linearise_blocks=True)
        """
        if value is not None:
            setattr(self, f"_{name}_arr", value)
        setattr(self, name, TMap(getattr(self, f"_{name}_arr"), linearise_blocks=True))

    @classmethod
    def _full_nan(cls, shape):
        return torch.full(shape, fill_value=np.nan)

    def _create_shapes(self):
        # get dimensions
        # n_obs: number of observations (length of the sequence)
        # n_batch: number of data sets that are processed independently
        # n_dim: dimensionality of the observations
        # chart_size: size of the parse chart: n_obs(n_obs+1)/2
        self.n_obs, self.n_batch, self.n_dim = self.observations.shape
        self.chart_size = TMap.size1d_from_n(self.n_obs)
        self.n_mix = len(self._log_mixture_weights)

    def _create_covariance_matrices(self):
        """covariance matrices with correct dimensions"""
        self.prior_mean = self._prior_mean.expand(self.n_dim)
        params = ["prior", "left_non_term", "right_non_term"]
        if not self.categorical:
            params += ["term_trans"]
        for base in params:
            # coupling
            if self.couple_non_term and base in ["left_non_term", "right_non_term"]:
                base_ = "non_term"
            else:
                base_ = base
            # get sqrt of covariance matrix
            sqrt = getattr(self, f"_{base_}_sqrt_cov")
            # get (or expand to) diagonal matrix if needed
            if len(sqrt.shape) < 2:
                sqrt = torch.diagflat(sqrt.expand(self.n_dim))
            # get diagonal regularisation
            diag = torch.diagflat(getattr(self, f"_{base_}_reg").expand(self.n_dim))
            setattr(self, f"{base}_cov", torch.matmul(sqrt, sqrt.t()) + diag ** 2)

    def _zero_L1_offset(self, arr, dim):
        """
        Computes the offset to the plane with zero L1 norm. Subtracting the offset normalises the array.
        """
        return arr.sum(dim=dim) / arr.shape[dim]

    def _create_mixtures(self):
        # enforce normalisation of mixture weights to zero L1 norm in log space
        # sum(x_i - sum(x_j)/n) = sum(x_i) - n sum(x_j)/n = 0
        self._log_mixture_weights.data -= self._zero_L1_offset(self._log_mixture_weights, dim=0)
        # construct transposition matrices
        self.left_T = torch.cat([torch.roll(torch.eye(self.n_dim),
                                            shifts=n,
                                            dims=0)[None] for n in self._left_transpositions_indices],
                                dim=0)
        self.right_T = torch.cat([torch.roll(torch.eye(self.n_dim),
                                             shifts=n,
                                             dims=0)[None] for n in self._right_transpositions_indices],
                                 dim=0)
        if self.trans_inv_prior:
            self.prior_T = torch.cat([torch.roll(torch.eye(self.n_dim),
                                                 shifts=n,
                                                 dims=0)[None] for n in range(self.n_dim)],
                                     dim=0)
        # expand to shape (n_mixture, n_split, n_batch, ...)
        # - don't expand variable dimension: this has to be done in inside/forward()
        # - it is not needed for outside
        # - mixture dimension is already present
        expand = (slice(None), None, None)
        self.left_T = self.left_T[expand]
        self.right_T = self.right_T[expand]

    def _create_tmaps(self, max_tree):
        # inside probabilities
        log_coef_shape = (self.chart_size, self.n_batch)
        mean_shape = (self.chart_size, self.n_batch, self.n_dim)
        cov_shape = (self.chart_size, self.n_batch, self.n_dim, self.n_dim)
        self._update_tmap("inside_mean", self._full_nan(mean_shape))
        self._update_tmap("inside_cov", self._full_nan(cov_shape))
        if not self.recompute_prec:
            self._update_tmap("left_split_prec", self._full_nan(cov_shape))
            self._update_tmap("right_split_prec", self._full_nan(cov_shape))
        self._update_tmap("inside_log_coef", self._full_nan(log_coef_shape))
        # outside probabilities
        self._update_tmap("outside_mean", self._full_nan(mean_shape))
        self._update_tmap("outside_cov", self._full_nan(cov_shape))
        self._update_tmap("outside_log_coef", self._full_nan(log_coef_shape))
        if self.multi_terminal:
            # multi terminal buffers
            self._update_tmap("multi_term_mean", self._full_nan(mean_shape))
            self._update_tmap("multi_term_cov", self._full_nan(cov_shape))
            self._update_tmap("multi_term_log_coef", self._full_nan(log_coef_shape))
            # multi-terminal distribution
            self._init_multi_term_poisson()
        if max_tree:
            self._update_tmap("max_split_indices", torch.full(log_coef_shape + (2,), fill_value=-1))
            self._update_tmap("max_split_LL", self._full_nan(log_coef_shape))
        else:
            # reinitialise to free up memory
            self._init_tmap("max_split_indices")
            self._init_tmap("max_split_LL")
            # set corresponding TMap to None
            self.max_split_indices = None
            self.max_split_LL = None

    def _init_multi_term_poisson(self):
        if self.multi_terminal:
            # multi-terminal distribution
            self.multi_terminal_lambda = self._log_multi_terminal_lambda.exp()
            self.multi_term_poisson = Poisson(self.multi_terminal_lambda)

    def _n_term_log_prob(self, n):
        if self.multi_terminal:
            return self.multi_term_poisson.log_prob(torch.ones(1) * (n - 1))
        else:
            return 0

    def _init_from_observations(self, observations: torch.Tensor, max_tree):
        """
        Initialise all necessary variable for a new set of observed data
        """
        # remember observations (convert to float32 in case it is double)
        observations = observations.float().to('cpu')
        # make sure observations have at least 3 dimensions (n_obs, n_batch, n_dim)
        if len(observations.shape) == 1:
            # interpret 1D input as single scalar data set (n_obs, 1, 1)
            observations = observations.reshape(observations.shape + (1, 1))
        elif len(observations.shape) == 2:
            # interpret 2D input as single multi-dimensional data set (n_obs, 1, n_dim)
            observations = observations.reshape(observations.shape[:1] + (1,) + observations.shape[1:])
        elif len(observations.shape) == 3:
            # nothing to do
            pass
        else:
            raise ValueError(f"Observations have more than 3 dimensions, "
                             f"don't known how to interpret shape: {observations.shape}")
        self.observations = observations
        # get correct shapes from observations
        self._create_shapes()
        # covariance matrices with correct dimensions
        self._create_covariance_matrices()
        # mixture components
        self._create_mixtures()
        # initialise TMap buffers
        self._create_tmaps(max_tree)
        # other stuff
        self._outside_uptodate = False

    def _assign_level_var(self, var, level, log_coef, mean, cov):
        getattr(self, var + "_log_coef").lslice(level)[:] = log_coef
        getattr(self, var + "_mean").lslice(level)[:] = mean
        getattr(self, var + "_cov").lslice(level)[:] = cov

    def _get_observation_mean_cov(self):
        """
        For non-categorial observations, returns the observations and the terminal transition matrix (expanded to
        correct dimension). For categorical data, adds prior counts to obtain Dirichlet and performs a Laplace
        approximation in log-space (see Milios D, Camoriano R, Michiardi P, et al (2018) Dirichlet-based Gaussian
        processes for large-scale calibrated classification. In: Advances in Neural Information Processing Systems)
        :return: tuple with (observation_means, observation_covs)
        """
        # if data is categorical transform Dirichlet counts into log space
        if self.categorical:
            # add prior counts
            obs = self.observations + self.prior_counts
            # get class-wise variances
            dirichlet_covs = torch.log(1 / obs + 1)
            # compute means
            observation_means = obs.log() - dirichlet_covs / 2
            # expand to diagonal covariance matrices
            observation_covs = torch.diag_embed(dirichlet_covs)
        else:
            observation_means = self.observations
            observation_covs = self.term_trans_cov.expand(self.n_obs,
                                                          self.n_batch,
                                                          *self.term_trans_cov.shape)
        return observation_means, observation_covs

    def _get_modes(self, log_coefs, means, precs=None, covs=None):
        assert ((precs is None) != (covs is None)) and ((precs is None) or (covs is None))
        if precs is not None:
            dist = MultivariateNormal(loc=means, precision_matrix=precs)
        else:
            dist = MultivariateNormal(loc=means, covariance_matrix=covs)
        # get maximum of distributions
        mode = dist.log_max()
        if self.checks:
            np.testing.assert_array_almost_equal(mode, dist.log_prob(means))
        # add scaling coefficients for different splits
        mode += log_coefs
        return mode

    def _multi_max(self, arr, n):
        """
        Take maximum along the first n dimensions and return values and indices
        """
        max_shape = arr.shape[:n]
        other_shape = arr.shape[n:]
        arr = arr.reshape(-1, *other_shape)
        # get max and indices
        max_split_LL, max_LL_indices = torch.max(arr, dim=0)
        # unravel indices, put into one array, and move new axis last
        return max_split_LL, np.moveaxis(np.array(np.unravel_index(max_LL_indices, max_shape)), 0, -1)

    @torch.no_grad()
    def _best_split(self, level, means, precs, log_coefs):
        """
        Get index of best split and transposition for given level and store
        """
        # get maximum of split distributions
        split_LL_mode = self._get_modes(log_coefs=log_coefs, means=means, precs=precs)
        # - need to take maximum over mixture and split dimensions (0, 1): max_shape
        # - but separately for the variables and batches: other_shape
        # --> reshape max_split_LL
        max_split_LL, max_indices = self._multi_max(split_LL_mode, 2)
        # assign max likelihood
        self.max_split_LL.lslice(level)[:] = max_split_LL
        # move new axis with indices last and assign
        self.max_split_indices.lslice(level)[:] = torch.from_numpy(max_indices)

    @torch.no_grad()
    def _best_term(self, level, means, covs, log_coefs):
        # TODO: reuse precision matrices
        # get maximum of multi-term distributions
        term_LL_mode = self._get_modes(log_coefs=log_coefs, means=means, covs=covs)
        # need to compare to LL from splitting
        term_better_indices = torch.where(term_LL_mode >= self.max_split_LL.lslice(level))
        # indicate termination by setting indices back to -1
        self.max_split_indices.lslice(level)[term_better_indices] = -1
        # replace corresponding LL values
        self.max_split_LL.lslice(level)[term_better_indices] = term_LL_mode[term_better_indices]

    def forward(self, *args, **kwargs):
        return self.compute_inside(*args, **kwargs)

    def compute_inside(self, observations, max_tree=True, reinit=True, progress=True):
        """
        Compute the likelihood of the provided observations. Along the way, compute inside probabilities.
        :param observations: observations
        :return: cross-entropy of observations (1/D * neg-log-likelihood)
        """
        if self.greedy_inside and not max_tree:
            warn(RuntimeWarning("'max_tree' implicitly set to True (greedy approximation requires max indices)"))
            max_tree = True
        if reinit:
            self._init_from_observations(observations=observations, max_tree=max_tree)
        # always init poisson to account for potentially changed parameter
        self._init_multi_term_poisson()
        # for multi terminal, keep running version per level to avoid in-place operations
        multi_term_mean = None
        multi_term_cov = None
        multi_term_log_coef_raw = None
        # get observation's mean and covariance
        observation_means, observation_covs = self._get_observation_mean_cov()
        # show progress?
        if progress:
            level_iter = tqdm(range(1, self.n_obs + 1), desc="computing inside")
        else:
            level_iter = range(1, self.n_obs + 1)
        # iterate through all levels bottom-up
        for level in level_iter:
            # initialise for first level, otherwise compute recursively
            if level == 1:
                # initialise all latent descriptive variables on the bottom level
                self._assign_level_var("inside", level,
                                       log_coef=self.terminal_log_prob + self._n_term_log_prob(1),
                                       mean=observation_means,
                                       cov=observation_covs)
                # initialise multi terminal
                if self.multi_terminal:
                    # same as inside but initial coefficient is 1 (log-coef. is 0)
                    # - need to explicitly expand dimensions (assigning new variable --> no broadcasting)
                    # - "raw" coefficients without terminal probability for recursion
                    multi_term_log_coef_raw = torch.zeros(self.n_obs, self.n_batch)
                    multi_term_mean = observation_means
                    multi_term_cov = observation_covs
            else:
                # compute mixture of Gaussians from all splitting possibilities
                # - the variables have a shape of (level, n_obs - level, n_batch, ...)
                # - the first 'level' axis is running over the possible splits
                # - the second axis runs over the different variables of the level, processed in parallel
                # - need: left and right mean, cov, and coefficients
                # - flip left long 'level' axis to have correct order for pairs matching up
                # - add one more leading dimension to account for mixture transitions
                left_mean = torch.flip(self.inside_mean.sblock[level, 1:], dims=(0,))[None]
                left_cov = torch.flip(self.inside_cov.sblock[level, 1:], dims=(0,))[None]
                left_log_coef = torch.flip(self.inside_log_coef.sblock[level, 1:], dims=(0,))[None]
                right_mean = self.inside_mean.eblock[level, 1:][None]
                right_cov = self.inside_cov.eblock[level, 1:][None]
                right_log_coef = self.inside_log_coef.eblock[level, 1:][None]

                # make broadcastable
                # - first four dimensions are: mixture, splits, variables, batches
                # - remaining dimensions are already of correct shape
                # - for mixture weights, first dimension is already present
                expand_shape = (None, None, None)
                left_non_term_cov = self.left_non_term_cov[(None,) + expand_shape]
                right_non_term_cov = self.right_non_term_cov[(None,) + expand_shape]
                log_mixture_weights = self.log_mixture_weights[(slice(None),) + expand_shape]
                # add variable dimension to transposition matrices
                left_T = self.left_T[:, None]
                right_T = self.right_T[:, None]

                # apply transpositions
                L = left_T.transpose(-2, -1).matmul(left_cov + left_non_term_cov).matmul(left_T)
                R = right_T.transpose(-2, -1).matmul(right_cov + right_non_term_cov).matmul(right_T)
                left_mean = torch.einsum('...ji,...j', left_T, left_mean)
                right_mean = torch.einsum('...ji,...j', right_T, right_mean)

                # split_* are the corresponding values for all the left/right split pairs
                # precision matrices
                if self.recompute_prec:
                    # recompute all precision matrices from scratch
                    left_split_precision = torch.inverse(L)
                    right_split_precision = torch.inverse(R)
                else:
                    # TODO adapt to mixtures
                    # get from memory (new ones are updated below)
                    left_split_precision = torch.flip(self.left_split_prec.sblock[level, 1:], dims=(0,))
                    right_split_precision = self.right_split_prec.eblock[level, 1:]
                    raise NotImplementedError("Needs to be adapted to mixture transitions")
                # coefficients
                split_log_coef_norm = MultivariateNormal(loc=left_mean, covariance_matrix=L + R)
                split_log_coef = left_log_coef + right_log_coef + split_log_coef_norm.log_prob(right_mean) + log_mixture_weights
                # add termination probability
                split_log_coef = split_log_coef + torch.log(1 - self.terminal_log_prob.exp())
                # precision/covariance
                split_prec = left_split_precision + right_split_precision
                split_cov = torch.inverse(split_prec)
                # mean
                split_mean = torch.einsum('...ab,...b->...a',
                                          split_cov,
                                          torch.einsum('...ab,...b->...a', left_split_precision, left_mean) +
                                          torch.einsum('...ab,...b->...a', right_split_precision, right_mean))

                # get index of best split (and transposition)
                if max_tree:
                    self._best_split(level=level, means=split_mean, precs=split_prec, log_coefs=split_log_coef)

                # approximate with single Gaussian
                if self.greedy_inside:
                    # pick the best mixture component and split
                    max_indices = self.max_split_indices.lslice(level)
                    advanced_index = (max_indices[:, :, 0],
                                      max_indices[:, :, 1],
                                      np.arange(split_log_coef.shape[-2])[:, None],
                                      np.arange(split_log_coef.shape[-1])[None, :])
                    # pick best mixture
                    approx_log_coef = split_log_coef[advanced_index]
                    approx_mean = split_mean[advanced_index]
                    approx_cov = split_cov[advanced_index]
                else:
                    # approximate mixtures with single Gaussian
                    approx = ApproximateMixture(log_weights=split_log_coef, means=split_mean, covariances=split_cov)
                    # approximate splits with single Gaussian
                    approx = ApproximateMixture(log_weights=approx.log_norm, means=approx.mean, covariances=approx.covariance)
                    approx_mean = approx.mean
                    approx_cov = approx.covariance
                    approx_log_coef = approx.log_norm

                # if generating multiple observed variables at once is allowed,
                # include terminating directly from this level
                if self.multi_terminal:
                    # - for level L with N variables, each of the N multi-terminal transitions generates L observations
                    # - the posterior distribution is the product of the L corresponding likelihood factors
                    # - each of the N+1 variables on the level before has already used a different set of L-1 factor
                    # - we can reuse those by selecting N of the N+1 variables and multiplying the missing likelihood factor
                    # - we need to decide whether we multiply the left-most or the right-most factor
                    # - here we reuse the N factors on the left and multiply the missing likelihood factor on the right
                    # TODO: reuse precision matrices
                    term_prod = PairwiseProduct(mean1=multi_term_mean[:-1],
                                                cov1=multi_term_cov[:-1],
                                                mean2=observation_means[level - 1:],
                                                cov2=observation_covs[level - 1:])
                    # multiply with coefficients from previous level to get overall coef and store for recursion
                    # - this should NOT include termination probability ("raw")
                    # - we need the raw coefficients for the recursive accumulation
                    multi_term_log_coef_raw = term_prod.log_norm + multi_term_log_coef_raw[:-1]
                    multi_term_mean = term_prod.mean
                    multi_term_cov = term_prod.cov
                    # add termination probabilities and Poisson over number of generated terminals
                    multi_term_log_coef = multi_term_log_coef_raw + self.terminal_log_prob + self._n_term_log_prob(level)

                    # check if multi-terminal is better than splitting
                    if max_tree:
                        self._best_term(level=level, means=multi_term_mean, covs=multi_term_cov, log_coefs=multi_term_log_coef)

                    # approximate the mixture of these distribution with those from splitting (computed above)
                    if self.greedy_inside:
                        # pick multi-terminal or split (which ever is better)
                        term_better = self.max_split_indices.lslice(level)[..., 0] == -1  # is -1 if terminal is better
                        # pick best mixture
                        term_approx_log_coef = torch.where(term_better, multi_term_log_coef, approx_log_coef)
                        term_approx_mean = torch.where(term_better[..., None], multi_term_mean, approx_mean)
                        term_approx_cov = torch.where(term_better[..., None, None], multi_term_cov, approx_cov)
                    else:
                        term_approx = ApproximateMixture(
                            log_weights=[approx_log_coef[None], multi_term_log_coef[None]],
                            means=[approx_mean[None], term_prod.mean[None]],
                            covariances=[approx_cov[None], term_prod.cov[None]],
                            cat=True)
                        term_approx_mean = term_approx.mean
                        term_approx_cov = term_approx.covariance
                        term_approx_log_coef = term_approx.log_norm
                    # assign
                    self._assign_level_var("inside", level,
                                           log_coef=term_approx_log_coef,
                                           mean=term_approx_mean,
                                           cov=term_approx_cov)
                else:
                    # directly assign mixture from splitting
                    self._assign_level_var("inside", level,
                                           log_coef=approx_log_coef,
                                           mean=approx_mean,
                                           cov=approx_cov)
            # store multi-terminal distributions
            if self.multi_terminal:
                self._assign_level_var("multi_term", level,
                                       log_coef=multi_term_log_coef_raw,
                                       mean=multi_term_mean,
                                       cov=multi_term_cov)
            if not self.recompute_prec:
                # TODO: adapt to mixtures
                # compute precision matrices for later iterations
                self.left_split_prec.lslice(level)[:] = torch.inverse(self.inside_cov.lslice(level) + self.left_non_term_cov)
                self.right_split_prec.lslice(level)[:] = torch.inverse(self.inside_cov.lslice(level) + self.right_non_term_cov)

        # product of the top inside probability with the prior distribution
        # apply transpositions
        if self.trans_inv_prior:
            if self.greedy_inside:
                raise NotImplementedError
            prior_mean = torch.einsum('...ij,...j', self.prior_T, self.prior_mean)
            # compute product (marginalise out latent variable); need to ensure correct broadcasting
            # - add mixture dimension to inside variables (first dimension)
            # - add batch dimension to prior mean (second dimension)
            mll_product = PairwiseProduct(mean1=self.inside_mean[0, self.n_obs][None], mean2=prior_mean[:, None],
                                          cov1=self.inside_cov[0, self.n_obs][None], cov2=self.prior_cov)
            # approximate mixture with single Gaussian (add uniform mixture weights)
            mll_product = ApproximateMixture(log_weights=mll_product.log_norm + np.log(1 / self.n_dim),
                                             means=mll_product.mean,
                                             covariances=mll_product.cov)
        else:
            # compute product (marginalise out latent variable)
            mll_product = PairwiseProduct(mean1=self.inside_mean[0, self.n_obs], mean2=self.prior_mean,
                                          cov1=self.inside_cov[0, self.n_obs], cov2=self.prior_cov)
        # add last inside coefficient
        self.marginal_log_likelihood = self.inside_log_coef[0, self.n_obs] + mll_product.log_norm
        # return mean over batches
        return -self.marginal_log_likelihood.mean()

    def _outside_product(self, start, end, left_right, approximate=True):
        """
        :param start: start index of variable
        :param end: end index of variable
        :param left_right: if "right", variable is generated as right child while left child is marginalised out; if
         "left" it is the inverse
        :param approximate: whether to approximate the resulting mixture with a single Gaussian or return components
        :return:
        """
        # need to reshape all tensors to (n_mixture, n_splits, n_batch..., ...)
        # - mixture dimension is already present in log_mixture_weights
        expand_shape = (None, None)
        log_mixture_weights = self.log_mixture_weights[(slice(None),) + expand_shape]
        # - select correct slice and add dimension for mixing/transpositions
        # - make covariance matrices broadcastable
        if left_right == "right":
            this_cov = self.right_non_term_cov[(None,) + expand_shape]
            other_cov = self.left_non_term_cov[(None,) + expand_shape]
            this_T = self.right_T
            other_T = self.left_T
            def slice_outside(tmap):
                return tmap.eslice[end, :start][None]
            def slice_inside(tmap):
                return tmap.eslice[start][None]
        elif left_right == "left":
            this_cov = self.left_non_term_cov[(None,) + expand_shape]
            other_cov = self.right_non_term_cov[(None,) + expand_shape]
            this_T = self.left_T
            other_T = self.right_T
            def slice_outside(tmap):
                return tmap.sslice[start, end - start:][None]
            def slice_inside(tmap):
                return tmap.sslice[end][None]
        else:
            raise ValueError(f"Unknown case '{left_right}'")
        # outside probabilities
        outside_means = slice_outside(self.outside_mean)
        outside_covs = slice_outside(self.outside_cov)
        outside_log_coefs = slice_outside(self.outside_log_coef)
        # inside probabilities
        inside_means = slice_inside(self.inside_mean)
        inside_covs = slice_inside(self.inside_cov)
        inside_log_coefs = slice_inside(self.inside_log_coef)
        # product of outside probability and Gaussian that results from marginalising out left/right child
        # - apply transpositions
        X = other_T.transpose(-2, -1).matmul(inside_covs + other_cov).matmul(other_T)
        inside_means = torch.einsum('...ji,...j', other_T, inside_means)
        prod = PairwiseProduct(mean1=outside_means, cov1=outside_covs,
                               mean2=inside_means, cov2=X)
        # Gaussian for computing coefficient
        norm = MultivariateNormal(loc=outside_means,
                                  covariance_matrix=outside_covs + X)
        # reuse inverse from MultivariateNormal for coefficient
        if self.checks:
            # make sure they are actually equal when explicitly computed
            # - only expect four decimals because inversion may be numerically quite unstable
            np.testing.assert_array_almost_equal(prod.sum_cov_inv, norm.precision_matrix, decimal=4)
        prod._sum_cov_inv = norm.precision_matrix
        # get coefficient, mean, and cov
        log_coefs = norm.log_prob(inside_means) + inside_log_coefs + outside_log_coefs + log_mixture_weights + torch.log(1 - self.terminal_log_prob.exp())
        # reverse transform of transpositions and assign
        means = torch.einsum('...ij,...j', this_T, prod.mean)
        covs = this_T.matmul(prod.cov).matmul(this_T.transpose(-2, -1)) + this_cov
        # approximate with a single Gaussian
        if approximate:
            if self.greedy_outside:
                split_LL_modes = self._get_modes(log_coefs=log_coefs, means=means, covs=covs)
                max_split_LL, max_indices = self._multi_max(split_LL_modes, 2)
                advanced_index = (max_indices[:, 0],
                                  max_indices[:, 1],
                                  np.arange(log_coefs.shape[-1]))
                return means[advanced_index], covs[advanced_index], log_coefs[advanced_index]
            else:
                # approximate mixtures with single Gaussian
                approx = ApproximateMixture(means=means, covariances=covs, log_weights=log_coefs)
                # approximate splits with single Gaussian
                approx = ApproximateMixture(means=approx.mean, covariances=approx.covariance, log_weights=approx.log_norm)
                return approx.mean, approx.covariance, approx.log_norm
        else:
            return means, covs, log_coefs

    def compute_outside(self, *args, **kwargs):
        with torch.no_grad():
            self.compute_outside_grad(*args, **kwargs)

    def compute_outside_grad(self, force=False, progress=True):
        # only proceed if update is needed or forced
        if self._outside_uptodate and not force:
            return
        self._outside_uptodate = True
        # show progress?
        if progress:
            start_iter = tqdm(range(0, self.n_obs), desc="computing outside")
        else:
            start_iter = range(0, self.n_obs)
        # go through all locations top-down (vectorisation not easily possible due to non-rectangular shapes)
        for start in start_iter:
            for end in reversed(range(start + 1, self.n_obs + 1)):
                # initialise for top element or compute recursively
                if start == 0 and end == self.n_obs:
                    # apply transpositions
                    if self.trans_inv_prior:
                        if self.greedy_outside:
                            raise NotImplementedError
                        mixture = ApproximateMixture(means=torch.einsum('...ij,...j', self.prior_T, self.prior_mean),
                                                     covariances=self.prior_cov)
                        prior_mean = mixture.mean
                        prior_cov = mixture.covariance
                    else:
                        # ...or directly use prior
                        prior_mean = self.prior_mean
                        prior_cov = self.prior_cov
                    # initialise / base case
                    self.outside_mean[0, self.n_obs] = prior_mean
                    self.outside_cov[0, self.n_obs] = prior_cov
                    self.outside_log_coef[0, self.n_obs] = 0
                else:
                    # variable can be generated as
                    # - a right child if start > 0 (i.e. there is also room for a complementing left child)
                    # - a left child if end > n_obs (i.e. there is also room for a complementing right child)
                    means_covs_coefs = []
                    if start > 0:
                        means_covs_coefs.append(self._outside_product(start, end, "right"))
                    if end < self.n_obs:
                        means_covs_coefs.append(self._outside_product(start, end, "left"))
                    # assign new outside probability
                    if len(means_covs_coefs) == 2:
                        right_means, right_covs, right_log_coefs = means_covs_coefs[0]
                        left_means, left_covs, left_log_coefs = means_covs_coefs[1]
                        if self.greedy_outside:
                            is_better = self._get_modes(log_coefs=right_log_coefs,
                                                        means=right_means,
                                                        covs=right_covs) > self._get_modes(log_coefs=left_log_coefs,
                                                                                           means=left_means,
                                                                                           covs=left_covs)
                            joint_approx_log_coef = torch.where(is_better, right_log_coefs, left_log_coefs)
                            joint_approx_mean = torch.where(is_better, right_means, left_means)
                            joint_approx_cov = torch.where(is_better, right_covs, left_covs)
                        else:
                            joint_approx = ApproximateMixture(means=[right_means[None], left_means[None]],
                                                              covariances=[right_covs[None], left_covs[None]],
                                                              log_weights=[right_log_coefs[None], left_log_coefs[None]],
                                                              cat=True)
                            joint_approx_log_coef = joint_approx.log_norm
                            joint_approx_mean = joint_approx.mean
                            joint_approx_cov = joint_approx.covariance
                        self.outside_mean[start, end] = joint_approx_mean
                        self.outside_cov[start, end] = joint_approx_cov
                        self.outside_log_coef[start, end] = joint_approx_log_coef
                    elif len(means_covs_coefs) == 1:
                        # only as right child
                        self.outside_mean[start, end] = means_covs_coefs[0][0]
                        self.outside_cov[start, end] = means_covs_coefs[0][1]
                        self.outside_log_coef[start, end] = means_covs_coefs[0][2]
                    else:
                        raise RuntimeError(f"{len(means_covs_coefs)} outside probabilities computed for {(start, end)} "
                                           f"(expected 1 or 2), this is a bug")
        # update marginal distributions per node
        self._update_nodes()

    def _update_nodes(self):
        # marginal distribution for nodes
        node_marginal = PairwiseProduct(mean1=self._inside_mean_arr, cov1=self._inside_cov_arr,
                                        mean2=self._outside_mean_arr, cov2=self._outside_cov_arr)
        self._update_tmap("node_mean", node_marginal.mean)
        self._update_tmap("node_cov", node_marginal.cov)
        self._update_tmap("node_log_coef",
                          self._inside_log_coef_arr +
                          self._outside_log_coef_arr +
                          node_marginal.log_norm -
                          self.marginal_log_likelihood)

    def max_tree(self):
        node_dict_list = []
        label_dict_list = []
        for batch_idx in range(self.n_batch):
            # get split info at the root node
            root_split = self.max_split_indices.top(1)[0, 1][batch_idx].detach().numpy()
            # node list contains node indices and split info
            node_list = [((0, self.max_split_indices.n), tuple(root_split))]
            # dicts to collect
            # - node --> [children, ...]
            # - node --> "label"
            node_dict = {}
            label_dict = {}
            # while there are unprocessed nodes
            while node_list:
                # get info for current node and pop from stack
                (parent_start, parent_end), current_node_split = node_list.pop()
                # get the splitting info --> -1 indicates terminals
                split_transposition, split_offset = current_node_split
                if split_offset == -1:
                    # is terminal
                    node_dict[(parent_start, parent_end)] = []
                    node_dict[(parent_start, parent_end)] = ""
                    continue
                # get split point from offset and construct left/right child
                split_point = parent_start + split_offset + 1
                left_start, left_end = (parent_start, split_point)
                right_start, right_end = (split_point, parent_end)
                # push children children with split info to stack
                left_child = ((left_start, left_end),
                              tuple(self.max_split_indices[left_start, left_end][batch_idx].detach().numpy()))
                right_child = ((right_start, right_end),
                               tuple(self.max_split_indices[right_start, right_end][batch_idx].detach().numpy()))
                node_list += [left_child, right_child]
                # add children and label to dicts
                node_dict[(parent_start, parent_end)] = [(left_start, left_end), (right_start, right_end)]
                label_dict[(parent_start, parent_end)] = f"{split_transposition}"
            node_dict_list.append(node_dict)
            label_dict_list.append(label_dict)
        return node_dict_list, label_dict_list

    def project_gradient(self):
        """
        Project the gradient of the log mixture weights to zero-L1-norm plane. This function must be called _after_
        backward() but _before_ calling the optimiser, as it ensures the gradient have the correct magnitude and are
        correctly scaled by optimisers such as Adam.
        """
        if self._log_mixture_weights.grad is not None:
            self._log_mixture_weights.grad -= self._zero_L1_offset(self._log_mixture_weights.grad, dim=0)
