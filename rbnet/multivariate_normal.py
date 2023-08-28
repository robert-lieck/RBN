#  Copyright (c) 2021 Robert Lieck

import math
from functools import cached_property
import operator
from typing import Union
from collections.abc import Iterable

import torch
from torch.distributions import MultivariateNormal as TorchMultivariateNormal
import numpy as np


class MultivariateNormal:
    """
    Represents a random variable with a multivariate normal distribution. It wraps PyTorch's
    :class:`torch.distributions.MultivariateNormal` (available via the :attr:`torch` property) but extends it with
    additional functionality.
    """
    __FULL__ = "__FULL__"
    __DIAG__ = "__DIAG__"
    __SCAL__ = "__SCAL__"

    @classmethod
    def _expand_scalar(cls, diag, dim):
        """
        Expand an array of scalars into an array of diagonal matrices.

        :param diag: Array of arbitrary shape (...) with scalars that specify the value on the diagonal of the matrices.
        :param dim: Dimensionality of the resulting matrices.
        :return: Array of shape (..., dim, dim) with diagonal matrices.

        .. testsetup::

            from rbnet.multivariate_normal import MultivariateNormal
            import torch

        .. testcode::

            diag = torch.arange(3, dtype=torch.float)
            print(diag)
            print(MultivariateNormal._expand_scalar(diag=diag, dim=3))

        .. testoutput::

            tensor([0., 1., 2.])
            tensor([[[0., 0., 0.],
                     [0., 0., 0.],
                     [0., 0., 0.]],

                    [[1., 0., 0.],
                     [0., 1., 0.],
                     [0., 0., 1.]],

                    [[2., 0., 0.],
                     [0., 2., 0.],
                     [0., 0., 2.]]])

        """
        diag_ = torch.zeros(diag.shape + (dim, dim))
        diag_[..., torch.arange(dim), torch.arange(dim)] = diag[..., None]
        return diag_

    @classmethod
    def _expand_diagonal(cls, diag, dim=None):
        """
        Expand an array of diagonals into an array of diagonal matrices.

        :param diag: Array of shape (..., dim) with entries that specify the values on the diagonal of the matrices.
        :param dim: Dimensionality of the resulting matrices.
        :return: Array of shape (..., dim, dim) with diagonal matrices.
        """

        if dim is None:
            dim = diag.shape[-1]
        else:
            if diag.shape[-1] != dim:
                raise ValueError(f"Last dimension of tensor has size {diag.shape[-1]}, "
                                 f"cannot expand to diagonal of size {dim}")
        mat_ = torch.zeros(diag.shape + (dim,))
        mat_[..., torch.arange(dim), torch.arange(dim)] = diag
        return mat_

    @classmethod
    def _to_tensor(cls, t):
        """
        Return `t` as a pytorch tensor. If `t` already is a tensor it is returned as is, otherwise ``torch.tensor(t)``
        is returned.
        """
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        return t

    def __init__(self, loc: torch.Tensor,
                 covariance_matrix: Union[torch.Tensor, None] = None,
                 precision_matrix: Union[torch.Tensor, None] = None,
                 scale_tril: Union[torch.Tensor, None] = None,
                 norm: Union[torch.Tensor, None] = None,
                 log_norm: Union[torch.Tensor, None] = None,
                 validate_args: Union[bool, None] = None,
                 dim: Union[int, bool] = False):
        """
        Initialise variable with given mean and covariance or precision matrix. :attr:`loc`

        :param loc:
        :param covariance_matrix:
        :param precision_matrix:
        :param scale_tril:
        :param norm:
        :param log_norm:
        :param validate_args:
        :param dim: explicitly specify the event dimensionality if `loc` is a (batch of) scalars; otherwise this is
         taken to correspond to the last axis of `loc`
        """
        # make sure at most one of 'norm' and 'log_norm' is provided
        if (norm is not None) + (log_norm is not None) > 1:
            raise ValueError("At most one of 'norm' and 'log_norm' may be specified.")
        if norm is None and log_norm is None:
            self._log_norm = self._to_tensor(0.)
        elif log_norm is not None:
            self._log_norm = self._to_tensor(log_norm)
        else:  # norm is not None
            self._log_norm = np.log(self._to_tensor(norm))

        # get the location
        self._loc = self._to_tensor(loc)
        if dim:
            # event dimensionality explicitly defined
            self._event_dim = dim
            # expand scalar locations accordingly
            self._expanded_loc = self._loc[..., None].expand(self._loc.shape + (dim,))
        else:
            # event dimensionality given by last axis of loc
            if self._loc.dim() < 1:
                # loc is scalar
                self._event_dim = self._loc.shape
            else:
                # loc is scalar
                self._event_dim = self._loc.shape[-1]
            self._expanded_loc = self._loc

        if (covariance_matrix is not None) + (scale_tril is not None) + (precision_matrix is not None) != 1:
            raise ValueError("Exactly one of covariance_matrix or precision_matrix or scale_tril must be specified.")

        if scale_tril is not None:
            self._scale_tril = mat = self._to_tensor(scale_tril)
        elif covariance_matrix is not None:
            self._covariance_matrix = mat = self._to_tensor(covariance_matrix)
        else:
            self._precision_matrix = mat = self._to_tensor(precision_matrix)

        self._mat_type = [self.__SCAL__, self.__DIAG__, self.__FULL__][mat.dim() - self._expanded_loc.dim() + 1]
        self._batch_shape = torch.broadcast_shapes(mat.shape[:len(self._expanded_loc.shape) - 1],
                                                   self._expanded_loc.shape[:-1])

        self._validate_args = validate_args
        self._torch = None

    def _expand_mat(self, mat: torch.Tensor) -> Union[torch.Tensor, None]:
        if self._mat_type == self.__SCAL__:
            mat = self._expand_scalar(mat, self._event_dim)
        elif self._mat_type == self.__DIAG__:
            mat = self._expand_diagonal(mat, self._event_dim)
        return mat

    @property
    def torch(self):
        if self._torch is None:
            if hasattr(self, "_scale_tril"):
                self._torch = TorchMultivariateNormal(loc=self._expanded_loc,
                                                      scale_tril=self._expand_mat(self._scale_tril),
                                                      validate_args=self._validate_args)
            elif hasattr(self, "_covariance_matrix"):
                self._torch = TorchMultivariateNormal(loc=self._expanded_loc,
                                                      covariance_matrix=self._expand_mat(self._covariance_matrix),
                                                      validate_args=self._validate_args)
            elif hasattr(self, "_precision_matrix"):
                self._torch = TorchMultivariateNormal(loc=self._expanded_loc,
                                                      precision_matrix=self._expand_mat(self._precision_matrix),
                                                      validate_args=self._validate_args)
        return self._torch

    def __mul__(self, other):
        if not isinstance(other, MultivariateNormal):
            return NotImplemented
        else:
            prod = PairwiseProduct(mean1=self._loc, cov1=self.torch.covariance_matrix,
                                   mean2=other._loc, cov2=other.torch.covariance_matrix)
            return MultivariateNormal(loc=prod.mean,
                                      scale_tril=prod.torch.scale_tril,
                                      log_norm=prod.log_norm + self._log_norm + other._log_norm)


class PairwiseProduct:

    def __init__(self, mean1, mean2, cov1=None, cov2=None, prec1=None, prec2=None):
        assert not (cov1 is None and prec1 is None), "either cov1 or prec1 needs to be provided"
        assert not (cov2 is None and prec2 is None), "either cov2 or prec2 needs to be provided"
        self.mean1 = mean1
        self.mean2 = mean2
        self._cov1 = cov1
        self._cov2 = cov2
        self._prec1 = prec1
        self._prec2 = prec2
        self._sum_cov = None
        self._sum_cov_inv = None
        self._cov = None
        self._prec = None
        self._torch = None

    @classmethod
    def _cost_comp(cls, cost1, cost2, comp):
        """
        Compares cost1 and cost2 using comp. Costs are numpy arrays [i, mm, mv, ma, va] with i indicating the number of
        matrix inversion, mm that of matrix-matrix multiplications, mv that of matrix-vector multiplications, ma that
        of matrix-matrix additions, and va that of vector-vector additions. Comparison is performed element-wise in this
        order, that is::

            if comp(i1, i2):
                return True
            elif not i1 == i2:
                return False

        continue with mm1 and mm2 etc...
        """
        for c1, c2 in zip(cost1, cost2):
            if comp(c1, c2):
                return True
            elif not c1 == c2:
                return False
        return False

    @classmethod
    def _cost_lt(cls, cost1, cost2):
        return cls._cost_comp(cost1=cost1, cost2=cost2, comp=operator.lt)

    @property
    def cov1(self):
        if self._cov1 is None:
            self._cov1 = torch.inverse(self.prec1)
        return self._cov1

    def _cov1_cost(self):
        return np.array([int(self._cov1 is None), 0, 0, 0, 0])

    @property
    def cov2(self):
        if self._cov2 is None:
            self._cov2 = torch.inverse(self.prec2)
        return self._cov2

    def _cov2_cost(self):
        return np.array([int(self._cov2 is None), 0, 0, 0, 0])

    @property
    def prec1(self):
        if self._prec1 is None:
            self._prec1 = torch.inverse(self.cov1)
        return self._prec1

    def _prec1_cost(self):
        return np.array([int(self._prec1 is None), 0, 0, 0, 0])

    @property
    def prec2(self):
        if self._prec2 is None:
            self._prec2 = torch.inverse(self.cov2)
        return self._prec2

    def _prec2_cost(self):
        return np.array([int(self._prec2 is None), 0, 0, 0, 0])

    @property
    def sum_cov(self):
        if self._sum_cov is None:
            self._sum_cov = self.cov1 + self.cov2
        return self._sum_cov

    def _sum_cov_cost(self):
        if self._sum_cov is None:
            return np.array([0, 0, 0, 1, 0]) + self._cov1_cost() + self._cov2_cost()
        else:
            return np.array([0, 0, 0, 0, 0])

    @property
    def sum_cov_inv(self):
        if self._sum_cov_inv is None:
            self._sum_cov_inv = torch.inverse(self.sum_cov)
        return self._sum_cov_inv

    def _sum_cov_inv_cost(self):
        if self._sum_cov_inv is None:
            return np.array([1, 0, 0, 0, 0]) + self._sum_cov_cost()
        else:
            return np.array([0, 0, 0, 0, 0])

    @property
    def prec(self):
        if self._prec is None:
            if self._cost_lt(self._prec_v1_cost(), self._prec_v2_cost()):
                self._prec = self.prec1 + self.prec2
            else:
                self._prec = torch.inverse(self.cov)
        return self._prec

    def _prec_v1_cost(self):
        """Cost of computing precision by adding prec1 and prec2"""
        return np.array([0, 0, 0, 1, 0]) + self._prec1_cost() + self._prec2_cost()

    def _prec_v2_cost(self):
        """
        Cost of computing precision by inverting cov (if cov is not present assume infinite costsfor tie-breaking).
        """
        if self._cov is None:
            return np.ones(5) * np.inf
        else:
            return np.array([1, 0, 0, 0, 0])

    def _prec_cost(self):
        if self._prec is None:
            return np.minimum(self._prec_v1_cost(), self._prec_v2_cost())
        else:
            return np.array([0, 0, 0, 0, 0])

    def _cov_v1(self):
        """Compute covariance from prec1 and prec2"""
        return torch.inverse(self.prec)

    def _cov_v1_cost(self):
        return np.array([1, 0, 0, 0, 0]) + self._prec_cost()

    def _cov_v2(self):
        """Compute covariance from cov1 and cov2"""
        return torch.matmul(torch.matmul(self.cov1, self.sum_cov_inv), self.cov2)

    def _cov_v2_cost(self):
        return np.array([0, 2, 0, 0, 0]) + self._cov1_cost() + self._cov2_cost() + self._sum_cov_inv_cost()

    @property
    def cov(self):
        """Compute covariance matrix using the least expensive method"""
        if self._cov is None:
            if self._cost_lt(self._cov_v1_cost(), self._cov_v2_cost()):
                self._cov = self._cov_v1()
            else:
                self._cov = self._cov_v2()
        return self._cov

    def _cov_cost(self):
        if self._cov is None:
            return np.minimum(self._cov_v1_cost(), self._cov_v2_cost())
        else:
            return np.array([0, 0, 0, 0, 0])

    def _mean_v1(self):
        """Compute mean from cov, prec1, and prec2"""
        return torch.einsum('...ab,...b->...a', self.cov,
                            torch.einsum('...ab,...b->...a', self.prec1, self.mean1) +
                            torch.einsum('...ab,...b->...a', self.prec2, self.mean2))

    def _mean_v1_cost(self):
        return np.array([0, 0, 3, 0, 1]) + self._cov_cost() + self._prec1_cost() + self._prec2_cost()

    def _mean_v2(self):
        """Compute mean from cov, prec1, and prec2"""
        return torch.einsum('...ab,...b->...a', self.cov2,
                            torch.einsum('...ab,...b->...a', self.sum_cov_inv, self.mean1)) + \
               torch.einsum('...ab,...b->...a', self.cov1,
                            torch.einsum('...ab,...b->...a', self.sum_cov_inv, self.mean2))

    def _mean_v2_cost(self):
        return np.array([0, 0, 4, 0, 1]) + self._sum_cov_inv_cost() + self._cov1_cost() + self._cov2_cost()

    @cached_property
    def mean(self):
        """Compute mean using the least expensive method"""
        if self._cost_lt(self._mean_v1_cost(), self._mean_v2_cost()):
            return self._mean_v1()
        else:
            return self._mean_v2()

    @cached_property
    def torch(self):
        """Return the product distribution as torch MultivariateNormal"""
        if self._torch is None:
            self._torch = TorchMultivariateNormal(loc=self.mean, covariance_matrix=self.cov)
        return self._torch

    @cached_property
    def log_norm(self):
        """Compute log-normalisation using the least expensive method"""
        # if both have equal costs use precision matrix, which is less expensive within TorchMultivariateNormal
        if self._cost_lt(self._sum_cov_cost(), self._sum_cov_inv_cost()):
            return TorchMultivariateNormal(loc=self.mean1, covariance_matrix=self.sum_cov).log_prob(self.mean2)
        else:
            return TorchMultivariateNormal(loc=self.mean1, precision_matrix=self.sum_cov_inv).log_prob(self.mean2)


class Product:

    def __init__(self,
                 means: torch.Tensor,
                 covariances: torch.Tensor = None,
                 precisions: torch.Tensor = None,
                 determinants: torch.Tensor = None,
                 scaled_means: torch.Tensor = None,
                 method: str = None):
        """
        Represents the product of N multivariate normal distributions over the same random variable. The inputs
        may additionally have an arbitrary number of batch dimensions (indicated as '...' below). The means and
        either the covariances or precisions have to be provided, the remaining parameters are computed from them
        (providing them will avoid recomputation). The 'method' argument determines which method is used for
        computing the normalisation factor. Methods return a triplet (log scaling factor (...), mean (...xD), covariance
        matrix (...xDxD) of the product), which are also available as log_norm, mean, covariance properties, respectively.

        :param means: Nx...xD array of means
        :param covariances: Nx...xDxD array of covariance matrices
        :param precisions: Nx...xDxD array of precision matrices
        :param determinants: Nx... array with determinants of the covariance matrices
        :param scaled_means: Nx...xD array with products of precision matrices and means
        :param method: method to use for computing the scaling factor (None/'default', 'iter', 'pair', 'commute')
        """
        self._means = means
        self.N = self._means.shape[0]
        self.D = self._means.shape[-1]
        # make sure we have both the covariance and precision matrices
        assert covariances is not None or precisions is not None
        if covariances is not None:
            self._covariances = covariances
            self._precisions = torch.inverse(covariances)
        else:
            self._covariances = torch.inverse(precisions)
            self._precisions = precisions
        # compute determinants
        if determinants is None:
            self._determinants = torch.det(self._covariances)
        else:
            self._determinants = determinants
        # compute scaled means
        if scaled_means is None:
            self._scaled_means = torch.einsum('n...ab,n...b->n...a', self._precisions, self._means)
        else:
            self._scaled_means = scaled_means
        # compute parameters of product distribution
        self.precision = self._precisions.sum(dim=0)
        self.covariance = torch.inverse(self.precision)
        self.mean = torch.einsum('...ab,...b->...a', self.covariance, self._scaled_means.sum(dim=0))
        self.det = torch.det(self.covariance)
        # compute normalisation factor
        if method is None or method == 'default':
            self.log_norm, _, _ = self.product()
        elif method == 'iter':
            self.log_norm, _, _ = self.iter_product(means=self._means, covariances=self._covariances)
        elif method == 'pair':
            self.log_norm = PairwiseProduct(mean1=self._means[0], cov1=self._covariances[0],
                                            mean2=self._means[1], cov2=self._covariances[1]).log_norm
        elif method == 'commute':
            self.log_norm, _, _ = self.commuting_product()
        else:
            raise ValueError(f"Unknown method '{method}'")

    def product(self):
        quad_factor = torch.einsum('n...a,n...a->n...', self._scaled_means, self._means).sum(dim=0)
        mixed_factor = torch.einsum('n...a,m...a->nm...',
                                    torch.einsum('...ab,n...b->n...a', self.covariance, self._scaled_means),
                                    self._scaled_means).sum(dim=(0, 1))
        exp_factor = -(quad_factor - mixed_factor) / 2
        div = self.det.log() - self._determinants.log().sum(dim=0)
        pi = math.log(2 * math.pi) * (-self.D * (self.N - 1))
        det_factor = (pi + div) / 2
        return det_factor + exp_factor, self.mean, self.covariance

    @classmethod
    def iter_product(cls, means, covariances):
        ret_log_fac = 0
        ret_mean = None
        ret_cov = None
        for m, cov in zip(means, covariances):
            if ret_mean is None and ret_cov is None:
                ret_mean = m
                ret_cov = cov
                continue
            pp = PairwiseProduct(mean1=ret_mean, cov1=ret_cov, mean2=m, cov2=cov)
            ret_log_fac += pp.log_norm
            ret_mean = pp.mean
            ret_cov = pp.cov
        return ret_log_fac, ret_mean, ret_cov

    def commuting_product(self):
        exp_factor = 0
        for i in range(self.N):
            for j in range(i + 1, self.N):
                cov_ij = self._covariances[j].matmul(self.precision).matmul(self._covariances[i])
                exp_factor += TorchMultivariateNormal(self._means[i], cov_ij).log_prob(self._means[j])
        det_factor = (math.log(2 * math.pi) * (self.D * (self.N - 1) * (self.N - 2) / 2) +
                      (self._determinants.log() * (self.N - 2)).sum(dim=0) -
                      self.det.log() * (self.N * (self.N - 1) / 2 - 1)
                      ) / 2
        return det_factor + exp_factor, self.mean, self.covariance


class ApproximateMixture:

    def __init__(self,
                 means: Union[torch.Tensor, Iterable[torch.Tensor]],
                 log_weights: Union[torch.Tensor, Iterable[torch.Tensor]] = None,
                 covariances: Union[torch.Tensor, Iterable[torch.Tensor]] = None,
                 cat=False):
        """
        Approximate a mixture of N multivariate normal distributions with a single one by matching moments (equivalent to
        minimising the KL-divergence or cross-entropy from the mixture to the approximation, and the neg-log-likelihood
        if means are data points).

        :param means: (N,X,D) array with means / locations of data points; X are arbitrary batch dimensions
        :param log_weights: (N,X) array of weights of the components (optional; default is to assume uniform weights)
        :param covariances: (N,X,D,D) array of covariance matrices of the components (optional; default is to assume
         zero covariance, which corresponds to the components being treated as single data points)
        :param cat: If true assume the inputs are iterables of tensors, which need to be concatenated first
        """
        # first concatenate inputs if requested
        if cat:
            means = torch.cat(tuple(means), dim=0)
            if log_weights is not None:
                log_weights = torch.cat(tuple(log_weights), dim=0)
            if covariances is not None:
                covariances = torch.cat(tuple(covariances), dim=0)
        # remember means
        self._means = means
        # get dimensions
        self.N = self._means.shape[0]
        self.D = self._means.shape[-1]
        # init uniform weights if not provided; get normalisation
        if log_weights is None:
            log_weights = (torch.ones(self._means.shape[:-1]) / self.N).log()
        self.log_norm = log_weights.logsumexp(dim=0)
        self.norm_log_weights = log_weights - self.log_norm
        self.norm_weights = self.norm_log_weights.exp()
        # compute mean
        self.mean = (self.norm_weights[:, ..., None] * self._means).sum(dim=0)
        # compute covariance component of means
        diff = self._means - self.mean[None, ..., :]
        mean_cov = (self.norm_weights[..., None, None] * diff[..., :, None] * diff[..., None, :]).sum(dim=0)
        # compute covariance
        if covariances is None:
            # only means component is non-zero (case for single data points)
            self.covariance = mean_cov
        else:
            # compute component of covariances and add to component of means
            self._cov_cov = (self.norm_weights[:, ..., None, None] * covariances).sum(dim=(0))
            self.covariance = mean_cov + self._cov_cov
