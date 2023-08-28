#  Copyright (c) 2021 Robert Lieck

from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import torch
from torch.distributions import MultivariateNormal as TorchMultivariateNormal
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import seaborn as sns

from rbnet.multivariate_normal import Product, ApproximateMixture, PairwiseProduct, MultivariateNormal

rng = np.random.default_rng(seed=0)


class TestMultivariateNormal(TestCase):

    def test_init(self):
        for _ in range(10):
            # use a random batch shape
            batch_shape = tuple(rng.integers(1, 4, rng.integers(0, 4)))
            # use random dimensionality
            dim = rng.integers(1, 5)
            # scalar versus full locations
            for loc_dim in [False, dim]:
                if loc_dim:
                    # full loc dimensions
                    loc = torch.Tensor(rng.uniform(-1, 1, batch_shape))
                    full_loc = torch.unsqueeze(loc, -1).expand(batch_shape + (dim,))
                else:
                    loc = torch.Tensor(rng.uniform(-1, 1, batch_shape + (dim,)))
                    full_loc = loc
                for mat_type in ["FULL", "DIAG", "SCAL"]:
                    if mat_type == "SCAL":
                        mat = torch.Tensor(rng.uniform(-1, 1, batch_shape)) ** 2
                        full_mat = MultivariateNormal._expand_scalar(mat, dim)
                    elif mat_type == "DIAG":
                        mat = torch.Tensor(rng.uniform(-1, 1, batch_shape + (dim,))) ** 2
                        full_mat = MultivariateNormal._expand_diagonal(mat, dim)
                    else:
                        mat = torch.Tensor(rng.uniform(-1, 1, batch_shape + (dim, dim)))
                        mat = mat @ torch.transpose(mat, -1, -2)
                        full_mat = mat
                    for cov_prec_tri in ["cov", "prec", "tri"]:
                        if cov_prec_tri == "cov":
                            n = MultivariateNormal(loc=loc, covariance_matrix=mat, dim=loc_dim, validate_args=True)
                            m = TorchMultivariateNormal(loc=full_loc, covariance_matrix=full_mat, validate_args=True)
                        elif cov_prec_tri == "prec":
                            n = MultivariateNormal(loc=loc, precision_matrix=mat, dim=loc_dim, validate_args=True)
                            m = TorchMultivariateNormal(loc=full_loc, precision_matrix=full_mat, validate_args=True)
                        else:
                            if mat_type == "FULL":
                                tri = torch.linalg.cholesky(mat)
                                full_tri = tri
                            elif mat_type == "DIAG":
                                tri = torch.sqrt(mat)
                                full_tri = MultivariateNormal._expand_diagonal(tri, dim)
                            else:
                                tri = torch.sqrt(mat)
                                full_tri = MultivariateNormal._expand_scalar(tri, dim)
                            n = MultivariateNormal(loc=loc, scale_tril=tri, dim=loc_dim, validate_args=True)
                            m = TorchMultivariateNormal(loc=full_loc, scale_tril=full_tri, validate_args=True)
                        assert_array_equal(n.torch.loc, m.loc)
                        assert_array_equal(n.torch._unbroadcasted_scale_tril, m._unbroadcasted_scale_tril)


class TestProduct(TestCase):

    def get_rnd_gaussians(self, N, D, commuting, batch_dims,
                          std_scale=1, min_var=0.01, min_weights=0.1, max_weights=1, mean_range=3,):
        if commuting and 2 <= D <= 3:
            # construct commuting covariance matrices by first getting random diagonal matrices
            # and then rotating all of them by the same random rotation matrix
            diag_cov = np.zeros((N, D, D))
            diag_cov[:, np.arange(D), np.arange(D)] = rng.uniform(-std_scale,
                                                                       std_scale,
                                                                       (N, D)) ** 2 + min_var
            # get random rotation matrix
            if D == 2:
                # for 2D only rotate about x and choose yz part as 2D rotation matrix
                r = R.from_euler('x', rng.uniform(0, 360), degrees=True).as_matrix()[1:, 1:]
            else:
                # for 3D rotate about 3 axes
                r = R.from_euler('xyz', rng.uniform(0, 360, 3), degrees=True).as_matrix()
            covariances = np.einsum('ba,nbc,cd->nad', r, diag_cov, r)
        else:
            covariances = rng.uniform(-std_scale, std_scale, (N, D, D))
            covariances = np.einsum('nax,nbx->nab', covariances, covariances)
            covariances += min_var * np.eye(D)[None, :, :]
        covariances = torch.from_numpy(covariances)
        means = rng.uniform(-mean_range, mean_range, (N, D))
        means = torch.from_numpy(means)
        weights = torch.from_numpy(rng.uniform(min_weights, max_weights, N) ** 2)
        weights /= weights.sum(dim=-1)

        # add batch dimensions with copies
        new_batch_dims = (None,) * len(batch_dims)  # to add batch dimensions
        col = (slice(None),)  # colon : to keep dimension
        batched_means = means[
            col + new_batch_dims + col
            ].expand(N, *batch_dims, D)
        batched_covariances = covariances[
            col + new_batch_dims + col + col
            ].expand(N, *batch_dims, D, D)
        batched_weights = weights[
            col + new_batch_dims
            ].expand(N, *batch_dims)
        return means, covariances, weights, batched_means, batched_covariances, batched_weights, new_batch_dims, col

    def test_product(self):
        debug = False
        if debug:
            do_plot = True
            batch_dim_list = [()]
            n_dimensions = [2]
            n_components = [2, 3]
            commute_list = [False]
            n_random_tests = 3
        else:
            do_plot = False
            batch_dim_list = [(), (1,), (2,), (1, 1), (1, 2), (2, 1), (2, 2)]
            n_dimensions = range(1, 5)
            n_components = range(2, 6)
            commute_list = [False, True]
            n_random_tests = 5

        def msg(*args, **kwargs):
            pass
            # print(*args, **kwargs)

        n_samples = 300
        plot_range = 5
        n_grid_points = 20000

        # different dimensions
        for D in n_dimensions:
            msg(f"D={D}")
            # only plot if 2D and requested
            _do_plot = do_plot
            if D != 2:
                _do_plot = False
            # always use approx. same number of points --> choose n_grid depending on D
            n_grid = int(np.power(n_grid_points, 1 / D))
            msg(f"n_grid={n_grid}")
            # different number of components
            for N in n_components:
                msg(f"N={N}")
                # commuting and non-commuting
                for commuting in commute_list:
                    msg(f"comm={commuting}")
                    if commuting and D > 3:
                        msg("skip")
                        continue
                    # for different additional batch dimensions
                    for batch_dims in batch_dim_list:
                        msg("batch:", batch_dims)
                        # multiple randomised tests
                        for _ in range(n_random_tests):
                            # define distributions
                            (means, covariances, weights,
                             batched_means, batched_covariances, batched_weights,
                             new_batch_dims, col) = self.get_rnd_gaussians(N=N,
                                                                           D=D,
                                                                           commuting=commuting,
                                                                           batch_dims=batch_dims)

                            # get product distribution
                            prod = Product(means=batched_means, covariances=batched_covariances)
                            # reinitialise by explicitly providing parameters
                            prod = Product(means=batched_means,
                                           precisions=prod._precisions,
                                           determinants=prod._determinants,
                                           scaled_means=prod._scaled_means)
                            batched_log_fac, batched_m, batched_cov = prod.product()
                            log_fac, m, cov = Product(means=means,
                                                      covariances=covariances).product()
                            # test that batching before and after amount to the same
                            if batch_dims != ():
                                np.testing.assert_array_almost_equal(log_fac[new_batch_dims].expand(*batch_dims),
                                                                     batched_log_fac)
                                np.testing.assert_array_almost_equal(m[new_batch_dims + col].expand(*batch_dims, D),
                                                                     batched_m)
                                np.testing.assert_array_almost_equal(cov[new_batch_dims + col + col].expand(*batch_dims,
                                                                                                            D, D),
                                                                     batched_cov)

                            # compare against results from alternative/specialised methods
                            if commuting:
                                # commutative matrices
                                prod = Product(means=batched_means, covariances=batched_covariances, method='commute')
                                np.testing.assert_array_almost_equal(batched_log_fac, prod.log_norm)
                                np.testing.assert_array_almost_equal(batched_m, prod.mean)
                                np.testing.assert_array_almost_equal(batched_cov, prod.covariance)
                            if N == 2:
                                # pairwise
                                prod = Product(means=batched_means, covariances=batched_covariances, method='pair')
                                np.testing.assert_array_almost_equal(batched_log_fac, prod.log_norm)
                                np.testing.assert_array_almost_equal(batched_m, prod.mean)
                                np.testing.assert_array_almost_equal(batched_cov, prod.covariance)
                                pp = PairwiseProduct(mean1=batched_means[0], cov1=batched_covariances[0],
                                                     mean2=batched_means[1], cov2=batched_covariances[1])
                                np.testing.assert_array_almost_equal(batched_log_fac, pp.log_norm)
                                np.testing.assert_array_almost_equal(batched_m, pp.mean)
                                np.testing.assert_array_almost_equal(batched_cov, pp.cov)
                            # general case by iterating pairwise
                            prod = Product(means=batched_means, covariances=batched_covariances, method='iter')
                            np.testing.assert_array_almost_equal(batched_log_fac, prod.log_norm)
                            np.testing.assert_array_almost_equal(batched_m, prod.mean)
                            np.testing.assert_array_almost_equal(batched_cov, prod.covariance)
                            iter_log_fac, iter_mean, iter_cov = Product.iter_product(means=batched_means,
                                                                                     covariances=batched_covariances)
                            np.testing.assert_array_almost_equal(batched_log_fac, iter_log_fac)
                            np.testing.assert_array_almost_equal(batched_m, iter_mean)
                            np.testing.assert_array_almost_equal(batched_cov, iter_cov)
                            # ValueError on bad method
                            self.assertRaises(ValueError, lambda: Product(means=batched_means,
                                                                          covariances=batched_covariances,
                                                                          method="bad method"))

                            # assert that approximate distribution minimises KL-divergence
                            # get parameters of the approximate distribution
                            batched_approx_mix = ApproximateMixture(means=batched_means,
                                                                    log_weights=batched_weights.log(),
                                                                    covariances=batched_covariances)
                            # require grad for tensors
                            batched_approx_mix.mean.requires_grad_(True)
                            batched_approx_mix.covariance.requires_grad_(True)
                            # get distribution object
                            batched_approx_mix_dist = TorchMultivariateNormal(
                                loc=batched_approx_mix.mean,
                                covariance_matrix=batched_approx_mix.covariance
                            )
                            # compute KL-divergence with mixture components and weighted sum of that
                            kld = 0
                            for comp_idx in range(N):
                                kld += batched_weights[comp_idx] * torch.distributions.kl_divergence(
                                    TorchMultivariateNormal(loc=batched_means[comp_idx],
                                                       covariance_matrix=batched_covariances[comp_idx]),
                                    batched_approx_mix_dist
                                )
                            # compute gradient
                            if batch_dims == ():
                                kld.backward()
                            else:
                                kld.backward(torch.ones(batch_dims))
                            # check it is zero
                            np.testing.assert_array_almost_equal(batched_approx_mix.mean.grad, 0)
                            np.testing.assert_array_almost_equal(batched_approx_mix.covariance.grad, 0)
                            # stop requiring grad for tensors
                            batched_approx_mix.mean.requires_grad_(False)
                            batched_approx_mix.covariance.requires_grad_(False)

                            # check that default corresponds to uniform weights and zero covariances
                            zero_batched_approx_mix = ApproximateMixture(
                                means=batched_means,
                                log_weights=(torch.ones_like(batched_weights) / N).log(),
                                covariances=torch.zeros_like(batched_covariances))
                            default_batched_approx_mix = ApproximateMixture(means=batched_means)
                            np.testing.assert_array_almost_equal(zero_batched_approx_mix.log_norm,
                                                                 default_batched_approx_mix.log_norm)
                            np.testing.assert_array_almost_equal(zero_batched_approx_mix.mean,
                                                                 default_batched_approx_mix.mean)
                            np.testing.assert_array_almost_equal(zero_batched_approx_mix.covariance,
                                                                 default_batched_approx_mix.covariance)

                            # define grid for heatmap and testing
                            # 1D space used for each dimension
                            grid = np.linspace(-plot_range, plot_range, n_grid, endpoint=True)
                            # meshgrid with D dimensions (tuple of coordinates)
                            xyz = np.meshgrid(*(grid,) * D, sparse=False)
                            # flatten and concatenate along new dimension to get grid of coordinates
                            locs = np.concatenate(tuple(l.flatten()[..., None] for l in xyz), axis=-1)

                            # compute value at grid points
                            normal = TorchMultivariateNormal(loc=means, covariance_matrix=covariances)
                            log_probs = normal.log_prob(torch.from_numpy(locs)[:, None, ...]).reshape(*(n_grid,) * D, -1)

                            # mixture
                            mixture_log_probs = (log_probs + weights[None, None, :]).logsumexp(dim=-1)
                            mixture_probs = mixture_log_probs.exp()
                            # approximate mixture
                            approx_mix = ApproximateMixture(means=means, log_weights=weights, covariances=covariances)
                            approx_mix_dist = TorchMultivariateNormal(
                                loc=approx_mix.mean,
                                covariance_matrix=approx_mix.covariance
                            )
                            approx_mixture_probs = approx_mix_dist.log_prob(
                                torch.from_numpy(locs)[:, None, ...]
                            ).reshape(*(n_grid,) * D).exp()

                            # product
                            product_log_probs = log_probs.sum(dim=-1)
                            product_probs = product_log_probs.exp()
                            # analytic product
                            prod_normal = TorchMultivariateNormal(loc=m, covariance_matrix=cov)
                            ana_prod_log_probs = prod_normal.log_prob(torch.from_numpy(locs)[:, None, ...])
                            ana_prod_log_probs = ana_prod_log_probs + log_fac
                            ana_prod_log_probs = ana_prod_log_probs.reshape(*(n_grid,) * D)
                            ana_prod_probs = ana_prod_log_probs.exp()

                            # test analytic product against explicitly computed for values on grid
                            # msg(ana_prod_probs / product_probs)
                            np.testing.assert_array_almost_equal(ana_prod_probs, product_probs)

                            # plot
                            if _do_plot:
                                fig, axes = plt.subplots(2, 2, figsize=(8, 8))
                                axes[0, 0].set_title("Mixture")
                                axes[0, 1].set_title("Product")
                                axes[1, 0].set_title("Mixture (Approximated)")
                                axes[1, 1].set_title("Product (Analytical)")

                                # generate samples
                                samples = normal.sample((n_samples,))
                                component = rng.choice(N, size=n_samples, p=weights)
                                samples = samples[np.arange(n_samples), component, :]

                                # plot distributions and samples
                                prob_list = [mixture_probs, product_probs, approx_mixture_probs, ana_prod_probs]
                                ax_list = [axes[0][0], axes[0][1], axes[1][0], axes[1][1]]
                                weights_as_labels = [True, False, True, False]
                                # vmax = np.concatenate(prob_list).max() # jointly normalise
                                # vmax = product_probs.max() # normalise product
                                vmax = None # do not normalise
                                for prob, ax, w_as_l in zip(prob_list, ax_list, weights_as_labels):
                                    ax.contourf(grid, grid, prob, 100, zorder=-10, cmap='Reds', vmin=0, vmax=vmax)
                                    g = sns.scatterplot(x=samples[:, 0],
                                                        y=samples[:, 1],
                                                        hue=component, s=10,
                                                        palette='muted',
                                                        ax=ax)
                                    if w_as_l:
                                        g.legend_.set_title("weights")
                                    for t, l in zip(g.legend_.texts, [f"{w:.3f}" if w_as_l else "" for w in weights]):
                                        t.set_text(l)
                                    ax.set_aspect('equal')
                                    ax.set_xlim(-plot_range, plot_range)
                                    ax.set_ylim(-plot_range, plot_range)

                                # show in tight layout
                                fig.tight_layout()
                                plt.show()

    def property_none(self, obj, props, not_none):
        """
        Test if properties of obj are None or not None.
        :param obj: object to test for
        :param props: all properties to test (both None and not None)
        :param not_none: properties that are expected to be None
        """
        # test that not_none is subset of props
        diff = set(not_none) - set(props)
        self.assertFalse(diff, f"some properties ({diff}) are not in list")
        # test for None / not None
        for p in props:
            self.assertTrue(hasattr(obj, p), f"{obj} has not attribute {p}")
            if p in not_none:
                self.assertIsNotNone(getattr(obj, p), f"{obj}.{p} is None ({getattr(obj, p)})")
            else:
                self.assertIsNone(getattr(obj, p), f"{obj}.{p} is not None ({getattr(obj, p)})")

    def test_pairwise_product(self):
        batch_dim_list = [(), (1,), (2,), (1, 1), (1, 2), (2, 1), (2, 2)]
        n_dimensions = range(1, 5)
        n_random_tests = 5
        for D in n_dimensions:
            for batch_dims in batch_dim_list:
                for _ in range(n_random_tests):
                    (means, covariances, weights,
                     batched_means, batched_covariances, batched_weights,
                     new_batch_dims, col) = self.get_rnd_gaussians(N=2, D=D, commuting=False, batch_dims=batch_dims)
                    precisions = torch.inverse(covariances)
                    # initialise from covariances and precisions
                    pp_cov = PairwiseProduct(mean1=means[0], mean2=means[1], cov1=covariances[0], cov2=covariances[1])
                    pp_prec = PairwiseProduct(mean1=means[0], mean2=means[1], prec1=precisions[0], prec2=precisions[1])
                    props = {"_cov1", "_cov2", "_prec1", "_prec2", "_sum_cov", "_sum_cov_inv", "_prec", "_cov"}
                    # check that the respective properties are (are not) present
                    self.property_none(pp_cov, props, {"_cov1", "_cov2"})
                    self.property_none(pp_prec, props, {"_prec1", "_prec2"})

                    # compute mean
                    np.testing.assert_array_almost_equal(pp_cov.mean, pp_prec.mean)
                    # for given covariance (first case) the mean should have been computed via the inverted the sum of
                    # covariances
                    self.property_none(pp_cov, props, {"_cov1", "_cov2", "_sum_cov", "_sum_cov_inv"})
                    # for given precisions (second case) the mean should have been computed via the covariance by
                    # inverting the precision
                    self.property_none(pp_prec, props, {"_prec1", "_prec2", "_prec", "_cov"})

                    # computing log_norm requires sum_cov or sum_cov_inv
                    np.testing.assert_array_almost_equal(pp_cov.log_norm, pp_prec.log_norm)
                    # first case does not change
                    self.property_none(pp_cov, props, {"_cov1", "_cov2", "_sum_cov", "_sum_cov_inv"})
                    # in second case cov1 and cov2 need also be computed
                    self.property_none(pp_prec, props, {"_prec1", "_prec2", "_prec", "_cov", "_cov1", "_cov2",
                                                             "_sum_cov"})

                    # compute covariance
                    np.testing.assert_array_almost_equal(pp_cov.cov, pp_prec.cov)
                    # in first case cov is computed from sum_cov_inv, in second case it is already computed
                    self.property_none(pp_cov, props, {"_cov1", "_cov2", "_sum_cov", "_sum_cov_inv", "_cov"})
                    self.property_none(pp_prec, props, {"_prec1", "_prec2", "_prec", "_cov", "_cov1", "_cov2",
                                                             "_sum_cov"})

                    # in fist case, prec is computed by inverting cov
                    np.testing.assert_array_almost_equal(pp_cov.prec, pp_prec.prec)
                    self.property_none(pp_cov, props, {"_cov1", "_cov2", "_sum_cov", "_sum_cov_inv", "_cov",
                                                            "_prec"})
                    # so prec1 and prec2 are the only properties not computed
                    np.testing.assert_array_almost_equal(pp_cov.prec1, pp_prec.prec1)
                    np.testing.assert_array_almost_equal(pp_cov.prec2, pp_prec.prec2)
                    self.property_none(pp_cov, props, props)

                    # in second case, computing sum_cov_inv forces computing the remaining properties
                    np.testing.assert_array_almost_equal(pp_cov.sum_cov_inv, pp_prec.sum_cov_inv)
                    self.property_none(pp_prec, props, props)

                    # all costs should be zero now (everything is computed)
                    cov_costs = np.concatenate(tuple(getattr(pp_cov, p + "_cost")() for p in props))
                    prec_costs = np.concatenate(tuple(getattr(pp_prec, p + "_cost")() for p in props))
                    np.testing.assert_array_equal(cov_costs, 0)
                    np.testing.assert_array_equal(prec_costs, 0)
