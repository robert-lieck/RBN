#  Copyright (c) 2021 Robert Lieck

from warnings import warn
from numbers import Number
from collections import namedtuple
from typing import Union

import numpy as np
import torch

from util import MultivariateNormal


class SampleRBN:

    def __init__(self,
                 prior_mean=(0.,), prior_std=(1.,),
                 left_non_terminal_offset=0., right_non_terminal_offset=0., terminal_offset=0.,
                 left_non_terminal_std=(0.1,), right_non_terminal_std=(0.1,), terminal_std=(0.01,),
                 left_rotations=(0,), right_rotations=(0,), rotation_weights=(1.,),
                 terminal_prob:Union[Number, tuple] = 0.7, multi_terminal_mean=None, multi_terminal_std=None,
                 terminal_prob_list=None,
                 n_dim=None):
        if isinstance(terminal_prob, Number):
            terminal_prob = (terminal_prob, terminal_prob)
        self.terminal_prob = np.array(terminal_prob)
        assert self.terminal_prob.shape == (2,), self.terminal_prob.shape
        assert np.sum(self.terminal_prob) > 1, self.terminal_prob
        self.terminal_prob_list = terminal_prob_list
        # get prior mean as tensor and expand if required
        prior_mean = torch.Tensor(prior_mean)
        if n_dim is not None and len(prior_mean) != n_dim:
            prior_mean = prior_mean.expand(n_dim)
        #
        self.left_non_terminal_offset = left_non_terminal_offset
        self.right_non_terminal_offset = right_non_terminal_offset
        self.terminal_offset = terminal_offset
        # construct transition noise distributions
        self._prior_normal = MultivariateNormal(loc=prior_mean,
                                                covariance_matrix=self._cov(prior_std, n_dim))
        self._left_non_terminal_normal = MultivariateNormal(loc=torch.zeros_like(prior_mean),
                                                            covariance_matrix=self._cov(left_non_terminal_std, n_dim))
        self._right_non_terminal_normal = MultivariateNormal(loc=torch.zeros_like(prior_mean),
                                                             covariance_matrix=self._cov(right_non_terminal_std, n_dim))
        self._terminal_normal = MultivariateNormal(loc=torch.zeros_like(prior_mean),
                                                   covariance_matrix=self._cov(terminal_std, n_dim))
        # non-terminal rotations
        self.left_rotations = np.array(left_rotations)
        self.right_rotations = np.array(right_rotations)
        self.rotation_weights = np.array(rotation_weights)
        self.rotation_weights /= self.rotation_weights.sum()
        assert self.rotation_weights.min() >= 0
        assert len(self.left_rotations) == len(self.right_rotations) == len(self.rotation_weights)
        # remember lambda for multi-terminal Poisson distribution
        self.multi_terminal_mean = multi_terminal_mean
        self.multi_terminal_std = multi_terminal_std

    def _cov(self, sqrt, n_dim=None):
        # make float tensor if necessary
        if not isinstance(sqrt, torch.Tensor):
            sqrt = torch.Tensor(sqrt)
        # create full matrix from diagonal, first expand if needed
        if len(sqrt.shape) < 2:
            if n_dim is None:
                raise ValueError("Need to provide 'n_dim' to expand")
            sqrt = torch.diagflat(sqrt.expand(n_dim))
        return torch.matmul(sqrt, sqrt.t())

    def sample_prior(self):
        return self._prior_normal.sample().numpy()

    def sample_non_terminal(self, parent):
        left = parent + self.left_non_terminal_offset + self._left_non_terminal_normal.sample().numpy()
        right = parent + self.right_non_terminal_offset + self._right_non_terminal_normal.sample().numpy()
        rotation_idx = np.random.choice(len(self.rotation_weights), p=self.rotation_weights)
        left = np.roll(left, self.left_rotations[rotation_idx])
        right = np.roll(right, self.right_rotations[rotation_idx])
        return left, right

    def sample_terminal(self, parent):
        n = self.multi_terminal_dist()
        return self._terminal_normal.sample((n,)).numpy() + parent

    def multi_terminal_dist(self):
        if self.multi_terminal_mean is None:
            return 1
        elif self.multi_terminal_std is None:
            # draw from Poisson with specified mean
            return int(1 + torch.poisson(torch.ones(1) * self.multi_terminal_mean))
        else:
            # draw from Binomial with specified mean and standard deviation
            # n p = mean, n p q = var
            # p = 1 - var/mean, n = mean/p
            # generate by drawing n time from Bernoulli and summing up
            p = 1 - self.multi_terminal_std ** 2 / self.multi_terminal_mean
            n = int(round(self.multi_terminal_mean / p))
            return int(torch.bernoulli(torch.ones(n) * p).sum())

    def min_max(self, min_size=0, max_size=np.inf, max_tries=10 ** 4, only_warn=False, info=True, **kwargs):
        # for printing
        print_width = 65
        hline = "+" + "-" * print_width
        HLINE = "+" + "=" * print_width
        if info:
            print(HLINE)
            # print(f"| generating with terminal probability p = {self.terminal_prob:.4f}")
            print(f"| generating with terminal probability p = {self.terminal_prob}")
            print(hline)
        # init variables
        s_high = 0
        s_low = 0
        n_high = 0
        n_low = 0
        n_total = 0
        best_ret = None
        best_diff = np.inf
        best_l = 0
        # make multiple tries of finding sequence within bounds
        max_exceeded = False
        for n_total in range(1, max_tries + 1):
            ret = self.generate(**kwargs)
            l = len(ret[0])
            # remember best result
            diff = min(abs(l - min_size), abs(l - max_size))
            if diff < best_diff:
                best_ret = ret
                best_diff = diff
                best_l = l
            # check for constraints
            if l < min_size:
                n_low += 1
                s_low += l
            elif l > max_size:
                n_high += 1
                s_high += l
            else:
                # constraints met --> this is best, break
                best_diff = 0
                best_ret = ret
                best_l = l
                break
        else:
            max_exceeded = True
        # print info
        if info:
            if max_exceeded:
                print(f"| FAILED generating a sequence within bounds: {min_size}/{max_size}, diff: {best_diff}")
            else:
                print(f"| generated sequence of length {best_l} (bounds: {min_size}/{max_size}, diff: {best_diff})")
            print(hline)
            print(f"|          {'total':>10} | {'low':>10} | {'high':>10}")
            print(hline)
            print(f"|   tries: {n_total:10} | {n_low:10} | {n_high:10}")
            print(f"| samples: {(s_low + s_high):10} | {s_low:10} | {s_high:10}")
            print(hline)
        if max_exceeded:
            msg = f"Exceeded maximum number of tries ({max_tries}). Check bounds and terminal probability."
            if only_warn:
                warn(RuntimeWarning(msg))
            else:
                raise RuntimeError(msg)
        # return best result
        return best_ret

    def generate(self, tree=True, collapse_multi_terminal=True, id_labels=False):
        # list of terminal and non-terminal nodes
        Node = namedtuple('Node', ['value', 'ID', 'term_prob'])
        terms = []
        non_terms = [Node(value=self.sample_prior(), ID=0, term_prob=0.)]
        # IDs for tree reconstruction
        n_generated_nodes = 0
        n_processed_nodes = 0
        id_dict = {}
        label_dict = {}
        while non_terms:
            parent_value, parent_id, parent_term_prob = non_terms.pop()
            if self.terminal_prob_list is not None:
                parent_term_prob = self.terminal_prob_list[n_processed_nodes]
            n_processed_nodes += 1
            label_dict[parent_id] = parent_value
            if np.random.uniform(0, 1) < parent_term_prob:
                # terminate from this node
                # create children (leaves) add to terminals
                child_values = list(self.sample_terminal(parent_value))
                child_ids = np.arange(n_generated_nodes + 1, n_generated_nodes + len(child_values) + 1)
                terms += [Node(value=v, ID=i, term_prob=None) for v, i in zip(child_values, child_ids)]
                n_generated_nodes += len(child_values)
                # remember parent child relation
                id_dict[parent_id] = list(child_ids)
                for ii, vv in zip(child_ids, child_values):
                    id_dict[ii] = []
                    label_dict[ii] = vv
            else:
                left_value, right_value = self.sample_non_terminal(parent_value)
                left_id, right_id = n_generated_nodes + 1, n_generated_nodes + 2
                n_generated_nodes += 2
                # create two new non-terminals (inverted order because pop takes from the end!)
                non_terms += [Node(value=right_value, ID=right_id, term_prob=self.terminal_prob[1]),
                              Node(value=left_value, ID=left_id, term_prob=self.terminal_prob[0])]
                # remember parent child relation and increment counter
                id_dict[parent_id] = [left_id, right_id]
                label_dict[left_id] = left_value
                label_dict[right_id] = right_value
        # reconstruct tree
        if tree:
            id_locations = {i: (idx, idx + 1) for idx, (v, i, p) in enumerate(terms)}

            # function to recursively get locations from IDs
            def get_loc(i):
                try:
                    return id_locations[i]
                except KeyError:
                    children_loc = np.array([get_loc(ii) for ii in id_dict[i]])
                    id_locations[i] = (np.min(children_loc[:, 0]), np.max(children_loc[:, 1]))
                    return id_locations[i]

            # transform ID dict into location dict
            node_dict = {get_loc(parent_id): [get_loc(ii) for ii in children_ids] for parent_id, children_ids in
                         id_dict.items()}
            # use IDs as labels
            if id_labels:
                label_dict = {get_loc(i): str(i) for i in id_dict.keys()}
            else:
                label_dict = {get_loc(i): label_dict[i] for i in id_dict.keys()}
            # collapse children of multi-terminal transitions
            if collapse_multi_terminal:
                multi_term_nodes = set(parent for parent, children in node_dict.items() if len(children) > 2)
                for node in multi_term_nodes:
                    for child in node_dict[node]:
                        del node_dict[child]
                        del label_dict[child]
                    node_dict[node] = []
        else:
            node_dict = None
            label_dict = None
        return np.array([v for v, i, p in terms]), node_dict, label_dict
