#  Copyright (c) 2021 Robert Lieck

import istarmap # patch Pool to be able to display progress
import multiprocessing.pool as mpp

import os
import pickle
import math
from warnings import warn

import matplotlib.pyplot as plt
from tqdm import tqdm
import ruptures as rpt
import torch
import pandas as pd
import seaborn as sns

from pyulib import next_color

from rbnbase import RBNBase
from plotting import plot_series, plot_model_tree, scape_plot
from util import hierarchical_clustering, f1_score_trees, get_timestamp, TMap, get_node_probs
from evaluation.pretrain_RBN import get_pretrained_params


def eval_rbn(data, actual_tree, data_kwargs, rbn_kwargs, noise_level,
             l_sequence, detailed_progress, RBN_marginal, multi_terminal_lambda,
             adaptive_noise_level=False, rescale_to_expected_n_nodes=False):
    kwargs = rbn_kwargs[(noise_level, l_sequence)].copy()
    # non-terminal transition variance to match data
    if adaptive_noise_level:
        assert data_kwargs['left_non_terminal_std'] == data_kwargs['right_non_terminal_std']
        assert 'non_term_sqrt_cov' in kwargs
        if (kwargs['non_term_sqrt_cov'],) != data_kwargs['left_non_terminal_std']:
            warn(RuntimeWarning(f"Setting non-terminal variance to value from data "
                                f"(was {kwargs['non_term_sqrt_cov']}; "
                                f"setting to {data_kwargs['left_non_terminal_std']})"))
        kwargs['non_term_sqrt_cov'] = data_kwargs['left_non_terminal_std']
    # parse data with RBN
    rbn = RBNBase(**kwargs)
    with torch.no_grad():
        rbn.compute_inside(torch.from_numpy(data), progress=detailed_progress)
    if RBN_marginal:
        rbn.compute_outside(progress=detailed_progress)
    # extract most likely tree
    (rbn_tree,), _ = rbn.max_tree()
    # compute scores
    rbn_max_score = f1_score_trees(actual=actual_tree, estimated=rbn_tree, n_obs=data.shape[0])
    if RBN_marginal:
        if rescale_to_expected_n_nodes:
            if multi_terminal_lambda is not None:
                expected_n_nodes = 2 * (l_sequence / multi_terminal_lambda) - 1
            else:
                expected_n_nodes = 2 * l_sequence - 1
        else:
            expected_n_nodes = None
        rbn_marginal_sccore = f1_score_trees(actual=actual_tree,
                                             estimated=get_node_probs(rbn, 0, expected_n_nodes=expected_n_nodes),
                                             n_obs=data.shape[0],
                                             probs=True)
    else:
        rbn_marginal_sccore = None
    return rbn, rbn_tree, rbn_max_score, rbn_marginal_sccore, noise_level


def main(l_sequence=50,
         n_samples=5,
         multi_terminal_lambda=5,
         file_suffix=None,
         compute_cpd_results=True,
         compute_rbn_results=True,
         RBN_marginal=True,
         rbn_parallel=True,
         n_cpu=None,
         plot_data=False,
         n_data_points=None,
         detailed_progress=False):
    if n_cpu is None:
        n_cpu = os.cpu_count()
    if file_suffix is None:
        file_suffix = f"_l{l_sequence}_n{n_samples}_m{multi_terminal_lambda}"
    # pre-trained RBNs for different noise levels (on 10 data samples)
    rbn_kwargs = {}
    for noise in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25]:
        rbn_kwargs[(noise, l_sequence)] = get_pretrained_params(noise)
        rbn_kwargs[(noise, l_sequence)]['left_transpositions'] = (0, 1)
        rbn_kwargs[(noise, l_sequence)]['right_transpositions'] = (0, 0)
    # pre-calibrated penalties for change point detection (on 100 data samples)
    cpd = rpt.Pelt(min_size=0, jump=1, custom_cost=rpt.costs.CostL2)
    pens = {
        (0.01, 50, 5): 1e-2,
        (0.05, 50, 5): 3.2e-2,
        (0.1, 50, 5): 1.8e-1,
        (0.15, 50, 5): 3.2e-1,
        (0.2, 50, 5): 3.2e-1,
        (0.25, 50, 5): 5.6e-1,
    }

    # load data
    with open(f'data{file_suffix}.pkl', 'rb') as file:
        data_list = pickle.load(file)
    if n_data_points is not None:
        data_list = data_list[:n_data_points]
    # iterate through data and compute scores
    cpd_score_list = []
    rbn_score_list = []
    # for plotting only
    cpd_tree_list = []
    cpd_result_list = []
    rbn_tree_list = []
    rbn_list = []

    if compute_cpd_results:
        for (data, actual_tree, actual_values, actual_cps), data_kwargs in tqdm(data_list, disable=detailed_progress, desc="CPD"):
            # noise level: standard deviation of terminal transition
            noise_level = data_kwargs['terminal_std'][0]
            if multi_terminal_lambda is None:
                res = None
            else:
                # compute change points with pre-calibrated penalty
                res = cpd.fit(data).predict(pen=pens[(noise_level, l_sequence, multi_terminal_lambda)])[:-1]
            # compute tree based on change points
            _, _, cpd_tree = hierarchical_clustering(data=data, change_points=res)
            # compute scores
            cpd_score = f1_score_trees(actual=actual_tree, estimated=cpd_tree, n_obs=data.shape[0])
            if multi_terminal_lambda is None:
                cpd_score_list.append(("HC",) + cpd_score + (noise_level,))
            else:
                cpd_score_list.append(("HC/CPD",) + cpd_score + (noise_level,))
            if plot_data:
                cpd_result_list.append(res)
                cpd_tree_list.append(cpd_tree)

    if compute_rbn_results:
        rbn_loop_results = []
        if rbn_parallel:
            arg_list = []
            for (data, actual_tree, actual_values, actual_cps), data_kwargs in data_list:
                # noise level: standard deviation of terminal transition
                noise_level = data_kwargs['terminal_std'][0]
                arg_list.append((data, actual_tree, data_kwargs, rbn_kwargs, noise_level,
                                 l_sequence, detailed_progress, RBN_marginal, multi_terminal_lambda,))
            with mpp.Pool(processes=n_cpu, maxtasksperchild=10) as pool:
                for result in tqdm(pool.istarmap(eval_rbn, arg_list), total=len(arg_list), disable=detailed_progress, desc="RBN parallel"):
                    rbn_loop_results.append(result)
        else:
            for (data, actual_tree, actual_values, actual_cps), data_kwargs in tqdm(data_list, disable=detailed_progress, desc="RBN"):
                # noise level: standard deviation of terminal transition
                noise_level = data_kwargs['terminal_std'][0]
                rbn_loop_results.append(eval_rbn(data, actual_tree, data_kwargs, rbn_kwargs, noise_level))
        for rbn, rbn_tree, rbn_max_score, rbn_marginal_sccore, noise_level in rbn_loop_results:
            rbn_score_list.append(("RBN max",) + rbn_max_score + (noise_level,))
            if RBN_marginal:
                rbn_score_list.append(("RBN marginal",) + rbn_marginal_sccore + (noise_level,))
            if plot_data:
                rbn_list.append(rbn)
                rbn_tree_list.append(rbn_tree)

    # plot tree and data
    if plot_data:
        for data_idx, ((data, actual_tree, actual_values, actual_cps), data_kwargs) in enumerate(data_list):
            # noise level: standard deviation of terminal transition
            noise_level = data_kwargs['terminal_std'][0]
            # print(data)
            fig, axes = plt.subplots(2, 2, figsize=(15, 10),
                                     sharex='col',
                                     gridspec_kw=dict(width_ratios=[0.03, 1],
                                                      hspace=0.02,
                                                      wspace=0.02))
            ax = axes[0, 1]
            plot_series(data,
                        fig_ax=(fig, [ax] + [axes[1, 1]] * 3),
                        node_dict=actual_tree,
                        separate=False,
                        tree_kwargs=dict(color=(0, 0.7, 0), ms=6, label="ground truth", term_line_kwargs=dict(ms=0)),
                        change_points=cpd_result_list[data_idx],
                        figsize=(15, 10),
                        data_cmap=plt.cm.cool,
                        tight_layout=False)
            ax.set_title(f"noise level: {noise_level}")
            if compute_cpd_results:
                # print("CPD:", cpd_score)
                color = next_color(ax)
                plot_model_tree(node_dict=cpd_tree_list[data_idx], ax=ax,
                                # color=(1, .7, 0),
                                color=color,
                                ls=(2, (2, 4)), ms=5, mec=(0, 0, 0, 0), mfc=color, label="HC/CPD",
                                term_line_kwargs=dict(ms=0))
            if compute_rbn_results:
                # print("RBN max:", rbn_max_score)
                color = next_color(ax)
                plot_model_tree(node_dict=rbn_tree_list[data_idx], ax=ax,
                                color=color,
                                # color=(1, 0, 0),
                                ls=(0, (2, 4)), ms=6, mec=color, mfc=(0, 0, 0, 0), label="RBN max",
                                term_line_kwargs=dict(ms=0))
                if RBN_marginal:
                    # print("RBN marginal:", rbn_marginal_sccore)
                    scape_plot(rbn_list[data_idx].node_log_coef.arr.exp().numpy()[:, 0], ax=ax, cmap='Greys',
                               colorbar=axes[0, 0],
                               cbar_kwargs=dict(ticklocation="left"),
                               log_scale=True)
            ax.legend()
            axes[1, 0].axis('off')
            axes[0, 1].set_yticks([])
            axes[1, 1].set_yticks([])
            plt.show()

    # time stamp for saving results
    timestamp = get_timestamp()

    # save or load results
    columns = ["method", "precision", "recall", "F1", "noise"]
    if compute_cpd_results:
        # transform to data frame
        cpd_score_list = pd.DataFrame(cpd_score_list, columns=columns)
        # save
        file_name = f"{timestamp}_CPD_scores{file_suffix}.pkl"
        with open(file_name, 'wb') as file:
            pickle.dump(cpd_score_list, file)
        print(f"saved CPD scores as {file_name}")
    else:
        with open(f"CPD_scores{file_suffix}.pkl", 'rb') as file:
            cpd_score_list = pickle.load(file)
    if compute_rbn_results:
        # transform to data frame
        rbn_score_list = pd.DataFrame(rbn_score_list, columns=columns)
        # save
        file_name = f"{timestamp}_RBN_scores{file_suffix}.pkl"
        with open(file_name, 'wb') as file:
            pickle.dump(rbn_score_list, file)
        print(f"saved RBN scores as {file_name}")
    else:
        with open(f"RBN_scores{file_suffix}.pkl", 'rb') as file:
            rbn_score_list = pickle.load(file)

    # joint scores
    scores = pd.concat((cpd_score_list, rbn_score_list), ignore_index=True)
    # unpack noise level (for old data)
    if isinstance(scores.at[0, 'noise'], tuple):
        for idx in scores.index:
            scores.at[idx, 'noise'] = scores.at[idx, 'noise'][0]

    # plot result
    metrics = ["precision", "recall", "F1"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(len(metrics) * 6, 4))
    for ax, y in zip(axes, metrics):
        sns.lineplot(x="noise", y=y,
                     hue="method",
                     err_style="bars",
                     data=scores, markers=True,
                     ax=ax)
    fig.tight_layout()
    file_name = f"{timestamp}_plot_scores{file_suffix}.pdf"
    fig.savefig(file_name)
    print(f"results saved as {file_name}")
    plt.show()


if __name__ == "__main__":
    main()
