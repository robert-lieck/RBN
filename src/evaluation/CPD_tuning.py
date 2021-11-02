#  Copyright (c) 2021 Robert Lieck

import istarmap # patch Pool
from multiprocessing import Pool

import os
import pickle
from itertools import product
from warnings import warn

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import ruptures as rpt
from tqdm import tqdm

from data_models import SampleRBN
from plotting import plot_series
from util import f1_score_cps, precision_recall_f1, get_timestamp


def eval_cp(search, cost, pen, data, bkps):
    algo = search(min_size=0, jump=1, custom_cost=cost()).fit(data.copy())
    res = algo.predict(pen=pen)[:-1]
    matches = f1_score_cps(actual=bkps, estimated=res, tolerance=0, return_matches=True)
    prec, rec, f1 = precision_recall_f1(**matches)
    return search, cost, pen, res, matches, prec, rec, f1


def main(generate_data=True,
         info=False,
         # generate_data=False,
         # compute_results=True,
         compute_results=False,
         plot_results=True,
         # plot_results=False,
         # plot_data=True,
         plot_data=False,
         parallel=True,
         # parallel=False,
         n_cpu=None,
         # detailed_progress=True,
         detailed_progress=False,
         timestamp=None,
         l_sequence=50,
         n_samples=500,
         max_l_sequence=None,
         multi_terminal_mean=5,
         file_suffix=None):
    if n_cpu is None:
        n_cpu = os.cpu_count()
    if timestamp is None:
        timestamp = get_timestamp()
    if max_l_sequence is None:
        max_l_sequence = int(np.floor(1.1 * l_sequence))
    if file_suffix is None:
        file_suffix = f"_l{l_sequence}_n{n_samples}_m{multi_terminal_mean}"

    # whether to generate data or load from file
    if generate_data:
        data_list = []
        data_iter = list(product(range(n_samples),
                                 [dict(terminal_std=(tstd,)) for tstd in [
                                     0.01, 0.05, 0.1, 0.15, 0.2, 0.25
                                 ]]))
    else:
        with open(f'data{file_suffix}.pkl', 'rb') as file:
            data_list = pickle.load(file)
        data_iter = list(product(range(len(data_list)), [None]))

    # create worker pool once (even if not used)
    with Pool(n_cpu) as pool:
        results = []
        # generate data and compute results
        for data_idx, data_kwargs in tqdm(data_iter, disable=detailed_progress):
            # get data
            if generate_data:
                # combine with default arguments
                data_kwargs = {**dict(prior_mean=(0.,), prior_std=(1.,), terminal_prob=0.6,
                                      left_non_terminal_std=(0.1,), right_non_terminal_std=(0.1,),
                                      terminal_std=None,  # to be replaced
                                      left_rotations=(0, 1), right_rotations=(0, 0), rotation_weights=(0.5, 0.5),
                                      # left_rotations=(1,), right_rotations=(0,), rotation_weights=(1.,),
                                      multi_terminal_mean=multi_terminal_mean,
                                      n_dim=3),
                               **data_kwargs}
                # generate data
                data, node_dict, label_dict = SampleRBN(**data_kwargs).min_max(min_size=l_sequence,
                                                                               max_size=max_l_sequence,
                                                                               info=info)
                # extract breakpoints
                bkps = np.unique(np.array([n for n, c in node_dict.items() if not c]).flatten())[1:-1]
                # store
                data_list.append(((data, node_dict, label_dict, bkps), data_kwargs))
            else:
                (data, node_dict, label_dict, bkps), data_kwargs = data_list[data_idx]
            data_kwarg_tuple = tuple((k, v) for k, v in data_kwargs.items())

            # compute results
            if compute_results:
                search_cost_pen_data_bkps = list(product([rpt.Pelt,
                                                          rpt.Binseg,
                                                          rpt.BottomUp,
                                                          rpt.Window
                                                          ],
                                                         [rpt.costs.CostL2,
                                                          rpt.costs.CostAR,
                                                          rpt.costs.CostRank
                                                          ],
                                                         [b * 10 ** e for e in
                                                          [-3, -2, -1, 0, 1, 2]
                                                          for b in
                                                          # [1, 2.2, 4.6]
                                                          [1, 1.8, 3.2, 5.6]
                                                          ] + [1e-2],
                                                         [data],
                                                         [bkps]))
                if parallel:
                    # only one large chunk per worker process
                    chunksize = int(np.ceil(len(search_cost_pen_data_bkps) / n_cpu))
                    for search, cost, pen, res, matches, prec, rec, f1 in tqdm(pool.istarmap(eval_cp,
                                                                                             search_cost_pen_data_bkps,
                                                                                             chunksize=chunksize),
                                                                               total=len(search_cost_pen_data_bkps),
                                                                               disable=not detailed_progress):
                        results.append([data_idx, search.__name__, cost.__name__, pen, res] +
                                       list(matches.values()) +
                                       [prec, rec, f1, data_kwarg_tuple])
                else:
                    for args in tqdm(search_cost_pen_data_bkps, disable=not detailed_progress):
                        search, cost, pen, res, matches, prec, rec, f1 = eval_cp(*args)
                        results.append([data_idx, search.__name__, cost.__name__, pen, res] +
                                       list(matches.values()) +
                                       [prec, rec, f1, data_kwarg_tuple])
                    # print(search, cost, pen)
                    # print(result_list)
                    # print(bkps)
                    # print(result)
                    # print(matches)
                    # print((prec, rec, f1))
                    # plot tree and data
                    # if plot_data:
                    #     fig, axes = plot_series(data, node_dict=node_dict, separate=True, tree_kwargs=dict(color=(0, 1, 0)),
                    #                             change_points=res)
                    #     plt.show()
            if plot_data:
                # print(data_kwargs['terminal_std'])
                fig, axes = plot_series(data, node_dict=node_dict, separate=False,
                                        tree_kwargs=dict(color=(0, 1, 0), term_fill_kwargs=dict(alpha=0.2)),
                                        figsize=(10, 5), tight_layout=False)
                axes[0].set_title(f"noise: {data_kwargs['terminal_std']}")
                plt.show()

    # save data
    if generate_data:
        file_name = f"{timestamp}_data{file_suffix}.pkl"
        print(f"writing data to {file_name} ...")
        with open(file_name, 'wb') as file:
            pickle.dump(data_list, file)
        print("DONE")

    if compute_results or plot_results:

        # load results from file if they were not computed
        if not compute_results:
            if generate_data:
                warn(RuntimeWarning("New data have been generated but results are not recomputed"))
            with open(f'results{file_suffix}.pkl', 'rb') as file:
                results = pickle.load(file)

        # convert results to data frame
        results = pd.DataFrame(results,
                               columns=["data ID", "search method", "cost function", "penalty", "change points"] +
                                       ["n_true_positives", "n_actual_positives", "n_estimated_positives"] +
                                       ["precision", "recall", "F1", "parameters"])

        # save results
        if compute_results:
            file_name = f"{timestamp}_results{file_suffix}.pkl"
            print(f"writing results to {file_name} ...")
            with open(file_name, 'wb') as file:
                pickle.dump(results, file)
            print("DONE")

        # plot results
        if plot_results:
            file_name = f"{timestamp}_plot_CPD_tuning{file_suffix}.pdf"
            print(f"plotting results and saving to {file_name} ...")
            parameters = results["parameters"].unique()
            common_parameters = set.intersection(*[set(p) for p in parameters])
            # print(results.columns)
            # print(results)
            # print(parameters)
            # print(common_parameters)
            metrics = ["precision", "recall", "F1"]
            fig, axes = plt.subplots(len(metrics), len(parameters), figsize=(len(parameters) * 5, 12))
            for ax_row, y in zip(axes, metrics):
                for ax, param in zip(ax_row, parameters):
                    param_results = results[results["parameters"] == param]
                    specific_params = set(param) - common_parameters
                    sns.lineplot(x="penalty", y=y,
                                 hue="search method", style="cost function",
                                 data=param_results, markers=True,
                                 ax=ax)
                    ax.set_xscale('log')
                    ax.set_title("|".join([f"{k}={v}" for k, v in specific_params]))
            fig.tight_layout()
            fig.savefig(file_name)
            print("DONE")
            plt.show()


if __name__ == "__main__":
    main()
