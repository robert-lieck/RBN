#  Copyright (c) 2021 Robert Lieck

from itertools import count
import math
import pickle

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from rbnbase import RBNBase


default_params = dict(terminal_log_prob=-0.66,
                      prior_mean=0,
                      prior_sqrt_cov=2,
                      non_term_sqrt_cov=0.5,
                      term_trans_sqrt_cov=0.5,
                      log_mixture_weights=(math.log(0.9), math.log(0.1)),
                      multi_terminal_lambda=8)


def get_pretrained_params(terminal_std):
    # parameter from pretrained RBNs
    if terminal_std == 0.01:
        rbn_kwargs = dict(prior_mean=0.007442277856171131,
                          prior_sqrt_cov=0.8436552882194519,
                          terminal_log_prob=-0.63204116,
                          term_trans_sqrt_cov=-0.001687273383140564,
                          non_term_sqrt_cov=0.103497713804245,
                          multi_terminal_lambda=math.exp(1.6104984283447266),
                          log_mixture_weights=[0.03780382, -0.03780382])
    elif terminal_std == 0.05:
        rbn_kwargs = dict(prior_mean=-0.007745358161628246,
                          prior_sqrt_cov=1.0408804416656494,
                          terminal_log_prob=-0.63135666,
                          term_trans_sqrt_cov=0.04814087226986885,
                          non_term_sqrt_cov=-0.1028837338089943,
                          multi_terminal_lambda=math.exp(1.6156957149505615),
                          log_mixture_weights=[0.03307785, -0.03307785])
    elif terminal_std == 0.1:
        rbn_kwargs = dict(prior_mean=0.5250821709632874,
                          prior_sqrt_cov=0.8863024115562439,
                          terminal_log_prob=-0.6337793,
                          term_trans_sqrt_cov=0.09903469681739807,
                          non_term_sqrt_cov=0.09550739824771881,
                          multi_terminal_lambda=math.exp(1.6011087894439697),
                          log_mixture_weights=[-0.02768178, 0.02768178])
    elif terminal_std == 0.15:
        rbn_kwargs = dict(prior_mean=-0.16925080120563507,
                          prior_sqrt_cov=1.0164692401885986,
                          terminal_log_prob=-0.62174577,
                          term_trans_sqrt_cov=0.1511688530445099,
                          non_term_sqrt_cov=0.10323083400726318,
                          multi_terminal_lambda=math.exp(1.8528882265090942),
                          log_mixture_weights=[-0.0497127, 0.0497127])
    elif terminal_std == 0.2:
        rbn_kwargs = dict(prior_mean=-0.14549808204174042,
                          prior_sqrt_cov=0.9265695214271545,
                          terminal_log_prob=-0.6340247,
                          term_trans_sqrt_cov=0.2024099975824356,
                          non_term_sqrt_cov=0.10538198798894882,
                          multi_terminal_lambda=math.exp(1.6627657413482666),
                          log_mixture_weights=[-0.17434898, 0.17434898])
    elif terminal_std == 0.25:
        rbn_kwargs = dict(prior_mean=0.06523402780294418,
                          prior_sqrt_cov=1.0010186433792114,
                          terminal_log_prob=-0.6319719,
                          term_trans_sqrt_cov=0.24559417366981506,
                          non_term_sqrt_cov=0.11163070797920227,
                          multi_terminal_lambda=math.exp(1.6337534189224243),
                          log_mixture_weights=[-0.3180804, 0.3180804])
    else:
        raise ValueError(f"Don't have pretrained parameters for terminal_std={terminal_std}")
    return rbn_kwargs


def main(l_sequence=50,
         n_samples=100,
         multi_terminal_lambda=5,
         file_suffix=None,
         terminal_std=None, # noise levels: 0.01, 0.05, 0.1, 0.15, 0.2, 0.25
         pick_first=None,
         max_epochs=10,
         pre_trained=False,
         print_params=False,
         plot_progress=False):
    if file_suffix is None:
        file_suffix = f"_l{l_sequence}_n{n_samples}_m{multi_terminal_lambda}"

    # parameter from pretrained RBNs
    if pre_trained:
        rbn_kwargs = get_pretrained_params(terminal_std)
    else:
        # default values to start training
        rbn_kwargs = default_params
    rbn_kwargs['left_transpositions'] = (0, 1)
    rbn_kwargs['right_transpositions'] = (0, 0)
    rbn = RBNBase(**rbn_kwargs)
    optimizer = torch.optim.Adam([dict(params=rbn.parameters(),
                                       lr=1e-2, betas=(0.5, 0.9),
                                       # lr=1e-3, betas=(0.9, 0.999)
                                       )])

    # load data
    with open(f'data{file_suffix}.pkl', 'rb') as file:
        data_list = pickle.load(file)
    # filter for noise level
    data_list = [d for d in data_list if d[1]['terminal_std'] == (terminal_std,)]
    if not data_list:
        raise RuntimeError(f"No data with terminal_std={terminal_std}")
    # pick the first ones for faster initial training
    if pick_first is not None:
        data_list = data_list[:pick_first]
    len(data_list)

    if plot_progress:
        # setup plot
        fig = plt.gcf()
        # lines = plt.plot(np.zeros((1, n_data)), '-o', linewidth=1, markersize=2)
        fig.show()
        fig.canvas.draw()

    # train RBN
    loss_list = []
    if max_epochs is None:
        epochs = count()
    else:
        epochs = range(max_epochs)
    for it in epochs:
        optimizer.zero_grad()
        epoch_loss = []
        bar = tqdm(data_list, desc=f"epoch {it + 1}")
        for (data, actual_tree, actual_values, actual_cps), data_kwargs in bar:
            loss = rbn.compute_inside(torch.from_numpy(data), progress=False)
            loss.backward()
            epoch_loss.append(loss.detach().clone().to('cpu').numpy())
            bar.set_postfix_str(f"average loss: {np.mean(epoch_loss)}", refresh=True)
        loss_list.append(epoch_loss)
        # print info and plot progress
        if print_params:
            print(f"terminal_std: {terminal_std}")
            rbn.print_params(print_derived=True)
        if plot_progress:
            loss_arr = np.array(loss_list)
            fig.clear()
            plt.plot(loss_arr, '-o', linewidth=1, markersize=2)
            plt.title(f"{terminal_std}")
            plt.pause(0.1)
            fig.canvas.draw()
        # update parameters
        optimizer.step()

    print(f"terminal_std: {terminal_std}")
    rbn.print_params(print_derived=True)


if __name__ == "__main__":
    main()
