#  Copyright (c) 2021 Robert Lieck

import matplotlib.pyplot as plt
import numpy as np
import torch

from plotting import scape_plot, plot_model_tree, plot_tree


def comparison(file, left_cbar=True):
    rbn = torch.load(file)
    batch_idx = 10

    fig, axes = plt.subplots(1, 2, figsize=(10,5),
                             gridspec_kw=dict(width_ratios=[0.03, 1] if left_cbar else [1, 0.03],
                                              hspace=0.01,
                                              wspace=0.01))
    if left_cbar:
        cbar_axis, scape_axis = axes
    else:
        scape_axis, cbar_axis = axes
    scape_axis.set_yticks([])
    scape_plot(arr=rbn.node_log_coef.arr.exp().clone().detach().numpy()[:, batch_idx],
               ax=scape_axis,
               time_slots=np.arange(rbn.node_log_coef.n + 1) * 2,  # scale up from half-note sampling to quarter display
               cmap='Greys',
               colorbar=cbar_axis,
               log_scale=True)
    if left_cbar:
        cbar_axis.yaxis.set_ticks_position('left')
    plot_tree(fig_ax=(fig, scape_axis),
              scaling=4,
              layout_kwargs=dict(bottom_align=True),
              plot_nodes=False,
              adjust_axes=False,
              plot_kwargs=dict(
                  # line_color=(0, 0.7, 0),
                  # color=(0.188, 0.451, 0.631),  # standard blue
                  color=(0.2, 0.5, 0.8),          # brighter blue
                  linestyle='-',
                  linewidth=2,
                  marker='o',
                  markersize=5,
                  label="expert annotation"
              ))
    max_tree = rbn.max_tree()
    plot_model_tree(node_dict=max_tree[0][batch_idx],
                    label_dict=max_tree[1][batch_idx],
                    label_first_child=True,
                    label_kwargs=dict(color=(1., 0.6, 0.1),
                                      fontsize=4,
                                      ha='center', va='center'),
                    label_offset=(-np.sqrt(2) / 2, np.sqrt(2) / 2),
                    scaling=2,
                    ax=scape_axis,
                    # color=(0.9, 0, 0),
                    # color=(0.882, 0.498, 0.169),  # standard orange
                    color=(1., 0.6, 0.1),        # brighter orange
                    lw=1, ls='--',
                    inner_kwargs=dict(ms=2),
                    leaf_kwargs=dict(ms=2),
                    term_fill_kwargs=dict(alpha=0.),
                    term_line_kwargs=dict(alpha=0.),
                    label="RBN estimate")
    scape_axis.legend()
    plt.show()
