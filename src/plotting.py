#  Copyright (c) 2021 Robert Lieck

from warnings import warn

import numpy as np
import torch
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_col

import pitchscapes.plotting as pt
from pitchscapes.plotting import counts_to_colors, scape_plot_from_array
from pitchscapes.util import start_end_to_center_width
from treeparser import TreeParser

from util import remap_to_color, __MUSIC__, __TIME_SERIES__, __COLOUR__, __TREE__
from rbnbase import Categorical

from util import TMap, default_dict


default_cmap = 'winter'
tree_BWV_846 = """[C [C [C [C] [C [G7/B [Dm7/C] [G7/B]] [C]]] [C [G [G/B [D7/C [Am] [D7/C]] [G/B]] [G [D7 [Cmaj7/B] [D7 [Am7] [D7]]] [G]]] [C [C/E [Bdim7/F [Dm/F [C#dim7/G] [Dm/F]] [Bdim7/F]] [C/E]] [C [G7 [Fmaj7/E] [G7 [Dm7] [G7]]] [C]]]]][C [G7 [G7 [Fmaj7 [C7] [Fmaj7]] [G7 [F#dim7] [G7 [Abdim7] [G7]]]] [G7 [G7 [Gsus64] [G7 [G7sus4] [G7]]] [G7 [Gsus64 [F#dim7/G] [Gsus64]] [G7 [G7sus4] [G7]]]] ] [C [G7/C [F/C [C7/C] [F/C]] [G7/C]] [C]] ] ]"""


def key_legen():
    pt.key_legend()


def get_average_scape(observations, offset=0):
    # average from observations (should be similar to terminal mean)
    n_obs = observations.shape[0]
    n_dim = observations.shape[-1]
    average_scape = TMap(torch.zeros((TMap.size1d_from_n(n_obs), n_dim)))
    average_scape_log = TMap(torch.zeros((TMap.size1d_from_n(n_obs), n_dim)))
    observations_log = Categorical.map(observations, dim=-1, offset=offset)
    for start in range(n_obs + 1):
        for end in range(n_obs + 1):
            if end <= start:
                continue
            average_scape[start, end] = observations[start:end].mean(dim=0)
            average_scape_log[start, end] = observations_log[start:end].mean(dim=0)
    average_scape_log.arr = Categorical.rmap(average_scape_log.arr, dim=-1, offset=offset)
    return average_scape, average_scape_log


@torch.no_grad()
def plot_overview(rbn, music_data, data_type, batch_idx, prec=False, plot_tree=True, plot_labels=True, log_scale=True, keep_as_log=True, ground_truth=None, squash_root=False, squash_bottom=False):
    # assert rbn.observations.shape[1] == 1, f"batch size is larger than 1, cannot display"
    observations = rbn.observations[:, batch_idx, :]
    multi_terminal = rbn.multi_terminal
    average_scape, average_scape_log = get_average_scape(observations)
    # row/col | Data | Inside | Outside | None | (Terminal)
    # Mean
    # Variance
    # Log Coef
    #
    # plus additional columns to add space
    fig, axes = plt.subplots(3, 14, figsize=(25, 8),
                             gridspec_kw=dict(width_ratios=[1, 0.05, 0.2] + ([1, 0.05, 0.2] * 4)[:-1],
                                              hspace=0.25, wspace=0.05,
                                              left=0.04, right=0.96, bottom=0.05, top=0.95))
    # remove axes from spacing columns
    for ax_row in axes:
        for idx, ax in enumerate(ax_row):
            if (idx + 1) % 3 == 0:
                ax.axis('off')

    # set row labels
    # - combined
    # for ax, l in zip(axes[:, 0], ["Mean (Data)", f"{'Precision' if prec else 'Variance'} (Average Scape)", "Log-Coef. (Log-Average Scape)"]):
    #     ax.set_ylabel(l)
    # - only for data
    for ax, l in zip(axes[:, 0], ["Data", "Average Scape", "Log-Average Scape"]):
        ax.set_ylabel(l)
    # - only for variables
    for ax, l in zip(axes[:, 3], ["Mean", 'Precision' if prec else 'Variance', "Log-Coef."]):
        ax.set_ylabel(l)

    # plot data
    data_axs = axes[:, 0]
    data_colbars = axes[:, 1]
    data_axs[0].set_title("Data")
    if data_type == __MUSIC__:
        for d, ax, cax in zip([music_data.colours[:, batch_idx, :],
                               counts_to_colors(average_scape.flatten('se')),
                               counts_to_colors(average_scape_log.flatten('se'))],
                              data_axs, data_colbars):
            scape_plot_from_array(arr=d, ax=ax)
            cax.axis('off')
    elif data_type == __TIME_SERIES__:
        data_axs[0].plot(observations.mean(dim=-1), 'o-', linewidth=0.5, markersize=1)
        data_colbars[0].axis('off')
        scape_plot(arr=average_scape.arr.detach().numpy(), ax=data_axs[1], cmap='Reds', colorbar=data_colbars[1])
        # scape_plot(arr=average_scape_log.arr.detach().numpy(), ax=data_axs[2], cmap='Reds', colorbar=data_colbars[2])
    elif data_type == __COLOUR__:
        for d, c in zip(observations.transpose(0, 1), [(1, 0, 0), (0, 1, 0), (0, 0, 1)]):
            data_axs[0].plot(d, 'o-', linewidth=0.5, markersize=1, c=c)
            data_colbars[0].axis('off')
        for d, ax, cax in zip([average_scape.flatten('se'),
                               average_scape_log.flatten('se')],
                              data_axs[1:], data_colbars[1:]):
            scape_plot_from_array(arr=d.numpy(), ax=ax)
            cax.axis('off')
    elif data_type == __TREE__:
        for d, c in zip(observations.transpose(0, 1), [(1, 0, 0), (0, 1, 0), (0, 0, 1)]):
            data_axs[0].plot(d, 'o-', linewidth=0.5, markersize=1, c=c)
            data_colbars[0].axis('off')
        if observations.shape[1] <= 3:
            for d, ax, cax in zip([
                remap_to_color(average_scape.flatten('se').numpy()),
                # remap_to_color(average_scape_log.flatten('se').numpy())
            ],
                    data_axs[1:], data_colbars[1:]):
                scape_plot_from_array(arr=d, ax=ax)
                cax.axis('off')
        else:
            warn(RuntimeWarning(f"Observations have {observations.shape[1]} dimensions. "
                                f"Can only visualise scapes for up to 3 dimensions."))
    else:
        raise ValueError(f"Unknown data type: {data_type}")

    # plot variables
    var_names = ["inside", "outside", "node"]
    if multi_terminal:
        var_names += ["multi_term"]
    for variable, main_axes, colour_axes in zip(var_names, axes[:, 3::3].transpose(), axes[:, 4::3].transpose()):
        main_axes[0].set_title(variable)
        # mean
        if data_type == __MUSIC__:
            scape_plot_from_array(
                arr=counts_to_colors(
                    (
                            getattr(rbn, variable + "_mean").flatten('se')
                            # + getattr(rbn, var + "_cov").flatten('se').diagonal(dim1=-2, dim2=-1) / 2  # correct mean of log-normal
                    ).detach().exp().numpy()[:, batch_idx]),  # TODO use normalise_log_normal
                ax=main_axes[0])
            colour_axes[0].axis('off')
        elif data_type == __TIME_SERIES__:
            scape_plot(arr=getattr(rbn, variable + "_mean").arr.detach().numpy()[:, batch_idx],
                       ax=main_axes[0], cmap='Reds', colorbar=colour_axes[0])
        elif data_type == __COLOUR__:
            if rbn.categorical:
                scape_plot_from_array(arr=Categorical.rmap(log_probs=getattr(rbn, variable + "_mean").flatten('se')[:, batch_idx],
                                                           dim=-1,
                                                           # offset=cat.offset.detach().numpy()
                                                           ).detach().numpy(),
                                      ax=main_axes[0])
            else:
                scape_plot_from_array(arr=remap_to_color(getattr(rbn, variable + "_mean").flatten('se')[:, batch_idx].detach().numpy()),
                                      ax=main_axes[0])
            colour_axes[0].axis('off')
        elif data_type == __TREE__:
            scape_plot_from_array(arr=remap_to_color(getattr(rbn, variable + "_mean").flatten('se')[:, batch_idx].detach().numpy()),
                                  ax=main_axes[0])
            colour_axes[0].axis('off')
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        # var
        var_prec = np.diagonal(getattr(rbn, variable + "_cov").arr.detach().numpy(), axis1=-2, axis2=-1).mean(axis=-1)
        if prec:
            var_prec = 1 / var_prec
        scape_plot(arr=var_prec, ax=main_axes[1], cmap=('Greys' if prec else 'Greys_r'), colorbar=colour_axes[1])
        # log coef
        x = getattr(rbn, variable + "_log_coef").arr.clone().detach().numpy()[:, batch_idx]
        use_log_scale = log_scale and not keep_as_log
        if not (log_scale and keep_as_log):
            x = np.exp(x)
        if squash_root:
            x[0] = x.mean()
        if squash_bottom:
            TMap(x).lslice(1)[:] = x.mean()
        scape_plot(arr=x, ax=main_axes[2], cmap='Greys', colorbar=colour_axes[2], log_scale=use_log_scale)

    # plot tree
    if ground_truth is not None:
        plot_model_tree(node_dict=ground_truth, ax=axes[2, 9], color=(0, 1, 0), lw=2)
    if plot_tree:
        # check if max_split_indices was assigned
        if rbn.max_split_indices is None:
            warn(RuntimeWarning("Cannot draw tree, max_split_indices is None"))
            return
        # check for ambiguous batch index
        # if rbn.n_batch > 1:
        #     warn(RuntimeWarning("Model has multiple batches, selecting only the first one"))
        node_dict, label_dict = rbn.max_tree()
        node_dict = node_dict[batch_idx]
        label_dict = label_dict[batch_idx]
        if not plot_labels:
            label_dict = None
        plot_model_tree(node_dict=node_dict, ax=axes[2, 9], label_dict=label_dict, color=(1, 0, 0), lw=1)


def plot_model_tree(node_dict, ax, label_dict=None, scaling=1.,
                    inner_kwargs=None, leaf_kwargs=None, label_kwargs=None,
                    term_fill_kwargs=None, term_line_kwargs=None, label=None, label_first_child=False,
                    label_offset=(0., 0.),
                    **kwargs):
    """
    Draw a tree
    :param node_dict: dictionary with parent --> [children, ...] dict; nodes are given by (start, end) tuple
    :param ax: axis to draw to
    :param label_dict: node --> label dict with labels to put on the nodes
    :param inner_kwargs: additional kwargs passed to plot() for drawing the inner nodes and branches of the tree
    :param leaf_kwargs: additional kwargs passed to plot() the marker for leaf nodes
    :param label_kwargs: additional kwargs passed to text() for drawing the labels
    :param term_fill_kwargs: additional kwargs passed to fill() for drawing terminal triangles
    :param term_line_kwargs: additional kwargs passed to plot() for drawing upper edges of terminal triangles
    :param kwargs: additional kwargs passed to inner_kwargs, leaf_kwargs, term_line_kwargs (color also used for term_fill_kwargs)
    :return:
    """
    # default color
    if 'color' not in kwargs:
        kwargs['color'] = (0, 0, 0)
    # assign defaults
    inner_kwargs = default_dict(default_dict(inner_kwargs, label=label, **kwargs), marker='o', ms=5)
    leaf_kwargs = default_dict(default_dict(leaf_kwargs, **kwargs), marker='o', ms=0)
    label_kwargs = default_dict(label_kwargs, fontdict=dict(fontsize=12))
    term_fill_kwargs = default_dict(term_fill_kwargs, lw=0, fc=kwargs['color'], alpha=0., ec=(0, 0, 0, 0))
    term_line_kwargs = default_dict(default_dict(term_line_kwargs, **kwargs), alpha=0., marker='o', ms=0)
    # plot tree
    for parent, children in node_dict.items():
        # get parent's coordinates
        parent_start, parent_end = parent
        parent_center, parent_width = start_end_to_center_width(parent_start, parent_end)
        # inner node or leaf?
        if children:
            # for inner nodes, draw the connections to all children
            for child_idx, child in enumerate(children):
                child_start, child_end = child
                child_center, child_width = start_end_to_center_width(child_start, child_end)
                # plot connection from parent to child
                ax.plot([scaling * parent_center, scaling * child_center],
                        [scaling * parent_width, scaling * child_width], **inner_kwargs)
                if 'label' in inner_kwargs:
                    del inner_kwargs['label']
                # plot labels at first child
                if label_first_child and child_idx == 0 and label_dict is not None:
                    ax.text(scaling * child_center + label_offset[0],
                            scaling * child_width + label_offset[1],
                            str(label_dict[parent]),
                            **label_kwargs)
        else:
            # for leaf nodes draw separate marker and filled triangle
            ax.plot([scaling * parent_center], [scaling * parent_width], **leaf_kwargs)
            x, y = (scaling * parent_start, scaling * parent_center, scaling * parent_end), (0, scaling * parent_width, 0)
            ax.fill(x, y, **term_fill_kwargs)
            ax.plot(x, y, **term_line_kwargs)
        # plot labels indicating transposition/mixture component
        if label_dict is not None and not label_first_child and parent in label_dict:
            ax.text(scaling * parent_center + label_offset[0],
                    scaling * parent_width + label_offset[1],
                    str(label_dict[parent]), **label_kwargs)


def plot_tree(tree=None, scaling=4, fig_ax=None, **kwargs):
    # get figure or create
    if fig_ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    else:
        fig, ax = fig_ax
    # tree to plot
    if tree is None:
        tree = tree_BWV_846
    # parse tree
    parser = TreeParser(array=tree, string_input=True)
    # plot
    kwargs = {**dict(ax=ax,
                   padding=1,
                   layout_kwargs=dict(bottom_align=True),
                   fontdict={'fontsize': 8},
                   textkwargs={'bbox': {'facecolor': 'white', 'pad': 1}},
                   scaling=(scaling, scaling),
                   offset=(scaling / 2, scaling),
                   plot_nodes=True,
                   adjust_axes=False),
              **kwargs}
    parser.plot(**kwargs)
    if fig_ax is None:
        fig.tight_layout()
    return fig, ax


def plot_series(data, fig_ax=None, node_dict=None, label_dict=None, separate=False, change_points=None, chpt_kwargs=None, tree_kwargs=None, data_cmap=None, tight_layout=True, **kwargs):
    # turn scalar data into size-1 array data
    if len(data.shape) == 1:
        data = data[:, None]
    # get number of rows (one for each dimension of the data)
    n_rows = data.shape[1]
    # add a row if the tree should be plotted
    if node_dict is not None:
        n_rows += 1
    # get figure/axes
    if fig_ax is None:
        kwargs = default_dict(kwargs, figsize=(25, 13), sharex='all')
        if separate:
            kwargs = default_dict(kwargs, gridspec_kw=dict(height_ratios=(n_rows - 1,) + (1,) * (n_rows - 1)))
            fig, axes = plt.subplots(n_rows, 1, **kwargs)
            axes = np.atleast_1d(axes)
        else:
            if node_dict is None:
                fig, ax = plt.subplots(1, 1, **kwargs)
                axes = [ax] * n_rows
            else:
                fig, axes = plt.subplots(2, 1, **kwargs)
                axes = [axes[0]] + [axes[1]] * (n_rows - 1)
    else:
        fig, axes = fig_ax
    for ax in axes:
        ax.set_xlim(0, data.shape[0])
    if node_dict:
        plot_model_tree(node_dict=node_dict, label_dict=label_dict, ax=axes[0], **default_dict(tree_kwargs))
        data_axes = axes[1:]
    else:
        data_axes = axes
    if data_cmap is not None:
        colors = data_cmap(np.linspace(0, 1, data.shape[1]))
    else:
        colors = [None] * data.shape[1]
    for idx, (ax, col) in enumerate(zip(data_axes, colors)):
        ax.plot(np.arange(data.shape[0]) + 0.5, data[:, idx], '-o', lw=1, ms=2, c=col)
        if change_points is not None:
            for chp in change_points:
                ax.axvline(x=chp, **default_dict(chpt_kwargs, color=(0, 0, 0, 0.5), lw=1, ls='--'))
    if tight_layout:
        fig.tight_layout()
    return fig, axes


def scape_plot(arr,
               ax,
               time_slots=None,
               vmin=None,
               vmax=None,
               cmap=None,
               colorbar=False,
               cbar_kwargs=None,
               log_scale=False):
    # get colour map
    if cmap is None:
        cmap = default_cmap
    if isinstance(cmap, str):
        cmap = cm.get_cmap(cmap)
    # convert multidimensional values to scalar values by taking the mean
    if len(arr.shape) > 1:
        arr = np.mean(arr, axis=tuple(range(1, len(arr.shape))))
    # wrap as TMap for processing
    tmap = TMap(arr)
    # convert to colours
    if log_scale:
        norm = mpl_col.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = mpl_col.Normalize(vmin=vmin, vmax=vmax)
    colours = cmap(norm(tmap.flatten('se')))
    # use integer time slots if not provided
    if time_slots is None:
        time_slots = np.arange(tmap.n + 1)
    # plot
    pt.scape_plot_from_array(colours, times=time_slots, ax=ax)
    # add colour bar
    if cbar_kwargs is None:
        cbar_kwargs = {}
    if colorbar is True:
        plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, **cbar_kwargs)
    elif not colorbar is False:
        plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=colorbar, **cbar_kwargs)
