import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import pytorch_lightning as pl
import matplotlib.pyplot as plt


_no_value = object()


class SequenceDataModule(pl.LightningDataModule):
    def __init__(self, sequences, val_split=0.2, test_split=0.1):
        super().__init__()
        self.sequences = [as_detached_tensor(s) for s in sequences]
        self.val_split = val_split
        self.test_split = test_split

    def setup(self, stage=None):
        total_samples = len(self.sequences)
        val_size = int(total_samples * self.val_split)
        test_size = int(total_samples * self.test_split)
        train_size = total_samples - val_size - test_size

        train_data, val_data, test_data = random_split(
            self.sequences, [train_size, val_size, test_size]
        )
        self.train_dataset = train_data
        self.val_dataset = val_data
        self.test_dataset = test_data

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=1, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1)


# from https://github.com/robert-lieck/pyulib
def normalize_non_zero(a, axis=_no_value, make_zeros_uniform=False, skip_type_check=False, check_positivity=True):
    """
    For the given ND array (NumPy or PyTorch), normalise each 1D array obtained by indexing the 'axis' dimension if
    the sum along the other dimensions (for that entry) is non-zero. Normalisation is performed in place.
    """
    # check array type (NumPy/PyTorch)
    if isinstance(a, np.ndarray):
        lib = "numpy"
        any = np.any
        all = np.all
        logical_not = np.logical_not
        ones_like = np.ones_like
    elif isinstance(a, torch.Tensor):
        lib = "pytorch"
        any = torch.any
        all = torch.all
        logical_not = torch.logical_not
        ones_like = torch.ones_like
    else:
        raise TypeError(f"Not implemented for arrays of type {type(a)}")
    # check that dtype is float (in place division of integer arrays silently rounds)
    if not skip_type_check:
        if ((lib == "numpy" and not np.issubdtype(a.dtype, np.floating)) or
                (lib == "pytorch" and not torch.is_floating_point(a))):
            raise TypeError(f"Cannot guarantee that normalisation works as expected on array of type '{a.dtype}'. "
                            f"Use 'skip_type_check=True' to skip this check.")
    # check positivity
    if check_positivity and any(a < 0):
        raise ValueError("Some elements are negative")
    # normalise over everything is axis is not provided
    if axis is _no_value:
        # if axis is not specified, keep old behaviour for compatibility
        warnings.warn("Not passing an explicit value to 'axis' is deprecated and will result in an error in the "
                      "future. The old behaviour of implicitly assuming the last axis is currently kept for "
                      "compatibility. The former default value of 'None' now results in normalising over "
                      "everything.", DeprecationWarning)
        axis = a.ndim - 1
    elif axis is None:
        # None normalises over everything
        axis = tuple(range(len(a.shape)))
    # make axis a tuple if it isn't
    if not isinstance(axis, tuple):
        axis = (axis,)

    # helper function to compute sum and non-zero entries
    def get_sum(a):
        if lib == "numpy":
            s = a.sum(axis=axis, keepdims=True)
        if lib == "pytorch":
            s = a.sum(dim=axis, keepdim=True)
        non_zero = (s != 0)
        # construct an index tuple to select the appropriate entries for normalisation (the dimensions specified by axis
        # have to be replaced by full slices ':' to broadcast normalisation along these dimensions)
        kwargs = dict(as_tuple=True) if lib == "pytorch" else {}
        non_zero_arr = tuple(slice(None) if idx in axis else n for idx, n in enumerate(non_zero.nonzero(**kwargs)))
        return s, non_zero, non_zero_arr

    # compute sum and non-zero entries
    s, non_zero, non_zero_arr = get_sum(a)

    # handle zero entries
    if not any(non_zero):
        # all entries are zero
        if make_zeros_uniform:
            # replace a with uniform array
            a = ones_like(a)
            s = get_sum(a)
        else:
            # nothing to normalise: directly return
            return a
    elif not all(non_zero):
        # some entries are zero
        if make_zeros_uniform:
            # create a uniform array, fill non-zero entries with those from a
            new_a = ones_like(a)
            new_a[non_zero_arr] = a[non_zero_arr]
            a = new_a
            s, non_zero, non_zero_arr = get_sum(a)

    # in-place replace non-zero entries by their normalised values
    a[non_zero_arr] = a[non_zero_arr] / s[non_zero_arr]
    # return array
    return a


def as_detached_tensor(t):
    if torch.is_tensor(t):
        return t.clone().detach()
    else:
        return torch.tensor(t)

def plot_vec(func, x_min=0, y_min=0, x_max=1, y_max=1, nx=10, ny=10):
    x, y = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
    x_vec, y_vec = func(x, y)
    plt.quiver(x, y, x_vec, y_vec)
    plt.show()


def plot_grad(func, x_min=0, y_min=0, x_max=1, y_max=1, nx=10, ny=10):
    x, y = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
    x = torch.tensor(x, requires_grad=True)
    y = torch.tensor(y, requires_grad=True)
    loss = func(x, y)
    loss.backward(torch.ones((nx, ny)))
    plt.quiver(x.detach(), y.detach(), x.grad, y.grad)
    plt.show()
