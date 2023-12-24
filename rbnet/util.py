import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import pytorch_lightning as pl


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
def normalize_non_zero(a, axis=None, skip_type_check=False):
    """
    For the given ND array (NumPy or PyTorch), normalise each 1D array obtained by indexing the 'axis' dimension if
    the sum along the other dimensions (for that entry) is non-zero. Normalisation is performed in place.
    """
    # check array type (NumPy/PyTorch)
    if isinstance(a, np.ndarray):
        lib = "numpy"
    elif isinstance(a, torch.Tensor):
        lib = "pytorch"
    else:
        raise TypeError(f"Not implemented for arrays of type {type(a)}")
    # check that dtype is float (in place division of integer arrays silently rounds)
    if not skip_type_check:
        if ((lib == "numpy" and not np.issubdtype(a.dtype, np.floating)) or
                (lib == "pytorch" and not torch.is_floating_point(a))):
            raise TypeError(f"Cannot guarantee that normalisation works as expected on array of type '{a.dtype}'. "
                            f"Use 'skip_type_check=True' to skip this check.")
    # normalise along last axis per default
    if axis is None:
        axis = a.ndim - 1
    # make axis a tuple if it isn't
    if not isinstance(axis, tuple):
        axis = (axis,)
    # compute sum along axis, keeping dimensions
    if lib == "numpy":
        s = a.sum(axis=axis, keepdims=True)
    if lib == "pytorch":
        s = a.sum(dim=axis, keepdim=True)
    # check for non-zero entries
    non_zero = (s != 0)
    if (lib == "numpy" and not np.any(non_zero)) or (lib == "pytorch" and not torch.any(non_zero)):
        # directly return if there are no non-zero entries
        return a
    # construct an index tuple to select the appropriate entries for normalisation (the dimensions specified by axis
    # have to be replaced by full slices ':' to broadcast normalisation along these dimensions)
    if lib == "pytorch":
        kwargs = dict(as_tuple=True)
    else:
        kwargs = {}
    non_zero_arr = tuple(slice(None) if idx in axis else n for idx, n in enumerate(non_zero.nonzero(**kwargs)))
    # in-place replace non-zero entries by their normalised values
    a[non_zero_arr] = a[non_zero_arr] / s[non_zero_arr]
    # return array
    return a


def as_detached_tensor(t):
    if torch.is_tensor(t):
        return t.clone().detach()
    else:
        return torch.tensor(t)
