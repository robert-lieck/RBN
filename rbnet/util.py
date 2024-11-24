import warnings
import sys
from functools import wraps

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from triangularmap import TMap

_no_value = object()

# always use strict zip if available
_buildin_zip = zip
if np.all([int(i) for i in sys.version.split(".")[:2]] >= [3, 10]):
    @wraps(zip)
    def zip(*args, **kwargs):
        kwargs = dict(strict=True) | kwargs
        yield from _buildin_zip(*args, **kwargs)
else:
    @wraps(zip)
    def zip(*args, **kwargs):
        yield from _buildin_zip(*args, **kwargs)


class TupleTMap(TMap):
    """
    A tuple of :class:`~TMap` objects. All getters/setters of attributes and items return/take tuples of values.
    Internally, a :class:`~TupleTMap` object actually maintains a tuple of :class:`~TMap` objects and delegates all
    calls to these objects.
    """

    def __init__(self, arrs, *args, **kwargs):
        # do NOT call super().__init__
        l = np.array([len(a) for a in arrs])
        if not np.all(l == l[0]):
            raise ValueError(f"Input arrays have different lengths: {l}")
        self._tmaps = tuple(TMap(a, *args, **kwargs) for a in arrs)

    def __getattribute__(self, item):
        if item in ['_tmaps']:
            return super().__getattribute__(item)
        ret = tuple(getattr(t, item) for t in self._tmaps)
        if callable(ret[0]):
            return lambda *args, **kwargs: tuple(r(*args, **kwargs) for r in ret)
        else:
            return ret

    def __setattr__(self, key, value):
        if key == '_tmaps':
            super().__setattr__(key, value)
        for t, v in zip(self._tmaps, value):
            setattr(t, key, v)

    def __getitem__(self, item):
        return tuple(t.__getitem__(item) for t in self._tmaps)

    def __setitem__(self, key, value):
        for t, v in zip(self._tmaps, value):
            t.__setitem__(key, v)


class ConstrainedModuleMixin:
    """
    A mixin class for modules with constraints to work cooperatively. Calling :meth:`~enforce_constraints` will
    recursively try to call this method on child modules.

    Typical usage would be to call :meth:`~enforce_constraints` on the top-level parent module before the forward pass
    and/or after an optimisation step.

    See :class:`~Prob` for an example.
    """

    def enforce_constraints(self, recurse=True):
        """
        Enforce constraints for module parameters and child modules.

        If modules have constrained parameters, they should override this method to enforce these constraints. If
        they also have child modules and ``recurse=True``, they should additionally call
        ``super().enforce_constraints()`` to recursively propagate the call.
        """
        if recurse:
            for module in self.children():
                try:
                    module.enforce_constraints()
                except AttributeError:
                    pass

    def remap(self, param, _top_level=True, prefix=None):
        for module in self.children():
            try:
                # try to use child for remapping
                rm = module.remap(param, _top_level=False)
            except (AttributeError, KeyError):
                # either has no remap function or could not find param
                pass
            else:
                if prefix is not None:
                    return f"{prefix}{rm}"
                else:
                    return rm
        if _top_level:
            # top-level call should return non-remapped param as fallback
            if prefix is not None:
                return str(param)
            else:
                return param
        else:
            # nested calls should signal they could not find param
            raise KeyError


class ConstrainedModuleList(torch.nn.ModuleList, ConstrainedModuleMixin):
    """A plain ModuleList with :class:`~ConstrainedModuleMixin` to be cooperative and not break recursive calls."""
    pass


class Prob(torch.nn.Module, ConstrainedModuleMixin):
    """
    A class for probability distributions that enforces positivity and normalisation constraints and projects the
    gradient in backward passes.

    :ivar p: probabilities
    :ivar dim: dimensions along which normalisation is applied
    """
    def __init__(self, p, dim=None, raise_zero_norms=True, *args, **kwargs):
        """
        :param p: initial probabilities
        :param dim: dimensions along which normalisation is to be applied or ``None`` for all dimensions
        :param args: passed on to super()__init__
        :param kwargs: passed on to super()__init__
        """
        super().__init__(*args, **kwargs)
        self.p = torch.nn.Parameter(ensure_is_floating_point(p,
                                                             "Probabilities 'p' must be of floating point type"))
        self.p.register_hook(self.project_grad)
        if dim is None:
            self.dim = tuple(range(len(self.p.shape)))
        else:
            self.dim = dim
        self.raise_zero_norms = raise_zero_norms
        self.enforce_constraints()

    def project_grad(self, grad):
        r"""
        Projects the gradient to the tangent space :math:`\bar{g} = g - 1/|g| \sum g`, registered as a hook on the
        parameter.

        :param grad: unconstrained gradient
        :return: projected gradient
        """
        shape = np.array(grad.shape)
        return grad - grad.sum(dim=self.dim, keepdim=True) / np.prod(shape[torch.tensor(self.dim)])

    def enforce_constraints(self, recurse=True):
        """
        Enforces positivity constraints (by clipping) and normalisation constraints.
        """
        with torch.no_grad():
            self.p.clip_(0)
            s = self.p.sum(dim=self.dim, keepdim=True)
            if self.raise_zero_norms and torch.any(s == 0):
                raise RuntimeError("Some normalisation constants are zero")
            self.p.div_(s)


class LogProb(torch.nn.Module, ConstrainedModuleMixin):

    def __init__(self, p=None, log_p=None, dim=None, raise_zero_norms=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if (p is None) == (log_p is None):
            raise ValueError("Have to provide exactly one of 'p' and 'log_p'")
        if p is not None:
            log_p = torch.log(torch.as_tensor(p))
        self.log_p = torch.nn.Parameter(log_p)
        self.log_p.register_hook(self.project_grad)
        if dim is None:
            self.dim = tuple(range(len(self.log_p.shape)))
        else:
            self.dim = dim
        self.raise_zero_norms = raise_zero_norms
        self.enforce_constraints()

    def remap(self, param, _top_level=True, prefix=None):
        if param is self.log_p:
            return self.p
        else:
            raise KeyError

    def enforce_constraints(self, recurse=True):
        with torch.no_grad():
            s = torch.logsumexp(self.log_p, dim=self.dim, keepdim=True)
            if self.raise_zero_norms and torch.any(torch.isinf(s)):
                raise RuntimeError("Some log-normalisation constants are inf")
            self.log_p.sub_(s)

    def project_grad(self, grad):
        shape = np.array(grad.shape)
        return grad - grad.sum(dim=self.dim, keepdim=True) / np.prod(shape[torch.tensor(self.dim)])

    @property
    def p(self):
        return torch.exp(self.log_p)

    @p.setter
    def p(self, value):
        self.log_p = torch.log(value)


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


def ensure_is_floating_point(t, msg=None):
    t = torch.tensor(t)
    if not torch.is_floating_point(t):
        if msg is None:
            msg = "Values must be of floating point type"
        raise TypeError(msg)
    return t


def as_detached_tensor(t):
    """
    Create a detached copy of tensor. If ``t`` already is a tensor, clone and detach it, otherwise create a new tensor.

    :param t: tensor to detach and copy
    :return: detached and copied tensor
    """
    if torch.is_tensor(t):
        return t.clone().detach()
    else:
        return torch.tensor(np.array(t))


def log_normalize(t, *args, **kwargs):
    """
    Normalise tensor ``t`` in log representation by computing :math:`t - \log \sum \exp t` using PyTorch logsumexp.

    :param t:
    :param args: positional arguments passed on to logsumexp
    :param kwargs: key-word arguments passed on to logsumexp
    :return: normalised tensor
    """
    return t - torch.logsumexp(t, *args, **kwargs)


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
    plt.quiver(x.detach().numpy(), y.detach().numpy(), x.grad, y.grad)
    plt.show()
