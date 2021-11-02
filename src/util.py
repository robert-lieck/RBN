#  Copyright (c) 2020 Robert Lieck
import re
from copy import deepcopy
import math
from collections import namedtuple
import time
import datetime
from warnings import warn

import torch
import numpy as np


__MUSIC__ = "music"
__TIME_SERIES__ = "time series"
__COLOUR__ = "colour"
__TREE__ = "tree"


class MultivariateNormal(torch.distributions.MultivariateNormal):

    def log_max(self):
        half_log_det = self._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        return -0.5 * (self._event_shape[0] * math.log(2 * math.pi)) - half_log_det


def get_node_probs(rbn, batch_idx=0, max_1=True, expected_n_nodes=None):
    # get probabilities
    probs = rbn.node_log_coef.arr.exp().detach().numpy()[:, batch_idx]
    # ensure they are not greater than one (due to approximations)
    if max_1:
        # make sure they actual reach 1 (should be the case for the root node)
        probs = probs / probs.max()
    else:
        # only rescale if they surpass 1
        probs = probs / max(1, probs.max())
    # rescale to match expected number of nodes
    if expected_n_nodes is not None:
        # rescale to [0, 1]
        probs = (probs - probs.min()) / (probs.max() - probs.min())
        # current expected number of nodes
        current_n_nodes = probs.sum()
        # required offset from zero to meet target
        alpha = (expected_n_nodes - current_n_nodes) / (len(probs) - current_n_nodes)
        # rescale
        probs = probs * (1 - alpha) + alpha
    # return as TMap
    return TMap(probs)


def remap_to_color(arr):
    n_missing = 3 - arr.shape[1]
    if n_missing > 0:
        arr = np.concatenate((arr, np.zeros((arr.shape[0], n_missing))), axis=1)
    min_val = arr.min()
    max_val = arr.max()
    return (arr - min_val) / (max_val - min_val)


def default_dict(default_dict=None, **kwargs):
    if default_dict is None:
        default_dict = {}
    return {**dict(**kwargs), **default_dict}


def print_options(linewidth=200, precision=3, **kwargs):
    np.set_printoptions(linewidth=linewidth, precision=precision, **kwargs)
    torch.set_printoptions(linewidth=linewidth, precision=precision, **kwargs)



def check_for_checkpoint(w, last_checkpoint):
    enabled = w['checkpoint']['checkbox'].value
    threshold = w['checkpoint']['time'].value * w['checkpoint']['factor'].value
    return enabled and time.time() - last_checkpoint >= threshold


def get_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def save_checkpoint(model, timestamp=None, prefix='', suffix='_model'):
    if timestamp is None:
        timestamp = get_timestamp()
    torch.save(model, f'{prefix}{timestamp}{suffix}.pt')
    torch.save(model.state_dict(), f'{prefix}{timestamp}{suffix}_state_dict.pt')


def pretty_arr(arr, indent="", indent_first=True):
    """Print given array with indentation"""
    if indent_first:
        first_indent = indent
    else:
        first_indent = ""
    return first_indent + str(arr).replace("\n", "\n" + indent)


MusicData = namedtuple('MusicData', 'observations, scapes, colours, times, names')


class TMap:
    """
    A wrapper around a 1D array that provides access in triangular (or ternary) coordinates. A 1D array of length
    N = n * (n + 1) / 2 is mapped to a triangular layout as follows (here with N=21 and n=6):

                           /\         depth  level
                          /0 \            0      6
                         /\  /\
                        /1 \/ 2\          1      5
                       /\  /\  /\
                      /3 \/4 \/5 \        2      4
                     /\  /\  /\  /\
                    /6 \/7 \/8 \/9 \      3      3
                   /\  /\  /\  /\  /\
                  /10\/11\/12\/13\/14\    4      2
                 /\  /\  /\  /\  /\  /\
                /15\/16\/17\/18\/19\/20\  5      1
               |   |   |   |   |   |   |
    start/end: 0   1   2   3   4   5   6

    Values can be accessed by start and end index (0 <= start < end <= n) as follows:
                   (0, 6)
                (0, 5) (1, 6)
             (0, 4) (1, 5) (2, 6)
          (0, 3) (1, 4) (2, 5) (3, 6)
       (0, 2) (1, 3) (2, 4) (3, 5) (4, 6)
    (0, 1) (1, 2) (2, 3) (3, 4) (4, 5) (5, 6)

    That is (start, end) is mapped to the linear index depth * (depth + 1) / 2 + end - level, where
    depth = n - (end - start) and level = n - depth. Advanced integer index arrays are processed in the same way and
    are applied to the underlying array following standard numpy rules (e.g. direct assignment works but otherwise a
    copy of the values is returned instead of a view). Additionally, slices by depth or level return views of the
    underlying array segment. Slices by start or end index are also supported but internally use advanced indexing, so
    they return a copy, not a view.
    """

    flatten_regex = re.compile("^(?P<outer_sign>[+-]?)(?P<outer>[sel])(?P<inner_sign>[+-]?)(?P<inner>[sel])$")

    class GetSetWrapper:
        """Wrapper class that delegates __getitem__ and __setitem__ to custom functions"""
        def __init__(self, getter, setter):
            self.getter = getter
            self.setter = setter

        def __getitem__(self, item):
            return self.getter(item)

        def __setitem__(self, key, value):
            self.setter(key, value)

    @classmethod
    def to_int(cls, i):
        """
        Convert i to integer, no matter whether it is a single number or a numpy array.
        :param i: number or array of numbers
        :return: integer or array of integers
        """
        if isinstance(i, np.ndarray):
            return i.astype(int)
        else:
            return int(i)

    @classmethod
    def size1d_from_n(cls, n):
        """
        Calculate the size N of the underlying 1D array for a given width n of the triangular map: N = n * (n + 1)) / 2.
        This function also works with arrays.
        :param n: Width (number of entries at the bottom of the map)
        :return: Length of underlying 1D array (total number of entries in the map)
        """
        return cls.to_int((n * (n + 1)) / 2)

    @classmethod
    def n_from_size1d(cls, N):
        """
        Calculate width n of the map given the size N of the underlying 1D array: n = (sqrt(8 * N + 1) - 1) / 2.
        Checks for valid size (i.e. if the resulting n is actually an integer) and raises a ValueError otherwise.
        This function also works with arrays.
        :param N: size of the underlying 1D array
        :return: width of the map
        """
        n = (np.sqrt(8 * N + 1) - 1) / 2
        if cls.size1d_from_n(np.floor(n)) != N:
            raise ValueError(f"{N} is not a valid size for a triangular map (n={n})")
        return cls.to_int(np.floor(n))

    def __init__(self, arr, linearise_blocks=False):
        self.n = self.n_from_size1d(len(arr))
        self.arr = arr
        try:
            self.value_shape = self.arr.shape[1:]
        except AttributeError:
            self.value_shape = ()
        self.linearise_blocks = linearise_blocks
        self.sslice = TMap.GetSetWrapper(getter=self.get_sslice, setter=self.set_sslice)
        self.eslice = TMap.GetSetWrapper(getter=self.get_eslice, setter=self.set_eslice)
        self.sblock = TMap.GetSetWrapper(getter=self.get_sblock, setter=self.set_sblock)
        self.eblock = TMap.GetSetWrapper(getter=self.get_eblock, setter=self.set_eblock)

    def _check(self, start, end):
        """
        Check whether 0 <= start < end < n. This function also works with arrays, in which case all start/end values
        have to pass the check. If the check is not passed, an IndexError is raised
        :param start: start index or array of indices
        :param end: end index or array of indices
        """
        if isinstance(start, np.ndarray) or isinstance(end, np.ndarray):
            c = start < end
            c = np.logical_and(c, 0 <= start)
            c = np.logical_and(c, end <= self.n)
            do_raise = not np.all(c)
        else:
            do_raise = not (0 <= start < end <= self.n)
        if do_raise:
            raise IndexError(f"Invalid indices for TMap with size n={self.n}\nstart: {start}\nend: {end}")

    def depth(self, start, end):
        """
        Compute the depth d corresponding to (start, end): d = n - (end - start). This function also works with arrays.
        :param start: start index or array of indices
        :param end: end index or array of indices
        :return: depth or array of depth values
        """
        return self.to_int(self.n - (end - start))

    def level(self, *args):
        """
        Compute the level from depth or (start, end). If (start, end) is given, the depth function is used to first
        compute the depth. This function also works with arrays.
        :param args: either one argument (depth) or two (start, end), which can also be arrays
        :return: depth or array of depth values
        """
        if len(args) == 1:
            depth = args[0]
        elif len(args) == 2:
            depth = self.depth(*args)
        else:
            raise TypeError(f"Expected one positional argument (depth) or two (start, end) to compute level but got "
                            f"{len(args)}")
        return self.n - depth

    def linear_from_start_end(self, start, end):
        """
        Compute the linear index (in the underlying 1D array) corresponding to a (start, end) pair. This function also
        works with arrays.
        :param start: start index or array of indices
        :param end: end index or array of indices
        :return: linear index or array of linear indices
        """
        depth = self.depth(start, end)
        level = self.level(depth)
        return self.to_int(depth * (depth + 1) / 2 + end - level)

    def __getitem__(self, item):
        """
        Get the item corresponding to (start, end) or the sub-map corresponding to slice start:end. For items, this
        function also works with arrays, resulting in advanced indexing of the underlying 1D array.
        :param item: (start, end) or slice
        :return: element or sub-map
        """
        if isinstance(item, slice):
            # return sub-map
            # function to concatenate slices
            if isinstance(self.arr, np.ndarray):
                cat = np.concatenate
            elif isinstance(self.arr, torch.Tensor):
                cat = torch.cat
            else:
                cat = lambda ls: sum(ls, [])
            # star/end and depth to start with
            start = item.start
            end = item.stop
            step = item.step
            start_depth = self.depth(start, end)
            # get new array of sub-map by concatenating slices
            arr_tuple = tuple(
                self.dslice(d)[slice(start, start + (d - start_depth) + 1, step)] for d in range(start_depth, self.n)
            )
            arr = cat(arr_tuple)
            # return new TMap
            return TMap(arr, linearise_blocks=self.linearise_blocks)
        else:
            # return element
            start, end = item
            self._check(start, end)
            linear_idx = self.linear_from_start_end(start, end)
            return self.arr[linear_idx]

    def __setitem__(self, key, value):
        """
        Set the item corresponding to (start, end). This function also works with arrays, resulting in advanced indexing
        of the underlying 1D array.
        :param key: (start, end)
        :value: value to set
        """
        start, end = key
        self._check(start, end)
        linear_idx = self.linear_from_start_end(start, end)
        self.arr[linear_idx] = value

    def copy(self):
        """
        Copy the map. If the underlying data is a lists, tuple, numpy array or pytorch tensor, the appropriate functions
        are called, otherwise a deepcopy of the data is made.
        :return: Copied map.
        """
        if isinstance(self.arr, np.ndarray):
            copy = self.arr.copy()
        elif isinstance(self.arr, torch.Tensor):
            copy = self.arr.detach().clone()
        elif isinstance(self.arr, list):
            copy = list(self.arr)
        elif isinstance(self.arr, tuple):
            copy = tuple(self.arr)
        else:
            copy = deepcopy(self.arr)
        return TMap(arr=copy, linearise_blocks=self.linearise_blocks)

    def top(self, depth=None):
        """
        Return the sub-map corresponding to the top-most levels. A view of the underlying data is used, so the returned
        TMap shares the same buffer and modification affect both objects.
        :param depth: How many levels from the top to include
        :return: sub-map
        """
        if depth is None:
            return self
        else:
            start, end = self.linear_start_end_from_level(self.level(depth - 1))
            return TMap(self.arr[:end + 1], linearise_blocks=self.linearise_blocks)

    def linear_start_end_from_level(self, level):
        """
        Compute the linear 1D start and end index corresponding to all values in the respective level of the map.
        Slicing the underlying array as arr[start:end + 1] will return a view of the values on the level.
        :param level: level for which to compute the indices
        :return: linear 1D start and end index
        """
        linear_start = self.linear_from_start_end(0, level)
        linear_end = self.linear_from_start_end(self.n - level, self.n)
        return linear_start, linear_end

    def lslice(self, level):
        """
        Slice the map at the given level, returning a view of the values.
        :param level: level to use for slicing
        :return: view of the values
        """
        linear_start, linear_end = self.linear_start_end_from_level(level)
        return self.arr[linear_start:linear_end + 1]

    def dslice(self, depth):
        """
        Slice the map at the given depth, returning a view of the values.
        :param depth: depth to use for slicing
        :return: view of the values
        """
        return self.lslice(self.level(depth))

    def end_indices_for_sslice(self, start):
        """
        Compute the end indices corresponding to a slice at the give start index.
        :param start: start index
        :return: integer array of end indices
        """
        return np.arange(start + 1, self.n + 1)

    def start_indices_for_eslice(self, end):
        """
        Compute the start indices corresponding to a slice at the give end index.
        :param end: end index
        :return: integer array of start indices
        """
        return np.arange(0, end)

    def get_sslice(self, item):
        """
        Return a slice for the given start index. Internally, advanced indexing is used, so the returned values are
        a copy, not a view. item can be a tuple to further slice down before retrieving the values.
        :param item: start index or tuple of start index and additional indices/slices
        :return: copy of slice at start index
        """
        if isinstance(item, tuple):
            start = item[0]
            s = item[1:]
            end_indices = self.end_indices_for_sslice(start)[s]
        else:
            start = item
            end_indices = self.end_indices_for_sslice(start)
        return self[start, end_indices]

    def set_sslice(self, key, value):
        """
        Like get_sslice but set value instead of returning values.
        """
        if isinstance(key, tuple):
            start = key[0]
            s = key[1:]
            end_indices = self.end_indices_for_sslice(start)[s]
        else:
            start = key
            end_indices = self.end_indices_for_sslice(start)
        self[start, end_indices] = value

    def get_eslice(self, item):
        """
        Return a slice for the given end index. Internally, advanced indexing is used, so the returned values are
        a copy, not a view. item can be a tuple to further slice down before retrieving the values.
        :param item: end index or tuple of end index and additional indices/slices
        :return: copy of slice at end index
        """
        if isinstance(item, tuple):
            end = item[0]
            s = item[1:]
            start_indices = self.start_indices_for_eslice(end)[s]
        else:
            end = item
            start_indices = self.start_indices_for_eslice(end)
        return self[start_indices, end]

    def set_eslice(self, key, value):
        """
        Like get_eslice but set value instead of returning values.
        """
        if isinstance(key, tuple):
            end = key[0]
            s = key[1:]
            start_indices = self.start_indices_for_eslice(end)[s]
        else:
            end = key
            start_indices = self.start_indices_for_eslice(end)
        self[start_indices, end] = value

    def get_sblock(self, item):
        """
        Return a block of sslices down from the specified level.
        """
        if isinstance(item, tuple):
            level = item[0]
            s = item[1:]
        else:
            level = item
            s = (slice(None), slice(None))
        start_indices = np.arange(0, self.n - level + 1)
        end_indices = np.concatenate(
            [np.flip(self.end_indices_for_sslice(start)[:level, None], axis=0) for start in start_indices],
            axis=1
        )
        start_indices = start_indices[None, :]
        linear_indices = self.linear_from_start_end(start_indices, end_indices)[s]
        if self.linearise_blocks:
            index = (linear_indices.flatten(),) + tuple([slice(None)] * len(self.value_shape))
            return self.arr[index].reshape(linear_indices.shape + self.value_shape)
        else:
            return self.arr[linear_indices]

    def set_sblock(self, key, value):
        """
        Like get_sblock but set value.
        """
        if isinstance(key, tuple):
            level = key[0]
            s = key[1:]
        else:
            level = key
            s = (slice(None), slice(None))
        start_indices = np.arange(0, self.n - level + 1)
        end_indices = np.concatenate(
            [np.flip(self.end_indices_for_sslice(start)[:level, None], axis=0) for start in start_indices],
            axis=1
        )
        start_indices = start_indices[None, :]
        linear_indices = self.linear_from_start_end(start_indices, end_indices)[s]
        if self.linearise_blocks:
            index = (linear_indices.flatten(),) + tuple([slice(None)] * len(self.value_shape))
            self.arr[index] = value
        else:
            self.arr[linear_indices] = value

    def get_eblock(self, item):
        """
        Return a block of eslices down from the specified level.
        """
        if isinstance(item, tuple):
            level = item[0]
            s = item[1:]
        else:
            level = item
            s = (slice(None), slice(None))
        end_indices = np.arange(level, self.n + 1)
        start_indices = np.concatenate(
            [self.start_indices_for_eslice(end)[-level:, None] for end in end_indices],
            axis=1
        )
        end_indices = end_indices[None, :]
        linear_indices = self.linear_from_start_end(start_indices, end_indices)[s]
        if self.linearise_blocks:
            index = (linear_indices.flatten(),) + tuple([slice(None)] * len(self.value_shape))
            return self.arr[index].reshape(linear_indices.shape + self.value_shape)
        else:
            return self.arr[linear_indices]

    def set_eblock(self, key, value):
        """
        Like get_sblock but set value.
        """
        if isinstance(key, tuple):
            level = key[0]
            s = key[1:]
        else:
            level = key
            s = (slice(None), slice(None))
        end_indices = np.arange(level, self.n + 1)
        start_indices = np.concatenate(
            [self.start_indices_for_eslice(end)[-level:, None] for end in end_indices],
            axis=1
        )
        end_indices = end_indices[None, :]
        linear_indices = self.linear_from_start_end(start_indices, end_indices)[s]
        if self.linearise_blocks:
            index = (linear_indices.flatten(),) + tuple([slice(None)] * len(self.value_shape))
            self.arr[index] = value
        else:
            self.arr[linear_indices] = value

    def flatten(self, order="-l+s"):
        """
        Return map in linear order. The different orders correspond to iteration using two nested for loops where the
        first letter indicates the outer dimension and the second the inner: s: start, e: end, l: level. A minus sign
        reverses the order of the respective dimension.
        :param order: string specifying order of linearisation: '+s+e', '+e+s', '+l+s' (+ can be omitted or replaced
        with -)
        :return: 1D array with values in given order
        """
        # get order info
        match = self.flatten_regex.match(order)
        if match is None:
            raise ValueError(f"Invalid order '{order}'")
        outer_dim = match['outer']
        inner_dim = match['inner']
        outer_sign = match['outer_sign']
        inner_sign = match['inner_sign']
        # check
        if (outer_dim, inner_dim) not in [('s', 'e'), ('e', 's'), ('l', 's')]:
            raise ValueError(f"Outer/inner dimension must be s/e, e/s or l/s but are {outer_dim}/{inner_dim}")
        # collect outer slices
        slices = []
        if outer_dim == 's':
            for start in range(self.n):
                slices.append(self.sslice[start])
        elif outer_dim == 'e':
            for end in range(1, self.n + 1):
                slices.append(self.eslice[end])
        else:
            assert outer_dim == 'l', outer_dim
            for level in range(1, self.n + 1):
                slices.append(self.lslice(level))
        # adjust sign for outer dimension
        if outer_sign == '-':
            slices = reversed(slices)
        else:
            assert not outer_sign or outer_sign == '+', outer_sign
        # adjust sign for inner dimension
        if inner_sign == '-':
            slices = [np.flip(s) for s in slices]
        else:
            assert not inner_sign or inner_sign == '+', inner_sign
        # concatenate and return
        if isinstance(self.arr, torch.Tensor):
            return torch.cat(tuple(slices))
        else:
            return np.concatenate(tuple(slices))

    def __repr__(self):
        return f"TMap(n={self.n}, {self.arr}, linearise_blocks={self.linearise_blocks})"

    def __str__(self):
        """
        Return a string representation of the map, consisting of consecutive dsclices.
        """
        s = ""
        for depth in range(self.n):
            if s:
                s += "\n"
            try:
                s += str(self.dslice(depth))
            except TypeError:
                s += "["
                linear_start, linear_end = self.linear_start_end_from_level(self.level(depth))
                s += " ".join([str(self.arr[idx]) for idx in range(linear_start, linear_end + 1)])
                s += "]"
        return s

    def pretty(self, cut=None, str_func=None, detach_pytorch=True, scf=None, pos=None, rnd=None):
        """
        :param cut: cut at specified level, printing only the bottom 'cut' levels of the map
        :param str_func: function to convert values to strings (default: str)
        :param detach_pytorch: whether to detach tensors if the underlying array is a pytorch tensor
        :param scf: kwargs to use np.format_float_scientific to format value
        :param pos: kwargs to use np.format_float_positional to format value
        :param rnd: kwargs to use np.around to format value
        :return: pretty-printed string

         ╳
        ╳0╳
       ╳0╳0╳
      ╳0╳0╳0╳
     ╳0╳0╳0╳0╳
    ╳0╳0╳0╳0╳0╳
   ╳0╳0╳0╳0╳0╳0╳
  ╳0╳0╳0╳0╳0╳0╳0╳
 ╳0╳0╳0╳0╳0╳0╳0╳0╳
╳0╳0╳0╳0╳0╳0╳0╳0╳0╳
│ │ │ │ │ │ │ │ │ │
0 1 2 3 4 5 6 7 8 9

                 ╳
                ╱ ╲
               ╳000╳
              ╱ ╲ ╱ ╲
             ╳000╳000╳
            ╱ ╲ ╱ ╲ ╱ ╲
           ╳000╳000╳000╳
          ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲
         ╳000╳000╳000╳000╳
        ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲
       ╳000╳000╳000╳000╳000╳
      ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲
     ╳000╳000╳000╳000╳000╳000╳
    ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲
   ╳000╳000╳000╳000╳000╳000╳000╳
  ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲
 ╳000╳000╳000╳000╳000╳000╳000╳000╳
 │   │   │   │   │   │   │   │   │
 0   1   2   3   4   5   6   7   8

                 ╱╲
                ╱00╲
               ╱╲  ╱╲
              ╱00╲╱00╲
             ╱╲  ╱╲  ╱╲
            ╱00╲╱00╲╱00╲
           ╱╲  ╱╲  ╱╲  ╱╲
          ╱00╲╱00╲╱00╲╱00╲
         ╱╲  ╱╲  ╱╲  ╱╲  ╱╲
        ╱00╲╱00╲╱00╲╱00╲╱00╲
       ╱╲  ╱╲  ╱╲  ╱╲  ╱╲  ╱╲
      ╱00╲╱00╲╱00╲╱00╲╱00╲╱00╲
     ╱╲  ╱╲  ╱╲  ╱╲  ╱╲  ╱╲  ╱╲
    ╱00╲╱00╲╱00╲╱00╲╱00╲╱00╲╱00╲
   ╱╲  ╱╲  ╱╲  ╱╲  ╱╲  ╱╲  ╱╲  ╱╲
  ╱00╲╱00╲╱00╲╱00╲╱00╲╱00╲╱00╲╱00╲
 ╱╲  ╱╲  ╱╲  ╱╲  ╱╲  ╱╲  ╱╲  ╱╲  ╱╲
╱00╲╱00╲╱00╲╱00╲╱00╲╱00╲╱00╲╱00╲╱00╲
│   │   │   │   │   │   │   │   │   │
0   1   2   3   4   5   6   7   8   9

                          ╱╲
                         ╱  ╲
                        ╱0000╲
                       ╱╲    ╱╲
                      ╱  ╲  ╱  ╲
                     ╱0000╲╱0000╲
                    ╱╲    ╱╲    ╱╲
                   ╱  ╲  ╱  ╲  ╱  ╲
                  ╱0000╲╱0000╲╱0000╲
                 ╱╲    ╱╲    ╱╲    ╱╲
                ╱  ╲  ╱  ╲  ╱  ╲  ╱  ╲
               ╱0000╲╱0000╲╱0000╲╱0000╲
              ╱╲    ╱╲    ╱╲    ╱╲    ╱╲
             ╱  ╲  ╱  ╲  ╱  ╲  ╱  ╲  ╱  ╲
            ╱0000╲╱0000╲╱0000╲╱0000╲╱0000╲
           ╱╲    ╱╲    ╱╲    ╱╲    ╱╲    ╱╲
          ╱  ╲  ╱  ╲  ╱  ╲  ╱  ╲  ╱  ╲  ╱  ╲
         ╱0000╲╱0000╲╱0000╲╱0000╲╱0000╲╱0000╲
        ╱╲    ╱╲    ╱╲    ╱╲    ╱╲    ╱╲    ╱╲
       ╱  ╲  ╱  ╲  ╱  ╲  ╱  ╲  ╱  ╲  ╱  ╲  ╱  ╲
      ╱0000╲╱0000╲╱0000╲╱0000╲╱0000╲╱0000╲╱0000╲
     ╱╲    ╱╲    ╱╲    ╱╲    ╱╲    ╱╲    ╱╲    ╱╲
    ╱  ╲  ╱  ╲  ╱  ╲  ╱  ╲  ╱  ╲  ╱  ╲  ╱  ╲  ╱  ╲
   ╱0000╲╱0000╲╱0000╲╱0000╲╱0000╲╱0000╲╱0000╲╱0000╲
  ╱╲    ╱╲    ╱╲    ╱╲    ╱╲    ╱╲    ╱╲    ╱╲    ╱╲
 ╱  ╲  ╱  ╲  ╱  ╲  ╱  ╲  ╱  ╲  ╱  ╲  ╱  ╲  ╱  ╲  ╱  ╲
╱0000╲╱0000╲╱0000╲╱0000╲╱0000╲╱0000╲╱0000╲╱0000╲╱0000╲
│     │     │     │     │     │     │     │     │     │
0     1     2     3     4     5     6     7     8     9
        """
        # get function to convert values to strings
        if str_func is None:
            if scf is not None:
                def str_func(val):
                    return np.format_float_scientific(val, **scf)
            elif pos is not None:
                def str_func(val):
                    return np.format_float_positional(val, **pos)
            elif rnd is not None:
                def str_func(val):
                    if rnd.setdefault("decimals", 0) <= 0:
                        try:
                            return str(int(np.around(val, **rnd)))
                        except ValueError:
                            return str(np.around(val, **rnd))
                    else:
                        return str(np.around(val, **rnd))
            else:
                str_func = str
            # if isinstance(self.arr, torch.Tensor) and detach_pytorch:
            #     def str_func(val):
            #         return str(val.detach().numpy()[0])
            # else:
            #     str_func = str
        # get values as strings
        str_slices = []
        max_width = -1
        for depth in range(self.n):
            str_slices.append([])
            # level corresponding to depth
            level = self.level(depth)
            # cut at level
            if cut is not None and level > cut:
                continue
            for val in self.dslice(depth):
                if isinstance(self.arr, torch.Tensor) and detach_pytorch:
                    val = val.detach().numpy()
                str_val = str_func(val)
                max_width = max(max_width, len(str_val))
                str_slices[-1].append(str_val)
        # width must be even
        max_width = int(2 * np.ceil(max_width / 2))
        # adjust width
        for sl_idx, sl in enumerate(str_slices):
            for idx, str_val in enumerate(sl):
                str_slices[sl_idx][idx] = str_val.rjust(max_width)
        # for x in str_slices:
        #     print(x)
        # generate triangular matrix
        lines_per_level = (max_width - 1) // 2 + 2
        s = ""
        for depth, sl in enumerate(str_slices):
            # level corresponding to depth
            level = self.level(depth)
            # cut at level
            if cut is not None and level > cut:
                continue
            # base indentation of this slice
            depth_indent = " " * lines_per_level * (level - 1)
            for line in range(lines_per_level - 1):
                # newline exept for first line
                if s:
                    s += "\n"
                # additional indent for this line of slice
                line_indent = " " * (lines_per_level - line - 1)
                # spacing within and in between cells
                within_cell_spacing = " " * 2 * line
                in_between_cell_spacing = " " * 2 * (lines_per_level - line - 1)
                # add indentation
                s += depth_indent + line_indent
                s += ("╱" + within_cell_spacing + "╲" + in_between_cell_spacing) * depth + "╱" + within_cell_spacing + "╲"
            s += "\n" + depth_indent + "╱" + "╲╱".join(sl) + "╲"
        return s

def f1_score_trees(actual, estimated, n_obs, tolerance=0, probs=False, return_matches=False):
    """
    Compute the precision, recall, and F1-score for an actual and an estimated tree structure
    """
    # list of the probabilities for the actual (ground-truth) nodes to be present in the estimated
    match_probabilities = []
    # go through all the actual nodes (represented by the start and end of their span)
    for actual_start, actual_end in actual:
        # probabilities to find node in estimate (with various offsets if tolerance > 0)
        p = np.zeros((2 * tolerance + 1, 2 * tolerance + 1))
        # index in p and offset for start
        for start_idx, start_offset in enumerate(range(-tolerance, tolerance + 1)):
            start = actual_start + start_offset
            # exclude out of range starts
            if not (0 <= start <= n_obs):
                continue
            # index in p and offset for end
            for end_index, end_offset in enumerate(range(-tolerance, tolerance + 1)):
                end = actual_end + end_offset
                # exclude out of range ends
                if not (0 <= end <= n_obs):
                    continue
                # exclude "inverted" spans (for low-level nodes)
                if start >= end:
                    continue
                # assign probability
                if probs:
                    # expect all possible nodes to have an expected probability
                    p[start_idx, end_index] = estimated[(start, end)]
                else:
                    # probability is 1 if node in expected ones, otherwise zero
                    p[start_idx, end_index] = int((start, end) in estimated)
        # check for multiple matches (more than one estimate within tolerance)
        if p.sum() > 1:
            warn(RuntimeWarning(f"Multiple estimated nodes within tolerance for node {(actual_start, actual_end)}\n"
                                f"{p}"))
        # probability of a "match" is that of not all possibilities not matching
        match_probabilities.append(1 - (1 - p).prod())
    # compute number of true positives ("matches"), actual positives, and estimated positives
    n_true_positives = sum(match_probabilities)
    n_actual_positives = len(match_probabilities)
    if probs:
        if isinstance(estimated, TMap):
            n_estimated_positives = estimated.arr.sum()
        else:
            n_estimated_positives = 0
            for s in range(n_obs):
                for e in range(s + 1, n_obs):
                    n_estimated_positives += estimated[(s, e)]
    else:
        n_estimated_positives = len(estimated)
    if return_matches:
        return dict(n_true_positives=n_true_positives,
                    n_actual_positives=n_actual_positives,
                    n_estimated_positives=n_estimated_positives)
    else:
        return precision_recall_f1(n_true_positives=n_true_positives,
                                   n_actual_positives=n_actual_positives,
                                   n_estimated_positives=n_estimated_positives)


def f1_score_cps(actual, estimated, tolerance=0, return_matches=False):
    """
    Compute the precision, recall, and F1-score for an actual and an estimated set of change points
    """
    n_matches = 0
    for bkpt in actual:
        match = False
        for offset in (range(-tolerance, tolerance + 1)):
            if bkpt + offset in estimated:
                if match:
                    warn(RuntimeWarning(f"Multiple estimated break points within tolerance for break point at {bkpt}"))
                match = True
        n_matches += int(match)
    if return_matches:
        return dict(n_true_positives=n_matches,
                    n_actual_positives=len(actual),
                    n_estimated_positives=len(estimated))
    else:
        return precision_recall_f1(n_true_positives=n_matches,
                                   n_actual_positives=len(actual),
                                   n_estimated_positives=len(estimated))


def precision_recall_f1(n_true_positives, n_actual_positives, n_estimated_positives):
    # compute recall and precision
    if n_actual_positives == 0:
        recall = 0
    else:
        recall = n_true_positives / n_actual_positives
    if n_estimated_positives == 0:
        precision = 0
    else:
        precision = n_true_positives / n_estimated_positives
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1


def hierarchical_clustering(data, change_points=None):
    """
    Hierarchically clusters adjacent data points by minial distance
    :param data: NxM data of N M-dimensional data points
    :param change_points: change points for initial segmentation (not including start and end of sequence)
    :return: tuple of
     - array of shape (N-1,2) with nodes start and end index
     - array of shape (N-1,4,2) with start and end indices of left child, parent, right_child, (NaN, NaN) for drawing the corresponding tree;
     - dict with (parent_start, parent_end): [children, ...] mappings
    """
    # remember nodes for plotting
    nodes = []  # for direct line plot
    node_dict = {}  # parent-->child relations for tree plot
    if change_points is None:
        # no change points provided --> start with single data points as sections
        n_obs = data.shape[0]
        # spans contain the start and end indices of the sections (is getting shorter as sections are merged)
        spans = np.zeros((n_obs, 2), dtype=int)
        spans[:, 0] = np.arange(n_obs)
        spans[:, 1] = np.arange(1, n_obs + 1)
        # means of the current sections (is getting shorter as sections are merged)
        means = data
    else:
        # change points provided --> use as initial spans
        spans = np.zeros((len(change_points) + 1, 2), dtype=int)
        # set start and end of sequence
        spans[0, 0] = 0
        spans[-1, 1] = data.shape[0]
        # set intermediate change points
        spans[1:, 0] = change_points
        spans[:-1, 1] = change_points
        # compute means for given spans
        means = np.array([data[s:e].mean(axis=0) for s, e in spans])
    # add leaf nodes
    for s, e in spans:
        node_dict[(s, e)] = []
    while len(means) > 1:
        # find best match: relative index (in means and spans, not data index) of best pair to merge
        left_idx = np.argmin(np.linalg.norm(means[:-1] - means[1:], axis=1))
        right_idx = left_idx + 1
        # get absolute start/end index (data index) of the new parent node and the child nodes
        new_node = spans[(left_idx, right_idx), (0, 1)]  # use advanced indexing to pick
        left_child = spans[left_idx]
        right_child = spans[right_idx]
        # get span width for weighting and compute weighted mean of new parent node
        left_width = left_child[1] - left_child[0]
        right_width = right_child[1] - right_child[0]
        new_mean = (left_width * means[left_idx] + right_width * means[right_idx]) / (left_width + right_width)
        # append nodes
        nodes += [[left_child, new_node, right_child, [np.nan, np.nan]]]
        node_dict[tuple(new_node)] = [tuple(left_child), tuple(right_child)]
        # print("parent/left/right:", new_node, left_child, right_child)
        # update data and spans
        means = np.concatenate((means[:left_idx],
                                [new_mean],
                                means[right_idx + 1:]))
        spans = np.concatenate((spans[:left_idx], [new_node], spans[right_idx + 1:]))
    nodes = np.array(nodes)
    return nodes[:, 1, :], nodes, node_dict


def pretty_print_chart(chart, exp=False, prec=3):
    if exp:
        map = TMap(chart.arr.exp())
    else:
        map = TMap(chart.arr)
    print(map.pretty(str_func=lambda n: np.format_float_scientific(n, precision=prec)))