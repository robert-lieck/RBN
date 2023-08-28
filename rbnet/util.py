import numpy as np


def normalize_non_zero(a, axis=None, skip_type_check=False):
    """For the given ND array, normalise each 1D array obtained by indexing the 'axis' dimension if the sum along the
    other dimensions (for that entry) is non-zero. Normalisation is performed in place."""
    # check that dtype is float (in place division of integer arrays silently rounds)
    if not skip_type_check:
        if not np.issubdtype(a.dtype, np.floating):
            raise TypeError(f"Cannot guarantee that normalisation works as expected on array of type '{a.dtype}'. "
                            f"Use 'skip_type_check=True' to skip this check.")
    # normalise along last axis per default
    if axis is None:
        axis = a.ndim - 1
    # make axis a tuple if it isn't
    if not isinstance(axis, tuple):
        axis = (axis,)
    # compute sum along axis, keeping dimensions
    s = a.sum(axis=axis, keepdims=True)
    # check for non-zero entries
    non_zero = (s != 0)
    if not np.any(non_zero):
        # directly return if there are no non-zero entries
        return a
    # construct an index tuple to select the appropriate entries for normalisation (the dimensions specified by axis
    # have to be replaced by full slices ':' to broadcast normalisation along these dimensions)
    non_zero_arr = tuple(slice(None) if idx in axis else n for idx, n in enumerate(non_zero.nonzero()))
    # in-place replace non-zero entries by their normalised values
    a[non_zero_arr] = a[non_zero_arr] / s[non_zero_arr]
    # return array
    return a