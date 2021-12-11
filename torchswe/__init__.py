#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""A shallow-water equation solver for pipeline landspill.
"""
__version__ = "0.2"

import os as _os
import logging as _logging
import functools as _functools
_logger = _logging.getLogger("torchswe")


def _dummy_function(*args, **kwargs):  #pylint: disable=unused-argument, useless-return
    """A dummy function for CuPy.

    Many functions in NumPy are not implemented in CuPy. However, most of them are not important.
    In order not to write another codepath for CuPy, we assign this dummy function to CuPy's
    corresponding attributes. Currenty, known functions

    - the member of the context manager: errstate
    - set_printoptions
    """
    _logger.debug("_dummy_function is called.")
    return None


class _DummyErrState:  # pylint: disable=too-few-public-methods
    """A dummy errstate context manager."""
    __enter__ = _dummy_function
    __exit__ = _dummy_function
    def __init__(self, *args, **kwargs):
        pass


# assume these two variables mean the code's running with Legate system
if "LEGATE_MAX_DIM" in _os.environ and "LEGATE_MAX_FIELDS" in _os.environ:
    from legate import numpy as nplike
elif "USE_CUPY" in _os.environ and _os.environ["USE_CUPY"] == "1":
    import cupy as nplike  # pylint: disable=import-error
    import cupyx  # pylint: disable=import-error
    nplike.errstate = _DummyErrState
    nplike.set_printoptions = _dummy_function
    nplike.sync = nplike.cuda.get_current_stream().synchronize
    nplike.get = nplike.ndarray.get
elif "USE_TORCH" in _os.environ and _os.environ["USE_TORCH"] == "1":
    import torch as nplike  # pylint: disable=import-error
    nplike.errstate = _DummyErrState
    nplike.ndarray = nplike.Tensor
    nplike.ndarray.astype = nplike.Tensor.to
    nplike.array = nplike.tensor
    nplike.nonzero = _functools.partial(nplike.nonzero, as_tuple=True)
    nplike.power = nplike.pow
    nplike.ndarray.__str__ = lambda self: f"{self.item()}"
    nplike.sync = nplike.cuda.synchronize

    if "TORCH_USE_CPU" in _os.environ and _os.environ["TORCH_USE_CPU"] == "1":
        nplike.set_default_tensor_type('torch.FloatTensor')
    else:
        nplike.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    import numpy as nplike
    nplike.sync = _dummy_function
    nplike.get = lambda arg: arg
