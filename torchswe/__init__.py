#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""A GPU shallow-water equation solver.
"""
import os
import logging
import functools
from .utils.misc import DummyErrState, dummy_function


# assume these two variables mean the code's running with Legate system
if "LEGATE_MAX_DIM" in os.environ and "LEGATE_MAX_FIELDS" in os.environ:
    from legate import numpy as nplike
elif "USE_CUPY" in os.environ and os.environ["USE_CUPY"] == "1":
    import cupy as nplike
    import cupyx
    nplike.errstate = DummyErrState
    nplike.set_printoptions = dummy_function
elif "USE_TORCH" in os.environ and os.environ["USE_TORCH"] == "1":
    import torch as _torch
    import torch as nplike
    nplike.errstate = DummyErrState
    nplike.ndarray = _torch.Tensor
    nplike.ndarray.astype = _torch.Tensor.to
    nplike.array = _torch.tensor
    nplike.nonzero = functools.partial(_torch.nonzero, as_tuple=True)
    nplike.power = _torch.pow
    nplike.ndarray.__str__ = lambda self: "{}".format(self.item())

    if "TORCH_USE_CPU" in os.environ and os.environ["TORCH_USE_CPU"] == "1":
        nplike.set_default_tensor_type('torch.FloatTensor')
    else:
        nplike.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    import numpy as nplike


__version__ = "0.1.dev2"
logger = logging.getLogger("torchswe")
