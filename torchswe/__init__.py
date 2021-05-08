#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""A shallow-water equation solver implemented with PyTorch.
"""
import os
import logging
from .utils.dummy import DummyErrState, dummy_function


# assume these two variables mean the code's running with Legate system
if "LEGATE_MAX_DIM" in os.environ and "LEGATE_MAX_FIELDS" in os.environ:
    from legate import numpy as nplike
elif "USE_CUPY" in os.environ and os.environ["USE_CUPY"] == "1":
    import cupy as nplike
    import cupyx
    nplike.errstate = DummyErrState
    nplike.set_printoptions = dummy_function
else:
    import numpy as nplike


__version__ = "0.1.dev1"
logger = logging.getLogger("torchswe")
