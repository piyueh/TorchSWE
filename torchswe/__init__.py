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

# assume these two variables mean the code's running with Legate system
if "LEGATE_MAX_DIM" in os.environ and "LEGATE_MAX_FIELDS" in os.environ:
    from legate import numpy as nplike
else:
    import numpy as nplike

__version__ = "0.1.dev1"
logger = logging.getLogger("torchswe")
