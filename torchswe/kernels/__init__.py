#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Compiled or JIT kernels for TorchSWE.
"""
import os as _os

if "LEGATE_MAX_DIM" in _os.environ and "LEGATE_MAX_FIELDS" in _os.environ:
    raise NotImplementedError("legate.numpy is deprecated.")
elif "USE_CUPY" in _os.environ and _os.environ["USE_CUPY"] == "1":
    from .cupy import get_discontinuous_flux
    from .cupy import central_scheme
    from .cupy import get_local_speed
    from .cupy import reconstruct
    from .cupy import reconstruct_cell_centers
elif "USE_TORCH" in _os.environ and _os.environ["USE_TORCH"] == "1":
    raise NotImplementedError("PyTorch is deprecated.")
else:
    from .cython import get_discontinuous_flux
    from .cython import central_scheme
    from .cython import get_local_speed
    from .cython import reconstruct
    from .cython import reconstruct_cell_centers
