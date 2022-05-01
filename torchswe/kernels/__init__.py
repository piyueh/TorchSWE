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

if "USE_CUPY" in _os.environ and _os.environ["USE_CUPY"] == "1":
    from .cupy import get_discontinuous_flux
    from .cupy import central_scheme
    from .cupy import get_local_speed
    from .cupy import reconstruct
    from .cupy import reconstruct_cell_centers
elif (
    ("LEGATE_MAX_DIM" in _os.environ and "LEGATE_MAX_FIELDS" in _os.environ) or
    ("USE_TORCH" in _os.environ and _os.environ["USE_TORCH"] == "1")
):
    from .cunumeric_flux import get_discontinuous_flux
    from .cunumeric_flux import central_scheme
    from .cunumeric_flux import get_local_speed
    from .cunumeric_reconstruction import reconstruct
    from .cunumeric_reconstruction import reconstruct_cell_centers
else:
    from .cython import get_discontinuous_flux
    from .cython import central_scheme
    from .cython import get_local_speed
    from .cython import reconstruct
    from .cython import reconstruct_cell_centers
