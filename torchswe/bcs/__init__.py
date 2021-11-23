#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Pi-Yueh Chuang <pychuang@pm.me>
#
# Distributed under terms of the BSD 3-Clause license.

"""This subpackage contain boundary-condition-related functions.
"""
import os as _os
from operator import itemgetter as _itemgetter
from mpi4py import MPI as _MPI
from torchswe.utils.config import BCConfig as _BCConfig
from torchswe.utils.data import Topography as _Topography
from torchswe.utils.data import States as _States

if "LEGATE_MAX_DIM" in _os.environ and "LEGATE_MAX_FIELDS" in _os.environ:
    raise NotImplementedError("legate.numpy is deprecated.")

if "USE_TORCH" in _os.environ and _os.environ["USE_TORCH"] == "1":
    raise NotImplementedError("PyTorch is deprecated.")

if "USE_CUPY" in _os.environ and _os.environ["USE_CUPY"] == "1":
    from ._cupy_const_extrap import const_extrap_factory  # pylint: disable=no-name-in-module
    from ._cupy_linear_extrap import linear_extrap_factory  # pylint: disable=no-name-in-module
    from ._cupy_const_val import const_val_factory  # pylint: disable=no-name-in-module
    from ._cupy_inflow import inflow_factory  # pylint: disable=no-name-in-module
else:
    from ._cython_const_extrap import const_extrap_factory  # pylint: disable=no-name-in-module
    from ._cython_linear_extrap import linear_extrap_factory  # pylint: disable=no-name-in-module
    from ._cython_const_val import const_val_factory  # pylint: disable=no-name-in-module


def get_ghost_cell_updaters(
    states: _States, topo: _Topography, bcs: _BCConfig, theta: float, tol: float, drytol: float
):
    """A function factory returning a function that updates all ghost cells.

    Arguments
    ---------
    states : torchswe.mpi.data.States
        The States instance that will be updated in the simulation.
    topo : torchswe.tuils.data.Topography
        Topography instance. Some boundary conditions require topography elevations.
    bcs : torchswe.utils.config.BCConfig
        The configuration instance of boundary conditions.
    tol : float
        The tolerance for depths to be considered completely dry cells.
    drytol : float
        The tolerance for depths to be considered wet but static cells.

    Returns
    -------
    A callable with signature `torchswe.utils.data.States = func(torchswe.utils.data.States)`.

    Notes
    -----
    The resulting functions modify the values in solution in-place. The return of this function is
    the same object as the one in input arguments. We return it just to comform the coding style.
    """

    bcs.check()
    funcs = []  # can be either a list or ordered dict, cannot be an unordered dict
    orientations = ["west", "east", "south", "north"]

    for ornt, bc in zip(orientations, _itemgetter(*orientations)(bcs)):

        # not on the physical boundary: skip
        # ----------------------------------
        if states.domain[ornt[0]] != _MPI.PROC_NULL:
            continue

        # special case: periodic BC
        # -------------------------
        # In MPI cases, periodic boundaries will be handled by internal exchange stage
        # Also, we're using Cartcomm, so periodic ranks are already configured in the beginning
        if bc.types[0] == "periodic":
            continue  # no need to continue this iteration as other components should be periodic

        # all other types of BCs
        # ----------------------
        for i, (bctp, bcv) in enumerate(zip(bc.types, bc.values)):

            # constant extrapolation BC (outflow)
            if bctp == "outflow":
                funcs.append(const_extrap_factory(ornt, i, states, topo, tol, drytol))

            # linear extrapolation BC
            elif bctp == "extrap":
                funcs.append(linear_extrap_factory(ornt, i, states, topo, tol, drytol))

            # constant, i.e., Dirichlet
            elif bctp == "const":
                funcs.append(const_val_factory(ornt, i, states, topo, theta, tol, drytol, bcv))

            # inflow, i.e., constant non-conservative variables
            elif bctp == "inflow":
                funcs.append(inflow_factory(ornt, i, states, topo, theta, tol, drytol, bcv))

            # this shouldn't happen because pydantic should have catched the error
            else:
                raise ValueError(f"{bctp} is not recognized.")

    # check the data model in case neighbors changed due to periodic BC
    states.check()

    # this is the function that will be retuned by this function factory
    def updater(soln: _States):
        for func in funcs:  # if funcs is an empty list, this will skip it
            func()
        return soln

    # store the functions as an attribute for debug
    updater.funcs = funcs

    return updater
