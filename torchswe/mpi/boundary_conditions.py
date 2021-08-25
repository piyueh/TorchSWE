#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""MPI version of bc-related routines.
"""
from torchswe.core.boundary_conditions import outflow_factory
from torchswe.core.boundary_conditions import linear_extrap_factory
from torchswe.core.boundary_conditions import constant_bc_factory
from torchswe.core.boundary_conditions import inflow_bc_factory
from torchswe.mpi.data import cal_rank_from_proc_loc as _cal_rank_from_proc_loc


def get_ghost_cell_updaters(bcs, states, topo=None):  # pylint: disable=invalid-name
    """Get a function that updates all ghost cells.

    This is a function factory. The return of this funciton is a function with signature:
        torchswe.utils.data.States = func(torchswe.utils.data.States)
    The update happens in-place, so the return of this function is not important. We return it
    just to comform the coding style.

    Arguments
    ---------
    bcs : torchswe.utils.config.BCConfig
        The configuration instance of boundary conditions.
    states : torchswe.mpi.data.States
        The States instance that will be updated in the simulation.
    topo : torchswe.tuils.data.Topography
        Topography instance. Some boundary conditions require topography elevations.

    Returns
    -------
    A callable with signature `torchswe.utils.data.States = func(torchswe.utils.data.States)`.
    """

    bcs.check()
    nngh = {"west": states.ny, "east": states.ny, "south": states.nx, "north": states.nx}
    funcs = {}

    for ornt in ["west", "east", "south", "north"]:

        if states[ornt] is not None:  # internal boundary between processes
            continue

        # special case: periodic BC
        if bcs[ornt].types[0] == "periodic": # implying all components are periodic
            states = _find_periodic_neighbor(states, ornt)
            continue  # no need to continue on the following code in this iteration

        # initialize the dictionary for this orientation
        funcs[ornt] = {}
        for i, key in enumerate(["w", "hu", "hv"]):

            # constant extrapolation BC (outflow)
            if bcs[ornt].types[i] == "outflow":
                funcs[ornt][key] = outflow_factory(states.ngh, ornt)

            # linear extrapolation BC
            elif bcs[ornt].types[i] == "extrap":
                funcs[ornt][key] = linear_extrap_factory(nngh[ornt], states.ngh, ornt, states.dtype)

            # constant, i.e., Dirichlet
            elif bcs[ornt].types[i] == "const":
                funcs[ornt][key] = constant_bc_factory(
                    bcs[ornt].values[i], nngh[ornt], states.ngh, ornt, states.dtype)

            # inflow, i.e., constant non-conservative variables
            elif bcs[ornt].types[i] == "inflow":
                topo.check()
                funcs[ornt][key] = inflow_bc_factory(
                    bcs[ornt].values[i], nngh[ornt], states.ngh, ornt, topo, i, states.dtype)

            # this shouldn't happen because pydantic should have catched the error
            else:
                raise ValueError("{} is not recognized.".format(bcs[ornt].types[i]))

    # check the data model in case neighbors changed due to periodic BC
    states.check()

    def updater(soln):
        for ornt in funcs:  # if funcs is an empty dictionary, this will skip it
            for key in ["w", "hu", "hv"]:
                soln.q[key] = funcs[ornt][key](soln.q[key])

        # exchange data on internal boundaries between MPI processes
        soln.exchange_data()
        return soln

    # store the functions as an attribute for debug
    updater.funcs = funcs

    return updater


def _find_periodic_neighbor(states, orientation):
    """Find the neighbor MPI process rank corresponding to periodic boundary."""
    # pylint: disable=invalid-name

    # aliases
    pny, pnx = states.proc_shape
    pj, pi = states.proc_loc

    # self.proc_loc = (pj, 0), target: (pj, pnx-1)
    if orientation == "west":
        assert pi == 0
        states.west = _cal_rank_from_proc_loc(pnx, pnx-1, pj)

    # self.proc_loc = (pj, pnx-1), target: (pj, 0)
    elif orientation == "east":
        assert pi == pnx - 1
        states.east = _cal_rank_from_proc_loc(pnx, 0, pj)

    # self.proc_loc = (0, pi), target: (pny-1, pi)
    elif orientation == "south":
        assert pj == 0
        states.south = _cal_rank_from_proc_loc(pnx, pi, pny-1)

    # self.proc_loc = (pny-1, pi), target: (0, pi)
    elif orientation == "north":
        assert pj == pny - 1
        states.north = _cal_rank_from_proc_loc(pnx, pi, 0)

    else:
        raise ValueError("\"orientation\" shold be one of: west, east, south, north.")

    # states should have been modified in-place; retrun it for coding style
    return states
