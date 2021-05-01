#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.
"""Functions related to updating ghost cells, i.e., boundary conditions.
"""
import numpy
from ..utils.dummydict import DummyDict
from ..utils.config import BCType, BCConfig

# to create corresponding slices for the left-hand-side of periodic BC
_periodic_slc_left = {
    "west": lambda n, c: (slice(c, c+1), slice(n, -n), slice(0, n)),
    "east": lambda n, c: (slice(c, c+1), slice(n, -n), slice(-n, None)),
    "north": lambda n, c: (slice(c, c+1), slice(-n, None), slice(n, -n)),
    "south": lambda n, c: (slice(c, c+1), slice(0, n), slice(n, -n))
}

# to create corresponding slices for the left-hand-side of periodic BC
_periodic_slc_right = {
    "west": lambda n, c: (slice(c, c+1), slice(n, -n), slice(-n-n, -n)),
    "east": lambda n, c: (slice(c, c+1), slice(n, -n), slice(n, n+n)),
    "north": lambda n, c: (slice(c, c+1), slice(n, n+n), slice(n, -n)),
    "south": lambda n, c: (slice(c, c+1), slice(-n-n, -n), slice(n, -n))
}

_extrap_seq = {
    "west": lambda n: numpy.arange(n, 0, -1),
    "east": lambda n: numpy.arange(1, n+1),
    "north": lambda n: numpy.arange(1, n+1).reshape((n, 1)),
    "south": lambda n: numpy.arange(n, 0, -1).reshape((n, 1))
}

_extrap_anchor = {
    "west": lambda n, c: (slice(c, c+1), slice(n, -n), slice(n, n+1)),
    "east": lambda n, c: (slice(c, c+1), slice(n, -n), slice(-n-1, -n)),
    "north": lambda n, c: (slice(c, c+1), slice(-n-1, -n), slice(n, -n)),
    "south": lambda n, c: (slice(c, c+1), slice(n, n+1), slice(n, -n))
}

_extrap_delta_slc = {
    "west": lambda n, c: (slice(c, c+1), slice(n, -n), slice(n+1, n+2)),
    "east": lambda n, c: (slice(c, c+1), slice(n, -n), slice(-n-2, -n-1)),
    "north": lambda n, c: (slice(c, c+1), slice(-n-2, -n-1), slice(n, -n)),
    "south": lambda n, c: (slice(c, c+1), slice(n+1, n+2), slice(n, -n))
}

_extrap_slc = {
    "west": lambda n, c: (slice(c, c+1), slice(n, -n), slice(0, n)),
    "east": lambda n, c: (slice(c, c+1), slice(n, -n), slice(-n, None)),
    "north": lambda n, c: (slice(c, c+1), slice(-n, None), slice(n, -n)),
    "south": lambda n, c: (slice(c, c+1), slice(0, n), slice(n, -n))
}

_inflow_topo_key = {
    "west": lambda c: "x",
    "east": lambda c: "x",
    "north": lambda c: "y",
    "south": lambda c: "y"
}

_inflow_topo_slc = {
    "west": (slice(None), slice(0, 1)),
    "east": (slice(None), slice(-1, None)),
    "north": (slice(-1, None), slice(None)),
    "south": (slice(0, 1), slice(None))
}


def periodic_factory(n_ghost, orientation, component):
    """A function factory to create a ghost-cell updating function.

    Arguments
    ---------
    n_ghost : int
        An integer for the number of ghost-cell layers outside each boundary.
    orientation : str
        A string of one of the following orientation: "west", "east", "north", or "south".
    component : int
        Which quantity will this function be applied to -- 0 for w, 1 for hu, and 2 for hv.

    Returns
    -------
    A function with a signature of f(numpy.ndarray) -> numpy.ndarray. Both the input and output
    arrays have a shape of (3, Ny+2*n_ghost, Nx+2*n_ghost).
    """

    slc_left = _periodic_slc_left[orientation](n_ghost, component)
    slc_right = _periodic_slc_right[orientation](n_ghost, component)

    def periodic(conserv_q):
        """Update the ghost cells with periodic BC.

        Arguments
        ---------
        conserv_q : a (3, Ny+2*n_ghost, Nx+2*n_ghost) numpy.ndarray

        Returns
        -------
        Updated conserv_q.
        """
        conserv_q[slc_left] = conserv_q[slc_right]
        return conserv_q

    periodic.slc_left = slc_left
    periodic.slc_right = slc_right

    return periodic


def outflow_factory(n_ghost, orientation, component):
    """A function factory to create a ghost-cell updating function.

    Arguments
    ---------
    n_ghost : int
        An integer for the number of ghost-cell layers outside each boundary.
    orientation : str
        A string of one of the following orientation: "west", "east", "north", or "south".
    component : int
        Which quantity will this function be applied to -- 0 for w, 1 for hu, and 2 for hv.

    Returns
    -------
    A function with a signature of f(numpy.ndarray) -> numpy.ndarray. Both the input and output
    arrays have a shape of (3, Ny+2*n_ghost, Nx+2*n_ghost).
    """

    anchor = _extrap_anchor[orientation](n_ghost, component)
    slc = _extrap_slc[orientation](n_ghost, component)

    def outflow_extrap(conserv_q):
        """Update the ghost cells with outflow BC using constant extrapolation.

        Arguments
        ---------
        conserv_q : a (3, Ny+2*n_ghost, Nx+2*n_ghost) numpy.ndarray

        Returns
        -------
        Updated conserv_q.
        """
        conserv_q[slc] = conserv_q[anchor]
        return conserv_q

    outflow_extrap.anchor = anchor
    outflow_extrap.slc = slc

    return outflow_extrap


def linear_extrap_factory(n_ghost, orientation, component):
    """A function factory to create a ghost-cell updating function.

    Arguments
    ---------
    n_ghost : int
        An integer for the number of ghost-cell layers outside each boundary.
    orientation : str
        A string of one of the following orientation: "west", "east", "north", or "south".
    component : int
        Which quantity will this function be applied to -- 0 for w, 1 for hu, and 2 for hv.

    Returns
    -------
    A function with a signature of f(numpy.ndarray) -> numpy.ndarray. Both the input and output
    arrays have a shape of (3, Ny+2*n_ghost, Nx+2*n_ghost).
    """

    seq = _extrap_seq[orientation](n_ghost)
    anchor = _extrap_anchor[orientation](n_ghost, component)
    delta_slc = _extrap_delta_slc[orientation](n_ghost, component)
    slc = _extrap_slc[orientation](n_ghost, component)

    def linear_extrap(conserv_q):
        """Update the ghost cells with outflow BC using linear extrapolation.

        Arguments
        ---------
        conserv_q : a (3, Ny+2*n_ghost, Nx+2*n_ghost) numpy.ndarray

        Returns
        -------
        Updated conserv_q.
        """
        delta = conserv_q[anchor] - conserv_q[delta_slc]
        conserv_q[slc] = conserv_q[anchor] + seq * delta
        return conserv_q

    linear_extrap.seq = seq
    linear_extrap.anchor = anchor
    linear_extrap.delta_slc = delta_slc
    linear_extrap.slc = slc

    return linear_extrap


def constant_bc_factory(const, n_ghost, orientation, component):
    """A function factory to create a ghost-cell updating function.

    Arguments
    ---------
    const : float
        The constant of either w, hu, or hv, depending the values in component.
    n_ghost : int
        An integer for the number of ghost-cell layers outside each boundary.
    orientation : str
        A string of one of the following orientation: "west", "east", "north", or "south".
    component : int
        Which quantity will this function be applied to -- 0 for w, 1 for hu, and 2 for hv.

    Returns
    -------
    A function with a signature of f(numpy.ndarray) -> numpy.ndarray. Both the input and output
    arrays have a shape of (3, Ny+2*n_ghost, Nx+2*n_ghost).
    """

    seq = _extrap_seq[orientation](n_ghost)
    anchor = _extrap_anchor[orientation](n_ghost, component)
    slc = _extrap_slc[orientation](n_ghost, component)

    def constant_bc(conserv_q):
        """Update the ghost cells with constant BC using linear extrapolation.

        Arguments
        ---------
        conserv_q : a (3, Ny+2*n_ghost, Nx+2*n_ghost) numpy.ndarray

        Returns
        -------
        Updated conserv_q.
        """
        delta = (const - conserv_q[anchor]) * 2.
        conserv_q[slc] = conserv_q[anchor] + seq * delta
        return conserv_q

    constant_bc.const = const
    constant_bc.seq = seq
    constant_bc.anchor = anchor
    constant_bc.slc = slc

    return constant_bc


def inflow_bc_factory(const, n_ghost, orientation, component, topo_cell_face):
    """A function factory to create a ghost-cell updating function.

    Arguments
    ---------
    const : float
        The constant of either h, u, or v, depending the values in component.
    n_ghost : int
        An integer for the number of ghost-cell layers outside each boundary.
    orientation : str
        A string of one of the following orientation: "west", "east", "north", or "south".
    component : int
        Which quantity will this function be applied to -- 0 for w, 1 for hu, and 2 for hv.
    topo_cell_face : a dict
        A dictionary containing the following key-value pairs
        - "x": a (3, Ny, Nx+1) numpy.ndarray holding the topography elevations at cell interfaces
          normal to x-direction.
        - "y": a (3, Ny+1, Nx) numpy.ndarray holding the topography elevations at cell interfaces
          normal to y-direction.

    Returns
    -------
    A function with a signature of f(numpy.ndarray) -> numpy.ndarray. Both the input and output
    arrays have a shape of (3, Ny+2*n_ghost, Nx+2*n_ghost).
    """

    seq = _extrap_seq[orientation](n_ghost)
    anchor = _extrap_anchor[orientation](n_ghost, component)
    slc = _extrap_slc[orientation](n_ghost, component)
    bckey = _inflow_topo_key[orientation](component)
    bcslc = _inflow_topo_slc[orientation]
    w_idx = _extrap_anchor[orientation](n_ghost, 0)

    def inflow_bc_depth(conserv_q):
        delta = (const + topo_cell_face[bckey][bcslc] - conserv_q[anchor]) * 2
        conserv_q[slc] = conserv_q[anchor] + seq * delta
        return conserv_q

    inflow_bc_depth.const = const
    inflow_bc_depth.seq = seq
    inflow_bc_depth.anchor = anchor
    inflow_bc_depth.slc = slc
    inflow_bc_depth.topo_cell_face = topo_cell_face
    inflow_bc_depth.bckey = bckey
    inflow_bc_depth.bcslc = bcslc

    def inflow_bc_velocity(conserv_q):
        depth = numpy.maximum(conserv_q[w_idx]-topo_cell_face[bckey][bcslc], 0.)
        delta = (const * depth - conserv_q[anchor]) * 2
        conserv_q[slc] = conserv_q[anchor] + seq * delta
        return conserv_q

    inflow_bc_velocity.const = const
    inflow_bc_velocity.seq = seq
    inflow_bc_velocity.anchor = anchor
    inflow_bc_velocity.slc = slc
    inflow_bc_velocity.topo_cell_face = topo_cell_face
    inflow_bc_velocity.bckey = bckey
    inflow_bc_velocity.bcslc = bcslc
    inflow_bc_velocity.w_idx = w_idx

    inflow_bc_depth.__doc__ = inflow_bc_velocity.__doc__ = \
    """Update the ghost cells for inflow boundary conditions.

    The inflow quantities are non-conservatives (i.e., h, u, and v) and are defined right on the
    boundary.

    Arguments
    ---------
    conserv_q: a (3, Ny+2*n_ghost, Nx+2*n_ghost) numpy.ndarray

    Returns
    -------
    Updated conserv_q.
    """  # noqa: E122

    if component == 0:
        return inflow_bc_depth

    return inflow_bc_velocity


def update_all_factory(bcs: BCConfig, n_ghost: int, topo_cell_face: DummyDict):
    """A factory to create a update-all ghost-cell updating function.

    Arguments
    ---------
    bcs : torchswe.utils.config.BCConfig
        The configuration from the "boundary" node in config.yaml.
    n_ghost : int
        Number of ghost cell layers at each boundary.
    topo_cell_face : a dict
        A dictionary containing the following key-value pairs
        - "x": a (3, Ny, Nx+1) numpy.ndarray holding the topography elevations at cell interfaces
          normal to x-direction.
        - "y": a (3, Ny+1, Nx) numpy.ndarray holding the topography elevations at cell interfaces
          normal to y-direction.

    Returns
    -------
    A function with a signature of f(numpy.ndarray) -> numpy.ndarray. Both the input and output
    arrays have a shape of (3, Ny+2*n_ghost, Nx+2*n_ghost).
    """

    # check periodicity
    check_periodicity(bcs)

    # initialize varaible
    functions = DummyDict({
        "west": [None, None, None], "east": [None, None, None],
        "north": [None, None, None], "south": [None, None, None],
    })

    for key in ["west", "east", "south", "north"]:

        # bctypes: {type: [...], values: [...]}
        for i, bctype in enumerate(bcs[key]["types"]):

            if bctype == BCType.PERIODIC:  # periodic BC
                functions[key][i] = periodic_factory(n_ghost, key, i)
            elif bctype == BCType.OUTFLOW:  # constant extrapolation BC (outflow)
                functions[key][i] = outflow_factory(n_ghost, key, i)
            elif bctype == BCType.EXTRAP:  # linear extrapolation BC
                functions[key][i] = linear_extrap_factory(n_ghost, key, i)
            elif bctype == BCType.CONST:  # constant/Dirichlet
                functions[key][i] = constant_bc_factory(bcs[key]["values"][i], n_ghost, key, i)
            elif bctype == BCType.INFLOW:  # inflow/constant non-conservative variables
                functions[key][i] = inflow_bc_factory(
                    bcs[key]["values"][i], n_ghost, key, i, topo_cell_face)
            else:  # others: error
                raise ValueError("{} is not recognized.".format(bctype))

    def update_all(conserv_q):
        """Update the ghost cells at all boundaries at once.

        Arguments
        ---------
        conserv_q : a (3, Ny+2*n_ghost, Nx+2*n_ghost) numpy.ndarray

        Returns
        -------
        Updated conserv_q.
        """

        for val in functions.values():
            for func in val:
                conserv_q = func(conserv_q)

        return conserv_q

    update_all.functions = functions

    return update_all


def check_periodicity(bcs: BCConfig):
    """Check whether periodic BCs match at corresponding boundary pairs.

    Arguments
    ---------
    bcs : torchswe.utils.config.BCConfig
        The configuration from the "boundary" node in config.yaml.

    Raises
    ------
    ValueError
        When periodic BCs do not match.
    """

    result = True

    for types in zip(bcs["west"]["types"], bcs["east"]["types"]):
        if any(t == BCType.PERIODIC for t in types):
            result = all(t == BCType.PERIODIC for t in types)

    for types in zip(bcs["north"]["types"], bcs["south"]["types"]):
        if any(t == BCType.PERIODIC for t in types):
            result = all(t == BCType.PERIODIC for t in types)

    if not result:
        raise ValueError("Periodic BCs do not match at boundaries and components.")
