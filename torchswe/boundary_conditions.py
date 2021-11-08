#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.
"""Functions related to updating ghost cells, i.e., boundary conditions.
"""
from operator import itemgetter as _itemgetter
from torchswe import nplike as _nplike
from torchswe.utils.config import BCConfig as _BCConfig
from torchswe.utils.data import Topography as _Topography
from torchswe.utils.data import States as _States
from torchswe.utils.misc import cal_rank_from_proc_loc as _cal_rank_from_proc_loc

_extrap_seq = {
    "west": lambda ngh, dtype: _nplike.arange(ngh, 0, -1, dtype=dtype).reshape((1, ngh)),
    "east": lambda ngh, dtype: _nplike.arange(1, ngh+1, dtype=dtype).reshape((1, ngh)),
    "north": lambda ngh, dtype: _nplike.arange(1, ngh+1, dtype=dtype).reshape((ngh, 1)),
    "south": lambda ngh, dtype: _nplike.arange(ngh, 0, -1, dtype=dtype).reshape((ngh, 1))
}

_extrap_anchor = {
    "west": lambda comp, ngh: (comp, slice(ngh, -ngh), slice(ngh, ngh+1)),       # shape (n, 1)
    "east": lambda comp, ngh: (comp, slice(ngh, -ngh), slice(-ngh-1, -ngh)),     # shape (n, 1)
    "north": lambda comp, ngh: (comp, slice(-ngh-1, -ngh), slice(ngh, -ngh)),    # shape (1, n)
    "south": lambda comp, ngh: (comp, slice(ngh, ngh+1), slice(ngh, -ngh))       # shape (1, n)
}

_extrap_delta_slc = {
    "west": lambda comp, ngh: (comp, slice(ngh, -ngh), slice(ngh+1, ngh+2)),     # shape (n, 1)
    "east": lambda comp, ngh: (comp, slice(ngh, -ngh), slice(-ngh-2, -ngh-1)),   # shape (n, 1)
    "north": lambda comp, ngh: (comp, slice(-ngh-2, -ngh-1), slice(ngh, -ngh)),  # shape (1, n)
    "south": lambda comp, ngh: (comp, slice(ngh+1, ngh+2), slice(ngh, -ngh))     # shape (1, n)
}

_target_slc = {
    "west": lambda comp, ngh: (comp, slice(ngh, -ngh), slice(0, ngh)),           # shape (n, ngh)
    "east": lambda comp, ngh: (comp, slice(ngh, -ngh), slice(-ngh, None)),       # shape (n, ngh)
    "north": lambda comp, ngh: (comp, slice(-ngh, None), slice(ngh, -ngh)),      # shape (ngh, n)
    "south": lambda comp, ngh: (comp, slice(0, ngh), slice(ngh, -ngh))           # shape (ngh, n)
}

_periodic_slc = {
    "west": lambda comp, ngh: (comp, slice(ngh, -ngh), slice(-2*ngh, -ngh)),     # shape (n, ngh)
    "east": lambda comp, ngh: (comp, slice(ngh, -ngh), slice(ngh, 2*ngh)),       # shape (n, ngh)
    "north": lambda comp, ngh: (comp, slice(ngh, 2*ngh), slice(ngh, -ngh)),      # shape (ngh, n)
    "south": lambda comp, ngh: (comp, slice(-2*ngh, -ngh), slice(ngh, -ngh))     # shape (ngh, n)
}

_inflow_topo_key = {
    "west": "xfcenters",
    "east": "xfcenters",
    "north": "yfcenters",
    "south": "yfcenters"
}

_inflow_topo_slc = {
    "west": (slice(None), slice(0, 1)),
    "east": (slice(None), slice(-1, None)),
    "north": (slice(-1, None), slice(None)),
    "south": (slice(0, 1), slice(None))
}


def periodic_factory(comp: int, ngh: int, ornt: str):
    """A function factory to create a ghost-cell updating function.

    Arguments
    ---------
    comp : int
        The component in a conservative vector. 0 for w, 1 for hu, and 2 for hv.
    ngh : int
        An integer for the number of ghost-cell layers outside each boundary.
    ornt : str
        A string of one of the following orientation: "west", "east", "north", or "south".

    Returns
    -------
    A function with a signature of f(nplike.ndarray) -> nplike.ndarray. Both the input and output
    arrays have a shape of (3, ny+2*ngh, nx+2*ngh).
    """

    target = _target_slc[ornt](comp, ngh)
    source = _periodic_slc[ornt](comp, ngh)

    def periodic(conserv_q):
        conserv_q[target] = conserv_q[source]
        return conserv_q

    periodic.target = target
    periodic.source = source

    return periodic


def outflow_factory(comp: int, ngh: int, ornt: str):
    """A function factory to create a ghost-cell updating function.

    Arguments
    ---------
    comp : int
        The component in a conservative vector. 0 for w, 1 for hu, and 2 for hv.
    ngh : int
        An integer for the number of ghost-cell layers outside each boundary.
    ornt : str
        A string of one of the following orientation: "west", "east", "north", or "south".

    Returns
    -------
    A function with a signature of f(nplike.ndarray) -> nplike.ndarray. Both the input and output
    arrays have a shape of (3, ny+2*ngh, nx+2*ngh).
    """

    target = _target_slc[ornt](comp, ngh)
    source = _extrap_anchor[ornt](comp, ngh)

    def outflow(conserv_q):
        conserv_q[target] = conserv_q[source]
        return conserv_q

    outflow.target = target
    outflow.source = source

    return outflow


def linear_extrap_factory(comp: int, ngh: int, ornt: str, dtype: str):
    """A function factory to create a ghost-cell updating function.

    Arguments
    ---------
    comp : int
        The component in a conservative vector. 0 for w, 1 for hu, and 2 for hv.
    ngh : int
        An integer for the number of ghost-cell layers outside each boundary.
    ornt : str
        A string of one of the following orientation: "west", "east", "north", or "south".
    dtype : str
        Either "float64" or "float32".

    Returns
    -------
    A function with a signature of f(nplike.ndarray) -> nplike.ndarray. Both the input and output
    arrays have a shape of (3, ny+2*ngh, nx+2*ngh).
    """

    target = _target_slc[ornt](comp, ngh)
    source = _extrap_anchor[ornt](comp, ngh)
    diff = _extrap_delta_slc[ornt](comp, ngh)
    seq = _extrap_seq[ornt](ngh, dtype)

    def linear_extrap(conserv_q):
        """Update the ghost cells with outflow BC using linear extrapolation.

        Arguments
        ---------
        conserv_q : a (3, ny+2*ngh, nx+2*ngh) nplike.ndarray

        Returns
        -------
        Updated conserv_q.
        """
        delta = conserv_q[source] - conserv_q[diff]
        conserv_q[target] = conserv_q[source] + seq * delta
        return conserv_q

    linear_extrap.target = target
    linear_extrap.source = source
    linear_extrap.diff = diff
    linear_extrap.seq = seq

    return linear_extrap


def constant_bc_factory(comp: int, ngh: int, ornt: str, dtype: str, const: float):
    """A function factory to create a ghost-cell updating function.

    Arguments
    ---------
    comp : int
        The component in a conservative vector. 0 for w, 1 for hu, and 2 for hv.
    ngh : int
        An integer for the number of ghost-cell layers outside each boundary.
    ornt : str
        A string of one of the following orientation: "west", "east", "north", or "south".
    dtype : str
        Either "float64" or "float32".
    const : float
        The constant of either w, hu, or hv, depending the values in component.

    Returns
    -------
    A function with a signature of f(nplike.ndarray) -> nplike.ndarray. Both the input and output
    arrays have a shape of (3, ny+2*ngh, nx+2*ngh).
    """

    target = _target_slc[ornt](comp, ngh)
    source = _extrap_anchor[ornt](comp, ngh)
    seq = _extrap_seq[ornt](ngh, dtype)

    def constant_bc(conserv_q):
        """Update the ghost cells with constant BC using linear extrapolation.

        Arguments
        ---------
        conserv_q : a (3, ny+2*ngh, nx+2*ngh) nplike.ndarray

        Returns
        -------
        Updated conserv_q.
        """
        delta = (const - conserv_q[source]) * 2.
        conserv_q[target] = conserv_q[source] + seq * delta
        return conserv_q

    constant_bc.target = target
    constant_bc.source = source
    constant_bc.seq = seq
    constant_bc.const = const

    return constant_bc


def inflow_bc_factory(comp: int, ngh: int, ornt: str, dtype: str, const: float, topo: _Topography):
    """A function factory to create a ghost-cell updating function.

    Arguments
    ---------
    comp : int
        The component in a conservative vector. 0 for w, 1 for hu, and 2 for hv.
    ngh : int
        An integer for the number of ghost-cell layers outside each boundary.
    ornt : str
        A string of one of the following orientation: "west", "east", "north", or "south".
    dtype : str
        Either "float64" or "float32".
    const : float
        The constant of either h, u, or v, depending the values in component.
    topo : torchswe.utils.data.Topography
        An instance of the topography data model.

    Returns
    -------
    A function with a signature of f(nplike.ndarray) -> nplike.ndarray. Both the input and output
    arrays have a shape of (3, ny+2*ngh, nx+2*ngh).
    """

    target = _target_slc[ornt](comp, ngh)
    source = _extrap_anchor[ornt](comp, ngh)
    w_source = _extrap_anchor[ornt](0, ngh)
    seq = _extrap_seq[ornt](ngh, dtype)
    topo_cache = topo[_inflow_topo_key[ornt]][_inflow_topo_slc[ornt]]

    def inflow_bc_depth(conserv_q):
        delta = (const + topo_cache - conserv_q[source]) * 2
        conserv_q[target] = conserv_q[source] + seq * delta
        return conserv_q

    inflow_bc_depth.target = target
    inflow_bc_depth.source = source
    inflow_bc_depth.seq = seq
    inflow_bc_depth.const = const
    inflow_bc_depth.topo_cache = topo_cache

    def inflow_bc_velocity(conserv_q):
        depth = _nplike.maximum(conserv_q[w_source]-topo_cache, 0.)
        delta = (const * depth - conserv_q[source]) * 2
        conserv_q[target] = conserv_q[source] + seq * delta
        return conserv_q

    inflow_bc_velocity.target = target
    inflow_bc_velocity.source = source
    inflow_bc_velocity.seq = seq
    inflow_bc_velocity.const = const
    inflow_bc_velocity.topo_cache = topo_cache
    inflow_bc_velocity.w_source = w_source

    inflow_bc_depth.__doc__ = inflow_bc_velocity.__doc__ = \
    """Update the ghost cells for inflow boundary conditions.

    The inflow quantities are non-conservatives (i.e., h, u, and v) and are defined right on the
    boundary.

    Arguments
    ---------
    conserv_q: a (3, ny+2*ngh, nx+2*ngh) nplike.ndarray

    Returns
    -------
    Updated conserv_q.
    """  # noqa: E122

    if comp == 0:
        return inflow_bc_depth

    return inflow_bc_velocity


def get_ghost_cell_updaters(bcs: _BCConfig, states: _States, topo: _Topography = None):
    """A function factory returning a function that updates all ghost cells.

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

    Notes
    -----
    The resulting functions modify the values in solution in-place. The return of this function is
    the same object as the one in input arguments. We return it just to comform the coding style.
    """

    bcs.check()
    funcs = {}
    orientations = ["west", "east", "south", "north"]

    for ornt, bc in zip(orientations, _itemgetter(*orientations)(bcs)):

        # not on the physical boundary: skip
        # ----------------------------------
        if states.domain.process[ornt] is not None:
            continue

        # special case: periodic BC
        # -------------------------
        # In MPI cases, periodic boundaries will be handled by internal exchange stage
        if bc.types[0] == "periodic":
            states = _find_periodic_neighbor(states, ornt)
            continue  # no need to continue this iteration as other components should be periodic

        # all other types of BCs
        # ----------------------
        funcs[ornt] = {}  # initialize the dictionary for this orientation
        for i, (bctp, bcv) in enumerate(zip(bc.types, bc.values)):

            # constant extrapolation BC (outflow)
            if bctp == "outflow":
                funcs[ornt][i] = outflow_factory(i, states.ngh, ornt)

            # linear extrapolation BC
            elif bctp == "extrap":
                funcs[ornt][i] = linear_extrap_factory(i, states.ngh, ornt, states.Q.dtype)

            # constant, i.e., Dirichlet
            elif bctp == "const":
                funcs[ornt][i] = constant_bc_factory(i, states.ngh, ornt, states.Q.dtype, bcv)

            # inflow, i.e., constant non-conservative variables
            elif bctp == "inflow":
                topo.check()
                funcs[ornt][i] = inflow_bc_factory(i, states.ngh, ornt, states.Q.dtype, bcv, topo)

            # this shouldn't happen because pydantic should have catched the error
            else:
                raise ValueError(f"{bctp} is not recognized.")

    # check the data model in case neighbors changed due to periodic BC
    states.check()

    # this is the function that will be retuned by this function factory
    def updater(soln: _States):
        for func in funcs.values():  # if funcs is an empty dictionary, this will skip it
            for i in range(3):
                soln.Q = func[i](soln.Q)
        return soln

    # store the functions as an attribute for debug
    updater.funcs = funcs

    return updater


def _find_periodic_neighbor(states: _States, orientation: str):
    """Find the neighbor MPI process rank corresponding to periodic boundary."""
    # pylint: disable=invalid-name

    # aliases
    pny, pnx = states.domain.process.proc_shape
    pj, pi = states.domain.process.proc_loc

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
