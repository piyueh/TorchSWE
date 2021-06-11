#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.
"""Functions related to updating ghost cells, i.e., boundary conditions.
"""
from torchswe import nplike as _nplike
from torchswe.utils.misc import DummyDtype as _DummyDtype
# pylint: disable=fixme

_extrap_seq = {
    "west": lambda n, ngh, dtype: _nplike.tile(
        _nplike.arange(ngh, 0, -1, dtype=dtype), (n, 1)),
    "east": lambda n, ngh, dtype: _nplike.tile(
        _nplike.arange(1, ngh+1, dtype=dtype), (n, 1)),
    "north": lambda n, ngh, dtype: _nplike.tile(
        _nplike.arange(1, ngh+1, dtype=dtype).reshape((ngh, 1)), (1, n)),
    "south": lambda n, ngh, dtype: _nplike.tile(
        _nplike.arange(ngh, 0, -1, dtype=dtype).reshape((ngh, 1)), (1, n))
}

_extrap_anchor = {
    "west": lambda ngh: (slice(ngh, -ngh), slice(ngh, ngh+1)),
    "east": lambda ngh: (slice(ngh, -ngh), slice(-ngh-1, -ngh)),
    "north": lambda ngh: (slice(-ngh-1, -ngh), slice(ngh, -ngh)),
    "south": lambda ngh: (slice(ngh, ngh+1), slice(ngh, -ngh))
}

_extrap_delta_slc = {
    "west": lambda ngh: (slice(ngh, -ngh), slice(ngh+1, ngh+2)),
    "east": lambda ngh: (slice(ngh, -ngh), slice(-ngh-2, -ngh-1)),
    "north": lambda ngh: (slice(-ngh-2, -ngh-1), slice(ngh, -ngh)),
    "south": lambda ngh: (slice(ngh+1, ngh+2), slice(ngh, -ngh))
}

_extrap_slc = {
    "west": lambda ngh: (slice(ngh, -ngh), slice(0, ngh)),
    "east": lambda ngh: (slice(ngh, -ngh), slice(-ngh, None)),
    "north": lambda ngh: (slice(-ngh, None), slice(ngh, -ngh)),
    "south": lambda ngh: (slice(0, ngh), slice(ngh, -ngh))
}

_inflow_topo_key = {
    "west": "xface",
    "east": "xface",
    "north": "yface",
    "south": "yface"
}

_inflow_topo_slc = {
    "west": (slice(None), slice(0, 1)),
    "east": (slice(None), slice(-1, None)),
    "north": (slice(-1, None), slice(None)),
    "south": (slice(0, 1), slice(None))
}


def periodic_factory(ngh: int, orientation: str):
    """A function factory to create a ghost-cell updating function.

    Arguments
    ---------
    ngh : int
        An integer for the number of ghost-cell layers outside each boundary.
    orientation : str
        A string of one of the following orientation: "west", "east", "north", or "south".

    Returns
    -------
    A function with a signature of f(nplike.ndarray) -> nplike.ndarray. Both the input and output
    arrays have a shape of (3, Ny+2*n_ghost, Nx+2*n_ghost).
    """

    def periodic_west(conserv_q):
        conserv_q[ngh:-ngh, :ngh] = conserv_q[ngh:-ngh, -ngh-ngh:-ngh]
        return conserv_q

    def periodic_east(conserv_q):
        conserv_q[ngh:-ngh, -ngh:] = conserv_q[ngh:-ngh, ngh:ngh+ngh]
        return conserv_q

    def periodic_south(conserv_q):
        conserv_q[:ngh, ngh:-ngh] = conserv_q[-ngh-ngh:-ngh, ngh:-ngh]
        return conserv_q

    def periodic_north(conserv_q):
        conserv_q[-ngh:, ngh:-ngh] = conserv_q[ngh:ngh+ngh, ngh:-ngh]
        return conserv_q

    candidates = {
        "west": periodic_west, "east": periodic_east,
        "south": periodic_south, "north": periodic_north, }

    return candidates[orientation]


def outflow_factory(ngh: int, orientation: str):
    """A function factory to create a ghost-cell updating function.

    Arguments
    ---------
    ngh : int
        An integer for the number of ghost-cell layers outside each boundary.
    orientation : str
        A string of one of the following orientation: "west", "east", "north", or "south".

    Returns
    -------
    A function with a signature of f(nplike.ndarray) -> nplike.ndarray. Both the input and output
    arrays have a shape of (3, Ny+2*n_ghost, Nx+2*n_ghost).
    """
    # TODO: Legate hasn't supported implict broadcasting; chage the code once they support it

    def outflow_west(conserv_q):
        for i in range(ngh):
            conserv_q[ngh:-ngh, i] = conserv_q[ngh:-ngh, ngh]
        return conserv_q

    def outflow_east(conserv_q):
        for i in range(1, ngh+1):
            conserv_q[ngh:-ngh, -i] = conserv_q[ngh:-ngh, -ngh-1]
        return conserv_q

    def outflow_south(conserv_q):
        for i in range(ngh):
            conserv_q[i, ngh:-ngh] = conserv_q[ngh, ngh:-ngh]
        return conserv_q

    def outflow_north(conserv_q):
        for i in range(1, ngh+1):
            conserv_q[-i, ngh:-ngh] = conserv_q[-ngh-1, ngh:-ngh]
        return conserv_q

    candidates = {
        "west": outflow_west, "east": outflow_east,
        "south": outflow_south, "north": outflow_north, }

    return candidates[orientation]


def linear_extrap_factory(n, n_ghost, orientation, dtype):
    """A function factory to create a ghost-cell updating function.

    Arguments
    ---------
    n : int
        Number of non-ghost cells along the target boundary.
    n_ghost : int
        An integer for the number of ghost-cell layers outside each boundary.
    orientation : str
        A string of one of the following orientation: "west", "east", "north", or "south".
    dtype : str
        Either "float64" or "float32".

    Returns
    -------
    A function with a signature of f(nplike.ndarray) -> nplike.ndarray. Both the input and output
    arrays have a shape of (3, Ny+2*n_ghost, Nx+2*n_ghost).
    """

    seq = _extrap_seq[orientation](n, n_ghost, dtype)
    anchor = _extrap_anchor[orientation](n_ghost)
    delta_slc = _extrap_delta_slc[orientation](n_ghost)
    slc = _extrap_slc[orientation](n_ghost)

    def linear_extrap(conserv_q):
        """Update the ghost cells with outflow BC using linear extrapolation.

        Arguments
        ---------
        conserv_q : a (3, Ny+2*n_ghost, Nx+2*n_ghost) nplike.ndarray

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


def constant_bc_factory(const, n, n_ghost, orientation, dtype):
    """A function factory to create a ghost-cell updating function.

    Arguments
    ---------
    n : int
        Number of non-ghost cells along the target boundary.
    const : float
        The constant of either w, hu, or hv, depending the values in component.
    n_ghost : int
        An integer for the number of ghost-cell layers outside each boundary.
    orientation : str
        A string of one of the following orientation: "west", "east", "north", or "south".
    dtype : str
        Either "float64" or "float32".

    Returns
    -------
    A function with a signature of f(nplike.ndarray) -> nplike.ndarray. Both the input and output
    arrays have a shape of (3, Ny+2*n_ghost, Nx+2*n_ghost).
    """

    seq = _extrap_seq[orientation](n, n_ghost, dtype)
    anchor = _extrap_anchor[orientation](n_ghost)
    slc = _extrap_slc[orientation](n_ghost)

    def constant_bc(conserv_q):
        """Update the ghost cells with constant BC using linear extrapolation.

        Arguments
        ---------
        conserv_q : a (3, Ny+2*n_ghost, Nx+2*n_ghost) nplike.ndarray

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


def inflow_bc_factory(const, n, n_ghost, orientation, topo, component, dtype):
    """A function factory to create a ghost-cell updating function.

    Arguments
    ---------
    n : int
        Number of non-ghost cells along the target boundary.
    const : float
        The constant of either h, u, or v, depending the values in component.
    n_ghost : int
        An integer for the number of ghost-cell layers outside each boundary.
    orientation : str
        A string of one of the following orientation: "west", "east", "north", or "south".
    topo : torchswe.utils.data.Topography
        An instance of the topography data model.
    component : int
        Which quantity will this function be applied to -- 0 for w, 1 for hu, and 2 for hv.
    dtype : str
        Either "float64" or "float32".

    Returns
    -------
    A function with a signature of f(nplike.ndarray) -> nplike.ndarray. Both the input and output
    arrays have a shape of (3, Ny+2*n_ghost, Nx+2*n_ghost).
    """

    seq = _extrap_seq[orientation](n, n_ghost, dtype)
    anchor = _extrap_anchor[orientation](n_ghost)
    slc = _extrap_slc[orientation](n_ghost)
    topo_cache = topo[_inflow_topo_key[orientation]]
    bcslc = _inflow_topo_slc[orientation]
    w_idx = _extrap_anchor[orientation](n_ghost)

    def inflow_bc_depth(conserv_q):
        delta = (const + topo_cache[bcslc] - conserv_q[anchor]) * 2
        conserv_q[slc] = conserv_q[anchor] + seq * delta
        return conserv_q

    inflow_bc_depth.const = const
    inflow_bc_depth.seq = seq
    inflow_bc_depth.anchor = anchor
    inflow_bc_depth.slc = slc
    inflow_bc_depth.topo_cache = topo_cache
    inflow_bc_depth.bcslc = bcslc

    def inflow_bc_velocity(conserv_q):
        depth = _nplike.maximum(conserv_q[w_idx]-topo_cache[bcslc], 0.)
        delta = (const * depth - conserv_q[anchor]) * 2
        conserv_q[slc] = conserv_q[anchor] + seq * delta
        return conserv_q

    inflow_bc_velocity.const = const
    inflow_bc_velocity.seq = seq
    inflow_bc_velocity.anchor = anchor
    inflow_bc_velocity.slc = slc
    inflow_bc_velocity.topo_cache = topo_cache
    inflow_bc_velocity.bcslc = bcslc
    inflow_bc_velocity.w_idx = w_idx

    inflow_bc_depth.__doc__ = inflow_bc_velocity.__doc__ = \
    """Update the ghost cells for inflow boundary conditions.

    The inflow quantities are non-conservatives (i.e., h, u, and v) and are defined right on the
    boundary.

    Arguments
    ---------
    conserv_q: a (3, Ny+2*n_ghost, Nx+2*n_ghost) nplike.ndarray

    Returns
    -------
    Updated conserv_q.
    """  # noqa: E122

    if component == 0:
        return inflow_bc_depth

    return inflow_bc_velocity


def get_ghost_cell_updaters(bcs, nx, ny, ngh, dtype, topo=None):  # pylint: disable=invalid-name
    """Get a function that updates all ghost cells.

    This is a function factory. The return of this funciton is a function with signature:
        torchswe.utils.data.States = func(torchswe.utils.data.States)
    The update happens in-place, so the return of this function is not important. We return it
    just to comform the coding style.

    Arguments
    ---------
    bcs : torchswe.utils.config.BCConfig
        The configuration instance of boundary conditions.
    nx, ny : int
        Numbers of non-ghost cells along x and y directions.
    ngh : int
        Number of ghost cell layers outside each boundary.
    dtype : str, nplike.float32, or nplike.float64
        Floating number precision.
    topo : torchswe.tuils.data.Topography
        Topography instance. Some boundary conditions require topography elevations.

    Returns
    -------
    A callable with signature `torchswe.utils.data.States = func(torchswe.utils.data.States)`.
    """

    bcs.check()
    dtype = _DummyDtype.validator(dtype)

    nngh = {"west": ny, "east": ny, "south": nx, "north": nx}
    funcs = {"w": {}, "hu": {}, "hv": {}}

    for i, key in enumerate(["w", "hu", "hv"]):
        for ornt in ["west", "east", "south", "north"]:
            # periodic BC
            if bcs[ornt].types[i] == "periodic":
                funcs[key][ornt] = periodic_factory(ngh, ornt)

            # constant extrapolation BC (outflow)
            elif bcs[ornt].types[i] == "outflow":
                funcs[key][ornt] = outflow_factory(ngh, ornt)

            # linear extrapolation BC
            elif bcs[ornt].types[i] == "extrap":
                funcs[key][ornt] = linear_extrap_factory(nngh[ornt], ngh, ornt, dtype)

            # constant, i.e., Dirichlet
            elif bcs[ornt].types[i] == "const":
                funcs[key][ornt] = constant_bc_factory(
                    bcs[ornt].values[i], nngh[ornt], ngh, ornt, dtype)

            # inflow, i.e., constant non-conservative variables
            elif bcs[ornt].types[i] == "inflow":
                topo.check()
                funcs[key][ornt] = inflow_bc_factory(
                    bcs[ornt].values[i], nngh[ornt], ngh, ornt, topo, i, dtype)

            # this shouldn't happen because pydantic should have catched the error
            else:
                raise ValueError("{} is not recognized.".format(bcs[ornt].types[i]))

    def updater(soln):
        for key in ["w", "hu", "hv"]:
            for ornt in ["west", "east", "south", "north"]:
                soln.q[key] = funcs[key][ornt](soln.q[key])
        return soln

    # store the functions as an attribute for debug
    updater.funcs = funcs

    return updater
