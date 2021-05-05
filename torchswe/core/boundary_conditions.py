#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.
"""Functions related to updating ghost cells, i.e., boundary conditions.
"""
from typing import Literal, Callable

from pydantic import conint
from torchswe import nplike
from torchswe.utils.config import BaseConfig, SingleBCConfig, BCConfig
from torchswe.utils.data import Topography, States

# to create corresponding slices for the left-hand-side of periodic BC
_periodic_slc_left = {
    "west": lambda ngh: (slice(ngh, -ngh), slice(0, ngh)),
    "east": lambda ngh: (slice(ngh, -ngh), slice(-ngh, None)),
    "north": lambda ngh: (slice(-ngh, None), slice(ngh, -ngh)),
    "south": lambda ngh: (slice(0, ngh), slice(ngh, -ngh))
}

# to create corresponding slices for the left-hand-side of periodic BC
_periodic_slc_right = {
    "west": lambda ngh: (slice(ngh, -ngh), slice(-ngh-ngh, -ngh)),
    "east": lambda ngh: (slice(ngh, -ngh), slice(ngh, ngh+ngh)),
    "north": lambda ngh: (slice(ngh, ngh+ngh), slice(ngh, -ngh)),
    "south": lambda ngh: (slice(-ngh-ngh, -ngh), slice(ngh, -ngh))
}

_extrap_seq = {
    "west": lambda ngh: numpy.arange(ngh, 0, -1),
    "east": lambda ngh: numpy.arange(1, ngh+1),
    "north": lambda ngh: numpy.arange(1, ngh+1).reshape((ngh, 1)),
    "south": lambda ngh: numpy.arange(ngh, 0, -1).reshape((ngh, 1))
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


def periodic_factory(n_ghost, orientation):
    """A function factory to create a ghost-cell updating function.

    Arguments
    ---------
    n_ghost : int
        An integer for the number of ghost-cell layers outside each boundary.
    orientation : str
        A string of one of the following orientation: "west", "east", "north", or "south".

    Returns
    -------
    A function with a signature of f(nplike.ndarray) -> nplike.ndarray. Both the input and output
    arrays have a shape of (3, Ny+2*n_ghost, Nx+2*n_ghost).
    """

    slc_left = _periodic_slc_left[orientation](n_ghost)
    slc_right = _periodic_slc_right[orientation](n_ghost)

    def periodic(conserv_q):
        """Update the ghost cells with periodic BC.

        Arguments
        ---------
        conserv_q : a (3, Ny+2*n_ghost, Nx+2*n_ghost) nplike.ndarray

        Returns
        -------
        Updated conserv_q.
        """
        conserv_q[slc_left] = conserv_q[slc_right]
        return conserv_q

    periodic.slc_left = slc_left
    periodic.slc_right = slc_right

    return periodic


def outflow_factory(n_ghost, orientation):
    """A function factory to create a ghost-cell updating function.

    Arguments
    ---------
    n_ghost : int
        An integer for the number of ghost-cell layers outside each boundary.
    orientation : str
        A string of one of the following orientation: "west", "east", "north", or "south".

    Returns
    -------
    A function with a signature of f(nplike.ndarray) -> nplike.ndarray. Both the input and output
    arrays have a shape of (3, Ny+2*n_ghost, Nx+2*n_ghost).
    """

    anchor = _extrap_anchor[orientation](n_ghost)
    slc = _extrap_slc[orientation](n_ghost)

    def outflow_extrap(conserv_q):
        """Update the ghost cells with outflow BC using constant extrapolation.

        Arguments
        ---------
        conserv_q : a (3, Ny+2*n_ghost, Nx+2*n_ghost) nplike.ndarray

        Returns
        -------
        Updated conserv_q.
        """
        conserv_q[slc] = conserv_q[anchor]
        return conserv_q

    outflow_extrap.anchor = anchor
    outflow_extrap.slc = slc

    return outflow_extrap


def linear_extrap_factory(n_ghost, orientation):
    """A function factory to create a ghost-cell updating function.

    Arguments
    ---------
    n_ghost : int
        An integer for the number of ghost-cell layers outside each boundary.
    orientation : str
        A string of one of the following orientation: "west", "east", "north", or "south".

    Returns
    -------
    A function with a signature of f(nplike.ndarray) -> nplike.ndarray. Both the input and output
    arrays have a shape of (3, Ny+2*n_ghost, Nx+2*n_ghost).
    """

    seq = _extrap_seq[orientation](n_ghost)
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


def constant_bc_factory(const, n_ghost, orientation):
    """A function factory to create a ghost-cell updating function.

    Arguments
    ---------
    const : float
        The constant of either w, hu, or hv, depending the values in component.
    n_ghost : int
        An integer for the number of ghost-cell layers outside each boundary.
    orientation : str
        A string of one of the following orientation: "west", "east", "north", or "south".

    Returns
    -------
    A function with a signature of f(nplike.ndarray) -> nplike.ndarray. Both the input and output
    arrays have a shape of (3, Ny+2*n_ghost, Nx+2*n_ghost).
    """

    seq = _extrap_seq[orientation](n_ghost)
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


def inflow_bc_factory(const, n_ghost, orientation, topo, component):
    """A function factory to create a ghost-cell updating function.

    Arguments
    ---------
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

    Returns
    -------
    A function with a signature of f(nplike.ndarray) -> nplike.ndarray. Both the input and output
    arrays have a shape of (3, Ny+2*n_ghost, Nx+2*n_ghost).
    """

    seq = _extrap_seq[orientation](n_ghost)
    anchor = _extrap_anchor[orientation](n_ghost)
    slc = _extrap_slc[orientation](n_ghost)
    topo_cache = topo[_inflow_topo_key[orientation]]
    bcslc = _inflow_topo_slc[orientation]
    w_idx = _extrap_anchor[orientation](n_ghost, 0)

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
        depth = nplike.maximum(conserv_q[w_idx]-topo_cache[bcslc], 0.)
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


class BoundaryGhostUpdaterOneBound(BaseConfig):
    """Ghost cell updaters on a single boundary."""
    ngh: conint(ge=2)
    orientation: Literal["west", "east", "south", "north"]
    w: Callable[[nplike.ndarray], nplike.ndarray]
    hu: Callable[[nplike.ndarray], nplike.ndarray]
    hv: Callable[[nplike.ndarray], nplike.ndarray]

    def __init__(self, bc: SingleBCConfig, ngh: int, orientation: str, topo: Topography):
        # validate data models for a sanity check
        bc.check()

        keymap = {0: "w", 1: "hu", 2: "hv"}
        kwargs = {}

        for i, bctype in enumerate(bc.types):
            if bctype == "periodic":  # periodic BC
                kwargs[keymap[i]] = periodic_factory(ngh, orientation)
            elif bctype == "outflow":  # constant extrapolation BC (outflow)
                kwargs[keymap[i]] = outflow_factory(ngh, orientation)
            elif bctype == "extrap":  # linear extrapolation BC
                kwargs[keymap[i]] = linear_extrap_factory(ngh, orientation)
            elif bctype == "const":  # constant/Dirichlet
                kwargs[keymap[i]] = constant_bc_factory(bc.values[i], ngh, orientation)
            elif bctype == "inflow":  # inflow/constant non-conservative variables
                topo.check()
                kwargs[keymap[i]] = inflow_bc_factory(bc.values[i], ngh, orientation, topo, i)
            else:  # this shouldn't happen because pydantic should have catched the error
                raise ValueError("{} is not recognized.".format(bctype))

        # initialize this data model instance and let pydantic validate the model
        super().__init__(ngh=ngh, orientation=orientation, **kwargs)

    def update_all(self, values: States):
        """Update all ghost cell of all components on this boundary.

        Arguments
        ---------
        values : torchswe.utils.data.States
            An instance of the States data model.

        Returns
        -------
        values : torchswe.utils.data.States
            The same input object. Values are updated in-place. Returning it just for coding style.
        """
        # values should be updated in-place; returning array again just for consistency in style
        values.q.w = self.w(values.q.w)
        values.q.hu = self.hu(values.q.hu)
        values.q.hv = self.hv(values.q.hv)
        return values


class BoundaryGhostUpdater(BaseConfig):
    """Ghost cell updaters for all boundaries and all components."""
    ngh: conint(ge=2)
    west: BoundaryGhostUpdaterOneBound
    east: BoundaryGhostUpdaterOneBound
    south: BoundaryGhostUpdaterOneBound
    north: BoundaryGhostUpdaterOneBound

    def __init__(self, bcs: BCConfig, ngh: int, topo: Topography):
        super().__init__(
            ngh=ngh,
            west=BoundaryGhostUpdaterOneBound(bcs.west, ngh, "west", topo),
            east=BoundaryGhostUpdaterOneBound(bcs.east, ngh, "east", topo),
            south=BoundaryGhostUpdaterOneBound(bcs.south, ngh, "south", topo),
            north=BoundaryGhostUpdaterOneBound(bcs.north, ngh, "north", topo)
        )

    def update_all(self, values: States):
        """Update all ghost cell of all components on all boundaries.

        Arguments
        ---------
        values : torchswe.utils.data.States
            An instance of the States data model.

        Returns
        -------
        values : torchswe.utils.data.States
            The same input object. Values are updated in-place. Returning it just for coding style.
        """
        # values should be updated in-place; returning array again just for consistency in style
        values = self.west.update_all(values)
        values = self.east.update_all(values)
        values = self.south.update_all(values)
        values = self.north.update_all(values)
        return values
