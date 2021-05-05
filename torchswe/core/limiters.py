#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Slope/flux limiters.
"""
from torchswe import nplike
from torchswe.utils.data import States, Gridlines


def minmod_slope(states: States, grid: Gridlines, theta: float, tol: float = 1e-12) -> States:
    """Calculate minmod slopes in x- and y-direction of quantity.

    Arguments
    ---------
    states : torchswe.utils.data.States
    grid : torchswe.utils.data.Gridlines
    theta: float
        Parameter to controll oscillation and dispassion. 1 <= theta <= 2.
    tol : float
        To control how small can be treat as zero.

    Returns
    -------
    states : torchswe.utils.data.States
        The same object as the input. Updated in-place. Returning it just for coding style.
    """

    states.slp.x.w = minmod_slope_x_one_comp(states.q.w, grid.x.delta, states.ngh, theta, tol)
    states.slp.x.hu = minmod_slope_x_one_comp(states.q.hu, grid.x.delta, states.ngh, theta, tol)
    states.slp.x.hv = minmod_slope_x_one_comp(states.q.hv, grid.x.delta, states.ngh, theta, tol)
    states.slp.y.w = minmod_slope_y_one_comp(states.q.w, grid.y.delta, states.ngh, theta, tol)
    states.slp.y.hu = minmod_slope_y_one_comp(states.q.hu, grid.y.delta, states.ngh, theta, tol)
    states.slp.y.hv = minmod_slope_y_one_comp(states.q.hv, grid.y.delta, states.ngh, theta, tol)

    return states


def minmod_slope_x_one_comp(q: nplike.ndarray, dx: float, ngh: int, theta: float, tol: float):
    """Minmod slope in x direction for only one conservative quantity.

    Arguments
    ---------
    q : nplike.ndarray
        (ny+2*ngh, nx+2*ngh) array of a conservative quantity.
    dx : float
        Cell size in x-direction. Assume an uniform grid.
    ngh : int
        Number of ghost cell layers outside each boundary.
    theta: float
        Parameter to controll oscillation and dispassion. 1 <= theta <= 2.
    tol : float
        To control how small can be treat as zero.

    Returns
    -------
    slpx : nplike.ndarray
        (ny, nx+2) array of the slopes in x-direction, including one ghost layer at west and east.
    """
    # pylint: disable=invalid-name

    cells = slice(ngh, -ngh)  # non-ghost cells, length ny or nx
    i = slice(ngh-1, q.shape[1]-ngh+1)  # i = one ghost outside each bound; length nx+2
    ip1 = slice(ngh, q.shape[1]-ngh+2)  # i + 1; length nx+2
    im1 = slice(ngh-2, q.shape[1]-ngh)  # i - 1; length nx+2

    denominator = q[cells, ip1] - q[cells, i]  # q_{j, i+1} - q_{j, i} for all j
    zeros = nplike.nonzero(nplike.logical_and(denominator > -tol, denominator < tol))

    with nplike.errstate(divide="ignore", invalid="ignore"):
        slpx = (q[cells, i] - q[cells, im1]) / denominator

    slpx[zeros] = 0.  # where q_{j, i+1} - q_{j, i} = 0

    slpx = nplike.maximum(
        nplike.minimum(
            nplike.minimum(theta*slpx, (1.+slpx)/2.),
            theta),
        0.
    )

    slpx *= denominator
    slpx /= dx

    return slpx


def minmod_slope_y_one_comp(q: nplike.ndarray, dy: float, ngh: int, theta: float, tol: float):
    """Minmod slope in x direction for only one conservative quantity.

    Arguments
    ---------
    q : nplike.ndarray
        (ny+2*ngh, nx+2*ngh) array of a conservative quantity.
    dy : float
        Cell size in y-direction. Assume an uniform grid.
    ngh : int
        Number of ghost cell layers outside each boundary.
    theta: float
        Parameter to controll oscillation and dispassion. 1 <= theta <= 2.
    tol : float
        To control how small can be treat as zero.

    Returns
    -------
    slpy : nplike.ndarray
        (ny+2, nx) array of the slopes in y-direction, including one ghost layer at west and east.
    """
    # pylint: disable=invalid-name

    cells = slice(ngh, -ngh)  # non-ghost cells, length ny or nx
    j = slice(ngh-1, q.shape[0]-ngh+1)  # j = one ghost outside each bound; length ny+2
    jp1 = slice(ngh, q.shape[0]-ngh+2)  # j + 1; length ny+2
    jm1 = slice(ngh-2, q.shape[0]-ngh)  # j - 1; length ny+2

    denominator = q[jp1, cells] - q[j, cells]  # q_{j+1, i} - q_{j, i} for all i
    zeros = nplike.nonzero(nplike.logical_and(denominator > -tol, denominator < tol))

    with nplike.errstate(divide="ignore", invalid="ignore"):
        slpy = (q[j, cells] - q[jm1, cells]) / denominator

    slpy[zeros] = 0.  # where q_{j+1, i} - q_{j, i} = 0

    slpy = nplike.maximum(
        nplike.minimum(
            nplike.minimum(theta*slpy, (1.+slpy)/2.),
            theta),
        0.
    )

    slpy *= denominator
    slpy /= dy

    return slpy
