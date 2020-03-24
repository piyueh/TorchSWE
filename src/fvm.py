#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the MIT license.

"""
Linear reconstruction.
"""
import torch
from .sources import source
from .limiters import minmod_limiter
from .reconstruction import discont_soln
from .misc import correct_negative_depth, get_huv, local_speed
from .flux import get_pde_flux
from .numerical_flux import central_scheme

def fvm(U, Bf, Bc, dBc, dx, Ngh, g, epsilon, theta):
    """Get the right-hand-side of a time-marching step with finite volume method.

    Args:
    -----
        U: a 3D torch.tensor of shape (3, Ny+2*Ngh, Nx+2Ngh) representing
            w, hu, and hv.
        Bf: a dictionary of the following key-value pairs
            x: a (Ny, Nx+1) torch.tensor representing the elevations at
                interfaces midpoint for those normal to x-direction. Must be
                calculated from the linear interpolation from corner elevations.
            y: a (Ny+1, Nx) torch.tensor representing the elevations at
                interfaces midpoint for those normal to y-direction. Must be
                calculated from the linear interpolation from corner elevations.
        Bc: a 2D torch.tensor of shape (Ny, Nx) representing elevations at cell
            centers, excluding ghost cells. Bc must be from the bilinear
            interpolation of elevation of cell coreners.
        dBc: a dictionary of the following key-value pair
            x: a 2d torch.tensor of shape (Ny, Nx) representing the topographic
                x-gradient at cell centers, i.e., $\partial B / \partial x$.
            y: a 2d torch.tensor of shape (Ny, Nx) representing the topographic
                y-gradient at cell centers, i.e., $\partial B / \partial y$.
        dx: a scalar of cell szie, assuming dx = dy and is uniform everywhere.
        Ngh: an integer, the number of ghost cells outside each boundary
        g: gravity
        epsilon: a very small number to avoid division by zero.

    Returns:
    --------
        f: a (3, Ny, Nx) torch.tensor to update the conservative variables in
            interior cell centers.
        max_dt: a scalar indicating the maximum safe time-step size.
    """

    # calculate source term
    S = source(U, dBc, Bc, Ngh, g)

    # calculate slopes of piecewise linear approximation
    dU = minmod_limiter(U, dx, Ngh, theta)

    # reconstruct solution at cell interfaces
    Uf = discont_soln(U, dU, dx, Ngh)
    Uf = correct_negative_depth(U, Bf, Uf, Ngh)

    # get non-conservative variables
    h, u, v = get_huv(Uf, Bf, epsilon)

    # get local speed
    a = local_speed(h, u, v, g)

    # get discontinuous PDE flux
    F, G = get_pde_flux(h, u, v, Uf, Bf, g)

    # get common/continuous numerical flux
    H = central_scheme(Uf, F, G, a)

    # get final right hand side
    f = (H["x"][:, :, :-1] - H["x"][:, :, 1:] +
         H["y"][:, :-1, :] - H["y"][:, 1:, :]) / dx + S

    # remove rounding errors
    zero_tensor = torch.tensor(0., device=f.device, dtype=f.dtype)
    f = torch.where(torch.abs(f)<=1e-10, zero_tensor, f)

    # obtain the maximum safe dt
    amax = torch.max(torch.max(a["xp"], -a["xm"])).item()
    bmax = torch.max(torch.max(a["yp"], -a["ym"])).item()
    max_dt = min(0.25*dx/amax, 0.25*dx/bmax)

    return f, max_dt
