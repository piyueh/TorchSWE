#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the MIT license.

"""Linear reconstruction.
"""
import torch
from .sources import source
from .limiters import minmod_limiter
from .reconstruction import discont_soln
from .misc import decompose_variables, local_speed
from .flux import fluxF, fluxG
from .numerical_flux import central_scheme

@torch.jit.script
def fvm(U, Bfx, Bfy, Bc, dBx, dBy,
        dx: float, Ngh: int, g: float, epsilon: float, theta: float):
    """Get the right-hand-side of a time-marching step with finite volume method.

    Args:
    -----
        U: a 3D torch.tensor of shape (3, Ny+2*Ngh, Nx+2Ngh) representing
            w, hu, and hv.
        Bfx: a (Ny, Nx+1) torch.tensor representing the elevations at the
            interface midpoints of those normal to x-direction. Must be
            calculated from the linear interpolation from corner elevations.
        Bfy: a (Ny+1, Nx) torch.tensor representing the elevations at the
            interface midpoints of those normal to y-direction. Must be
            calculated from the linear interpolation from corner elevations.
        Bc: a 2D torch.tensor of shape (Ny, Nx) representing elevations at cell
            centers, excluding ghost cells. Bc must be from the bilinear
            interpolation of elevation of cell coreners.
        dBx: a 2d torch.tensor of shape (Ny, Nx) representing the topographic
            x-gradient at cell centers, i.e., $\partial B / \partial x$.
        dBy: a 2d torch.tensor of shape (Ny, Nx) representing the topographic
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
    S = source(U, dBx, dBy, Bc, Ngh, g)

    # calculate slopes of piecewise linear approximation
    dUx, dUy = minmod_limiter(U, dx, Ngh, theta)

    # reconstruct solution at cell interfaces
    Ufxm, Ufxp, Ufym, Ufyp = discont_soln(U, dUx, dUy, Bfx, Bfy, dx, Ngh)

    # get non-conservative variables
    hxm, uxm, vxm = decompose_variables(Ufxm, Bfx, epsilon)
    hxp, uxp, vxp = decompose_variables(Ufxp, Bfx, epsilon)
    hym, uym, vym = decompose_variables(Ufym, Bfy, epsilon)
    hyp, uyp, vyp = decompose_variables(Ufyp, Bfy, epsilon)

    # get local speed
    axm, axp, aym, ayp = local_speed(
        hxm, hxp, hym, hyp, uxm, uxp, vym, vyp, g)

    # get discontinuous PDE flux
    Fxm = fluxF(hxm, uxm, vxm, Ufxm[0], Bfx, g)
    Fxp = fluxF(hxp, uxp, vxp, Ufxp[0], Bfx, g)
    Gym = fluxG(hym, uym, vym, Ufym[0], Bfy, g)
    Gyp = fluxG(hyp, uyp, vyp, Ufyp[0], Bfy, g)

    # get common/continuous numerical flux
    Hx = central_scheme(Ufxm, Ufxp, Fxm, Fxp, axm, axp)
    Hy = central_scheme(Ufym, Ufyp, Gym, Gyp, aym, ayp)

    # get final right hand side
    f = (Hx[:, :, :-1] - Hx[:, :, 1:] + Hy[:, :-1, :] - Hy[:, 1:, :]) / dx + S

    # remove rounding errors
    zero_tensor = torch.tensor(0., device=f.device, dtype=f.dtype)
    f = torch.where(torch.abs(f)<=1e-10, zero_tensor, f)

    # obtain the maximum safe dt
    amax = torch.max(torch.max(axp, -axm)).item()
    bmax = torch.max(torch.max(ayp, -aym)).item()
    max_dt = min(0.25*dx/amax, 0.25*dx/bmax)

    return f, max_dt
