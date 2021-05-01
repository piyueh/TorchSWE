#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Linear reconstruction.
"""
import torch

@torch.jit.script
def correct_negative_depth(U, Bfx, Bfy, Ufxm, Ufxp, Ufym, Ufyp, Ngh: int):
    """Fix negative interface depth.

    Args:
    -----
        U: a (3, Ny+2*Ngh, Nx+2*Ngh) torch.tensor of conservative variables at
            cell centers.
        Bfx: a (Ny, Nx+1) torch.tensor representing the elevations at the
            interface midpoints of those normal to x-direction. Must be
            calculated from the linear interpolation from corner elevations.
        Bfy: a (Ny+1, Nx) torch.tensor representing the elevations at the
            interface midpoints of those normal to y-direction. Must be
            calculated from the linear interpolation from corner elevations.
        Ufxm: a (3, Ny, Nx+1) torch.tensor representing the U values at the
            left sides of the cell interfaces normal to x-direction.
        Ufxp: a (3, Ny, Nx+1) torch.tensor representing the U values at the
            right sides of the cell interfaces normal to x-direction.
        Ufym: a (3, Ny+1, Nx) torch.tensor representing the U values at the
            bottom sides of the cell interfaces normal to y-direction.
        Ufyp: a (3, Ny+1, Nx) torch.tensor representing the U values at the
            top sides of the cell interfaces normal to y-direction.
        Ngh: an integer of the number of ghost cells at each boundary.

    Returns:
    --------
        Fixed Ufxm, Ufxp, Ufym, Ufyp, i.e., positivity preserving.
    """

    # aliases
    Ny, Nx = U.shape[1]-2*Ngh, U.shape[2]-2*Ngh

    # fix the case when the left depth of an interface is negative
    j, i = torch.where(Ufxm[0, :, :]<Bfx)
    Ufxm[0, j, i] = Bfx[j, i]
    j, i = j[i!=0], i[i!=0] # to avoid those i - 1 = -1
    Ufxp[0, j, i-1] = 2 * U[0, j+Ngh, i-1+Ngh] - Bfx[j, i]

    # fix the case when the right depth of an interface is negative
    j, i = torch.where(Ufxp[0, :, :]<Bfx)
    Ufxp[0, j, i] = Bfx[j, i]
    j, i = j[i!=Nx], i[i!=Nx] # to avoid i + 1 = Nx + 1
    Ufxm[0, j, i+1] = 2 * U[0, j+Ngh, i+Ngh] - Bfx[j, i]

    # fix the case when the bottom depth of an interface is negative
    j, i = torch.where(Ufym[0, :, :]<Bfy)
    Ufym[0, j, i] = Bfy[j, i]
    j, i = j[j!=0], i[j!=0] # to avoid j - 1 = -1
    Ufyp[0, j-1, i] = 2 * U[0, j-1+Ngh, i+Ngh] - Bfy[j, i]

    # fix the case when the top depth of an interface is negative
    j, i = torch.where(Ufyp[0, :, :]<Bfy)
    Ufyp[0, j, i] = Bfy[j, i]
    j, i = j[j!=Ny], i[j!=Ny] # to avoid j + 1 = Ny + 1
    Ufym[0, j+1, i] = 2 * U[0, j+Ngh, i+Ngh] - Bfy[j, i]

    # fix tiny tolerance due to numerical rounding error
    j, i = torch.where(Ufxp[0]<Bfx)
    Ufxp[0, j, i] = Bfx[j, i]

    j, i = torch.where(Ufxm[0]<Bfx)
    Ufxm[0, j, i] = Bfx[j, i]

    j, i = torch.where(Ufyp[0]<Bfy)
    Ufyp[0, j, i] = Bfy[j, i]

    j, i = torch.where(Ufym[0]<Bfy)
    Ufym[0, j, i] = Bfy[j, i]

    return Ufxm, Ufxp, Ufym, Ufyp

@torch.jit.script
def discont_soln(U, dUx, dUy, Bfx, Bfy, dx: float, Ngh: int):
    """Discontinuous solution at interfaces with piecewise linear approximation.

    Note, the returned interface values have not been fixed for postivity
    preserving.

    Args:
    -----
        U: a 3D torch.tensor of shape (3, Ny+2*Ngh, Nx+2*Ngh) representing w,
            hu, and hv at cell centers; the extra cells are ghost cells.
        dUx: a (3, Ny, Nx+2) torch.tensor for $\partial U / \partial x$ at cell
            centers. Only one layer of ghost cells is included at each domain
            boundary in x direction.
        dUy: a (3, Ny+2, Nx) torch.tensor for $\partial U / \partial y$ at cell
            centers. Only one layer of ghost cells is included at each domain
            boundary in y direction.
        Bfx: a (Ny, Nx+1) torch.tensor representing the elevations at the
            interface midpoints of those normal to x-direction. Must be
            calculated from the linear interpolation from corner elevations.
        Bfy: a (Ny+1, Nx) torch.tensor representing the elevations at the
            interface midpoints of those normal to y-direction. Must be
            calculated from the linear interpolation from corner elevations.
        dx: a scalar of cell szie, assuming dx = dy and is uniform everywhere.
        Ngh: an integer, number of ghost cells outside each boundary

    Returns:
    --------
        Ufxm: a (3, Ny, Nx+1) torch.tensor representing the U values at the
            left sides of the cell interfaces normal to x-direction.
        Ufxp: a (3, Ny, Nx+1) torch.tensor representing the U values at the
            right sides of the cell interfaces normal to x-direction.
        Ufym: a (3, Ny+1, Nx) torch.tensor representing the U values at the
            bottom sides of the cell interfaces normal to y-direction.
        Ufyp: a (3, Ny+1, Nx) torch.tensor representing the U values at the
            top sides of the cell interfaces normal to y-direction.
    """

    dx2 = dx / 2

    # left value at each x-interface
    Ufxm = U[:, Ngh:-Ngh, Ngh-1:-Ngh] + dUx[:, :, :-1] * dx2

    # right value at each x-interface
    Ufxp = U[:, Ngh:-Ngh, Ngh:-Ngh+1] - dUx[:, :, 1:] * dx2

    # bottom value at each y-interface
    Ufym = U[:, Ngh-1:-Ngh, Ngh:-Ngh] + dUy[:, :-1, :] * dx2

    # top value at each y-interface
    Ufyp = U[:, Ngh:-Ngh+1, Ngh:-Ngh] - dUy[:, 1:, :] * dx2

    # fix non-physical negative path
    Ufxm, Ufxp, Ufym, Ufyp = correct_negative_depth(
        U, Bfx, Bfy, Ufxm, Ufxp, Ufym, Ufyp, Ngh)

    return Ufxm, Ufxp, Ufym, Ufyp
