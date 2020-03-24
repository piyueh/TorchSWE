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

def discont_soln(U, dU, dx, Ngh):
    """Discontinuous solution at interfaces with piecewise linear approximation.

    Note, the returned interface values have not been fixed for postivity
    preserving.

    Args:
    -----
        U: a 3D torch.tensor of shape (3, Ny+2*Ngh, Nx+2*Ngh) representing w,
            hu, and hv at cell centers; the extra cells are ghost cells.
        dU: a dictionary of the following key-value pairs
            x: a (3, Ny, Nx+2) torch.tensor for $\partial U / \partial x$
                at cell centers. Only one layer of ghost cells is included at
                each domain boundary in x direction.
            y: a (3, Ny+2, Nx) torch.tensor for $\partial U / \partial y$
                at cell centers. Only one layer of ghost cells is included at
                each domain boundary in y direction.
        dx: a scalar of cell szie, assuming dx = dy and is uniform everywhere.
        Ngh: an integer, number of ghost cells outside each boundary

    Returns:
    --------
        Uf: a dictionary of the following key-value pairs
            xm: a (3, Ny, Nx+1) torch.tensor representing the U values at the
                left sides of the cell interfaces normal to x-direction.
            xp: a (3, Ny, Nx+1) torch.tensor representing the U values at the
                right sides of the cell interfaces normal to x-direction.
            ym: a (3, Ny+1, Nx) torch.tensor representing the U values at the
                bottom sides of the cell interfaces normal to y-direction.
            yp: a (3, Ny+1, Nx) torch.tensor representing the U values at the
                top sides of the cell interfaces normal to y-direction.
    """

    Uf = {}

    # left value at each x-interface
    Uf["xm"] = U[:, Ngh:-Ngh, Ngh-1:-Ngh] + dU["x"][:, :, :-1] * dx / 2

    # right value at each x-interface
    Uf["xp"] = U[:, Ngh:-Ngh, Ngh:-Ngh+1] - dU["x"][:, :, 1:] * dx / 2

    # bottom value at each y-interface
    Uf["ym"] = U[:, Ngh-1:-Ngh, Ngh:-Ngh] + dU["y"][:, :-1, :] * dx / 2

    # top value at each y-interface
    Uf["yp"] = U[:, Ngh:-Ngh+1, Ngh:-Ngh] - dU["y"][:, 1:, :] * dx / 2

    # sanity check
    assert Uf["xm"].shape == (3, U.shape[1]-2*Ngh, U.shape[2]-2*Ngh+1)
    assert Uf["xp"].shape == (3, U.shape[1]-2*Ngh, U.shape[2]-2*Ngh+1)
    assert Uf["ym"].shape == (3, U.shape[1]-2*Ngh+1, U.shape[2]-2*Ngh)
    assert Uf["yp"].shape == (3, U.shape[1]-2*Ngh+1, U.shape[2]-2*Ngh)

    return Uf
