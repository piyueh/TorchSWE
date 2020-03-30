#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the MIT license.

"""
Slope/flux limiters.
"""
import torch

@torch.jit.script
def minmod_limiter(U, dx: float, Ngh: int, theta: float=1.3):
    """Calculating slope based on minmod.

    Args:
    -----
        U: a 3D (3, Ny+2*Ngh, Nx+2*Ngh) torch.tensor of conservative variables
            at cell centers.
        dx: a scalar of cell szie, assuming dx = dy and is uniform everywhere.
        theta: parameter to controll oscillation and dispassion.

    Return:
    -------
        dU: a dictionary of the following key-value pairs
            x: a (3, Ny, Nx+2) torch.tensor for $\partial U / \partial x$
                at cell centers. Only one layer of ghost cells is included at
                each domain boundary in x direction.
            y: a (3, Ny+2, Nx) torch.tensor for $\partial U / \partial y$
                at cell centers. Only one layer of ghost cells is included at
                each domain boundary in y direction.
    """

    # sanity check
    assert Ngh >= 2
    assert isinstance(dx, float)

    # for convenience
    Ny = U.shape[1] - 2 * Ngh
    Nx = U.shape[2] - 2 * Ngh
    zero = torch.tensor(0., dtype=U.dtype, device=U.device)
    theta = torch.tensor(theta, dtype=U.dtype, device=U.device)

    numerator = U[:, Ngh:-Ngh, Ngh-1:-Ngh+1]-U[:, Ngh:-Ngh, Ngh-2:-Ngh] # U_{i} - U_{i-1}
    denominator = U[:, Ngh:-Ngh, Ngh:Nx+Ngh+2]-U[:, Ngh:-Ngh, Ngh-1:-Ngh+1] # U_{i+1} - U_{i}
    r = numerator / denominator
    phi = torch.max(torch.min(torch.min(theta*r, (1.+r)/2.), theta), zero)
    dUx = phi * denominator / dx

    numerator = U[:, Ngh-1:-Ngh+1, Ngh:-Ngh]-U[:, Ngh-2:-Ngh, Ngh:-Ngh] # U_{j} - U_{j-1}
    denominator = U[:, Ngh:Ny+Ngh+2, Ngh:-Ngh]-U[:, Ngh-1:-Ngh+1, Ngh:-Ngh] # U_{j+1} - U_{j}
    r = numerator / denominator
    phi = torch.max(torch.min(torch.min(theta*r, (1.+r)/2.), theta), zero)
    dUy = phi * denominator / dx

    return {"x": dUx, "y": dUy}
