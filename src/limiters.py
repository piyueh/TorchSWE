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

def minmod(z):
    """Minmod function.

    Args:
    -----
        z: a N-dimensional torch.tensor; minmod is applied to the last dimension.

    Return:
    -------
        Minmod results.
    """
    # TODO: check the performance. Code may look clean but I doubt its efficiency.

    axis = len(z.shape) - 1
    results = torch.zeros(z.shape[:-1], device=z.device, dtype=z.dtype)
    results = torch.where(torch.all(z>0, dim=axis), torch.min(z, dim=axis)[0], results)
    results = torch.where(torch.all(z<0, dim=axis), torch.max(z, dim=axis)[0], results)

    return results

def minmod_limiter(U, dx, Ngh, theta=1.3):
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

    # init the returned dictionary
    dU = {}

    # x-direction
    aux = torch.stack(
        [
            theta*(U[:, Ngh:-Ngh, Ngh-1:-Ngh+1]-U[:, Ngh:-Ngh, Ngh-2:-Ngh])/dx,
            (U[:, Ngh:-Ngh, Ngh:Nx+Ngh+2]-U[:, Ngh:-Ngh, Ngh-2:-Ngh])/(dx*2.),
            theta*(U[:, Ngh:-Ngh, Ngh:Nx+Ngh+2]-U[:, Ngh:-Ngh, Ngh-1:-Ngh+1])/dx
        ], dim=3
    )
    dU["x"] = minmod(aux)

    # y-direction
    aux = torch.stack(
        [
            theta*(U[:, Ngh-1:-Ngh+1, Ngh:-Ngh]-U[:, Ngh-2:-Ngh, Ngh:-Ngh])/dx,
            (U[:, Ngh:Ny+Ngh+2, Ngh:-Ngh]-U[:, Ngh-2:-Ngh, Ngh:-Ngh])/(dx*2.),
            theta*(U[:, Ngh:Ny+Ngh+2, Ngh:-Ngh]-U[:, Ngh-1:-Ngh+1, Ngh:-Ngh])/dx
        ], dim=3
    )
    dU["y"] = minmod(aux)

    return dU
