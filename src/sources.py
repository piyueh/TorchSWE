#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the MIT license.

"""Source terms.
"""
from typing import Dict
import torch

@torch.jit.script
def source(U, dBx, dBy, Bc, Ngh: int, g: float) -> torch.Tensor:
    """Source term

    Note, the mesh arrangement must be Y first and then X. For example, U[0, j, i]
    is the w of j-th cell in y-direction and i-th cell in x-direction.

    Args:
    -----
        U: a 3D torch.tensor of shape (3, Ny+2*Ngh, Nx+2Ngh) representing
            w, hu, and hv.
        dBx: a 2d torch.tensor of shape (Ny, Nx) representing the topographic
            x-gradient at cell centers, i.e., $\partial B / \partial x$.
        dBy: a 2d torch.tensor of shape (Ny, Nx) representing the topographic
            y-gradient at cell centers, i.e., $\partial B / \partial y$.
        Bc: a 2D torch.tensor of shape (Ny, Nx) representing elevations at cell
            centers, excluding ghost cells. Bc must be from the bilinear
            interpolation of elevation of cell coreners.
        Ngh: an integer, the number of ghost cells outside each boundary
        g: gravity

    Returns:
    --------
        S: a 3D torch.tensor of shape (3, Ny, Nx) representing the source terms
            for continuity equation and momentum equations for each cell.
    """

    Nx: int = U.shape[2] - 2 * Ngh
    Ny: int = U.shape[1] - 2 * Ngh

    S: torch.Tensor = torch.zeros((3, Ny, Nx), device="cuda", dtype=U.dtype)

    gH: torch.Tensor = -g * (U[0, Ngh:-Ngh, Ngh:-Ngh] - Bc)
    S[1, :, :] = gH * dBx
    S[2, :, :] = gH * dBy

    return S
