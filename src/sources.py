#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the MIT license.

"""
Source terms.
"""
import torch

def source(U, dBc, Bc, Ngh, g):
    """Source term

    Note, the mesh arrangement must be Y first and then X. For example, U[0, j, i]
    is the w of j-th cell in y-direction and i-th cell in x-direction.

    Args:
    -----
        U: a 3D torch.tensor of shape (3, Ny+2*Ngh, Nx+2Ngh) representing
            w, hu, and hv.
        dBc: a dictionary of the following key-value pair
            x: a 2d torch.tensor of shape (Ny, Nx) representing the topographic
                x-gradient at cell centers, i.e., $\partial B / \partial x$.
            y: a 2d torch.tensor of shape (Ny, Nx) representing the topographic
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

    Nx = U.shape[2] - 2 * Ngh
    Ny = U.shape[1] - 2 * Ngh

    S = torch.zeros((3, Ny, Nx), device="cuda", dtype=U.dtype)

    gH = -g * (U[0, Ngh:-Ngh, Ngh:-Ngh] - Bc)
    S[1, :, :] = gH * dBc["x"]
    S[2, :, :] = gH * dBc["y"]

    return S
