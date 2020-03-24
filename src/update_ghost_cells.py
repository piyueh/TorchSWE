#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the MIT license.

"""
Functions related to updating ghost cells.
"""
import torch

def periodic(U, Ngh):
    """Update the ghost cells with periodic BC.

    Note we ignore the ghost cells at corners. They shouldn't be used in this
    numerical method.

    Theoretically speaking, U is modified in-place, but we still return it.

    Args:
    -----
        U: a (3, Ny+2*Ngh, Nx+2*Ngh) torch.tensor of conservative variables.
        Ngh: an integer for the number of ghost cells outside each boundary.

    Returns:
    --------
        Updated U.
    """

    U[:, Ngh:-Ngh, :Ngh] = U[:, Ngh:-Ngh, -Ngh-Ngh:-Ngh]
    U[:, Ngh:-Ngh, -Ngh:] = U[:, Ngh:-Ngh, Ngh:Ngh+Ngh]
    U[:, :Ngh, Ngh:-Ngh] = U[:, -Ngh-Ngh:-Ngh, Ngh:-Ngh]
    U[:, -Ngh:, Ngh:-Ngh] = U[:, Ngh:Ngh+Ngh, Ngh:-Ngh]

    return U

def linear_extrap(U, Ngh):
    """Update the ghost cells with outflow BC using linear extrapolation.

    Ghost cells should have the same cell size as the first/last interior cell does.
    Note we ignore the ghost cells at corners. They shouldn't be used in this
    numerical method.

    Theoretically speaking, U is modified in-place, but we still return it.

    Args:
    -----
        U: a (3, Ny+2*Ngh, Nx+2*Ngh) torch.tensor of conservative variables.
        Ngh: an integer for the number of ghost cells outside each boundary.

    Returns:
    --------
        Updated U.
    """

    seq_f = torch.arange(1, Ngh+1, device=U.device)
    seq_b = torch.arange(Ngh, 0, -1, device=U.device)

    # west
    delta = U[:, Ngh:-Ngh, Ngh+1][:, :, None] - U[:, Ngh:-Ngh, Ngh][:, :, None]
    U[:, Ngh:-Ngh, :Ngh] = U[:, Ngh:-Ngh, Ngh][:, :, None] - seq_b * delta

    # east
    delta = U[:, Ngh:-Ngh, -Ngh-1][:, :, None] - U[:, Ngh:-Ngh, -Ngh-2][:, :, None]
    U[:, Ngh:-Ngh, -Ngh:] = U[:, Ngh:-Ngh, -Ngh-1][:, :, None] + seq_f * delta

    # south
    delta = U[:, Ngh+1, Ngh:-Ngh][:, None, :] - U[:, Ngh, Ngh:-Ngh][:, None, :]
    U[:, :Ngh, Ngh:-Ngh] = U[:, Ngh, Ngh:-Ngh][:, None, :] - seq_b[:, None] * delta

    # north
    delta = U[:, -Ngh-1, Ngh:-Ngh][:, None, :] - U[:, -Ngh-2, Ngh:-Ngh][:, None, :]
    U[:, -Ngh:, Ngh:-Ngh] = U[:, -Ngh-1, Ngh:-Ngh][:, None, :] + seq_f[:, None] * delta

    return U
