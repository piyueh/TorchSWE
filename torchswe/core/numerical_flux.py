#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the MIT license.

"""Functions to calculate numerical/common flux.
"""
import torch

@torch.jit.script
def central_scheme(Ufm, Ufp, Fm, Fp, am, ap):
    """A central scheme to calculate numerical flux at interfaces.

    Should not be used explicitly. It is supposed to be called by other schemes.

    Args:
    -----
        Ufm: either a (3, Ny, Nx+1) or (3, Ny+1, Nx) torch.tensor, depending on
            whether it's in x or y direction; the discontinuous conservative
            quantities at the left or the bottom of cell interfaces.
        Ufp: either a (3, Ny, Nx+1) or (3, Ny+1, Nx) torch.tensor, depending on
            whether it's in x or y direction; the discontinuous conservative
            quantities at the right or the top of cell interfaces.
        Fm: ither a (3, Ny, Nx+1) or (3, Ny+1, Nx) torch.tensor, depending on
            whether it's in x or y direction; the flux at the left or the bottom
            of cell interfaces.
        Fp: ither a (3, Ny, Nx+1) or (3, Ny+1, Nx) torch.tensor, depending on
            whether it's in x or y direction; the flux at the right or the top
            of cell interfaces.
        am: ither a (Ny, Nx+1) or (Ny+1, Nx) torch.tensor, depending on whether
            it's in x or y direction; the local speed at the left or the bottom
            of cell interfaces.
        ap: ither a (Ny, Nx+1) or (Ny+1, Nx) torch.tensor, depending on whether
            it's in x or y direction; the local speed at the right or the top of
            cell interfaces.

    Returns:
    --------
        H: ither a (3, Ny, Nx+1) or (3, Ny+1, Nx) torch.tensor, depending on
            whether it's in x or y direction; the numerical (common) flux at the
            cell interfaces.
    """

    # for convenience
    zero_tensor = torch.tensor(0, device=Fm.device, dtype=Fm.dtype)
    one_tensor = torch.tensor(1, device=Fm.device, dtype=Fm.dtype)

    # flux in x direction
    H = ap * Fm - am * Fp + ap * am * (Ufp - Ufm)
    denominator = ap - am
    denominator[denominator==0] = one_tensor # to avoid division by zero
    H /= denominator

    return H
