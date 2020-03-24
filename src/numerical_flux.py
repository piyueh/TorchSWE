#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the MIT license.

"""
Functions to calculate numerical/common flux.
"""
import torch

def central_scheme(Uf, F, G, a):
    """A central scheme to calculate numerical flux at interfaces.

    Should not be used explicitly. It is supposed to be called by other schemes.

    Args:
    -----
        Uf: a dictionary of the following key-value pairs
            xm: a (3, Ny, Nx+1) torch.tensor representing the U values at the
                left sides of the cell interfaces normal to x-direction.
            xp: a (3, Ny, Nx+1) torch.tensor representing the U values at the
                right sides of the cell interfaces normal to x-direction.
            ym: a (3, Ny+1, Nx) torch.tensor representing the U values at the
                bottom sides of the cell interfaces normal to y-direction.
            yp: a (3, Ny+1, Nx) torch.tensor representing the U values at the
                top sides of the cell interfaces normal to y-direction.
        F: a dictionary of the following key-value pairs
            xm: a (3, Ny, Nx+1) torch.tensor representing the flux at the
                left sides of the cell interfaces normal to x-direction.
            xp: a (3, Ny, Nx+1) torch.tensor representing the flux at the
                right sides of the cell interfaces normal to x-direction.
        G: a dictionary of the following key-value pairs
            ym: a (3, Ny+1, Nx) torch.tensor representing the flux at the
                bottom sides of the cell interfaces normal to y-direction.
            yp: a (3, Ny+1, Nx) torch.tensor representing the flux at the
                top sides of the cell interfaces normal to y-direction.
        a: a dictionary of the following key-value pairs
            xm: a (Ny, Nx+1) torch.tensor representing the local speed at the
                left sides of the cell interfaces normal to x-direction.
            xp: a (Ny, Nx+1) torch.tensor representing the local speed at the
                right sides of the cell interfaces normal to x-direction.
            ym: a (Ny+1, Nx) torch.tensor representing the local speed at the
                bottom sides of the cell interfaces normal to y-direction.
            yp: a (Ny+1, Nx) torch.tensor representing the local speed at the
                top sides of the cell interfaces normal to y-direction.

    Returns:
    --------
        H: a diction of the following key-value pairs:
            x: a (3, Ny, Nx+1) torch.tensor representing the common numerical
                flux at cell interfaces normal to x-direction.
            y: a (3, Ny+1, Nx) torch.tensor representing the common numerical
                flux at cell interfaces normal to y-direction.
    """

    # aliases for convinience
    ap, am, bp, bm = a["xp"], a["xm"], a["yp"], a["ym"]
    Fp, Fm, Gp, Gm = F["xp"], F["xm"], G["yp"], G["ym"]

    # for convenience
    zero_tensor = torch.tensor(0, device=Fm.device, dtype=Fm.dtype)

    # flux in x direction
    Hx = ap * Fm - am * Fp + ap * am * (Uf["xp"] - Uf["xm"])
    denominator = ap - am
    # a sanity check: if denominator == 0, numerators should also == 0
    assert torch.allclose(Hx[:, denominator==0], zero_tensor)
    denominator[denominator==0] = 1. # to avoid division by zero
    Hx /= denominator

    # flux in x direction
    Hy = bp * Gm - bm * Gp + bp * bm * (Uf["yp"] - Uf["ym"])
    denominator = bp - bm
    # a sanity check: if denominator == 0, numerators should also == 0
    assert torch.allclose(Hy[:, denominator==0], zero_tensor)
    denominator[denominator==0] = 1. # to avoid division by zero
    Hy /= denominator

    return {"x": Hx, "y": Hy}
