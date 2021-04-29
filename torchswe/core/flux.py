#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the MIT license.

"""Functions for calculating discontinuous flux.
"""
import torch

@torch.jit.script
def fluxF(h, u, v, w, B, g: float):
    """Calculting the PDE flux in x direction.

    Note that when we calculate h**2, we use W*W-W*B-B*W+B*B to lower down
    the effect of rounding error. The results will be slightly different than
    using h * h naively, or even different from W^2 - 2WB + B^2.

    Args:
    -----
        h: a (Ny, Nx+1) torch.tensor of depth defined at cell interfaces of
            those normal to x direction.
        u: a (Ny, Nx+1) torch.tensor of u velocity defined at cell interfaces
            of those normal to x direction.
        v: a (Ny, Nx+1) torch.tensor of v velocity defined at cell interfaces
            of those normal to x direction.
        w: a (Ny, Nx+1) torch.tensor of w defined at cell interfaces of those
            normal to x direction. (w = h + B)
        B: a (Ny, Nx+1) torch.tensor of elevation defined at cell interfaces of
            those normal to x direction. (B = w - h)
        g: gravity.

    Returns:
    --------
        F: a (3, Ny, Nx+1) torch.tensor of PDE flux.
    """

    hu = h * u

    F = torch.stack(
        [
            hu,
            hu * u + g * (w * w - w * B - B * w + B * B) / 2.,
            hu * v
        ], dim=0
    )

    return F

@torch.jit.script
def fluxG(h, u, v, w, B, g: float):
    """Calculting the PDE flux in y direction.

    Note that when we calculate h**2, we use W*W-W*B-B*W+B*B to lower down
    the effect of rounding error. The results will be slightly different than
    using h * h naively, or even different from W^2 - 2WB + B^2.

    Args:
    -----
        h: a (Ny+1, Nx) torch.tensor of depth defined at cell interfaces of
            those normal to y direction.
        u: a (Ny+1, Nx) torch.tensor of u velocity defined at cell interfaces
            of those normal to y direction.
        v: a (Ny+1, Nx) torch.tensor of v velocity defined at cell interfaces
            of those normal to y direction.
        w: a (Ny+1, Nx) torch.tensor of w defined at cell interfaces of those
            normal to y direction. (w = h + B)
        B: a (Ny+1, Nx) torch.tensor of elevation defined at cell interfaces of
            those normal to y direction. (B = w - h)
        g: gravity.

    Returns:
    --------
        G: a (3, Ny+1, Nx) torch.tensor of PDE flux.
    """

    hv = h * v

    G = torch.stack(
        [
            hv,
            hv * u,
            hv * v + g * (w * w - w * B - B * w + B * B) / 2.,
        ], dim=0
    )

    return G
