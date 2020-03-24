#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the MIT license.

"""
Functions for calculating discontinuous flux.
"""
import torch


def fluxF(h, u, v, w, B, g):
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

def fluxG(h, u, v, w, B, g):
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

def get_pde_flux(h, u, v, Uf, Bf, g):
    """

    Args:
    -----
        h: a dictionary of the following key-value pairs
            xm: a (Ny, Nx+1) torch.tensor of dpeths at the left sides of the
                cell interfaces normal to x-direction.
            xp: a (Ny, Nx+1) torch.tensor of dpeths at the right sides of the
                cell interfaces normal to x-direction.
            ym: a (Ny+1, Nx) torch.tensor of dpeths at the bottom sides of the
                cell interfaces normal to y-direction.
            yp: a (Ny+1, Nx) torch.tensor of dpeths at the top sides of the
                cell interfaces normal to y-direction.
        u: a dictionary of the following key-value pairs
            xm: a (Ny, Nx+1) torch.tensor of u-velocity at the left sides of
                the cell interfaces normal to x-direction.
            xp: a (Ny, Nx+1) torch.tensor of u-velocity at the right sides of
                the cell interfaces normal to x-direction.
            ym: a (Ny+1, Nx) torch.tensor of u-velocity at the bottom sides of
                the cell interfaces normal to y-direction.
            yp: a (Ny+1, Nx) torch.tensor of u-velocity at the top sides of the
                cell interfaces normal to y-direction.
        v: a dictionary of the following key-value pairs
            xm: a (Ny, Nx+1) torch.tensor of v-velocity at the left sides of
                the cell interfaces normal to x-direction.
            xp: a (Ny, Nx+1) torch.tensor of v-velocity at the right sides of
                the cell interfaces normal to x-direction.
            ym: a (Ny+1, Nx) torch.tensor of v-velocity at the bottom sides of
                the cell interfaces normal to y-direction.
            yp: a (Ny+1, Nx) torch.tensor of v-velocity at the top sides of the
                cell interfaces normal to y-direction.
        Uf: a dictionary of the following key-value pairs
            xm: a (3, Ny, Nx+1) torch.tensor representing the U values at the
                left sides of the cell interfaces normal to x-direction.
            xp: a (3, Ny, Nx+1) torch.tensor representing the U values at the
                right sides of the cell interfaces normal to x-direction.
            ym: a (3, Ny+1, Nx) torch.tensor representing the U values at the
                bottom sides of the cell interfaces normal to y-direction.
            yp: a (3, Ny+1, Nx) torch.tensor representing the U values at the
                top sides of the cell interfaces normal to y-direction.
        Bf: a dictionary of the following key-value pairs
            x: a (Ny, Nx+1) torch.tensor representing the elevations at
                interfaces midpoint for those normal to x-direction. Must be
                calculated from the linear interpolation from corner elevations.
            y: a (Ny+1, Nx) torch.tensor representing the elevations at
                interfaces midpoint for those normal to y-direction. Must be
                calculated from the linear interpolation from corner elevations.
        g: gravity

    Returns:
        F: a dictionary with the following key-value pairs
            xm: a (Ny, Nx+1) torch.tensor of PDE flux at the left sides of the
                cell interfaces normal to x-direction.
            xp: a (Ny, Nx+1) torch.tensor of PDE flux at the right sides of the
                cell interfaces normal to x-direction.
        G: a dictionary with the following key-value pairs
            ym: a (Ny+1, Nx) torch.tensor of PDE flux at the bottom sides of
                the cell interfaces normal to y-direction.
            yp: a (Ny+1, Nx) torch.tensor of PDE flux at the top sides of the
                cell interfaces normal to y-direction.
    """

    F = {}
    F["xm"] = fluxF(h["xm"], u["xm"], v["xm"], Uf["xm"][0], Bf["x"], g)
    F["xp"] = fluxF(h["xp"], u["xp"], v["xp"], Uf["xp"][0], Bf["x"], g)

    G = {}
    G["ym"] = fluxG(h["ym"], u["ym"], v["ym"], Uf["ym"][0], Bf["y"], g)
    G["yp"] = fluxG(h["yp"], u["yp"], v["yp"], Uf["yp"][0], Bf["y"], g)

    return F, G
