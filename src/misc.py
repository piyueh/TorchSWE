#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the MIT license.

"""
Misc. functions.
"""
import torch

class CFLWarning(Warning):
    """A category of Warning for custome controll of warning action."""
    def __init__(self, *args, **kwargs):
        """initializataion"""
        super(CFLWarning, self).__init__(*args, **kwargs)

def check_timestep_size(dt, max_dt):
    """Check if dt is smaller than maximum safe dt.

    If not, add a runtime warning. Users can decide whether to turn the warning
    into a blocking exception in their program with warnings module:

            warnings.simplefilter("error", CFLWarning)

    Args:
    -----
        dt: a scalar time-step size.
        max_dt: a scalar; maximum safe dt.

    Returns:
    --------
        N/A
    """

    if dt > max_dt:
        warnings.warn(
            "dt={} exceeds the maximum safe value of {}".format(dt, max_dt),
            CFLWarning)

def correct_negative_depth(U, Bf, Uf, Ngh):
    """Fix negative interface depth.

    Args:
    -----
        U: a (3, Ny+2*Ngh, Nx+2*Ngh) torch.tensor of conservative variables at
            cell centers.
        Bf: a dictionary of the following key-value pairs
            x: a (Ny, Nx+1) torch.tensor representing the elevations at
                interfaces midpoint for those normal to x-direction. Must be
                calculated from the linear interpolation from corner elevations.
            y: a (Ny+1, Nx) torch.tensor representing the elevations at
                interfaces midpoint for those normal to y-direction. Must be
                calculated from the linear interpolation from corner elevations.
        Uf: a dictionary of the following key-value pairs
            xm: a (3, Ny, Nx+1) torch.tensor representing the U values at the
                left sides of the cell interfaces normal to x-direction.
            xp: a (3, Ny, Nx+1) torch.tensor representing the U values at the
                right sides of the cell interfaces normal to x-direction.
            ym: a (3, Ny+1, Nx) torch.tensor representing the U values at the
                bottom sides of the cell interfaces normal to y-direction.
            yp: a (3, Ny+1, Nx) torch.tensor representing the U values at the
                top sides of the cell interfaces normal to y-direction.
        Ngh: an integer of the number of ghost cells at each boundary.

    Returns:
    --------
        Fixed Uf, i.e., positivity preserving.
    """

    # aliases
    Ny, Nx = U.shape[1]-2*Ngh, U.shape[2]-2*Ngh

    # sanity check
    assert Bf["x"].shape == (Ny, Nx+1)
    assert Bf["y"].shape == (Ny+1, Nx)
    assert Uf["xp"].shape == (3, Ny, Nx+1)
    assert Uf["xm"].shape == (3, Ny, Nx+1)
    assert Uf["yp"].shape == (3, Ny+1, Nx)
    assert Uf["ym"].shape == (3, Ny+1, Nx)

    # fix the case when the left depth of an interface is negative
    j, i = torch.where(Uf["xm"][0, :, :]<Bf["x"])
    Uf["xm"][0, j, i] = Bf["x"][j, i]
    j, i = j[i!=0], i[i!=0] # to avoid those i - 1 = -1
    Uf["xp"][0, j, i-1] = 2 * U[0, j+Ngh, i-1+Ngh] - Bf["x"][j, i]

    # fix the case when the right depth of an interface is negative
    j, i = torch.where(Uf["xp"][0, :, :]<Bf["x"])
    Uf["xp"][0, j, i] = Bf["x"][j, i]
    j, i = j[i!=Nx], i[i!=Nx] # to avoid i + 1 = Nx + 1
    Uf["xm"][0, j, i+1] = 2 * U[0, j+Ngh, i+Ngh] - Bf["x"][j, i]

    # fix the case when the bottom depth of an interface is negative
    j, i = torch.where(Uf["ym"][0, :, :]<Bf["y"])
    Uf["ym"][0, j, i] = Bf["y"][j, i]
    j, i = j[j!=0], i[j!=0] # to avoid j - 1 = -1
    Uf["yp"][0, j-1, i] = 2 * U[0, j-1+Ngh, i+Ngh] - Bf["y"][j, i]

    # fix the case when the top depth of an interface is negative
    j, i = torch.where(Uf["yp"][0, :, :]<Bf["y"])
    Uf["yp"][0, j, i] = Bf["y"][j, i]
    j, i = j[j!=Ny], i[j!=Ny] # to avoid j + 1 = Ny + 1
    Uf["ym"][0, j+1, i] = 2 * U[0, j+Ngh, i+Ngh] - Bf["y"][j, i]

    # fix tiny tolerance due to numerical rounding error
    j, i = torch.where(Uf["xp"][0]<Bf["x"])
    assert torch.allclose(Uf["xp"][0, j, i], Bf["x"][j, i], 1e-6, 1e-10)
    Uf["xp"][0, j, i] = Bf["x"][j, i]

    j, i = torch.where(Uf["xm"][0]<Bf["x"])
    assert torch.allclose(Uf["xm"][0, j, i], Bf["x"][j, i], 1e-6, 1e-10)
    Uf["xm"][0, j, i] = Bf["x"][j, i]

    j, i = torch.where(Uf["yp"][0]<Bf["y"])
    assert torch.allclose(Uf["yp"][0, j, i], Bf["y"][j, i], 1e-6, 1e-10)
    Uf["yp"][0, j, i] = Bf["y"][j, i]

    j, i = torch.where(Uf["ym"][0]<Bf["y"])
    assert torch.allclose(Uf["ym"][0, j, i], Bf["y"][j, i], 1e-6, 1e-10)
    Uf["ym"][0, j, i] = Bf["y"][j, i]

    return Uf

def decompose_variables(U, B, epsilon):
    """Decompose conservative variables to dpeth and velocity.

    Args:
    -----
        U: depends on the use case, it can be a torch.tensor with a shape of
            either (3, Ny, Nx), (3, Ny+1, Nx), or (3, Ny, Nx+1).
        B: a 2D torch.tensor with a shape of either (Ny, Nx), (Ny+1, Nx) or
            (Ny, Nx+1), depending on U.
        epsilon: a very small number to avoid division by zero.

    Returns:
    --------
        h: a (Ny, Nx), (Ny, Nx+1) or (Ny+1, Nx) torch.tensor of depth.
        u: a (Ny, Nx), (Ny, Nx+1) or (Ny+1, Nx) torch.tensor of u velocity.
        v: a (Ny, Nx), (Ny, Nx+1) or (Ny+1, Nx) torch.tensor of v velocity.
    """

    h = U[0, :, :] - B
    h4 = torch.pow(h, 4)
    denominator = torch.sqrt(
        h4+torch.max(h4, torch.tensor(epsilon, device=h4.device, dtype=h4.dtype)))

    u = h * U[1, :, :] * (2**0.5) / denominator
    v = h * U[2, :, :] * (2**0.5) / denominator

    return h, u, v

def get_huv(Uf, B, epsilon):
    """Obtain all discontinuous h, u, and v at cell interfaces.

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
        B: a dictionary of the following key-value pairs
            x: a (Ny, Nx+1) torch.tensor representing the elevations at
                interfaces midpoint for those normal to x-direction. Must be
                calculated from the linear interpolation from corner elevations.
            y: a (Ny+1, Nx) torch.tensor representing the elevations at
                interfaces midpoint for those normal to y-direction. Must be
                calculated from the linear interpolation from corner elevations.
        epsilon: a very small number to avoid division by zero.

    Returns:
    --------
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
    """

    h = {}
    u = {}
    v = {}

    h["xm"], u["xm"], v["xm"] = decompose_variables(Uf["xm"], B["x"], epsilon)
    h["xp"], u["xp"], v["xp"] = decompose_variables(Uf["xp"], B["x"], epsilon)
    h["ym"], u["ym"], v["ym"] = decompose_variables(Uf["ym"], B["y"], epsilon)
    h["yp"], u["yp"], v["yp"] = decompose_variables(Uf["yp"], B["y"], epsilon)

    return h, u, v

def local_speed(h, u, v, g):
    """Calculate local speed a.

    Args:
    -----
        h: a dictionary of the following key-value pairs
            xm: a (Ny, Nx+1) torch.tensor of depth at the left of interfaces.
            xp: a (Ny, Nx+1) torch.tensor of depth at the right of interfaces.
            ym: a (Ny+1, Nx) torch.tensor of depth at the bottom of interfaces.
            yp: a (Ny+1, Nx) torch.tensor of depth at the top of interfaces.
        u: a dictionary of at least the following key-value pairs
            xm: a (Ny, Nx+1) torch.tensor of u velocity at the left of interfaces.
            xp: a (Ny, Nx+1) torch.tensor of u velocity at the right of interfaces.
        v: a dictionary of at least the following key-value pairs
            ym: a (Ny+1, Nx) torch.tensor of v velocity at the bottom of interfaces.
            yp: a (Ny+1, Nx) torch.tensor of v velocity at the top of interfaces.
        g: gravity

    Returns:
    --------
        a: a dictionary of the following key-value pairs
            xm: a (Ny, Nx+1) torch.tensor representing the local speed at the
                left sides of the cell interfaces normal to x-direction.
            xp: a (Ny, Nx+1) torch.tensor representing the local speed at the
                right sides of the cell interfaces normal to x-direction.
            ym: a (Ny+1, Nx) torch.tensor representing the local speed at the
                bottom sides of the cell interfaces normal to y-direction.
            yp: a (Ny+1, Nx) torch.tensor representing the local speed at the
                top sides of the cell interfaces normal to y-direction.
    """

    # aliases
    hm, hp, hb, ht = h["xm"], h["xp"], h["ym"], h["yp"]
    um, up = u["xm"], u["xp"]
    vb, vt = v["ym"], v["yp"]

    assert torch.all(hm>=0.)
    assert torch.all(hp>=0.)
    assert torch.all(hb>=0.)
    assert torch.all(ht>=0.)

    a = {}
    a["xp"] = torch.max(
        torch.max(up+torch.sqrt(hp*g), um+torch.sqrt(hm*g)),
        torch.tensor(0, device=up.device, dtype=up.dtype))
    a["xm"] = torch.min(
        torch.min(up-torch.sqrt(hp*g), um-torch.sqrt(hm*g)),
        torch.tensor(0, device=up.device, dtype=up.dtype))
    a["yp"] = torch.max(
        torch.max(vt+torch.sqrt(ht*g), vb+torch.sqrt(hb*g)),
        torch.tensor(0, device=vt.device, dtype=vt.dtype))
    a["ym"] = torch.min(
        torch.min(vt-torch.sqrt(ht*g), vb-torch.sqrt(hb*g)),
        torch.tensor(0, device=vt.device, dtype=vt.dtype))

    return a
