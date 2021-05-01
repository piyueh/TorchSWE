#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Misc. functions.
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

@torch.jit.script
def decompose_variables(U, B, epsilon: float):
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

@torch.jit.script
def local_speed(hxm, hxp, hym, hyp, uxm, uxp, vym, vyp, g: float):
    """Calculate local speed a.

    Args:
    -----
        hxm: a (Ny, Nx+1) torch.tensor of depth at the left of interfaces.
        hxp: a (Ny, Nx+1) torch.tensor of depth at the right of interfaces.
        hym: a (Ny+1, Nx) torch.tensor of depth at the bottom of interfaces.
        hyp: a (Ny+1, Nx) torch.tensor of depth at the top of interfaces.
        uxm: a (Ny, Nx+1) torch.tensor of u velocity at the left of interfaces.
        uxp: a (Ny, Nx+1) torch.tensor of u velocity at the right of interfaces.
        vym: a (Ny+1, Nx) torch.tensor of v velocity at the bottom of interfaces.
        vyp: a (Ny+1, Nx) torch.tensor of v velocity at the top of interfaces.
        g: gravity

    Returns:
    --------
        axm: a (Ny, Nx+1) torch.tensor representing the local speed at the
            left sides of the cell interfaces normal to x-direction.
        axp: a (Ny, Nx+1) torch.tensor representing the local speed at the
            right sides of the cell interfaces normal to x-direction.
        aym: a (Ny+1, Nx) torch.tensor representing the local speed at the
            bottom sides of the cell interfaces normal to y-direction.
        ayp: a (Ny+1, Nx) torch.tensor representing the local speed at the
            top sides of the cell interfaces normal to y-direction.
    """

    hpg_sqrt = torch.sqrt(hxp*g)
    hmg_sqrt = torch.sqrt(hxm*g)
    hbg_sqrt = torch.sqrt(hym*g)
    htg_sqrt = torch.sqrt(hyp*g)
    zero_tensor = torch.tensor(0, device=hxm.device, dtype=hxm.dtype)

    axm = torch.min(torch.min(uxp-hpg_sqrt, uxm-hmg_sqrt), zero_tensor)
    axp = torch.max(torch.max(uxp+hpg_sqrt, uxm+hmg_sqrt), zero_tensor)
    aym = torch.min(torch.min(vyp-htg_sqrt, vym-hbg_sqrt), zero_tensor)
    ayp = torch.max(torch.max(vyp+htg_sqrt, vym+hbg_sqrt), zero_tensor)

    return axm, axp, aym, ayp
