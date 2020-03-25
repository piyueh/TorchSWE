#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the MIT license.

"""
Functions related to updating ghost cells, i.e., boundary conditions.
"""
import torch

# to create corresponding slices for the left-hand-side of periodic BC
_periodic_slc_left = {
    "west": lambda n: [slice(None), slice(n, -n), slice(0, n)],
    "east": lambda n: [slice(None), slice(n, -n), slice(-n, None)],
    "north": lambda n: [slice(None), slice(-n, None), slice(n, -n)],
    "south": lambda n: [slice(None), slice(0, n), slice(n, -n)]
}

# to create corresponding slices for the left-hand-side of periodic BC
_periodic_slc_right = {
    "west": lambda n: [slice(None), slice(n, -n), slice(-n-n, -n)],
    "east": lambda n: [slice(None), slice(n, -n), slice(n, n+n)],
    "north": lambda n: [slice(None), slice(n, n+n), slice(n, -n)],
    "south": lambda n: [slice(None), slice(-n-n, -n), slice(n, -n)]
}

_extrap_seq = {
    "west": lambda n, dev: torch.arange(n, 0, -1, device=dev),
    "east": lambda n, dev: torch.arange(1, n+1, device=dev),
    "north": lambda n, dev: torch.arange(1, n+1, device=dev).reshape((n, 1)),
    "south": lambda n, dev: torch.arange(n, 0, -1, device=dev).reshape((n, 1))
}

_extrap_anchor = {
    "west": lambda n: [slice(None), slice(n, -n), slice(n, n+1)],
    "east": lambda n: [slice(None), slice(n, -n), slice(-n-1, -n)],
    "north": lambda n: [slice(None), slice(-n-1, -n), slice(n, -n)],
    "south": lambda n: [slice(None), slice(n, n+1), slice(n, -n)]
}

_extrap_delta_slc = {
    "west": lambda n: [slice(None), slice(n, -n), slice(n+1, n+2)],
    "east": lambda n: [slice(None), slice(n, -n), slice(-n-2, -n-1)],
    "north": lambda n: [slice(None), slice(-n-2, -n-1), slice(n, -n)],
    "south": lambda n: [slice(None), slice(n+1, n+2), slice(n, -n)]
}

_extrap_slc = {
    "west": lambda n: [slice(None), slice(n, -n), slice(0, n)],
    "east": lambda n: [slice(None), slice(n, -n), slice(-n, None)],
    "north": lambda n: [slice(None), slice(-n, None), slice(n, -n)],
    "south": lambda n: [slice(None), slice(0, n), slice(n, -n)]
}

_const_slc = {
    "west": lambda n: [slice(None), slice(n, -n), slice(0, n)],
    "east": lambda n: [slice(None), slice(n, -n), slice(-n, None)],
    "north": lambda n: [slice(None), slice(-n, None), slice(n, -n)],
    "south": lambda n: [slice(None), slice(0, n), slice(n, -n)]
}

_allowed_orient = ["west", "east", "north", "south"]
_allowed_bc_type = ["periodic", "extrapolation", "constant"]


def periodic_factory(Ngh, orientation):
    """A function factory to create a ghost-cell updating function.

    Args:
    -----
        Ngh: an integer for the number of ghost cells outside each boundary.
        orientation: a string of one of the following orientation -- "west",
            "east", "north", or "south".

    Returns:
    --------
        A function with a signature of f(torch.tensor) -> torch.tensor. Both the
        input and output tensors have a shape of (3, Ny+2*Ngh, Nx+2*Ngh).
    """

    slc_left = _periodic_slc_left[orientation](Ngh)
    slc_right = _periodic_slc_right[orientation](Ngh)

    def periodic(U):
        """Update the ghost cells with periodic BC.

        Note we ignore the ghost cells at corners. They shouldn't be used in this
        numerical method.

        Theoretically speaking, U is modified in-place, but we still return it.

        This function has two member attributes: slc_left and slc_right, which
        are lists of three Python native slice object. Users can manually check
        the values of these slices for debugging.

        Args:
        -----
            U: a (3, Ny+2*Ngh, Nx+2*Ngh) torch.tensor of conservative variables.

        Returns:
        --------
            Updated U.
        """

        U[slc_left] = U[slc_right]

        return U

    periodic.slc_left = slc_left
    periodic.slc_right = slc_right

    return periodic

def linear_extrap_factory(Ngh, orientation, device):
    """A function factory to create a ghost-cell updating function.

    Args:
    -----
        Ngh: an integer for the number of ghost cells outside each boundary.
        orientation: a string of one of the following orientation -- "west",
            "east", "north", or "south".
        device: where new tensors will be created.

    Returns:
    --------
        A function with a signature of f(torch.tensor) -> torch.tensor. Both the
        input and output tensors have a shape of (3, Ny+2*Ngh, Nx+2*Ngh).
    """

    seq = _extrap_seq[orientation](Ngh, device)
    anchor = _extrap_anchor[orientation](Ngh)
    delta_slc = _extrap_delta_slc[orientation](Ngh)
    slc = _extrap_slc[orientation](Ngh)

    def linear_extrap(U):
        """Update the ghost cells with outflow BC using linear extrapolation.

        Ghost cells should have the same cell size as the first/last interior
        cell does. Note we ignore the ghost cells at corners. They shouldn't be
        used in this numerical method.

        Theoretically speaking, U is modified in-place, but we still return it.

        This function has two member attributes: slc_left and slc_right, which
        are lists of three Python native slice object. Users can manually check
        the values of these slices for debugging.

        Args:
        -----
            U: a (3, Ny+2*Ngh, Nx+2*Ngh) torch.tensor of conservative variables.

        Returns:
        --------
            Updated U.
        """

        # west
        delta = U[anchor] - U[delta_slc]
        U[slc] = U[anchor] + seq * delta

        return U

    linear_extrap.seq = seq
    linear_extrap.anchor = anchor
    linear_extrap.delta_slc = delta_slc
    linear_extrap.slc = slc

    return linear_extrap

def constant_bc_factory(const, Ngh, orientation):
    """A function factory to create a ghost-cell updating function.

    Args:
    -----
        const: a length-3 1D torch.tensor of the constant values of w, hu, and
            hv.
        Ngh: an integer for the number of ghost cells outside each boundary.
        orientation: a string of one of the following orientation -- "west",
            "east", "north", or "south".
        device: where new tensors will be created.

    Returns:
    --------
        A function with a signature of f(torch.tensor) -> torch.tensor. Both the
        input and output tensors have a shape of (3, Ny+2*Ngh, Nx+2*Ngh).
    """

    const = const.reshape((-1, 1, 1))
    slc = _const_slc[orientation](Ngh)

    def constant_bc(U):
        """Update the ghost cells with constant BC using linear extrapolation.

        Ghost cells should have the same cell size as the first/last interior
        cell does. Note we ignore the ghost cells at corners. They shouldn't be
        used in this numerical method.

        Theoretically speaking, U is modified in-place, but we still return it.

        This function has two member attributes: slc_left and slc_right, which
        are lists of three Python native slice object. Users can manually check
        the values of these slices for debugging.

        Args:
        -----
            U: a (3, Ny+2*Ngh, Nx+2*Ngh) torch.tensor of conservative variables.

        Returns:
        --------
            Updated U.
        """

        U[slc] = const

        return U

    constant_bc.const = const
    constant_bc.slc = slc

    return constant_bc

def update_all_factory(bcs, Ngh, device=None, const={}):
    """A factory to create a update-all ghost-cell updating function.

    Args:
    -----
        bcs: a dictionary of the following four key-value pair -- ("west", type),
            ("east", type), ("north", type), and ("south", type). The value of
            type can be found in _allowed_bc_type.
        Ngh: number of ghost cell layers at each boundary.
        device: if any of the boundaries has extrapolation BC, device must be
            specified. Otherwise, the value of device is not used.
        const: a dictionary of (orientation, const) pairs. orientation is the
            key in bcs where a constant BC is requested. const is a length-3 1D
            torch.tensor. If none of the BCs is constant BC, then this argument
            does not matter.

    Returns:
    --------
        A function with a signature of f(torch.tensor) -> torch.tensor. Both the
        input and output tensors have a shape of (3, Ny+2*Ngh, Nx+2*Ngh).
    """

    functions = {}
    for key, value in bcs.items():

        # sanity check
        assert key in _allowed_orient

        if value == "periodic":
            functions[key] = periodic_factory(Ngh, key)
        elif value == "extrapolation":
            if device is None:
                raise ValueError(
                    "Extrapolation BC requires a valied device argument.")
            functions[key] = linear_extrap_factory(Ngh, key, device)
        elif value == "constant":
            if key not in const:
                raise ValueError(
                    "Can't find corresponding constant values in the argument const.")
            functions[key] = constant_bc_factory(const[key], Ngh, key)
        else:
            raise ValueError("{} is not recognized.".format(value))

    if bcs["west"] == "periodic":
        assert bcs["east"] == "periodic"

    if bcs["east"] == "periodic":
        assert bcs["west"] == "periodic"

    if bcs["north"] == "periodic":
        assert bcs["south"] == "periodic"

    if bcs["south"] == "periodic":
        assert bcs["north"] == "periodic"

    def update_all(U):
        """Update the ghost cells at all boundaries at once.

        Ghost cells should have the same cell size as the first/last interior
        cell does. Note we ignore the ghost cells at corners. They shouldn't be
        used in this numerical method.

        Theoretically speaking, U is modified in-place, but we still return it.

        Individual updating function of each boundary can be found in the
        member attribute, functions, which is a dictionary.

        Args:
        -----
            U: a (3, Ny+2*Ngh, Nx+2*Ngh) torch.tensor of conservative variables.

        Returns:
        --------
            Updated U.
        """

        for func in functions.values():
            U = func(U)

        return U

    update_all.functions = functions

    return update_all
