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
    "west": lambda n, c: [slice(c, c+1), slice(n, -n), slice(0, n)],
    "east": lambda n, c: [slice(c, c+1), slice(n, -n), slice(-n, None)],
    "north": lambda n, c: [slice(c, c+1), slice(-n, None), slice(n, -n)],
    "south": lambda n, c: [slice(c, c+1), slice(0, n), slice(n, -n)]
}

# to create corresponding slices for the left-hand-side of periodic BC
_periodic_slc_right = {
    "west": lambda n, c: [slice(c, c+1), slice(n, -n), slice(-n-n, -n)],
    "east": lambda n, c: [slice(c, c+1), slice(n, -n), slice(n, n+n)],
    "north": lambda n, c: [slice(c, c+1), slice(n, n+n), slice(n, -n)],
    "south": lambda n, c: [slice(c, c+1), slice(-n-n, -n), slice(n, -n)]
}

_extrap_seq = {
    "west": lambda n, dev: torch.arange(n, 0, -1, device=dev),
    "east": lambda n, dev: torch.arange(1, n+1, device=dev),
    "north": lambda n, dev: torch.arange(1, n+1, device=dev).reshape((n, 1)),
    "south": lambda n, dev: torch.arange(n, 0, -1, device=dev).reshape((n, 1))
}

_extrap_anchor = {
    "west": lambda n, c: [slice(c, c+1), slice(n, -n), slice(n, n+1)],
    "east": lambda n, c: [slice(c, c+1), slice(n, -n), slice(-n-1, -n)],
    "north": lambda n, c: [slice(c, c+1), slice(-n-1, -n), slice(n, -n)],
    "south": lambda n, c: [slice(c, c+1), slice(n, n+1), slice(n, -n)]
}

_extrap_delta_slc = {
    "west": lambda n, c: [slice(c, c+1), slice(n, -n), slice(n+1, n+2)],
    "east": lambda n, c: [slice(c, c+1), slice(n, -n), slice(-n-2, -n-1)],
    "north": lambda n, c: [slice(c, c+1), slice(-n-2, -n-1), slice(n, -n)],
    "south": lambda n, c: [slice(c, c+1), slice(n+1, n+2), slice(n, -n)]
}

_extrap_slc = {
    "west": lambda n, c: [slice(c, c+1), slice(n, -n), slice(0, n)],
    "east": lambda n, c: [slice(c, c+1), slice(n, -n), slice(-n, None)],
    "north": lambda n, c: [slice(c, c+1), slice(-n, None), slice(n, -n)],
    "south": lambda n, c: [slice(c, c+1), slice(0, n), slice(n, -n)]
}

_allowed_orient = ["west", "east", "north", "south"]
_allowed_bc_type = ["periodic", "extrap", "const"]


def periodic_factory(Ngh, orientation, component):
    """A function factory to create a ghost-cell updating function.

    Args:
    -----
        Ngh: an integer for the number of ghost cells outside each boundary.
        orientation: a string of one of the following orientation -- "west",
            "east", "north", or "south".
        component: an integer indicating this BC will be applied to which
            component -- 0 for w, 1 for hu, and 2 for hv.

    Returns:
    --------
        A function with a signature of f(torch.tensor) -> torch.tensor. Both the
        input and output tensors have a shape of (3, Ny+2*Ngh, Nx+2*Ngh).
    """

    slc_left = _periodic_slc_left[orientation](Ngh, component)
    slc_right = _periodic_slc_right[orientation](Ngh, component)

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

def linear_extrap_factory(Ngh, orientation, component, device):
    """A function factory to create a ghost-cell updating function.

    Args:
    -----
        Ngh: an integer for the number of ghost cells outside each boundary.
        orientation: a string of one of the following orientation -- "west",
            "east", "north", or "south".
        component: an integer indicating this BC will be applied to which
            component -- 0 for w, 1 for hu, and 2 for hv.
        device: where new tensors will be created.

    Returns:
    --------
        A function with a signature of f(torch.tensor) -> torch.tensor. Both the
        input and output tensors have a shape of (3, Ny+2*Ngh, Nx+2*Ngh).
    """

    seq = _extrap_seq[orientation](Ngh, device)
    anchor = _extrap_anchor[orientation](Ngh, component)
    delta_slc = _extrap_delta_slc[orientation](Ngh, component)
    slc = _extrap_slc[orientation](Ngh, component)

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

def constant_bc_factory(const, Ngh, orientation, component, device):
    """A function factory to create a ghost-cell updating function.

    Args:
    -----
        const: a scalar of either w, hu, or hv.
        Ngh: an integer for the number of ghost cells outside each boundary.
        orientation: a string of one of the following orientation -- "west",
            "east", "north", or "south".
        component: an integer indicating this BC will be applied to which
            component -- 0 for w, 1 for hu, and 2 for hv.
        device: where new tensors will be created.

    Returns:
    --------
        A function with a signature of f(torch.tensor) -> torch.tensor. Both the
        input and output tensors have a shape of (3, Ny+2*Ngh, Nx+2*Ngh).
    """

    seq = _extrap_seq[orientation](Ngh, device)
    anchor = _extrap_anchor[orientation](Ngh, component)
    slc = _extrap_slc[orientation](Ngh, component)

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

        delta = (const - U[anchor]) * 2.
        U[slc] = U[anchor] + seq * delta

        return U

    constant_bc.const = const
    constant_bc.seq = seq
    constant_bc.anchor = anchor
    constant_bc.slc = slc

    return constant_bc

def update_all_factory(bcs, Ngh, device=None):
    """A factory to create a update-all ghost-cell updating function.

    Args:
    -----
        bcs: the "boundary conditions" node from the YAML config.
        Ngh: number of ghost cell layers at each boundary.
        device: if any of the boundaries has extrapolation/const BC, device must
            be specified. Otherwise, the value of device is not used.

    Returns:
    --------
        A function with a signature of f(torch.tensor) -> torch.tensor. Both the
        input and output tensors have a shape of (3, Ny+2*Ngh, Nx+2*Ngh).
    """

    # check periodicity
    check_periodicity(bcs)

    # initialize varaible
    functions = {
        "west": [None, None, None], "east": [None, None, None],
        "north": [None, None, None], "south": [None, None, None],
    }

    for key, bctypes in bcs.items():

        # sanity check: should be one of west, east, north, south
        assert key in _allowed_orient

        # bctypes: {type: [...], values: [...]}
        for i, bctype in enumerate(bctypes["types"]):

            # periodic BC
            if bctype == "periodic":
                functions[key][i] = periodic_factory(Ngh, key, i)

            # linear extrapolation BC
            elif bctype == "extrap":
                assert device is not None
                functions[key][i] = linear_extrap_factory(Ngh, key, i, device)

            # constant/Dirichlet
            elif bctype == "const":
                assert device is not None
                functions[key][i] = constant_bc_factory(bctypes["values"][i], Ngh, key, i, device)

            # others: error
            else:
                raise ValueError("{} is not recognized.".format(bctype))

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

        for val in functions.values():
            for func in val:
                U = func(U)

        return U

    update_all.functions = functions

    return update_all

def check_periodicity(bcs):
    """Check whether periodic BCs match at corresponding boundary pairs.

    Args:
    -----
        bcs: the "boundary conditions" node from the YAML config.

    Returns:
    --------
        Raise an exception if not matching, otherwise returns nothing.
    """

    result = True

    for types in zip(bcs["west"]["types"], bcs["east"]["types"]):
        if any([t=="periodic" for t in types]):
            result = all([t=="periodic" for t in types])

    for types in zip(bcs["north"]["types"], bcs["south"]["types"]):
        if any([t=="periodic" for t in types]):
            result = all([t=="periodic" for t in types])

    if not result:
        raise ValueError("Periodic BCs do not match at boundaries and components.")

# a simple test
if __name__ == "__main__":

    bcs = {
        "west": {
            "types": ["extrap", "const", "periodic"],
            "values": [None, 1000., None],
        },
        "east": {
            "types": ["const", "extrap", "periodic"],
            "values": [-100., None, None],
        },
        "north": {
            "types": ["extrap", "periodic", "const"],
            "values": [None, None, -3.],
        },
        "south": {
            "types": ["const", "periodic", "extrap"],
            "values": [999., None, None],
        },
    }

    func = update_all_factory(bcs, 2, "cpu")
    U = torch.arange(0, 216, dtype=torch.float64, device="cpu").reshape((3, 8, 9))
    U = func(U)

    ans = torch.tensor(
        [[[   0.,    1., 3936., 3933., 3930., 3927., 3924.,    7.,    8.],
          [   9.,   10., 1978., 1977., 1976., 1975., 1974.,   16.,   17.],
          [  18.,   19.,   20.,   21.,   22.,   23.,   24., -224., -472.],
          [  27.,   28.,   29.,   30.,   31.,   32.,   33., -233., -499.],
          [  36.,   37.,   38.,   39.,   40.,   41.,   42., -242., -526.],
          [  45.,   46.,   47.,   48.,   49.,   50.,   51., -251., -553.],
          [  54.,   55.,   56.,   57.,   58.,   59.,   60.,   61.,   62.],
          [  63.,   64.,   65.,   66.,   67.,   68.,   69.,   70.,   71.]],

         [[  72.,   73.,  110.,  111.,  112.,  113.,  114.,   79.,   80.],
          [  81.,   82.,  119.,  120.,  121.,  122.,  123.,   88.,   89.],
          [3724., 1908.,   92.,   93.,   94.,   95.,   96.,   97.,   98.],
          [3697., 1899.,  101.,  102.,  103.,  104.,  105.,  106.,  107.],
          [3670., 1890.,  110.,  111.,  112.,  113.,  114.,  115.,  116.],
          [3643., 1881.,  119.,  120.,  121.,  122.,  123.,  124.,  125.],
          [ 126.,  127.,   92.,   93.,   94.,   95.,   96.,  133.,  134.],
          [ 135.,  136.,  101.,  102.,  103.,  104.,  105.,  142.,  143.]],

         [[ 144.,  145.,  146.,  147.,  148.,  149.,  150.,  151.,  152.],
          [ 153.,  154.,  155.,  156.,  157.,  158.,  159.,  160.,  161.],
          [ 167.,  168.,  164.,  165.,  166.,  167.,  168.,  164.,  165.],
          [ 176.,  177.,  173.,  174.,  175.,  176.,  177.,  173.,  174.],
          [ 185.,  186.,  182.,  183.,  184.,  185.,  186.,  182.,  183.],
          [ 194.,  195.,  191.,  192.,  193.,  194.,  195.,  191.,  192.],
          [ 198.,  199., -197., -198., -199., -200., -201.,  205.,  206.],
          [ 207.,  208., -585., -588., -591., -594., -597.,  214.,  215.]]],
        dtype=torch.float64, device="cpu")

    assert torch.allclose(U, ans)
