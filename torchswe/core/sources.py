#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Source terms.
"""


def topography_gradient(w, dtopo_x, dtopo_y, topo_cntr, n_ghost, gravity):
    """Momentim sources due to topographic changes.

    Notes
    -----
    - The mesh arrangement must be Y first and then X. For example, w[j, i] is the w of j-th
      cell in y-direction and i-th cell in x-direction.
    - topo_cntr must be from the bilinear interpolation of elevation at vertices.
    - dtopo_x & dtopo_y must be calculated from central difference using the elevations at vertices.

    Arguments
    ---------
    w : (Ny+2*n_ghost, Nx+2*n_ghost) numpy.ndarray
        Water elevation, including ghost cells.
    dtopo_x, dtopo_y : (Ny, Nx) numpy.ndarray
        Topography derivative w.r.t. x and y at cell centers, excluding ghost cells.
    topo_cntr : (Ny, Nx) numpy.ndarray
        Topography elevations at cell centers, excluding ghost cells.
    n_ghost : int
        The number of ghost cell layers outside each boundary
    gravity : float
        Gravity in m/s^2.

    Returns
    -------
    source_hu, source_hv : (Ny, Nx) numpy.ndarray
        The source terms for the momentum eqautions of hu and hv respectively.
    """

    internal = slice(n_ghost, -n_ghost)
    gravity_depth = - gravity * (w[internal, internal] - topo_cntr)
    return dtopo_x * gravity_depth, dtopo_y * gravity_depth
