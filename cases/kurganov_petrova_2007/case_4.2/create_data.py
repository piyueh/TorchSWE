#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Create topography and I.C. file for case 5.3 in Xing and Shu (2005).

Note, the elevation data is defined at vertices, rather than at cell centers.
"""
import pathlib
import numpy
import h5py
from torchswe.utils.config import get_config
# pylint: disable=invalid-name, too-many-locals


def get_topo(x, y):
    """Topography."""
    topo = 7. * numpy.exp(-8.*numpy.power(x-0.3, 2)-60.*numpy.power(y-0.1, 2)) / 32.
    topo -= (1. * numpy.exp(-30.*numpy.power(x+0.1, 2)-90.*numpy.power(y+0.2, 2)) / 8.)

    loc = numpy.logical_and(numpy.abs(y) <= 0.5, (x-(y-1.)/2.) <= 0.)
    topo[loc] += numpy.power(y[loc], 2)

    loc = numpy.logical_and(numpy.abs(y) > 0.5, (x-(y-1.)/2.) <= 0.)
    topo[loc] += (numpy.power(y[loc], 2) + numpy.sin(numpy.pi*x[loc]) / 10.)

    loc = (x - (y - 1.) / 2. > 0.)
    topo[loc] += numpy.maximum(numpy.power(y[loc], 2)+numpy.sin(numpy.pi*x[loc])/10., 1./8.)

    return topo


def main():
    """Main function"""

    case = pathlib.Path(__file__).expanduser().resolve().parent
    config = get_config(case)

    # aliases
    nx, ny = config.spatial.discretization
    dtype = config.params.dtype

    # gridlines at vertices
    x = numpy.linspace(*config.spatial.domain[:2], nx+1, dtype=dtype)
    y = numpy.linspace(*config.spatial.domain[2:], ny+1, dtype=dtype)

    # write topography at vertices
    topo = get_topo(*numpy.meshgrid(x, y))

    # write topography file
    with h5py.File(case.joinpath(config.topo.file), "w") as root:
        root.create_dataset(config.topo.xykeys[0], x.shape, x.dtype, x)
        root.create_dataset(config.topo.xykeys[1], y.shape, y.dtype, y)
        root.create_dataset(config.topo.key, topo.shape, topo.dtype, topo)

    # gridlines at cell centers
    x = (x[1:] + x[:-1]) / 2.
    y = (y[1:] + y[:-1]) / 2.

    # topography elevation at cell centers
    _, y2d = numpy.meshgrid(x, y)
    topo = (topo[:-1, :-1] + topo[1:, :-1] + topo[:-1, 1:] + topo[1:, 1:]) / 4.  # cell center topo

    # i.c., all zeros
    ic = numpy.zeros((3, ny, nx), dtype=dtype)

    # i.c.: w
    ic[0] = numpy.maximum(topo, 0.25)

    # i.c.: hu
    loc = (numpy.abs(y2d) <= 0.5)
    ic[1][loc] = (ic[0][loc] - topo[loc]) * 0.5

    # write initial condition file
    with h5py.File(case.joinpath(config.ic.file), "w") as root:
        root.create_dataset(config.ic.xykeys[0], x.shape, x.dtype, x)
        root.create_dataset(config.ic.xykeys[1], y.shape, y.dtype, y)
        for i in range(3):
            root.create_dataset(config.ic.keys[i], ic[i].shape, ic[i].dtype, ic[i])


if __name__ == "__main__":
    import sys
    sys.exit(main())
