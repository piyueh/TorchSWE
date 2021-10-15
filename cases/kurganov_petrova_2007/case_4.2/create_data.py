#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Create topography and I.C. file for case 5.3 in Xing and Shu (2005).

Note, the elevation data in the resulting NetCDF file is defined at vertices,
instead of cell centers. But the I.C. is defined at cell centers.
"""
import pathlib
import yaml
import numpy
from torchswe.utils.netcdf import write
# pylint: disable=invalid-name, too-many-locals


def topo(x, y):
    """Topography."""
    B = 7. * numpy.exp(-8.*numpy.power(x-0.3, 2)-60.*numpy.power(y-0.1, 2)) / 32.
    B -= (1. * numpy.exp(-30.*numpy.power(x+0.1, 2)-90.*numpy.power(y+0.2, 2)) / 8.)

    loc = numpy.logical_and(numpy.abs(y) <= 0.5, (x-(y-1.)/2.) <= 0.)
    B[loc] += numpy.power(y[loc], 2)

    loc = numpy.logical_and(numpy.abs(y) > 0.5, (x-(y-1.)/2.) <= 0.)
    B[loc] += (numpy.power(y[loc], 2) + numpy.sin(numpy.pi*x[loc]) / 10.)

    loc = (x - (y - 1.) / 2. > 0.)
    B[loc] += numpy.maximum(numpy.power(y[loc], 2)+numpy.sin(numpy.pi*x[loc])/10., 1./8.)

    return B


def main():
    """Main function"""

    case = pathlib.Path(__file__).expanduser().resolve().parent

    with open(case.joinpath("config.yaml"), 'r', encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    # alias
    nx, ny = config.spatial.discretization
    dtype = config.params.dtype
    xlim, ylim = config.spatial.domain[:2], config.spatial.domain[2:]

    # gridlines at vertices
    x = numpy.linspace(*xlim, nx+1, dtype=dtype)
    y = numpy.linspace(*ylim, ny+1, dtype=dtype)

    # write topography at vertices
    B = topo(*numpy.meshgrid(x, y))
    write(case.joinpath(config.topo.file), (x, y), {"elevation": B})

    # gridlines at cell centers
    dx, dy = (xlim[1] - xlim[0]) / nx,  (ylim[1] - ylim[0]) / ny
    x = numpy.linspace(xlim[0]+dx/2., xlim[1]-dx/2., nx, dtype=dtype)
    y = numpy.linspace(ylim[0]+dy/2., ylim[1]-dy/2., ny, dtype=dtype)

    # topography elevation at cell centers
    _, Yc = numpy.meshgrid(x, y)
    Bc = (B[:-1, :-1] + B[1:, :-1] + B[:-1, 1:] + B[1:, 1:]) / 4.

    # i.c., all zeros
    ic = numpy.zeros((3, ny, nx), dtype=dtype)

    # i.c.: w
    ic[0] = numpy.maximum(Bc, 0.25)

    # i.c.: hu
    loc = (numpy.abs(Yc) <= 0.5)
    ic[1][loc] = (ic[0][loc] - Bc[loc]) * 0.5

    # initial conditions, defined on cell centers
    write(case.joinpath(config.ic.file), (x, y), {"w": ic[0], "hu": ic[1], "hv": ic[2]})


if __name__ == "__main__":
    import sys
    sys.exit(main())
