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


def main():
    """Main function"""
    # pylint: disable=invalid-name

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

    # topogeaphy elevation
    X, Y = numpy.meshgrid(x, y)
    B = 0.8 * numpy.exp(-5.*numpy.power(X-0.9, 2)-50.*numpy.power(Y-0.5, 2))

    # write topography
    write(case.joinpath(config.topo.file), (x, y), {"elevation": B})

    # gridlines at cell centers
    dx, dy = (xlim[1] - xlim[0]) / nx,  (ylim[1] - ylim[0]) / ny
    x = numpy.linspace(xlim[0]+dx/2., xlim[1]-dx/2., nx, dtype=dtype)
    y = numpy.linspace(ylim[0]+dy/2., ylim[1]-dy/2., ny, dtype=dtype)

    # initialize i.c., all zeros
    ic = numpy.zeros((3, ny, nx), dtype=dtype)

    # I.C.: w
    Xc, _ = numpy.meshgrid(x, y)
    ic[0] = 1.0
    ic[0][(Xc >= 0.05)*(Xc <= 0.15)] += 0.01

    # write I.C. file
    write(case.joinpath(config.ic.file), (x, y), {"w": ic[0], "hu": ic[1], "hv": ic[2]})

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
