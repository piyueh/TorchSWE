#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Create topography for case 3.1.5 in Delestre et al. (2013).

Note, the elevation data in the resulting NetCDF file is defined at vertices,
instead of cell centers.
"""
import pathlib
import yaml
import numpy
from torchswe.utils.netcdf import write


def main():
    """Main function"""

    case = pathlib.Path(__file__).expanduser().resolve().parent

    with open(case.joinpath("config.yaml"), 'r', encoding="utf-8") as fobj:
        config = yaml.load(fobj, Loader=yaml.Loader)

    # alias
    nx, ny = config.spatial.discretization
    dtype = config.params.dtype
    xlim, ylim = config.spatial.domain[:2], config.spatial.domain[2:]

    # gridlines at vertices
    x = numpy.linspace(*xlim, nx+1, dtype=dtype)
    y = numpy.linspace(*ylim, ny+1, dtype=dtype)

    # create 1D version of B first
    topo_vert = numpy.zeros_like(x)
    loc = (x >= 8.) * (x <= 12.)
    topo_vert[loc] = 0.2 - 0.05 * numpy.power(x[loc]-10., 2)

    # make it 2D
    topo_vert = numpy.tile(topo_vert, (y.size, 1))

    # write topography file
    write(case.joinpath(config.topo.file), (x, y), {"elevation": topo_vert})

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
