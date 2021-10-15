#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Create a topography file for the case 4.1 and smooth topography in Xing and Shu (2005).

Note, the elevation data in the resulting NetCDF file is defined at vertices,
instead of cell centers.
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

    # write topography
    write(
        case.joinpath(config.topo.file), (x, y),
        {"elevation": numpy.tile(5.*numpy.exp(-0.4*((x-5.)**2)), (ny+1, 1))})

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
