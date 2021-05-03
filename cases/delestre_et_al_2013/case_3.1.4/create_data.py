#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Create topography for case 3.1.4 in Delestre et al. (2013).

Note, the elevation data in the resulting NetCDF file is defined at vertices,
instead of cell centers.
"""
import pathlib
import yaml
import numpy
from torchswe.utils.config import Config
from torchswe.utils.netcdf import write_cf


def main():
    """Main function"""

    case = pathlib.Path(__file__).expanduser().resolve().parent

    with open(case.joinpath("config.yaml"), 'r') as fobj:
        config: Config = yaml.load(fobj, Loader=yaml.Loader)

    x = numpy.linspace(
        config.spatial.domain[0], config.spatial.domain[1],
        config.spatial.discretization[0]+1, dtype=numpy.float64)
    y = numpy.linspace(
        config.spatial.domain[2], config.spatial.domain[3],
        config.spatial.discretization[1]+1, dtype=numpy.float64)

    # create 1D version of B first
    topo_vert = numpy.zeros_like(x)

    loc = (x >= 8.) * (x <= 12.)
    topo_vert[loc] = 0.2 - 0.05 * numpy.power(x[loc]-10., 2)

    # make it 2D
    topo_vert = numpy.tile(topo_vert, (config.spatial.discretization[1]+1, 1))

    # write topography file
    write_cf(
        case.joinpath(config.topo.file), {"x": x, "y": y},
        {config.topo.key: topo_vert},
        options={config.topo.key: {"units": "m"}})

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
