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
from torchswe.utils.config import Config
from torchswe.utils.netcdf import write_cf


def main():
    """Main function"""
    # pylint: disable=invalid-name

    case = pathlib.Path(__file__).expanduser().resolve().parent

    with open(case.joinpath("config.yaml"), 'r') as f:
        config: Config = yaml.load(f, Loader=yaml.Loader)

    x = numpy.linspace(
        config.spatial.domain[0], config.spatial.domain[1],
        config.spatial.discretization[0]+1, dtype=numpy.float64)
    y = numpy.linspace(
        config.spatial.domain[2], config.spatial.domain[3],
        config.spatial.discretization[1]+1, dtype=numpy.float64)

    B = numpy.tile(5.*numpy.exp(-0.4*((x-5.)**2)), (config.spatial.discretization[1]+1, 1))

    write_cf(
        case.joinpath(config.topo.file), {"x": x, "y": y},
        {config.topo.key: B},
        options={config.topo.key: {"units": "m"}})

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
