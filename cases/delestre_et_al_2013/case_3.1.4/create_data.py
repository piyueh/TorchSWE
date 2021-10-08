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
from torchswe.utils.io import create_topography_file


def main():
    """Main function"""

    case = pathlib.Path(__file__).expanduser().resolve().parent

    with open(case.joinpath("config.yaml"), 'r') as fobj:
        config = yaml.load(fobj, Loader=yaml.Loader)

    # gridlines
    spatial = config.spatial
    x = numpy.linspace(*spatial.domain[:2], spatial.discretization[0]+1, dtype=config.params.dtype)
    y = numpy.linspace(*spatial.domain[2:], spatial.discretization[1]+1, dtype=config.params.dtype)

    # create 1D version of B first
    topo_vert = numpy.zeros_like(x)
    loc = (x >= 8.) * (x <= 12.)
    topo_vert[loc] = 0.2 - 0.05 * numpy.power(x[loc]-10., 2)

    # make it 2D
    topo_vert = numpy.tile(topo_vert, (y.size, 1))

    # write topography file
    create_topography_file(case.joinpath(config.topo.file), (x, y), topo_vert)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
