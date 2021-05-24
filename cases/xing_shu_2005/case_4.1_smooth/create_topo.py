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
from torchswe.utils.data import get_gridlines
from torchswe.utils.io import create_topography_file


def main():
    """Main function"""
    # pylint: disable=invalid-name

    case = pathlib.Path(__file__).expanduser().resolve().parent

    with open(case.joinpath("config.yaml"), 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    # gridlines; ignore temporal axis
    grid = get_gridlines(*config.spatial.discretization, *config.spatial.domain, [], config.dtype)


    # topography, defined on cell vertices
    B = numpy.tile(5.*numpy.exp(-0.4*((grid.x.vert-5.)**2)), (grid.y.n+1, 1))
    create_topography_file(case.joinpath(config.topo.file), [grid.x.vert, grid.y.vert], B)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
