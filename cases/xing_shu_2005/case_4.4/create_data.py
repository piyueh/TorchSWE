#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Create topography and I.C. file for case 4.4 in Xing and Shu (2005).

Note, the elevation data in the resulting NetCDF file is defined at vertices,
instead of cell centers. But the I.C. is defined at cell centers.
"""
import pathlib
import yaml
import numpy
from torchswe.utils.data import get_gridlines, get_empty_whuhvmodel
from torchswe.utils.io import create_topography_file, create_soln_snapshot_file


def main():
    """Main function"""
    # pylint: disable=invalid-name

    case = pathlib.Path(__file__).expanduser().resolve().parent

    with open(case.joinpath("config.yaml"), 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    # gridlines; ignore temporal axis
    grid = get_gridlines(*config.spatial.discretization, *config.spatial.domain, [], config.dtype)

    # create 1D version of B first and then tile it to 2D
    B = numpy.zeros_like(grid.x.vert)
    B = numpy.where(numpy.abs(grid.x.vert-750.) <= 1500./8., 8, B)
    B = numpy.tile(B, (grid.y.n+1, 1))

    # write topography file
    create_topography_file(case.joinpath(config.topo.file), [grid.x.vert, grid.y.vert], B)

    # initialize i.c., all zeros
    ic = get_empty_whuhvmodel(*config.spatial.discretization, config.dtype)

    # i.c.: w
    Xc, _ = numpy.meshgrid(grid.x.cntr, grid.y.cntr)
    ic.w = numpy.where(Xc <= 750., 20., 15.)

    # write I.C. file
    ic.check()
    create_soln_snapshot_file(case.joinpath(config.ic.file), grid, ic)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
