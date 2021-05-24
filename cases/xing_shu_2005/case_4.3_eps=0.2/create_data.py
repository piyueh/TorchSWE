#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Create topography and I.C. file for case 4.3 and epsilon=0.2 in Xing and Shu (2005).

Note, the elevation data in the resulting NetCDF file is defined at vertices,
instead of cell centers. But the I.C. is defined at cell centers.
"""
import pathlib
import yaml
import numpy
from torchswe.utils.data import get_empty_whuhvmodel, get_gridlines
from torchswe.utils.io import create_soln_snapshot_file, create_topography_file


def main():
    """Main function"""
    # pylint: disable=invalid-name

    case = pathlib.Path(__file__).expanduser().resolve().parent

    with open(case.joinpath("config.yaml"), 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    # gridlines; ignore temporal axis
    grid = get_gridlines(*config.spatial.discretization, *config.spatial.domain, [], config.dtype)

    # create 1D version of B first
    B = numpy.zeros_like(grid.x.vert)
    loc = (grid.x.vert >= 1.4) * (grid.x.vert <= 1.6)  # use multiplication to do logical_and
    B[loc] = (numpy.cos(10.*numpy.pi*(grid.x.vert[loc]-1.5))+1.) / 4.
    B = numpy.tile(B, (grid.y.n+1, 1))  # make it 2D

    # write topography file
    create_topography_file(case.joinpath(config.topo.file), [grid.x.vert, grid.y.vert], B)

    # initialize i.c., all zeros
    ic = get_empty_whuhvmodel(*config.spatial.discretization, config.dtype)

    # i.c.: w
    Xc, _ = numpy.meshgrid(grid.x.cntr, grid.y.cntr)
    ic.w = numpy.ones_like(Xc)
    ic.w[(Xc >= 1.1) * (Xc <= 1.2)] += 0.2

    # write I.C. file
    create_soln_snapshot_file(case.joinpath(config.ic.file), grid, ic)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
