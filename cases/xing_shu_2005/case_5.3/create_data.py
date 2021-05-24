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

    # topogeaphy elevation
    X, Y = numpy.meshgrid(grid.x.vert, grid.y.vert)
    B = 0.8 * numpy.exp(-5.*numpy.power(X-0.9, 2)-50.*numpy.power(Y-0.5, 2))

    # write topography file
    create_topography_file(case.joinpath(config.topo.file), [grid.x.vert, grid.y.vert], B)

    # initialize i.c., all zeros
    ic = get_empty_whuhvmodel(*config.spatial.discretization, config.dtype)

    # I.C.: w
    Xc, _ = numpy.meshgrid(grid.x.cntr, grid.y.cntr)
    ic.w[...] = 1.0
    ic.w[(Xc >= 0.05)*(Xc <= 0.15)] += 0.01

    # write I.C. file
    ic.check()
    create_soln_snapshot_file(case.joinpath(config.ic.file), grid, ic)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
