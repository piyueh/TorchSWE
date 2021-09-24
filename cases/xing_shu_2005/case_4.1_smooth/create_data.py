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
from mpi4py import MPI
from torchswe.utils.init import get_process, get_gridline, get_domain
from torchswe.utils.io import create_topography_file


def main():
    """Main function"""
    # pylint: disable=invalid-name

    size = MPI.COMM_WORLD.Get_size()
    assert size == 1, "This script expects non-parallel execution environment."

    case = pathlib.Path(__file__).expanduser().resolve().parent

    with open(case.joinpath("config.yaml"), 'r', encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    # gridlines
    spatial = config.spatial
    domain = get_domain(
        process=get_process(MPI.COMM_WORLD, *spatial.discretization),
        x=get_gridline("x", 1, 0, spatial.discretization[0], *spatial.domain[:2], config.dtype),
        y=get_gridline("y", 1, 0, spatial.discretization[1], *spatial.domain[2:], config.dtype)
    )


    # topography, defined on cell vertices
    create_topography_file(
        case.joinpath(config.topo.file),
        [domain.x.vertices, domain.y.vertices],
        numpy.tile(5.*numpy.exp(-0.4*((domain.x.vertices-5.)**2)), (domain.y.n+1, 1))
    )

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
