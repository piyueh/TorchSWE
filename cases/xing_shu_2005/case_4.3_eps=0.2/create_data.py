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
from mpi4py import MPI
from torchswe.utils.init import get_empty_whuhvmodel, get_process, get_gridline, get_domain
from torchswe.utils.io import create_soln_snapshot_file, create_topography_file


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

    # create 1D version of B first
    B = numpy.zeros_like(domain.x.vertices)
    loc = (domain.x.vertices >= 1.4) * (domain.x.vertices <= 1.6)  # this is logical and
    B[loc] = (numpy.cos(10.*numpy.pi*(domain.x.vertices[loc]-1.5)) + 1.) / 4.
    B = numpy.tile(B, (domain.y.n+1, 1))  # make it 2D

    # write topography file
    create_topography_file(
        case.joinpath(config.topo.file), [domain.x.vertices, domain.y.vertices], B)

    # initialize i.c., all zeros
    ic = get_empty_whuhvmodel(*config.spatial.discretization, config.dtype)

    # i.c.: w
    Xc, _ = numpy.meshgrid(domain.x.centers, domain.y.centers)
    ic.w = numpy.ones_like(Xc)
    ic.w[(Xc >= 1.1) * (Xc <= 1.2)] += 0.2

    # write I.C. file
    create_soln_snapshot_file(case.joinpath(config.ic.file), domain, ic)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
