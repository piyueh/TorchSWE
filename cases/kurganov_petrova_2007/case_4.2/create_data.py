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
from mpi4py import MPI
from torchswe.utils.init import get_empty_whuhvmodel, get_process, get_gridline, get_domain
from torchswe.utils.io import create_soln_snapshot_file, create_topography_file
# pylint: disable=invalid-name, too-many-locals


def topo(x, y):
    """Topography."""
    B = 7. * numpy.exp(-8.*numpy.power(x-0.3, 2)-60.*numpy.power(y-0.1, 2)) / 32.
    B -= (1. * numpy.exp(-30.*numpy.power(x+0.1, 2)-90.*numpy.power(y+0.2, 2)) / 8.)

    loc = numpy.logical_and(numpy.abs(y) <= 0.5, (x-(y-1.)/2.) <= 0.)
    B[loc] += numpy.power(y[loc], 2)

    loc = numpy.logical_and(numpy.abs(y) > 0.5, (x-(y-1.)/2.) <= 0.)
    B[loc] += (numpy.power(y[loc], 2) + numpy.sin(numpy.pi*x[loc]) / 10.)

    loc = (x - (y - 1.) / 2. > 0.)
    B[loc] += numpy.maximum(numpy.power(y[loc], 2)+numpy.sin(numpy.pi*x[loc])/10., 1./8.)

    return B


def main():
    """Main function"""

    size = MPI.COMM_WORLD.Get_size()
    assert size == 1, "This script expects non-parallel execution environment."

    case = pathlib.Path(__file__).expanduser().resolve().parent

    with open(case.joinpath("config.yaml"), 'r', encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    # gridlines
    spatial = config.spatial
    domain = get_domain(
        process=get_process(MPI.COMM_WORLD, *spatial.discretization),
        x=get_gridline("x", 1, 0, spatial.discretization[0], *spatial.domain[:2], config.params.dtype),
        y=get_gridline("y", 1, 0, spatial.discretization[1], *spatial.domain[2:], config.params.dtype)
    )

    # topography, defined on cell vertices
    B = topo(*numpy.meshgrid(domain.x.vertices, domain.y.vertices))

    create_topography_file(
        case.joinpath(config.topo.file), [domain.x.vertices, domain.y.vertices], B)

    # topography elevation at cell centers
    _, Yc = numpy.meshgrid(domain.x.centers, domain.y.centers)
    Bc = (B[:-1, :-1] + B[1:, :-1] + B[:-1, 1:] + B[1:, 1:]) / 4.

    # i.c., all zeros
    ic = get_empty_whuhvmodel(*config.spatial.discretization, config.params.dtype)

    # i.c.: w
    ic.w = numpy.maximum(Bc, 0.25)

    # i.c.: hu
    loc = (numpy.abs(Yc) <= 0.5)
    ic.hu[loc] = (ic.w[loc] - Bc[loc]) * 0.5

    # initial conditions, defined on cell centers
    ic.check()
    create_soln_snapshot_file(case.joinpath(config.ic.file), domain, ic)


if __name__ == "__main__":
    import sys
    sys.exit(main())
