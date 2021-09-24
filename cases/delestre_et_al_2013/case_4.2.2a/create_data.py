#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Create topography and I.C. file for case 4.2.2-1 in Delestre et al., 2013.

Note, the elevation data in the resulting NetCDF file is defined at vertices,
instead of cell centers. But the I.C. is defined at cell centers.
"""
import pathlib
import yaml
import numpy
from mpi4py import MPI
from torchswe.utils.init import get_empty_whuhvmodel, get_process, get_gridline, get_domain
from torchswe.utils.io import create_soln_snapshot_file, create_topography_file


def topo(x, y, h0=0.1, L=4., a=1.):
    """Topography."""
    # pylint: disable=invalid-name

    r = numpy.sqrt((x-L/2.)**2+(y-L/2.)**2)
    return - h0 * (1. - (r / a)**2)


def exact_soln(x, y, t, g=9.81, h0=0.1, L=4., a=1., r0=0.8):
    """Exact solution."""
    # pylint: disable=invalid-name, too-many-arguments

    omega = numpy.sqrt(8.*h0*g) / a
    A = (a**2 - r0**2) / (a**2 + r0**2)
    z = topo(x, y, h0, L, a)

    C0 = 1. - A * numpy.cos(omega*t)
    C1 = numpy.sqrt((1.-A*A)) / C0
    C2 = (1. - A * A) / (C0**2)

    h = h0 * (C1 - 1. - ((numpy.sqrt((x-L/2.)**2+(y-L/2.)**2) / a)**2) * (C2 - 1.)) - z
    h[h < 0.] = 0.

    return \
        h + z, \
        h * 0.5 * omega * A * (x - L/2.) * numpy.sin(omega*t) / C0, \
        h * 0.5 * omega * A * (y - L/2.) * numpy.sin(omega*t) / C0


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
        case.joinpath(config.topo.file), [domain.x.vertices, domain.x.vertices],
        topo(*numpy.meshgrid(domain.x.vertices, domain.x.vertices)))

    # initial conditions, defined on cell centers
    ic = get_empty_whuhvmodel(*config.spatial.discretization, config.dtype)
    ic.w, ic.hu, ic.hv = exact_soln(*numpy.meshgrid(domain.x.centers, domain.x.centers), 0.)
    ic.check()
    create_soln_snapshot_file(case.joinpath(config.ic.file), domain, ic)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
