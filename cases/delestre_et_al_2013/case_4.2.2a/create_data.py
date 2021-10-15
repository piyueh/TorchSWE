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
from torchswe.utils.netcdf import write


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

    return numpy.concatenate((
        (h + z)[None, ...],
        (h * 0.5 * omega * A * (x - L/2.) * numpy.sin(omega*t) / C0)[None, ...],
        (h * 0.5 * omega * A * (y - L/2.) * numpy.sin(omega*t) / C0)[None, ...]
    ), 0)


def main():
    """Main function"""
    # pylint: disable=invalid-name

    case = pathlib.Path(__file__).expanduser().resolve().parent

    with open(case.joinpath("config.yaml"), 'r', encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    # alias
    nx, ny = config.spatial.discretization
    dtype = config.params.dtype
    xlim, ylim = config.spatial.domain[:2], config.spatial.domain[2:]

    # gridlines at vertices
    x = numpy.linspace(*xlim, nx+1, dtype=dtype)
    y = numpy.linspace(*ylim, ny+1, dtype=dtype)

    # write topography
    write(case.joinpath(config.topo.file), (x, y), {"elevation": topo(*numpy.meshgrid(x, y))})

    # gridlines at cell centers
    dx, dy = (xlim[1] - xlim[0]) / nx,  (ylim[1] - ylim[0]) / ny
    x = numpy.linspace(xlim[0]+dx/2., xlim[1]-dx/2., nx, dtype=dtype)
    y = numpy.linspace(ylim[0]+dy/2., ylim[1]-dy/2., ny, dtype=dtype)

    # write initial conditions, defined on cell centers
    ic = exact_soln(*numpy.meshgrid(x, y), 0.)
    write(case.joinpath(config.ic.file), (x, y), {"w": ic[0], "hu": ic[1], "hv": ic[2]})

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
