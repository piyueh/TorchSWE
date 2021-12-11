#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Create topography and I.C. file for case 4.2.2-2 in Delestre et al., 2013.

Note, the elevation data is defined at vertices, rather than at cell centers.
"""
import pathlib
import numpy
import h5py
from torchswe.utils.config import get_config


def get_topo(x, y, h0=0.1, L=4., a=1.):
    """Topography."""
    # pylint: disable=invalid-name

    r = numpy.sqrt((x-L/2.)**2+(y-L/2.)**2)
    return - h0 * (1. - (r / a)**2)


def exact_soln(x, y, t, g=9.81, h0=0.1, L=4., a=1., eta=0.5):
    """Exact solution."""
    # pylint: disable=invalid-name, too-many-arguments

    omega = numpy.sqrt(2.*h0*g) / a
    z = get_topo(x, y, h0, L, a)
    cot = numpy.cos(omega*t)
    sot = numpy.sin(omega*t)

    h = eta * h0 * (2 * (x - L / 2) * cot + 2 * (y - L / 2.) * sot - eta) / (a * a) - z
    h[h < 0.] = 0.

    return numpy.concatenate((
        (h + z)[None, ...],
        (- h * eta * omega * sot)[None, ...],
        (h * eta * omega * cot)[None, ...]
    ), 0)


def main():
    """Main function"""
    # pylint: disable=invalid-name

    case = pathlib.Path(__file__).expanduser().resolve().parent
    config = get_config(case)

    # aliases
    nx, ny = config.spatial.discretization
    dtype = config.params.dtype

    # gridlines at vertices
    x = numpy.linspace(*config.spatial.domain[:2], nx+1, dtype=dtype)
    y = numpy.linspace(*config.spatial.domain[2:], ny+1, dtype=dtype)

    # topography
    topo = get_topo(*numpy.meshgrid(x, y))

    # write topography file
    with h5py.File(case.joinpath(config.topo.file), "w") as root:
        root.create_dataset(config.topo.xykeys[0], x.shape, x.dtype, x)
        root.create_dataset(config.topo.xykeys[1], y.shape, y.dtype, y)
        root.create_dataset(config.topo.key, topo.shape, topo.dtype, topo)

    # gridlines at cell centers
    x = (x[1:] + x[:-1]) / 2.
    y = (y[1:] + y[:-1]) / 2.

    # initial conditions, defined on cell centers
    ic = exact_soln(*numpy.meshgrid(x, y), 0.)

    # write topography file
    with h5py.File(case.joinpath(config.ic.file), "w") as root:
        root.create_dataset(config.ic.xykeys[0], x.shape, x.dtype, x)
        root.create_dataset(config.ic.xykeys[1], y.shape, y.dtype, y)
        for i in range(3):
            root.create_dataset(config.ic.keys[i], ic[i].shape, ic[i].dtype, ic[i])

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
