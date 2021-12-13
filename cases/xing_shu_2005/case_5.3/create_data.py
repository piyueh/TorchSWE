#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Create topography and I.C. file for case 5.3 in Xing and Shu (2005).

Note, the elevation data is defined at vertices, rather than at cell centers.
"""
import pathlib
import numpy
import h5py
from torchswe.utils.config import get_config


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

    # topogeaphy elevation
    x2d, y2d = numpy.meshgrid(x, y)
    topo = 0.8 * numpy.exp(-5.*numpy.power(x2d-0.9, 2)-50.*numpy.power(y2d-0.5, 2))

    # write topography file
    with h5py.File(case.joinpath(config.topo.file), "w") as root:
        root.create_dataset(config.topo.xykeys[0], x.shape, x.dtype, x)
        root.create_dataset(config.topo.xykeys[1], y.shape, y.dtype, y)
        root.create_dataset(config.topo.key, topo.shape, topo.dtype, topo)

    # gridlines at cell centers
    x = (x[1:] + x[:-1]) / 2.
    y = (y[1:] + y[:-1]) / 2.

    # initialize i.c., all zeros
    ic = numpy.zeros((3, ny, nx), dtype=dtype)

    # i.c.: w
    x2d, _ = numpy.meshgrid(x, y)
    ic[0] = 1.0
    ic[0][(x2d >= 0.05)*(x2d <= 0.15)] += 0.01

    # write initial condition file
    with h5py.File(case.joinpath(config.ic.file), "w") as root:
        root.create_dataset(config.ic.xykeys[0], x.shape, x.dtype, x)
        root.create_dataset(config.ic.xykeys[1], y.shape, y.dtype, y)
        for i in range(3):
            root.create_dataset(config.ic.keys[i], ic[i].shape, ic[i].dtype, ic[i])

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
