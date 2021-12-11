#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Create topography for case 3.1.5 in Delestre et al. (2013).

Note, the elevation data is defined at vertices, rather than at cell centers.
"""
import pathlib
import numpy
import h5py
from torchswe.utils.config import get_config


def main():
    """Main function"""

    case = pathlib.Path(__file__).expanduser().resolve().parent
    config = get_config(case)

    # alias
    nx, ny = config.spatial.discretization
    dtype = config.params.dtype

    # gridlines at vertices
    x = numpy.linspace(*config.spatial.domain[:2], nx+1, dtype=dtype)
    y = numpy.linspace(*config.spatial.domain[2:], ny+1, dtype=dtype)

    # create 1D version of B first
    topo = numpy.zeros_like(x)
    loc = (x >= 8.) * (x <= 12.)
    topo[loc] = 0.2 - 0.05 * numpy.power(x[loc]-10., 2)

    # make it 2D
    topo = numpy.tile(topo, (y.size, 1))

    # write topography file
    with h5py.File(case.joinpath(config.topo.file), "w") as root:
        root.create_dataset(config.topo.xykeys[0], x.shape, x.dtype, x)
        root.create_dataset(config.topo.xykeys[1], y.shape, y.dtype, y)
        root.create_dataset(config.topo.key, topo.shape, topo.dtype, topo)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
