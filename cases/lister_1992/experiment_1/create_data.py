#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Create topography for an inclined plate with an angle of 2.5 degree..
"""
import pathlib
import numpy
import h5py
from torchswe.utils.config import get_config
# pylint: disable=invalid-name


def main():
    """Main function"""

    case = pathlib.Path(__file__).expanduser().resolve().parent
    config = get_config(case)

    # aliases
    nx, ny = config.spatial.discretization

    # gridlines
    xi = numpy.linspace(1.2, 0.0, nx+1, dtype=config.params.dtype)  # coordinate along the plane
    x = 1. - xi * numpy.cos(numpy.pi*2.5/180.)  # coordinates in flow direction but horizontal
    y = numpy.linspace(-0.3, 0.3, ny+1, dtype=config.params.dtype)

    # elevation
    topo = numpy.tile(xi*numpy.sin(2.5*numpy.pi/180.), (y.size, 1))

    # write topography file
    with h5py.File(case.joinpath(config.topo.file), "w") as root:
        root.create_dataset(config.topo.xykeys[0], x.shape, x.dtype, x)
        root.create_dataset(config.topo.xykeys[1], y.shape, y.dtype, y)
        root.create_dataset(config.topo.key, topo.shape, topo.dtype, topo)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
