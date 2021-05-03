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
from torchswe.utils.config import Config
from torchswe.utils.netcdf import write_cf
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

    case = pathlib.Path(__file__).expanduser().resolve().parent

    with open(case.joinpath("config.yaml"), 'r') as f:
        config: Config = yaml.load(f, Loader=yaml.Loader)

    x = numpy.linspace(
        config.spatial.domain[0], config.spatial.domain[1],
        config.spatial.discretization[0]+1, dtype=numpy.float64)
    y = numpy.linspace(
        config.spatial.domain[2], config.spatial.domain[3],
        config.spatial.discretization[1]+1, dtype=numpy.float64)

    # 2D X, Y for temporarily use
    X, Y = numpy.meshgrid(x, y)

    # topogeaphy elevation
    B = topo(X, Y)

    # write topography file
    write_cf(
        case.joinpath(config.topo.file), {"x": x, "y": y},
        {config.topo.key: B}, options={config.topo.key: {"units": "m"}})

    # x and y for cell centers
    xc = (x[:-1] + x[1:]) / 2.
    yc = (y[:-1] + y[1:]) / 2.
    Xc, Yc = numpy.meshgrid(xc, yc)
    Bc = (B[:-1, :-1] + B[1:, :-1] + B[:-1, 1:] + B[1:, 1:]) / 4.
    assert Bc.shape == Xc.shape, "{} vs {}".format(Bc.shape, Xc.shape)

    # I.C.: w
    w = numpy.maximum(Bc, 0.25)

    # I.C.: hu & hv
    hu = numpy.zeros_like(Bc)
    hv = numpy.zeros_like(Bc)

    loc = (numpy.abs(Yc) <= 0.5)
    hu[loc] = (w[loc] - Bc[loc]) * 0.5

    # write I.C. file
    write_cf(
        case.joinpath(config.ic.file), {"x": xc, "y": yc}, dict(zip(config.ic.keys, [w, hu, hv])),
        options=dict(
            zip(config.ic.keys, [{"units": "m"}, {"units": "m2 s-1"}, {"units": "m2 s-1"}])))


if __name__ == "__main__":
    import sys
    sys.exit(main())
