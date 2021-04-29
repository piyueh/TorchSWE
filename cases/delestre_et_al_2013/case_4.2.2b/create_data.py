#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the MIT license.

"""
Create topography and I.C. file for case 4.2.2-2 in Delestre et al., 2013.

Note, the elevation data in the resulting NetCDF file is defined at vertices,
instead of cell centers. But the I.C. is defined at cell centers.
"""
import pathlib
import yaml
import numpy
from torchswe.utils.netcdf import write_cf


def topo(x, y, h0=0.1, L=4., a=1.):
    """Topography."""
    # pylint: disable=invalid-name

    r = numpy.sqrt((x-L/2.)**2+(y-L/2.)**2)
    return - h0 * (1. - (r / a)**2)


def exact_soln(x, y, t, g=9.81, h0=0.1, L=4., a=1., eta=0.5):
    """Exact solution."""
    # pylint: disable=invalid-name, too-many-arguments

    omega = numpy.sqrt(2.*h0*g) / a
    z = topo(x, y, h0, L, a)
    cot = numpy.cos(omega*t)
    sot = numpy.sin(omega*t)

    h = eta * h0 * (2 * (x - L / 2) * cot + 2 * (y - L / 2.) * sot - eta) / (a * a) - z
    h[h < 0.] = 0.

    return h + z, - h * eta * omega * sot, h * eta * omega * cot


def main():
    """Main function"""
    # pylint: disable=invalid-name

    case = pathlib.Path(__file__).expanduser().resolve().parent

    with open(case.joinpath("config.yaml"), 'r') as f:
        config = yaml.load(f, Loader=yaml.CLoader)

    x = numpy.linspace(
        config["domain"]["west"], config["domain"]["east"],
        config["discretization"]["Nx"]+1, dtype=numpy.float64)
    y = numpy.linspace(
        config["domain"]["south"], config["domain"]["north"],
        config["discretization"]["Ny"]+1, dtype=numpy.float64)

    # 2D X, Y for temporarily use
    X, Y = numpy.meshgrid(x, y)

    # topogeaphy elevation
    B = topo(X, Y)

    # write topography file
    write_cf(
        case.joinpath(config["topography"]["file"]), x, y,
        {config["topography"]["key"]: B},
        options={config["topography"]["key"]: {"units": "m"}})

    # x and y for cell centers
    xc = (x[:-1] + x[1:]) / 2.
    yc = (y[:-1] + y[1:]) / 2.
    Xc, Yc = numpy.meshgrid(xc, yc)

    # I.C.: w, hu, hv
    w, hu, hv = exact_soln(Xc, Yc, 0.)

    # write I.C. file
    write_cf(
        case.joinpath(config["ic"]["file"]), xc, yc,
        dict(zip(config["ic"]["keys"], [w, hu, hv])),
        options=dict(zip(config["ic"]["keys"], [
            {"units": "m"}, {"units": "m2 s-1"}, {"units": "m2 s-1"}])))

    return 0


if __name__ == "__main__":

    # execute the main function
    import sys
    sys.exit(main())
