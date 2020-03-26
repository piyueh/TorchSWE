#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the MIT license.

"""
Create topography and I.C. file for case 4.4 in Xing and Shu (2005).

Note, the elevation data in the resulting NetCDF file is defined at vertices,
instead of cell centers. But the I.C. is defined at cell centers.
"""
import os
import yaml
import numpy

def main():
    """Main function"""

    # it's users' responsibility to make sure TorchSWE package can be found
    from TorchSWE.utils.netcdf import write_cf

    me = os.path.abspath(__file__)
    case = os.path.dirname(me)

    with open(os.path.join(case, "config.yaml"), 'r') as f:
        config = yaml.load(f, Loader=yaml.CLoader)

    x = numpy.linspace(
        config["domain"]["west"], config["domain"]["east"],
        config["discretization"]["Nx"]+1, dtype=numpy.float64)
    y = numpy.linspace(
        config["domain"]["south"], config["domain"]["north"],
        config["discretization"]["Ny"]+1, dtype=numpy.float64)

    # create 1D version of B first
    B = numpy.zeros_like(x)
    B = numpy.where(numpy.abs(x-750.)<=1500./8., 8, B)

    # make it 2D
    B = numpy.tile(B, (config["discretization"]["Ny"]+1, 1))

    # write topography file
    write_cf(
        os.path.join(case, config["topography"]["file"]), x, y,
        {config["topography"]["key"]: B},
        options={config["topography"]["key"]: {"units": "m"}})

    # x and y for cell centers
    xc = (x[:-1] + x[1:]) / 2.
    yc = (y[:-1] + y[1:]) / 2.

    # I.C.: w
    w = numpy.ones_like(xc)
    w = numpy.where(xc<=750., 20., 15.)
    w = numpy.tile(w, (config["discretization"]["Ny"], 1))

    # I.C.: hu & hv
    hu = numpy.zeros_like(xc)
    hv = numpy.zeros_like(xc)
    hu = numpy.tile(hu, (config["discretization"]["Ny"], 1))
    hv = numpy.tile(hv, (config["discretization"]["Ny"], 1))

    # write I.C. file
    data = dict(zip(config["ic"]["keys"], [w, hu, hv]))
    write_cf(
        os.path.join(case, config["ic"]["file"]), xc, yc,
        dict(zip(config["ic"]["keys"], [w, hu, hv])),
        options=dict(zip(config["ic"]["keys"], [
            {"units": "m"}, {"units": "m2 s-1"}, {"units": "m2 s-1"}])))


if __name__ == "__main__":
    import sys

    # when execute this script directly, make sure TorchSWE can be found
    pkg_path = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    sys.path.append(pkg_path)

    # execute the main function
    main()
