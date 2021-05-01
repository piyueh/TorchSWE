#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""
Create topography for case 3.1.4 in Delestre et al. (2013).

Note, the elevation data in the resulting NetCDF file is defined at vertices,
instead of cell centers.
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

    loc = (x >= 8.) * (x <= 12.)
    B[loc] = 0.2 - 0.05 * numpy.power(x[loc]-10., 2)

    # make it 2D
    B = numpy.tile(B, (config["discretization"]["Ny"]+1, 1))

    # write topography file
    write_cf(
        os.path.join(case, config["topography"]["file"]), x, y,
        {config["topography"]["key"]: B},
        options={config["topography"]["key"]: {"units": "m"}})


if __name__ == "__main__":
    import sys

    # when execute this script directly, make sure TorchSWE can be found
    pkg_path = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    sys.path.append(pkg_path)

    # execute the main function
    main()
