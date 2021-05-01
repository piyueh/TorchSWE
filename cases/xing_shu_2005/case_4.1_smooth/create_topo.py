#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""
Create a topography file for the case 4.1 and smooth topography in Xing and Shu (2005).

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

    B = numpy.tile(
        5.*numpy.exp(-0.4*((x-5.)**2)), (config["discretization"]["Ny"]+1, 1))

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
