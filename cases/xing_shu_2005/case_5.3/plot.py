#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""
Plot.
"""
import os
import numpy
from matplotlib import pyplot

def main():
    """Plot and compare to analytical solutions."""

    # it's users' responsibility to make sure TorchSWE package can be found
    from TorchSWE.utils.netcdf import read_cf

    # read simulation data
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solutions.nc")
    sim_data, _ = read_cf(filename, ["w"])

    # 2D coordinates
    x = sim_data["x"]
    y = sim_data["y"]
    X, Y = numpy.meshgrid(x, y)

    # get solutions except the one at T=0
    W = sim_data["w"][1:, :, :]

    # time labels
    t = [0.12, 0.24, 0.36, 0.48, 0.6]

    # contour line range
    n = 32
    r = [
        numpy.linspace(0.999703, 1.00629, n),
        numpy.linspace(0.994836, 1.01604, n),
        numpy.linspace(0.988582, 1.0117, n),
        numpy.linspace(0.990344, 1.00497, n),
        numpy.linspace(0.995065, 1.0056, n)]

    # contour lines: to compare with Xing & Shu
    for i in range(5):
        pyplot.figure(figsize=(10, 4), dpi=166)
        pyplot.contour(X, Y, W[i], r[i], linewidths=1)
        pyplot.title("Xing & Shu (2005) case 5.3: water level @ T={} sec".format(t[i]))
        pyplot.xlabel("x (m)")
        pyplot.ylabel("y (m)")
        pyplot.xlim(0., 2.)
        pyplot.ylim(0., 1.)
        pyplot.colorbar()
        pyplot.tight_layout()
        pyplot.savefig("water_level_contourline_t={}.png".format(t[i]), dpi=166, bbox_inches="tight")

    # contourf
    for i in range(5):
        pyplot.figure(figsize=(10, 4), dpi=166)
        pyplot.contourf(X, Y, W[i], 128)
        pyplot.title("Xing & Shu (2005) case 5.3: water level @ T={} sec".format(t[i]))
        pyplot.xlabel("x (m)")
        pyplot.ylabel("y (m)")
        pyplot.xlim(0., 2.)
        pyplot.ylim(0., 1.)
        pyplot.colorbar()
        pyplot.tight_layout()
        pyplot.savefig("water_level_contour_t={}.png".format(t[i]), dpi=166, bbox_inches="tight")

if __name__ == "__main__":
    import sys

    # when execute this script directly, make sure TorchSWE can be found
    pkg_path = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    sys.path.append(pkg_path)

    # execute the main function
    main()
