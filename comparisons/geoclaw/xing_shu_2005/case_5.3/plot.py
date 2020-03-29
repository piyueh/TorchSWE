#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the MIT license.

"""

"""
import os
import numpy
from matplotlib import pyplot
from clawpack import pyclaw

def main():
    """Main plotting function."""

    # paths
    casepath = os.path.dirname(os.path.abspath(__file__))
    outputpath = os.path.join(casepath, "_output")

    # time
    t = [0., 0.12, 0.24, 0.36, 0.48, 0.6]

    # contour line range
    n = 32
    r = [
        numpy.linspace(1., 1.01, n),
        numpy.linspace(0.999703, 1.00629, n),
        numpy.linspace(0.994836, 1.01604, n),
        numpy.linspace(0.988582, 1.0117, n),
        numpy.linspace(0.990344, 1.00497, n),
        numpy.linspace(0.995065, 1.0056, n)]

    # to store all grid patch
    x = numpy.linspace(0., 2., 2001, dtype=numpy.float64)
    x = (x[:-1] + x[1:]) / 2.
    y = numpy.linspace(0., 1., 1001, dtype=numpy.float64)
    y = (y[:-1] + y[1:]) / 2.
    X, Y = numpy.meshgrid(x, y)

    # plot each figure
    for frameno in range(6):

        # new memory space
        H = numpy.zeros_like(X).flatten()
        B = numpy.zeros_like(X).flatten()

        # solution
        soln = pyclaw.Solution()
        soln.read(frameno, outputpath, file_format="ascii", read_aux=True)

        # fill the global solution patch-to-patch
        for state in soln.states:

            # make sure AMR was not used ...
            if state.patch.level != 1:
                raise RuntimeError("This test case should only have 1 level of grid.")

            # identify the region/indices
            right = (X <= state.patch.upper_global[0]).flatten()
            left = (X >= state.patch.lower_global[0]).flatten()
            top = (Y <= state.patch.upper_global[1]).flatten()
            bottom = (Y >= state.patch.lower_global[1]).flatten()

            # copy to slices
            H[right*left*top*bottom] = state.q[0, :, :].T.flatten()
            B[right*left*top*bottom] = state.aux[0, :, :].T.flatten()

        # go back to 2D shape
        H = H.reshape(1000, 2000)
        B = B.reshape(1000, 2000)
        W = H + B

        # contour lines: to compare with Xing & Shu
        pyplot.figure(figsize=(10, 4), dpi=166)
        pyplot.contour(X, Y, W, r[frameno], linewidths=1)
        pyplot.title("Xing & Shu (2005) case 5.3: water level @ T={} sec".format(t[frameno]))
        pyplot.xlabel("x (m)")
        pyplot.ylabel("y (m)")
        pyplot.xlim(0., 2.)
        pyplot.ylim(0., 1.)
        pyplot.colorbar()
        pyplot.tight_layout()
        pyplot.savefig("water_level_contourline_t={}.png".format(t[frameno]), dpi=166, bbox_inches="tight")

        # a new figure
        pyplot.figure(figsize=(10, 4), dpi=166)
        pyplot.contourf(X, Y, W, 128)
        pyplot.title("Xing & Shu (2005) case 5.3: water level @ T={} sec".format(t[frameno]))
        pyplot.xlabel("x (m)")
        pyplot.ylabel("y (m)")
        pyplot.xlim(0., 2.)
        pyplot.ylim(0., 1.)
        pyplot.colorbar()
        pyplot.tight_layout()
        pyplot.savefig("water_level_contour_t={}.png".format(t[frameno]), dpi=166, bbox_inches="tight")

if __name__ == "__main__":
    main()
