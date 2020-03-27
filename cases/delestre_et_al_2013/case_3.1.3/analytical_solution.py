#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the MIT license.

"""Comparison to the analytical solutions of subcritical flow benchmark.
"""
import os
import numpy

def topo(x):
    """Calculate the topography elevation.

    Args:
    -----
        x: a numpy.ndarray

    Retruns:
    --------
        b: a numpy ndarray; the elevation at x
    """

    b = numpy.zeros_like(x)

    loc = (x >= 8.) * (x <= 12.)
    b[loc] = 0.2 - 0.05 * numpy.power(x[loc]-10., 2)

    return b

def get_coeffs(b, gL, hL, g):
    """Coefficients of the quadratic and constant terms in the Bernoulli relation.

    Args:
    -----
        b: a scalar or 1 1D numpy.ndarray; the topo elevation at target locations.
        gL: a scalar; the conservative quantity hu at the left boundary.
        hL: a scalar; the depth h at the left boundary.
        g: gravity.
    """

    C0 = gL * gL / (2. * g)
    C1 = b - C0 / (hL * hL) - hL

    return C0, C1

def main(N=1000):
    """Exact solution to 1D subcritical flow.

    Args:
    -----
        N: an integer; resolution of the analytical solution.

    Returns:
        x: 1D numpy.ndarray; coordinates.
        b: 1D numpy.ndarray; topography elevation.
        h: 1D numpy.ndarray; water depth.
        w: 1D numpy.ndarray; water depth + topography elecation.
    """

    # it's users' responsibility to make sure TorchSWE package can be found
    from TorchSWE.utils.netcdf import read_cf

    # read simulation data
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solutions.nc")
    sim_data, _ = read_cf(filename, ["w"])
    x = sim_data["x"]
    w = sim_data["w"][-1, :, :] # only keep the soln at the last time
    w = numpy.mean(w, 0) # use the average in y direction

    # get a set of analytical solution for error
    b_ana = topo(x)
    h_ana = numpy.zeros_like(x)
    C0, C1 = get_coeffs(b_ana, 4.42, 2.0, 9.81)
    for i, c1 in enumerate(C1):
        h_ana[i] = numpy.roots([1.0, c1, 0., C0])[0]
    w_ana = h_ana + b_ana

    # get another set of solution for plotting
    x_plot = numpy.linspace(0., 25., 1000, dtype=numpy.float64)
    b_plot = topo(x_plot)
    h_plot = numpy.zeros_like(x_plot)
    C0, C1 = get_coeffs(b_plot, 4.42, 2.0, 9.81)
    for i, c1 in enumerate(C1):
        h_plot[i] = numpy.roots([1.0, c1, 0., C0])[0]
    w_plot = h_plot + b_plot

    # relative L1 error
    w_err = numpy.abs((w-w_ana)/w_ana)

    # total volume per unit y
    vol = w.sum() * (x[1] - x[0])
    vol_ana = w_ana.sum() * (x[1] - x[0])
    print("Total volume per y: analytical -- {} m^2; ".format(vol_ana) +
          "simulation -- {} m^2".format(vol))

    # plot
    from matplotlib import pyplot

    pyplot.figure()
    pyplot.plot(x_plot, b_plot, "k-", lw=4, label="Topography elevation (m)")
    pyplot.plot(x_plot, w_plot, "k-", lw=2, label="Analytical solution")
    pyplot.plot(x, w, ls='', marker='x', ms=5, alpha=0.6, label="Simulation solution")
    pyplot.title("Subcritical flow benchmark: w (water level + topography elevation")
    pyplot.xlabel("x (m)")
    pyplot.ylabel("Water dpeth + Topography elevation (m)")
    pyplot.grid()
    pyplot.legend()
    pyplot.savefig("simulation_vs_analytical_w.png", dpi=166)

    pyplot.figure()
    pyplot.plot(x, w_err, "k-", lw=2, label="Analytical solution")
    pyplot.title("Subcritical flow benchmark: relative L1 error of w")
    pyplot.xlabel("x (m)")
    pyplot.ylabel(r"$\left|\left(w_{simulation}-w_{analytical}\right)/w_{analytical}\right|$")
    pyplot.grid()
    pyplot.legend()
    pyplot.savefig("simulation_vs_analytical_w_L1.png", dpi=166)

if __name__ == "__main__":
    import sys

    # when execute this script directly, make sure TorchSWE can be found
    pkg_path = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    sys.path.append(pkg_path)

    # execute the main function
    main()
