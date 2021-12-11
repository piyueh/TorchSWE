#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Comparison to the analytical solutions of subcritical flow benchmark.
"""
import pathlib
import numpy
import h5py
from matplotlib import pyplot
# pylint: disable=invalid-name, too-many-locals, too-many-statements


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


def get_coeffs(b, q0, hL, g):
    """Coefficients of the quadratic and constant terms in the Bernoulli relation.

    Args:
    -----
        b: a scalar or 1 1D numpy.ndarray; the topo elevation at target locations.
        q0: a scalar; the conservative quantity hu at x = 0 (the left) boundary.
        hL: a scalar; the depth h at the x = L (the right) boundary.
        g: gravity.
    """

    C0 = q0 * q0 / (2. * g)
    C1 = b - C0 / (hL * hL) - hL

    return C0, C1


def get_analytical_solution(x):
    """Get the analytical solution.
    """

    b = topo(x)
    h = numpy.zeros_like(x)
    c0, c1 = get_coeffs(b, 4.42, 2.0, 9.81)
    for i, c1i in enumerate(c1):
        h[i] = numpy.roots([1.0, c1i, 0., c0])[0]
    w = h + b
    return w, h, b


def main():
    """Plot and compare to analytical solutions."""

    # read simulation data
    case = pathlib.Path(__file__).expanduser().resolve().parent

    with h5py.File(case.joinpath("solutions.h5"), "r") as root:
        x = root["grid/x/c"][...]
        dx = root["grid/x"].attrs["delta"]
        assert abs(x[1]-x[0]-dx) < 1e-10

        w = root["30/states/w"][...]
        h = root["30/states/h"][...]
        hu = root["30/states/hu"][...]

    # make sure y direction does not have variance
    assert numpy.allclose(w, w[0].reshape(1, -1))
    assert numpy.allclose(h, h[0].reshape(1, -1))
    assert numpy.allclose(hu, hu[0].reshape(1, -1))

    # average in y direction
    w = w.mean(axis=0)  # pylint: disable=no-member
    h = h.mean(axis=0)  # pylint: disable=no-member
    hu = hu.mean(axis=0)  # pylint: disable=no-member

    # get a set of analytical solution for error calculations
    w_ana, _, b_ana = get_analytical_solution(x)

    # get another set of solutions for plotting
    x_plot = numpy.linspace(0., 25., 1000, dtype=numpy.float64)
    w_plot, h_plot, b_plot = get_analytical_solution(x_plot)

    # relative L1 error
    w_err = numpy.abs((w-w_ana)/w_ana)

    # total volume per unit y
    print(f"Total volume per y: analytical -- {w_ana.sum()*dx} m^2; simulation -- {w.sum()*dx} m^2")

    # figure folder
    case.joinpath("figs").mkdir(exist_ok=True)

    # plots
    pyplot.figure()
    pyplot.plot(x_plot, b_plot, "k-", lw=4, label="Topography elevation (m)")
    pyplot.plot(x_plot, w_plot, "k-", lw=2, label="Analytical solution")
    pyplot.plot(x, w, ls='', marker='x', ms=5, alpha=0.6, label="Simulation solution")
    pyplot.title("Subcritical flow: water level")
    pyplot.xlabel("x (m)")
    pyplot.ylabel("Water level (m)")
    pyplot.grid()
    pyplot.legend()
    pyplot.savefig(case.joinpath("figs", "simulation_vs_analytical_w.png"), dpi=166)

    pyplot.figure()
    pyplot.plot(x_plot, h_plot, "k-", lw=2, label="Analytical solution")
    pyplot.plot(x, w-b_ana, ls='', marker='x', ms=5, alpha=0.6, label="Simulation solution")
    pyplot.title("Subcritical flow: water depth")
    pyplot.xlabel("x (m)")
    pyplot.ylabel("Water depth (m)")
    pyplot.grid()
    pyplot.legend()
    pyplot.savefig(case.joinpath("figs", "simulation_vs_analytical_h.png"), dpi=166)

    pyplot.figure()
    pyplot.plot(x_plot, numpy.ones_like(x_plot)*4.42, "k-", lw=2, label="Analytical solution")
    pyplot.plot(x, hu, ls='', marker='x', ms=5, alpha=0.6, label="Simulation solution")
    pyplot.title("Subcritical flow: discharge")
    pyplot.xlabel("x (m)")
    pyplot.ylabel("Discharge " r"($q=hu$)" " (m)")
    pyplot.grid()
    pyplot.legend()
    pyplot.savefig(case.joinpath("figs", "simulation_vs_analytical_hu.png"), dpi=166)

    pyplot.figure()
    pyplot.semilogy(x, w_err, "k-", lw=2)
    pyplot.title("Subcritical flow: relative L1 error of w")
    pyplot.xlabel("x (m)")
    pyplot.ylabel(r"$\left|\left(w_{simulation}-w_{analytical}\right)/w_{analytical}\right|$")
    pyplot.grid()
    pyplot.savefig(case.joinpath("figs", "simulation_vs_analytical_w_L1.png"), dpi=166)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
