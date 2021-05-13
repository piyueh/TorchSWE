#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Comparison to the analytical solutions of transcritical flow w/ shock benchmark.
"""
import pathlib
import numpy
from matplotlib import pyplot
from torchswe.utils.netcdf import read_cf
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


def solve_shock_loc(bM, hM, q0, hL, g):
    """Solve the location of shock.

    Args:
    -----
        bM: maximum topo elevation.
        hM: water depth at where the topo elevation is maximum.
        q0: a scalar; the conservative quantity hu at x = 0 (the left) boundary.
        hL: a scalar; the depth h at x = L (the right) boundary.
        g: gravity.

    Returns:
    --------
        The x-coordinate of the shock.
    """

    X = numpy.array([[hM], [0.33], [0.]], dtype=numpy.float64)

    q0 = numpy.array(q0, dtype=numpy.float64)
    g = numpy.array(g, dtype=numpy.float64)
    hL = numpy.array(hL, dtype=numpy.float64)
    hM = numpy.array(hM, dtype=numpy.float64)
    bM = numpy.array(bM, dtype=numpy.float64)

    q02 = q0 * q0
    c0 = q02 / (2. * g)
    c1 = c0 / (hM * hM)
    c2 = c0 / (hL * hL)

    J = numpy.zeros((3, 3), dtype=numpy.float64)
    F = numpy.zeros((3, 1), dtype=numpy.float64)

    # solve with Newton-Raphson method
    norm = 1000.
    while norm > 1e-16:

        # function value
        F[0, 0] = numpy.power(X[0], 3) + (X[2] - c1 - hM - bM) * numpy.power(X[0], 2) + c0
        F[1, 0] = numpy.power(X[1], 3) + (X[2] - c2 - hL) * numpy.power(X[1], 2) + c0
        F[2, 0] = q02 * (1. / X[0] - 1. / X[1]) + g * (
            numpy.power(X[0], 2) - numpy.power(X[1], 2)) / 2.

        # Jacobian
        J[0, 0] = 3 * numpy.power(X[0], 2) + 2 * (X[2] - c1 - hM - bM) * X[0]
        J[0, 2] = numpy.power(X[0], 2)
        J[1, 1] = 3 * numpy.power(X[1], 2) + 2 * (X[2] - c2 - hL) * X[1]
        J[1, 2] = numpy.power(X[1], 2)
        J[2, 0] = - q02 / numpy.power(X[0], 2) + g * X[0]
        J[2, 1] = q02 / numpy.power(X[1], 2) - g * X[1]

        dX = numpy.matmul(numpy.linalg.inv(J), F)
        X = X - dX
        norm = numpy.linalg.norm(dX)

    return inverse_bump_topo(X[-1, 0])  # x is a 3 x 1 array, i.e., column vector


def solve_hcr(q0, g):
    """Calculate the critical h.

    For 1D problem, q is theoretically constant everywhere if no mass gain/loss.
    So the q at the left boundary can be treated as the q everywhere.

    Args:
    -----
        q0: a scalar; the conservative quantity hu at x = 0 (the left) boundary.
        g: gravity.

    Returns:
    --------
        hcr: the critical h.
    """

    hcr = (q0 * q0 / g)**(1. / 3.)

    return hcr


def inverse_bump_topo(b):
    """Solve the location given an elevation -- only applies to the bump.

    Returns:
    --------
        x: the coordinate of the 2nd critical point.
    """

    # topography eq. of the bump: b(x) = 0.2 - 0.05(x-10)^2
    x = numpy.sort(numpy.roots([1., -20., 100.+b/0.05-4.]))[::-1][0]

    return x


def get_trsnscritical_coeffs(b, bM, hM, q0, g):
    """2nd- & 0th-order term coefficients of transcritical Bernoulli relation.

    Args:
    -----
        b: a scalar or 1 1D numpy.ndarray; the topo elevation at target locations.
        bM: maximum topo elevation.
        hM: water depth at where the topo elevation is maximum.
        q0: a scalar; the conservative quantity hu at x = 0 (the left) boundary.
        g: gravity.

    Returns:
    --------
        C0: the coefficient of the constant term of the polynomial.
        C2: the coefficient of the quadratic term of the polynomial.
    """

    g2 = 2. * g
    q02 = q0 * q0

    C0 = q02 / g2
    C2 = b - q02 / (g2 * hM * hM) - hM - bM

    return C0, C2


def get_subcritical_coeffs(b, q0, hL, g):
    """2nd- & 0th-order term coefficients of subcritical Bernoulli relation.

    Args:
    -----
        b: a scalar or 1 1D numpy.ndarray; the topo elevation at target locations.
        q0: a scalar; the conservative quantity hu at x = 0 (the left) boundary.
        hL: a scalar; the depth h at x = L (the right) boundary.
        g: gravity.

    Returns:
    --------
        C0: the coefficient of the constant term of the polynomial.
        C2: the coefficient of the quadratic term of the polynomial.
    """

    C0 = q0 * q0 / (2. * g)
    C2 = b - C0 / (hL * hL) - hL

    return C0, C2


def get_analytical(x, bM, hM, xsh, q0, hL, g):
    """Get the analytical solution of transcritical flow w/ shock.

    Args:
    -----
        x: coordinates of evaluation points.
        bM: the maximum topo elevation.
        hM: the critical h; also the h at the maximum topo elevation.
        xsh: the location of the shock.
        q0: a scalar; the conservative quantity hu at x = 0 (the left) boundary.
        hL: a scalar; the depth h at x = L (the right) boundary.
        g: gravity.

    Returns:
        b, h, w
    """

    # first regime: transcritical w/o shock
    x_trans = x[x <= xsh]
    b_trans = topo(x_trans)
    h_trans = numpy.zeros_like(x_trans)
    C0, C2 = get_trsnscritical_coeffs(b_trans, bM, hM, q0, g)
    for i, c2 in enumerate(C2):
        R = numpy.sort(numpy.roots([1.0, c2, 0., C0]))[::-1]
        h_trans[i] = R[0] if x_trans[i] <= 10. else R[1]

    # second regime: subcritical
    x_sub = x[x >= xsh]
    b_sub = topo(x_sub)
    h_sub = numpy.zeros_like(x_sub)
    C0, C2 = get_subcritical_coeffs(b_sub, q0, hL, g)
    for i, c2 in enumerate(C2):
        h_sub[i] = numpy.roots([1.0, c2, 0., C0])[0]

    b = numpy.hstack([b_trans, b_sub])
    h = numpy.hstack([h_trans, h_sub])
    w = h + b

    # we simply assume no x is right at xsh ... usually we're not that "lucky"
    assert b.shape[0] == x.shape[0]

    return b, h, w


def main():
    """Plot and compare to analytical solutions."""

    # get the critical depth; also represents h at b = 0.2 (& x = 10)
    hcr = solve_hcr(0.18, 9.81)
    xsh = solve_shock_loc(0.2, hcr, 0.18, 0.33, 9.81)

    # read simulation data
    filename = pathlib.Path(__file__).expanduser().resolve().parent.joinpath("solutions.nc")
    sim_data, _ = read_cf(filename, ["w", "hu"])
    x = sim_data["x"]
    w = sim_data["w"][-1, :, :]  # only keep the soln at the last time
    w = numpy.mean(w, 0)  # use the average in y direction
    hu = sim_data["hu"][-1, :, :]
    hu = numpy.mean(hu, 0)

    # get a set of analytical solution for error
    b_ana, _, w_ana = get_analytical(x, 0.2, hcr, xsh, 0.18, 0.33, 9.81)

    # get another set of solution for plotting
    x_plot = numpy.linspace(0., 25., 2500)
    b_plot, h_plot, w_plot = get_analytical(x_plot, 0.2, hcr, xsh, 0.18, 0.33, 9.81)

    # relative L1 error
    w_err = numpy.abs((w-w_ana)/w_ana)

    # total volume of w per unit y
    vol = w.sum() * (x[1] - x[0])
    vol_ana = w_ana.sum() * (x[1] - x[0])
    print("Total volume per y: analytical -- {} m^2; ".format(vol_ana) +
          "simulation -- {} m^2".format(vol))

    # plot
    pyplot.figure()
    pyplot.plot(x_plot, b_plot, "k-", lw=4, label="Topography elevation (m)")
    pyplot.plot(x_plot, w_plot, "k-", lw=2, label="Analytical solution")
    pyplot.plot(x, w, ls='', marker='x', ms=5, alpha=0.6, label="Simulation solution")
    pyplot.title("Transcritical flow w/ shock: water level")
    pyplot.xlabel("x (m)")
    pyplot.ylabel("Water level (m)")
    pyplot.grid()
    pyplot.legend()
    pyplot.savefig("simulation_vs_analytical_w.png", dpi=166)

    pyplot.figure()
    pyplot.plot(x_plot, h_plot, "k-", lw=2, label="Analytical solution")
    pyplot.plot(x, w-b_ana, ls='', marker='x', ms=5, alpha=0.6, label="Simulation solution")
    pyplot.title("Transcritical flow w/ shock: water depth")
    pyplot.xlabel("x (m)")
    pyplot.ylabel("Water depth (m)")
    pyplot.grid()
    pyplot.legend()
    pyplot.savefig("simulation_vs_analytical_h.png", dpi=166)

    pyplot.figure()
    pyplot.plot(x_plot, numpy.ones_like(x_plot)*0.18, "k-", lw=2, label="Analytical solution")
    pyplot.plot(x, hu, ls='', marker='x', ms=5, alpha=0.6, label="Simulation solution")
    pyplot.title("Transcritical flow w/ shock: discharge")
    pyplot.xlabel("x (m)")
    pyplot.ylabel("Discharge " r"($q=hu$)" " (m)")
    pyplot.grid()
    pyplot.legend()
    pyplot.savefig("simulation_vs_analytical_hu.png", dpi=166)

    pyplot.figure()
    pyplot.semilogy(x, w_err, "k-", lw=2)
    pyplot.title("Transcritical flow w/ shock: relative L1 error of w")
    pyplot.xlabel("x (m)")
    pyplot.ylabel(r"$\left|\left(w_{simulation}-w_{analytical}\right)/w_{analytical}\right|$")
    pyplot.grid()
    pyplot.savefig("simulation_vs_analytical_w_L1.png", dpi=166)


if __name__ == "__main__":
    import sys
    sys.exit(main())
