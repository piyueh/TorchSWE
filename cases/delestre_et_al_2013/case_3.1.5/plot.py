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


def get_analytical_solution(x):
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

    # get the critical depth; also represents h at b = 0.2 (& x = 10)
    hcr = solve_hcr(0.18, 9.81)  # critical depth
    xsh = solve_shock_loc(0.2, hcr, 0.18, 0.33, 9.81)  # location of the shock

    # first regime: transcritical w/o shock
    x_trans = x[x <= xsh]
    b_trans = topo(x_trans)
    h_trans = numpy.zeros_like(x_trans)
    C0, C2 = get_trsnscritical_coeffs(b_trans, 0.2, hcr, 0.18, 9.81)
    for i, c2 in enumerate(C2):
        R = numpy.sort(numpy.roots([1.0, c2, 0., C0]))[::-1]
        h_trans[i] = R[0] if x_trans[i] <= 10. else R[1]

    # second regime: subcritical
    x_sub = x[x >= xsh]
    b_sub = topo(x_sub)
    h_sub = numpy.zeros_like(x_sub)
    C0, C2 = get_subcritical_coeffs(b_sub, 0.18, 0.33, 9.81)
    for i, c2 in enumerate(C2):
        h_sub[i] = numpy.roots([1.0, c2, 0., C0])[0]

    b = numpy.hstack([b_trans, b_sub])
    h = numpy.hstack([h_trans, h_sub])
    w = h + b

    # we simply assume no grid point is right located at xsh ... usually we're not that "lucky"
    assert b.shape[0] == x.shape[0]

    return b, h, w


def main():
    """Plot and compare to analytical solutions."""

    # read simulation data
    case = pathlib.Path(__file__).expanduser().resolve().parent

    with h5py.File(case.joinpath("solutions.h5"), "r") as root:
        x = root["grid/x/c"][...]
        dx = root["grid/x"].attrs["delta"]
        assert abs(x[1]-x[0]-dx) < 1e-10

        w = root["50/states/w"][...]
        h = root["50/states/h"][...]
        hu = root["50/states/hu"][...]

    # make sure y direction does not have variance
    assert numpy.allclose(w, w[0].reshape(1, -1))
    assert numpy.allclose(h, h[0].reshape(1, -1))
    assert numpy.allclose(hu, hu[0].reshape(1, -1))

    # average in y direction
    w = w.mean(axis=0)  # pylint: disable=no-member
    h = h.mean(axis=0)  # pylint: disable=no-member
    hu = hu.mean(axis=0)  # pylint: disable=no-member

    # get a set of analytical solution for error
    b_ana, _, w_ana = get_analytical_solution(x)

    # get another set of solution for plotting
    x_plot = numpy.linspace(0., 25., 2500, dtype=numpy.float64)
    b_plot, h_plot, w_plot = get_analytical_solution(x_plot)

    # relative L1 error
    w_err = numpy.abs((w-w_ana)/w_ana)

    # total volume of w per unit y
    print(f"Total volume per y: analytical -- {w_ana.sum()*dx} m^2; simulation -- {w.sum()*dx} m^2")

    # figure folder
    case.joinpath("figs").mkdir(exist_ok=True)

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
    pyplot.savefig(case.joinpath("figs", "simulation_vs_analytical_w.png"), dpi=166)

    pyplot.figure()
    pyplot.plot(x_plot, h_plot, "k-", lw=2, label="Analytical solution")
    pyplot.plot(x, w-b_ana, ls='', marker='x', ms=5, alpha=0.6, label="Simulation solution")
    pyplot.title("Transcritical flow w/ shock: water depth")
    pyplot.xlabel("x (m)")
    pyplot.ylabel("Water depth (m)")
    pyplot.grid()
    pyplot.legend()
    pyplot.savefig(case.joinpath("figs", "simulation_vs_analytical_h.png"), dpi=166)

    pyplot.figure()
    pyplot.plot(x_plot, numpy.ones_like(x_plot)*0.18, "k-", lw=2, label="Analytical solution")
    pyplot.plot(x, hu, ls='', marker='x', ms=5, alpha=0.6, label="Simulation solution")
    pyplot.title("Transcritical flow w/ shock: discharge")
    pyplot.xlabel("x (m)")
    pyplot.ylabel("Discharge " r"($q=hu$)" " (m)")
    pyplot.grid()
    pyplot.legend()
    pyplot.savefig(case.joinpath("figs", "simulation_vs_analytical_hu.png"), dpi=166)

    pyplot.figure()
    pyplot.semilogy(x, w_err, "k-", lw=2)
    pyplot.title("Transcritical flow w/ shock: relative L1 error of w")
    pyplot.xlabel("x (m)")
    pyplot.ylabel(r"$\left|\left(w_{simulation}-w_{analytical}\right)/w_{analytical}\right|$")
    pyplot.grid()
    pyplot.savefig(case.joinpath("figs", "simulation_vs_analytical_w_L1.png"), dpi=166)


if __name__ == "__main__":
    import sys
    sys.exit(main())
