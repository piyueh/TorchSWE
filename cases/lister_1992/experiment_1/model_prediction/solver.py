#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Solving eq. 2.8 in Lister 1992
"""
# pylint: disable=invalid-name
import os

if "USE_CUPY" in os.environ and os.environ["USE_CUPY"] == "1":
    import cupy as nplike
    from cupyx.scipy import sparse
    from cupyx.scipy.sparse import linalg
else:
    import numpy as nplike
    from scipy import sparse
    from scipy.sparse import linalg


def convection_step(h, dx, t_cur, t_up):
    """Solve the first step in the splitting method -- convection.

    We use a simple upwind scheme. It is actually equivalent to Osher's scheme due to:
        1. We only have one eigenfunction, which is a(h) = 3h^2, because this is a scalar problem.
        2. The a+(h) = max(a(h), 0) = a(h) = 3h^2
        3. The a-(h) = min(a(h), 0) = min(3h^2, 0) = 0
        4. The Osher's flux at i+1/2 => f(h_i) + int_{h_i}^{h_{i+1}} a-(h) dh = f(h_i)
        5. So dF/dx at cell i is (f(h_i) - f(h_{i-1}) / dx => simple upwind with one direction.

    Arguments
    ---------
    h : nplike.ndarray of shape (ny+2, nx+2)
        Unknown variables with ghost cells.
    dx : float
        Cell size in x direction.
    t_cur : float
        The current time.
    t_up : float
        The upper limit the time can go. Used to adjust time step.

    Returns
    -------
    h : nplike.ndarray of shape (ny+2, nx+2)
        Updated unknown variables.
    dt : float
        The adaptive timestep used in this step
    save : bool
        Indicate this is the time to save output
    """

    h3 = nplike.power(h, 2)  # temporarily use h3 to store h^2
    dt = dx / nplike.max(3.*h3)

    if dt >= (t_up - t_cur):
        dt = t_up - t_cur
        save = True
    else:
        save = False

    h3 *= h
    h[1:-1, 1:-1] -= (h3[1:-1, 1:-1] - h3[1:-1, :-2]) * dt / dx

    # update ghost cells -- Neumann
    h[0, 1:-1] = h[1, 1:-1]
    h[-1, 1:-1] = h[-2, 1:-1]
    h[1:-1, 0] = h[1:-1, 1]
    h[1:-1, -1] = h[1:-1, -2]

    return h, dt, save


def prapare_spm_pattern(nx, ny):
    """Prepare the row and col indices of the sparse matrix in diffusion term.

    Arguments
    ---------
    nx, ny : int
        Number of cells (excluding ghost cells).

    Returns
    -------
    row : nplike.ndarray
        A 1D array holding the row indices of non-zeros.
    col : nplike.ndarray
        A 1D array holding the column indices of non-zeros.
    """

    # row indices (ghost cells not yet pruned)
    row = nplike.tile(nplike.arange(0, nx*ny, dtype=int).reshape((-1, 1)), (1, 5)).flatten()

    # col indices (ghost cells not yet pruned)
    c1 = nplike.full((ny, nx), -999, dtype=int)
    c2 = nplike.full((ny, nx), -999, dtype=int)
    c3 = nplike.full((ny, nx), -999, dtype=int)
    c4 = nplike.full((ny, nx), -999, dtype=int)
    center = nplike.arange(0, nx*ny, dtype=int).reshape((ny, nx))

    c1[1:, :] = (center[1:, :] - nx)
    c2[:, 1:] = (center[:, 1:] - 1)
    c3[:, :-1] = (center[:, :-1] + 1)
    c4[:-1, :] = (center[:-1, :] + nx)

    c1 = c1.reshape((-1, 1))
    c2 = c2.reshape((-1, 1))
    c3 = c3.reshape((-1, 1))
    c4 = c4.reshape((-1, 1))
    center = center.reshape((-1, 1))
    col = nplike.concatenate((c1, c2, center, c3, c4), 1).flatten()

    # remove ghost cells
    cond = (col != -999)
    row = nplike.extract(cond, row)
    col = nplike.extract(cond, col)

    return row, col


def diffusion_step(h, row, col, dt, dx, dy):
    """Solve the second step in the splitting method -- diffusion.

    Arguments
    ---------
    h : nplike.ndarray of shape (ny+2, nx+2)
        Unknown variables with ghost cells.
    idxs : nplike.ndarray of shape (ny+2, nx+2)
        Mappings from j, i to indices in the flatten matrix.
    dt : float
        Time step size.
    dx : float
        Cell size in x direction.

    Returns
    -------
    h : nplike.ndarray of shape (ny+2, nx+2)
        Updated unknown variables.
    """

    # shape without ghost cells
    shape = h.shape[0] - 2, h.shape[1] - 2

    cx = nplike.power((h[1:-1, 1:]+h[1:-1, :-1])/2., 3) * dt / dx / dx  # (ny, nx+1)
    cy = nplike.power((h[1:, 1:-1]+h[:-1, 1:-1])/2., 3) * dt / dy / dy  # (ny+1, nx)

    c1 = cy[:-1, :]  # (ny, nx), for H(j-1, i)
    c2 = cx[:, :-1]  # (ny, nx), for H(j, i-1)
    c3 = - (1. + cx[:, :-1] + cx[:, 1:] + cy[:-1, :] + cy[1:, :])  # (ny, nx), for H(j, i)
    c4 = cx[:, 1:]  # (ny, nx), for H(j, i+1)
    c5 = cy[1:, :]  # (ny, nx), for H(j+1, i)

    # when j=0, there's no j-1
    c3[0, :] += c1[0, :]
    c1[0, :] = float("NaN")

    # when j=ny-1, there's no j+1
    c3[-1, :] += c5[-1, :]
    c5[-1, :] = float("NaN")

    # when i=0, there's no i-1
    c3[:, 0] += c2[:, 0]
    c2[:, 0] = float("NaN")

    # when i=nx-1, there's no i+1
    c3[:, -1] += c4[:, -1]
    c4[:, -1] = float("NaN")

    # coefficients
    c = nplike.concatenate((
        c1.reshape((-1, 1)), c2.reshape((-1, 1)), c3.reshape((-1, 1)),
        c4.reshape((-1, 1)), c5.reshape((-1, 1))
    ), 1).flatten()

    c = nplike.extract(nplike.logical_not(nplike.isnan(c)), c)
    assert len(c) == len(row), f"{len(c)}, {len(row)}"
    assert len(c) == len(col), f"{len(c)}, {len(col)}"

    # coefficient matrix
    c = sparse.csr_matrix((c, (row, col)), shape=(shape[0]*shape[1], shape[0]*shape[1]))

    # solve
    c1 = sparse.dia_matrix((1./c.diagonal(), 0), shape=c.shape)  # preconditioner; borrow variable
    cx, cy = linalg.cg(  # borrow the variable names
        A=c, b=-h[1:-1, 1:-1].flatten(), x0=h[1:-1, 1:-1].flatten(),
        tol=1e-10, M=c1, atol=1e-10
    )
    h[1:-1, 1:-1] = cx.reshape(shape)
    assert cy == 0, f"CG solver diverged with info code {cy}"

    # update ghost cells -- Neumann
    h[0, 1:-1] = h[1, 1:-1]
    h[-1, 1:-1] = h[-2, 1:-1]
    h[1:-1, 0] = h[1:-1, 1]
    h[1:-1, -1] = h[1:-1, -2]

    return h


def solve(runtime):
    """Solve the system.
    """

    # make sure (0., 0.) is at some cell's center
    assert (runtime["nx"] - 3) % 6 == 0
    assert runtime["ny"] % 2 == 1

    # point source indices (j, i)
    loc = (runtime["ny"] // 2, (runtime["nx"] - 3) // 6)

    # dimensionless grid spacing
    dX = 1.2 / runtime["nx"] / runtime["xstar"]
    dY = 0.6 / runtime["ny"] / runtime["ystar"]
    dT = 0.9 * dX  # initial time step size

    # dimensionless unknown variables w/ ghost cells
    H = nplike.zeros((runtime["ny"]+2, runtime["nx"]+2))

    # prepare spm pattern
    pattern = prapare_spm_pattern(runtime["nx"], runtime["ny"])

    # final solution holder
    h = nplike.zeros((len(runtime["output"]), runtime["ny"], runtime["nx"]))

    # time contorl
    T = 0

    for ti, T_out in enumerate(runtime["output"]):
        # dimensionaless output time
        T_out /= runtime["tstar"]

        while True:
            # add source
            H[loc[0], loc[1]] += dT / dX / dY

            # solve for this time step
            H, dT, save = convection_step(H, dX, T, T_out)
            H = diffusion_step(H, *pattern, dT, dX, dY)

            # update current time and counter
            T += dT
            print(T, T*runtime["tstar"], nplike.count_nonzero(H), nplike.sum(H)*dX*dY)

            # save
            if save:
                h[ti, ...] = H[1:-1, 1:-1] * runtime["hstar"]
                break

    # prepare gridlines for convenience (reuse variables)
    x = nplike.linspace(-0.2, 1.0, runtime["nx"]+1)
    x = (x[:-1] + x[1:]) / 2.
    y = nplike.linspace(-0.3, 0.3, runtime["ny"]+1)
    y = (y[:-1] + y[1:]) / 2.

    return x, y, h


def dimensionless_factor(alpha, theta, Q, nu, g, *args, **kwargs):
    """Calculate factors that make variables dimensionless.

    Arguments
    ---------
    alpha : float
        The type of point sources. Currently only support alpha=1.
    theta : float
        The angle (in radian) of the inclined plane.
    Q : float
        Volumetric flow rate, in m^3/s.
    nu : float
        Kinematic viscosity, in m^2/s.
    g : float
        Gravitational acceleration, in m/s^2.
    *args, **kwargs :
        To capture unused arguments.

    Returns
    -------
    xstar, ystar, hstar, tstar : float
        Factors to make variables dimensionless.
    """

    c = 1. / nplike.tan(theta)
    r = g * nplike.sin(theta) / (3. * nu)
    xstar = (Q * c**(2*alpha+1) / r**alpha)**(1. / (alpha + 3.))
    ystar = (Q * c**(2*alpha+1) / r**alpha)**(1. / (alpha + 3.))
    hstar = (Q * c**(2*alpha+1) / r**alpha)**(1. / (alpha + 3.)) / c
    tstar = (c**5 / (Q * r**3))**(1. / (alpha + 3.))

    return xstar, ystar, hstar, tstar


if __name__ == "__main__":
    import pathlib

    # configuration of the problem
    config = {
        "alpha": 1,
        "theta": 2.5 * nplike.pi / 180.,
        "g": 9.81,
        "nu": 1.13e-3,
        "Q": 1.48e-6,
    }

    # prepare parameters for numerical solver
    data = {
        "nx": 963,
        "ny": 483,
        "output": [32., 59., 122., 271., 486., 727.]
    }
    data["xstar"], data["ystar"], data["hstar"], data["tstar"] = dimensionless_factor(**config)

    # run
    solution = solve(data)

    # save
    case = pathlib.Path(__file__).expanduser().resolve().parent
    nplike.savez(case.joinpath("solution.npz"), x=solution[0], y=solution[1], h=solution[2])
