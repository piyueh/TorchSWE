#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Linear reconstruction.
"""


cdef _minmod_slope_kernel = cupy.ElementwiseKernel(
    "T s1, T s2, T s3, T theta, T dx",
    "T slp",
    """
        T denominator = s3 - s2;
        slp = (s2 - s1) / denominator;
        slp = min(slp*theta, (1.0 + slp) / 2.0);
        slp = min(slp, theta);
        slp = max(slp, 0.);
        slp *= denominator;
        slp /= dx;
    """,
    "minmod_slope_kernel",
)


@cython.boundscheck(False)  # deactivate bounds checking
def minmod_slope(object states, double theta):
    """Calculate the slope of using minmod limiter.

    Arguments
    ---------
    states : torchswe.utils.data.State
        The instance of States holding quantities.
    theta : float
        The parameter adjusting the dissipation.

    Returns
    -------
    states : torchswe.utils.data.State
        The same object as the input.
    """

    # alias
    cdef Py_ssize_t ngh = states.ngh
    cdef double dx = states.domain.x.delta
    cdef double dy = states.domain.y.delta
    Q = states.Q  # pylint: disable=invalid-name

    # kernels
    states.slpx = _minmod_slope_kernel(
        Q[:, ngh:-ngh, :-ngh], Q[:, ngh:-ngh, 1:-1], Q[:, ngh:-ngh, ngh:], theta, dx)
    states.slpy = _minmod_slope_kernel(
        Q[:, :-ngh, ngh:-ngh], Q[:, 1:-1, ngh:-ngh], Q[:, ngh:, ngh:-ngh], theta, dy)

    return states


def correct_negative_depth(states):
    """Fix negative depth on the both sides of cell faces.

    Arguments
    ---------
    states : torchswe.utils.data.States

    Returns:
    --------
    states : torchswe.utils.data.States
        The same object as the input. Changed inplace. Returning it just for coding style.

    Notes
    -----
    Instead of fixing the w+h as described in Keuganov & Petrova, 2007, we fix h directly, However,
    this is only valid when the topography elevation at a cell center is exactly the linear
    interpolation of elevations at cell interfaces.
    """

    # aliases
    ngh = states.ngh
    nx, ny = states.domain.x.n, states.domain.y.n

    # fix the case when the left depth of an interface is negative
    j, i = cupy.nonzero(states.face.x.minus.U[0] < 0.)
    states.face.x.minus.U[0, j, i] = 0.
    j, i = j[i != 0], i[i != 0]  # to avoid those i - 1 = -1
    states.face.x.plus.U[0, j, i-1] = 2 * states.U[0, j+ngh, i-1+ngh]

    # fix the case when the right depth of an interface is negative
    j, i = cupy.nonzero(states.face.x.plus.U[0] < 0.)
    states.face.x.plus.U[0, j, i] = 0.
    j, i = j[i != nx], i[i != nx]  # to avoid i + 1 = nx + 1
    states.face.x.minus.U[0, j, i+1] = 2 * states.U[0, j+ngh, i+ngh]

    # fix the case when the bottom depth of an interface is negative
    j, i = cupy.nonzero(states.face.y.minus.U[0] < 0.)
    states.face.y.minus.U[0, j, i] = 0.
    j, i = j[j != 0], i[j != 0]  # to avoid j - 1 = -1
    states.face.y.plus.U[0, j-1, i] = 2 * states.U[0, j-1+ngh, i+ngh]

    # fix the case when the top depth of an interface is negative
    j, i = cupy.nonzero(states.face.y.plus.U[0] < 0.)
    states.face.y.plus.U[0, j, i] = 0.
    j, i = j[j != ny], i[j != ny]  # to avoid j + 1 = Ny + 1
    states.face.y.minus.U[0, j+1, i] = 2 * states.U[0, j+ngh, i+ngh]

    return states


def reconstruct(states, runtime, config):
    """Reconstructs quantities at cell interfaces and centers.

    The following quantities in `states` are updated in this function:
        1. non-conservative quantities defined at cell centers (states.U)
        2. discontinuous non-conservative quantities defined at cell interfaces
        3. discontinuous conservative quantities defined at cell interfaces

    Arguments
    ---------
    states : torchswe.utils.data.States
    runtime : torchswe.utils.misc.DummyDict
    config : torchswe.utils.config.Config

    Returns
    -------
    states : torchswe.utils.data.States
        The same object as the input. Returning it just for coding style. The values are actually
        updated in-place.
    """

    ngh = states.ngh
    dx_half = states.domain.x.delta / 2.
    dy_half = states.domain.y.delta / 2.

    # calculate non-conservative quantities at cell centers
    states.U[0, ngh:-ngh, ngh:-ngh] = states.Q[0, ngh:-ngh, ngh:-ngh] - runtime.topo.centers

    wet = (states.U[0] > config.params.drytol)
    states.U[1:, ...] = 0.
    states.U[1:, wet] = states.Q[1:, wet] / states.U[0, wet]

    # get slopes
    states = minmod_slope(states, config.params.theta)

    # get discontinuous conservatice quantities at cell faces
    states.face.x.minus.Q = states.Q[:, ngh:-ngh, ngh-1:-ngh] + states.slpx[:, :, :-1] * dx_half
    states.face.x.plus.Q = states.Q[:, ngh:-ngh, ngh:-ngh+1] - states.slpx[:, :, 1:] * dx_half
    states.face.y.minus.Q = states.Q[:, ngh-1:-ngh, ngh:-ngh] + states.slpy[:, :-1, :] * dy_half
    states.face.y.plus.Q = states.Q[:, ngh:-ngh+1, ngh:-ngh] - states.slpy[:, 1:, :] * dy_half

    # get depth at cell interfaces
    for ornt in ["x", "y"]:
        for sign in ["minus", "plus"]:
            states.face[ornt][sign].U[0] = \
                states.face[ornt][sign].Q[0] - runtime.topo[ornt+"fcenters"]

    # fix negative depths at cell interfaces
    states = correct_negative_depth(states)

    # get non-consercative variables at cell faces
    for ornt in ["x", "y"]:
        for sign in ["minus", "plus"]:

            # fix rounding errors
            loc = cupy.nonzero(states.face[ornt][sign].U[0] < runtime.tol)
            states.face[ornt][sign].U[0, loc[0], loc[1]] = 0.

            # calculate velocities at wet interfaces
            loc = (states.face[ornt][sign].U[0] > config.params.drytol)
            states.face[ornt][sign].U[1:] = 0.
            states.face[ornt][sign].U[1:, loc] = \
                states.face[ornt][sign].Q[1:, loc] / states.face[ornt][sign].U[0, loc]

            # re-calculate conservative quantities at cell interfaces
            states.face[ornt][sign].Q[0] = \
                   states.face[ornt][sign].U[0] + runtime.topo[ornt+"fcenters"]
            states.face[ornt][sign].Q[1:] = \
                states.face[ornt][sign].U[0] * states.face[ornt][sign].U[1:]

    return states
