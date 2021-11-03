# Cython implementation of flux calculations
import numpy
from torchswe.utils.data import States as _States
cimport numpy
cimport cython
numpy.seterr(divide="ignore", invalid="ignore")


# fused floating point type; uses as the template varaible type in C++
ctypedef fused fptype:
    cython.float
    cython.double

ctypedef fused confptype:
    const cython.float
    const cython.double


@cython.boundscheck(False)  # deactivate bounds checking
@cython.wraparound(False)  # deactivate negative indexing.
cdef void get_discontinuous_flux_x(
    fptype[:, :, ::1] F,
    confptype[:, :, ::1] Q, confptype[:, :, ::1] U, const double gravity
) nogil:
    """Kernel of calculating discontinuous flux in x direction (in-place).
    """
    cdef Py_ssize_t nx = Q.shape[2]
    cdef Py_ssize_t ny = Q.shape[1]
    cdef Py_ssize_t k, j, i
    cdef fptype grav2 = gravity / 2.0

    # F[0] = hu
    for j in range(ny):
        for i in range(nx):
            F[0, j, i] = Q[1, j, i]

    # F[1] = hu * u + g/2 * h * h
    for j in range(ny):
        for i in range(nx):
            F[1, j, i] = Q[1, j, i] * U[1, j, i] + grav2 * (U[0, j, i] * U[0, j, i])

    # F[2] = hu * v
    for j in range(ny):
        for i in range(nx):
            F[2, j, i] = Q[1, j, i] * U[2, j, i]


@cython.boundscheck(False)  # deactivate bounds checking
@cython.wraparound(False)  # deactivate negative indexing.
cdef void get_discontinuous_flux_y(
    fptype[:, :, ::1] F,
    confptype[:, :, ::1] Q, confptype[:, :, ::1] U, const double gravity
) nogil:
    """Kernel of calculating discontinuous flux in y direction.
    """
    cdef Py_ssize_t nx = Q.shape[2]
    cdef Py_ssize_t ny = Q.shape[1]
    cdef Py_ssize_t k, j, i
    cdef fptype grav2 = gravity / 2.0

    # F[0] = hv
    for j in range(ny):
        for i in range(nx):
            F[0, j, i] = Q[2, j, i]

    # F[1] = u * hv
    for j in range(ny):
        for i in range(nx):
            F[1, j, i] = U[1, j, i] * Q[2, j, i]

    # F[2] = hv * v + g/2 * h * h
    for j in range(ny):
        for i in range(nx):
            F[2, j, i] = Q[2, j, i] * U[2, j, i] + grav2 * U[0, j, i] * U[0, j, i]


@cython.boundscheck(False)  # deactivate bounds checking
@cython.wraparound(False)  # deactivate negative indexing.
def get_discontinuous_flux(object states, double gravity):
    """Calculting the discontinuous fluxes on the both sides at cell faces.

    Arguments
    ---------
    states : torchswe.utils.data.States
    gravity : float

    Returns
    -------
    states : torchswe.utils.data.States
        The same object as the input. Changed inplace. Returning it just for coding style.
    """
    # aliases to reduce calls in the generated c/c++ code
    xm = states.face.x.minus
    xp = states.face.x.plus
    ym = states.face.y.minus
    yp = states.face.y.plus

    xmF = xm.F
    xpF = xp.F
    ymF = ym.F
    ypF = yp.F
    xmQ = xm.Q
    xpQ = xp.Q
    ymQ = ym.Q
    ypQ = yp.Q
    xmU = xm.U
    xpU = xp.U
    ymU = ym.U
    ypU = yp.U

    dtype = xmF.dtype

    if dtype == numpy.single:
        # face normal to x-direction: [hu, hu^2 + g(h^2)/2, huv]
        get_discontinuous_flux_x[cython.float, cython.float](xmF, xmQ, xmU, gravity)
        get_discontinuous_flux_x[cython.float, cython.float](xpF, xpQ, xpU, gravity)

        # face normal to y-direction: [hv, huv, hv^2+g(h^2)/2]
        get_discontinuous_flux_y[cython.float, cython.float](ymF, ymQ, ymU, gravity)
        get_discontinuous_flux_y[cython.float, cython.float](ypF, ypQ, ypU, gravity)
    elif dtype == numpy.double:
        # face normal to x-direction: [hu, hu^2 + g(h^2)/2, huv]
        get_discontinuous_flux_x[cython.double, cython.double](xmF, xmQ, xmU, gravity)
        get_discontinuous_flux_x[cython.double, cython.double](xpF, xpQ, xpU, gravity)

        # face normal to y-direction: [hv, huv, hv^2+g(h^2)/2]
        get_discontinuous_flux_y[cython.double, cython.double](ymF, ymQ, ymU, gravity)
        get_discontinuous_flux_y[cython.double, cython.double](ypF, ypQ, ypU, gravity)
    else:
        raise RuntimeError(f"Arrays are using an unrecognized dtype: {dtype}.")

    return states


@cython.boundscheck(False)  # deactivate bounds checking
@cython.wraparound(False)  # deactivate negative indexing.
cdef void central_scheme_kernel(
    fptype[:, :, ::1] H,
    confptype[:, :, ::1] Qm, confptype[:, :, ::1] Qp, confptype[:, :, ::1] Fm,
    confptype[:, :, ::1] Fp, confptype[:, ::1] Am, confptype[:, ::1] Ap
) nogil:
    """Kernel calculating common/numerical flux.
    """
    cdef Py_ssize_t nx = Qm.shape[2]
    cdef Py_ssize_t ny = Qm.shape[1]
    cdef Py_ssize_t k, j, i
    cdef fptype denominator
    cdef fptype coeff

    for k in range(3):
        for j in range(ny):
            for i in range(nx):
                denominator = Ap[j, i] - Am[j, i]

                # NOTE ============================================================================
                # If `demoninator` is zero, then both `Ap` and `Am` should also be zero.
                # =================================================================================
                if denominator == 0.0:
                    continue  # implying H[k, j, i] is simply 0

                coeff = Ap[j, i] * Am[j, i]

                H[k, j, i] = (
                    Ap[j, i] * Fm[k, j, i] - Am[j, i] * Fp[k, j, i] +
                    coeff * (Qp[k, j, i] - Qm[k, j, i])
                ) / denominator


@cython.boundscheck(False)  # deactivate bounds checking
@cython.wraparound(False)  # deactivate negative indexing.
def central_scheme(object states):
    """A central scheme to calculate numerical flux at interfaces.

    Arguments
    ---------
    states : torchswe.utils.data.States

    Returns
    -------
    states : torchswe.utils.data.States
        The same object as the input. Updated in-place. Returning it just for coding style.
    """
    # aliases to reduce calls in the generated c/c++ code
    x = states.face.x
    y = states.face.y
    xm = x.minus
    xp = x.plus
    ym = y.minus
    yp = y.plus

    xmF = xm.F
    xpF = xp.F
    ymF = ym.F
    ypF = yp.F
    xmQ = xm.Q
    xpQ = xp.Q
    ymQ = ym.Q
    ypQ = yp.Q
    xma = xm.a
    xpa = xp.a
    yma = ym.a
    ypa = yp.a
    xH = x.H
    yH = y.H

    dtype = yH.dtype

    if dtype == numpy.single:
        central_scheme_kernel[cython.float, cython.float](xH, xmQ, xpQ, xmF, xpF, xma, xpa)
        central_scheme_kernel[cython.float, cython.float](yH, ymQ, ypQ, ymF, ypF, yma, ypa)
    elif dtype == numpy.double:
        central_scheme_kernel[cython.double, cython.double](xH, xmQ, xpQ, xmF, xpF, xma, xpa)
        central_scheme_kernel[cython.double, cython.double](yH, ymQ, ypQ, ymF, ypF, yma, ypa)
    else:
        raise RuntimeError(f"Arrays are using an unrecognized dtype: {dtype}.")

    return states


@cython.boundscheck(False)  # deactivate bounds checking
@cython.wraparound(False)  # deactivate negative indexing.
cdef void local_speed_kernel(
    fptype[:, ::1] am, fptype[:, ::1] ap,
    confptype[:, ::1] hm, confptype[:, ::1] hp,
    confptype[:, ::1] um, confptype[:, ::1] up,
    const double gravity
) nogil:
    cdef Py_ssize_t ny = ap.shape[0]
    cdef Py_ssize_t nx = ap.shape[1]
    cdef Py_ssize_t i, j
    cdef fptype sqrt_ghm, sqrt_ghp

    for j in range(ny):
        for i in range(nx):
            sqrt_ghp = (hp[j, i] * gravity)**0.5
            sqrt_ghm = (hm[j, i] * gravity)**0.5
            ap[j, i] = max(max(up[j, i]+sqrt_ghp, um[j, i]+sqrt_ghm), 0.0)
            am[j, i] = min(min(up[j, i]-sqrt_ghp, um[j, i]-sqrt_ghm), 0.0)


@cython.boundscheck(False)  # deactivate bounds checking
@cython.wraparound(False)  # deactivate negative indexing.
def get_local_speed(object states, double gravity):
    """Calculate local speeds on the two sides of cell faces.

    Arguments
    ---------
    states : torchswe.utils.data.States
    gravity : float
        Gravity in m / s^2.

    Returns
    -------
    states : torchswe.utils.data.States
        The same object as the input. Changed inplace. Returning it just for coding style.
    """
    x = states.face.x
    xp = x.plus
    xm = x.minus

    y = states.face.y
    yp = y.plus
    ym = y.minus

    dtype = xp.a.dtype

    if dtype == numpy.single:
        local_speed_kernel[cython.float, cython.float](
            xm.a, xp.a, xm.U[0], xp.U[0], xm.U[1], xp.U[1], gravity)
        local_speed_kernel[cython.float, cython.float](
            ym.a, yp.a, ym.U[0], yp.U[0], ym.U[2], yp.U[2], gravity)
    elif dtype == numpy.double:
        local_speed_kernel[cython.double, cython.double](
            xm.a, xp.a, xm.U[0], xp.U[0], xm.U[1], xp.U[1], gravity)
        local_speed_kernel[cython.double, cython.double](
            ym.a, yp.a, ym.U[0], yp.U[0], ym.U[2], yp.U[2], gravity)
    else:
        raise RuntimeError(f"Arrays are using an unrecognized dtype: {dtype}.")

    return states
