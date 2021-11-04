# Cython implementation of flux calculations
# TODO: once cython 0.3 is released, use `const cython.floating` for read-only buffers


cdef void get_discontinuous_flux_x(
    cython.floating[:, :, ::1] F,
    cython.floating[:, :, ::1] Q,  # TODO: read-only buffer
    cython.floating[:, :, ::1] U,  # TODO: read-only buffer
    const double gravity
) nogil except *:
    """Kernel of calculating discontinuous flux in x direction (in-place).
    """
    cdef Py_ssize_t nx = Q.shape[2]
    cdef Py_ssize_t ny = Q.shape[1]
    cdef Py_ssize_t k, j, i
    cdef cython.floating grav2 = gravity / 2.0

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


cdef void get_discontinuous_flux_y(
    cython.floating[:, :, ::1] F,
    cython.floating[:, :, ::1] Q,  # TODO: read-only buffer
    cython.floating[:, :, ::1] U,  # TODO: read-only buffer
    const double gravity
) nogil except *:
    """Kernel of calculating discontinuous flux in y direction.
    """
    cdef Py_ssize_t nx = Q.shape[2]
    cdef Py_ssize_t ny = Q.shape[1]
    cdef Py_ssize_t k, j, i
    cdef cython.floating grav2 = gravity / 2.0

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
        get_discontinuous_flux_x[cython.float](xmF, xmQ, xmU, gravity)
        get_discontinuous_flux_x[cython.float](xpF, xpQ, xpU, gravity)

        # face normal to y-direction: [hv, huv, hv^2+g(h^2)/2]
        get_discontinuous_flux_y[cython.float](ymF, ymQ, ymU, gravity)
        get_discontinuous_flux_y[cython.float](ypF, ypQ, ypU, gravity)
    elif dtype == numpy.double:
        # face normal to x-direction: [hu, hu^2 + g(h^2)/2, huv]
        get_discontinuous_flux_x[cython.double](xmF, xmQ, xmU, gravity)
        get_discontinuous_flux_x[cython.double](xpF, xpQ, xpU, gravity)

        # face normal to y-direction: [hv, huv, hv^2+g(h^2)/2]
        get_discontinuous_flux_y[cython.double](ymF, ymQ, ymU, gravity)
        get_discontinuous_flux_y[cython.double](ypF, ypQ, ypU, gravity)
    else:
        raise RuntimeError(f"Arrays are using an unrecognized dtype: {dtype}.")

    return states


cdef void central_scheme_kernel(
    cython.floating[:, :, ::1] H,
    cython.floating[:, :, ::1] Qm,  # TODO: read-only buffer
    cython.floating[:, :, ::1] Qp,  # TODO: read-only buffer
    cython.floating[:, :, ::1] Fm,  # TODO: read-only buffer
    cython.floating[:, :, ::1] Fp,  # TODO: read-only buffer
    cython.floating[:, ::1] Am,  # TODO: read-only buffer
    cython.floating[:, ::1] Ap  # TODO: read-only buffer
) nogil except *:
    """Kernel calculating common/numerical flux.
    """
    cdef Py_ssize_t nx = Qm.shape[2]
    cdef Py_ssize_t ny = Qm.shape[1]
    cdef Py_ssize_t k, j, i
    cdef cython.floating denominator
    cdef cython.floating coeff

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
        central_scheme_kernel[cython.float](xH, xmQ, xpQ, xmF, xpF, xma, xpa)
        central_scheme_kernel[cython.float](yH, ymQ, ypQ, ymF, ypF, yma, ypa)
    elif dtype == numpy.double:
        central_scheme_kernel[cython.double](xH, xmQ, xpQ, xmF, xpF, xma, xpa)
        central_scheme_kernel[cython.double](yH, ymQ, ypQ, ymF, ypF, yma, ypa)
    else:
        raise RuntimeError(f"Arrays are using an unrecognized dtype: {dtype}.")

    return states


cdef void local_speed_kernel(
    cython.floating[:, ::1] am,
    cython.floating[:, ::1] ap,
    cython.floating[:, ::1] hm,  # TODO: read-only buffer
    cython.floating[:, ::1] hp,  # TODO: read-only buffer
    cython.floating[:, ::1] um,  # TODO: read-only buffer
    cython.floating[:, ::1] up,  # TODO: read-only buffer
    const double gravity
) nogil except *:
    cdef Py_ssize_t ny = ap.shape[0]
    cdef Py_ssize_t nx = ap.shape[1]
    cdef Py_ssize_t i, j
    cdef cython.floating sqrt_ghm, sqrt_ghp

    for j in range(ny):
        for i in range(nx):
            sqrt_ghp = (hp[j, i] * gravity)**0.5
            sqrt_ghm = (hm[j, i] * gravity)**0.5
            ap[j, i] = max(max(up[j, i]+sqrt_ghp, um[j, i]+sqrt_ghm), 0.0)
            am[j, i] = min(min(up[j, i]-sqrt_ghp, um[j, i]-sqrt_ghm), 0.0)


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
        local_speed_kernel[cython.float](
            xm.a, xp.a, xm.U[0], xp.U[0], xm.U[1], xp.U[1], gravity)
        local_speed_kernel[cython.float](
            ym.a, yp.a, ym.U[0], yp.U[0], ym.U[2], yp.U[2], gravity)
    elif dtype == numpy.double:
        local_speed_kernel[cython.double](
            xm.a, xp.a, xm.U[0], xp.U[0], xm.U[1], xp.U[1], gravity)
        local_speed_kernel[cython.double](
            ym.a, yp.a, ym.U[0], yp.U[0], ym.U[2], yp.U[2], gravity)
    else:
        raise RuntimeError(f"Arrays are using an unrecognized dtype: {dtype}.")

    return states
