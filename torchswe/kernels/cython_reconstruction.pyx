# vim:fenc=utf-8
# vim:ft=pyrex
# TODO: once cython 0.3 is released, use `const cython.floating` for read-only buffers


cdef inline cython.floating non_zero_slope(
    cython.floating qi,
    cython.floating qim1,
    cython.floating denominator,
    cython.floating theta
) nogil except *:
    """Uitlity function helping calculate slops.

    Note: the delta (dx or dy) is already eliminated, because (diff/dx) * (dx/2) = diff / 2.
    However, this is based on the assumption that dx (or dy) is a constant, i.e., uniform grid.
    """
    cdef cython.floating slp = (qi - qim1) / denominator
    slp = max(min(min(theta*slp, (1.0+slp)/2.0), theta), 0.)
    slp *= denominator
    slp /= 2.0
    return slp


cdef void extrapolate_minmod_x(
    cython.floating[:, :, ::1] xmQ,
    cython.floating[:, :, ::1] xpQ,
    cython.floating[:, :, ::1] cQ,  # TODO: read-only buffer
    const double theta,
    const Py_ssize_t ngh
) nogil except *:
    cdef Py_ssize_t ny = cQ.shape[1] - 2 * ngh
    cdef Py_ssize_t nx = cQ.shape[2] - 2 * ngh
    cdef Py_ssize_t i, j, k
    cdef Py_ssize_t jgh, igh_i, igh_im1, igh_ip1, im

    cdef cython.floating qi
    cdef cython.floating slp
    cdef cython.floating ip1mi

    # initialize xmQ
    for k in range(3):
        for j in range(ny):
            for i in range(nx+1):
                xmQ[k, j, i] = cQ[k, j+ngh, i+ngh-1]

    # initialize xpQ
    for k in range(3):
        for j in range(ny):
            for i in range(nx+1):
                xpQ[k, j, i] = cQ[k, j+ngh, i+ngh]

    for k in range(3):
        jgh = ngh
        for j in range(ny):
            igh_i = ngh - 1; igh_im1 = ngh - 2; igh_ip1 = ngh; im = 0

            # edge @ igh_i = ngh - 1 (the ghost cell immediately next to the left boundary)
            qi = cQ[k, jgh, igh_i]
            ip1mi = cQ[k, jgh, igh_ip1] - qi
            if ip1mi != 0.:  # if it's exactly zero, imnplying a zero slope, no need to exrtapolate
                xmQ[k, j, im] += non_zero_slope(qi, cQ[k, jgh, igh_im1], ip1mi, theta)
            igh_i += 1; igh_im1 +=1; igh_ip1 += 1; im += 1

            # internal cells
            for i in range(nx):
                qi = cQ[k, jgh, igh_i]
                ip1mi = cQ[k, jgh, igh_ip1] - qi
                if ip1mi != 0.0:
                    slp = non_zero_slope(qi, cQ[k, jgh, igh_im1], ip1mi, theta)
                    xmQ[k, j, im] += slp
                    xpQ[k, j, i] -= slp
                igh_i += 1; igh_im1 +=1; igh_ip1 += 1; im += 1

            # edge @ igh_i = nx + ngh (the ghost cell immediately next to the right boundary)
            qi = cQ[k, jgh, igh_i]
            ip1mi = cQ[k, jgh, igh_ip1] - qi
            if ip1mi != 0.:
                xpQ[k, j, nx] -= non_zero_slope(qi, cQ[k, jgh, igh_im1], ip1mi, theta)

            # update counter
            jgh += 1


cdef void extrapolate_minmod_y(
    cython.floating[:, :, ::1] ymQ,
    cython.floating[:, :, ::1] ypQ,
    cython.floating[:, :, ::1] cQ,  # TODO: read-only buffer
    const double theta,
    const Py_ssize_t ngh
) nogil except *:
    cdef Py_ssize_t ny = cQ.shape[1] - 2 * ngh
    cdef Py_ssize_t nx = cQ.shape[2] - 2 * ngh
    cdef Py_ssize_t i, j, k
    cdef Py_ssize_t igh, jgh_i, jgh_im1, jgh_ip1, jm

    cdef cython.floating qi
    cdef cython.floating slp
    cdef cython.floating ip1mi

    # initialize ymQ
    for k in range(3):
        for j in range(ny+1):
            for i in range(nx):
                ymQ[k, j, i] = cQ[k, j+ngh-1, i+ngh]

    # initialize ypQ
    for k in range(3):
        for j in range(ny+1):
            for i in range(nx):
                ypQ[k, j, i] = cQ[k, j+ngh, i+ngh]

    for k in range(3):
        jgh_i = ngh - 1; jgh_im1 = ngh - 2; jgh_ip1 = ngh; jm = 0

        # edge case jgh_i = ngh-1 (the ghost cell immediately below the bottom boundary)
        igh = ngh
        for i in range(nx):
            qi = cQ[k, jgh_i, igh]
            ip1mi = cQ[k, jgh_ip1, igh] - qi
            if ip1mi != 0.:
                ymQ[k, jm, i] += non_zero_slope(qi, cQ[k, jgh_im1, igh], ip1mi, theta)
            igh += 1
        jgh_i += 1; jgh_im1 += 1; jgh_ip1 += 1; jm += 1

        # internal
        for j in range(ny):
            igh = ngh
            for i in range(nx):
                qi = cQ[k, jgh_i, igh]
                ip1mi = cQ[k, jgh_ip1, igh] - qi
                if ip1mi != 0.0:
                    slp = non_zero_slope(qi, cQ[k, jgh_im1, igh], ip1mi, theta)
                    ymQ[k, jm, i] += slp
                    ypQ[k, j, i] -= slp
                igh += 1
            jgh_i += 1; jgh_im1 += 1; jgh_ip1 += 1; jm += 1

        # edge case jgh_i = ny+ngh (the ghost cell immediately on top of the upper boundary)
        igh = ngh
        for i in range(nx):
            qi = cQ[k, jgh_i, igh]
            ip1mi = cQ[k, jgh_ip1, igh] - qi
            if ip1mi != 0.:
                ypQ[k, ny, i] -= non_zero_slope(qi, cQ[k, jgh_im1, igh], ip1mi, theta)
            igh += 1


cdef inline void recnstrt_face_depth(
    cython.floating[:, :, ::1] U,
    cython.floating[:, :, ::1] Q,  # TODO: read-only buffer
    cython.floating[:, ::1] B,  # TODO: read-only buffer
) nogil except *:
    cdef Py_ssize_t ny = U.shape[1]
    cdef Py_ssize_t nx = U.shape[2]
    cdef Py_ssize_t i, j
    for j in range(ny):
        for i in range(nx):
            U[0, j, i] = Q[0, j, i] - B[j, i]


cdef inline void fix_face_negative_depth_x(
    cython.floating[:, :, ::1] mU,
    cython.floating[:, :, ::1] pU,
    cython.floating[:, ::1] cH  # TODO: read-only buffer
) nogil except *:
    """Fix negative depths at cell faces normal to x axis.

    Arguments
    ---------
    mU : memoryview with shape (3, ny, nx+1)
    pU : memoryview with shape (3, ny, nx+1)
    cH : memoryview with shape (ny+2, nx+2)
    """
    cdef Py_ssize_t ny = cH.shape[0] - 2
    cdef Py_ssize_t nx = cH.shape[1] - 2
    cdef Py_ssize_t i, j, im
    cdef cython.floating cHij_2, mUij, pUij

    for j in range(ny):
        # minus sign at the most left cell face
        mUij = mU[0, j, 0]
        cHij_2 = cH[j+1, 0] * 2.0
        if mUij < 0.0:
            mU[0, j, 0] = 0.0
        elif mUij > cHij_2:
            mU[0, j, 0] = cHij_2

        # the left and right faces of non-ghost cells
        im = 1
        for i in range(nx):
            mUij = mU[0, j, im]
            cHij_2 = cH[j+1, i+1] * 2.0
            if mUij < 0.0:
                mU[0, j, im] = 0.0
                pU[0, j, i] = cHij_2
            elif mUij > cHij_2:
                mU[0, j, im] = cHij_2
                pU[0, j, i] = 0.0
            im += 1

        # plus sign at the most left cell face
        pUij = pU[0, j, nx]
        cHij_2 = cH[j+1, nx+1] * 2.0
        if pUij < 0.0:
            pU[0, j, nx] = 0.0
        elif pUij > cHij_2:
            pU[0, j, nx] = cHij_2


cdef inline void fix_face_negative_depth_y(
    cython.floating[:, :, ::1] mU,
    cython.floating[:, :, ::1] pU,
    cython.floating[:, ::1] cH  # TODO: read-only buffer
) nogil except *:
    """Fix negative depths at cell faces normal to y axis.

    Arguments
    ---------
    mU : memoryview with shape (3, ny+1, nx)
    pU : memoryview with shape (3, ny+1, nx)
    cH : memoryview with shape (ny+2, nx+2)
    """
    cdef Py_ssize_t ny = cH.shape[0] - 2
    cdef Py_ssize_t nx = cH.shape[1] - 2
    cdef Py_ssize_t i, j, jm
    cdef cython.floating cHij_2, mUij, pUij

    # minus sign at the most bottom cell face
    for i in range(nx):
        mUij = mU[0, 0, i]
        cHij_2 = cH[0, i+1] * 2.0
        if mUij < 0.0:
            mU[0, 0, i] = 0.0
        elif mUij > cHij_2:
            mU[0, 0, i] = cHij_2

    # the bottom and top faces of non-ghost cells
    jm = 1
    for j in range(ny):
        for i in range(nx):
            mUij = mU[0, jm, i]
            cHij_2 = cH[j+1, i+1] * 2.0
            if mUij < 0.0:
                mU[0, jm, i] = 0.0
                pU[0, j, i] = cHij_2
            elif mUij > cHij_2:
                mU[0, jm, i] = cHij_2
                pU[0, j, i] = 0.0
        jm += 1

    # plus sign at the most top cell face
    for i in range(nx):
        pUij = pU[0, ny, i]
        cHij_2 = cH[ny+1, i+1] * 2.0
        if pUij < 0.0:
            pU[0, ny, i] = 0.0
        elif pUij > cHij_2:
            pU[0, ny, i] = cHij_2


cdef inline void recnstrt_face_velocity(
    cython.floating[:, :, ::1] U,
    cython.floating[:, :, ::1] Q,  # TODO: read-only buffer
    const double drytol,
    const double tol
) nogil except *:

    cdef Py_ssize_t n1 = U.shape[1]
    cdef Py_ssize_t n2 = U.shape[2]
    cdef Py_ssize_t i, j

    for j in range(n1):
        for i in range(n2):
            if U[0, j, i] < tol:  # smaller than floating point tolerance, treat as a dry cell
                U[0, j, i] = 0.0
                U[1, j, i] = 0.0
                U[2, j, i] = 0.0
            elif U[0, j, i] < drytol:  # a wet cell but not flowing
                U[1, j, i] = 0.0
                U[2, j, i] = 0.0
            else:  # a wet and flowing cell
                U[1, j, i] = Q[1, j, i] / U[0, j, i]
                U[2, j, i] = Q[2, j, i] / U[0, j, i]


cdef inline void recnstrt_face_conservatives(
    cython.floating[:, :, ::1] Q,
    cython.floating[:, :, ::1] U,  # TODO: read-only buffer
    cython.floating[:, ::1] B,  # TODO: read-only buffer
) nogil except *:

    cdef Py_ssize_t n1 = U.shape[1]
    cdef Py_ssize_t n2 = U.shape[2]
    cdef Py_ssize_t i, j

    for j in range(n1):
        for i in range(n2):
            Q[0, j, i] = U[0, j, i] + B[j, i]  # w = h + b

    for j in range(n1):
        for i in range(n2):
            Q[1, j, i] = U[0, j, i] * U[1, j, i]  # hu = h * u

    for j in range(n1):
        for i in range(n2):
            Q[2, j, i] = U[0, j, i] * U[2, j, i]  # hv = h * v


cdef void reconstruct_kernel(
    cython.floating[:, :, ::1] xmQ, cython.floating[:, :, ::1] xpQ,
    cython.floating[:, :, ::1] ymQ, cython.floating[:, :, ::1] ypQ,
    cython.floating[:, :, ::1] xmU, cython.floating[:, :, ::1] xpU,
    cython.floating[:, :, ::1] ymU, cython.floating[:, :, ::1] ypU,
    cython.floating[:, :, ::1] cQ,  # TODO: read-only buffer
    cython.floating[:, ::1] cH,   # TODO: read-only buffer
    cython.floating[:, ::1] cB,  # TODO: read-only buffer
    cython.floating[:, ::1] xB,  # TODO: read-only buffer
    cython.floating[:, ::1] yB,   # TODO: read-only buffer
    const Py_ssize_t ngh, const double drytol, const double theta, const double tol
) nogil except *:

    # extrapolate conservative quantites to cell faces; got xmQ, xpQ, ymQ, and ypQ
    extrapolate_minmod_x(xmQ, xpQ, cQ, theta, ngh)
    extrapolate_minmod_y(ymQ, ypQ, cQ, theta, ngh)

    # calculate depths at xm, xp, ym, and yp
    recnstrt_face_depth(xmU, xmQ, xB)
    recnstrt_face_depth(xpU, xpQ, xB)
    recnstrt_face_depth(ymU, ymQ, yB)
    recnstrt_face_depth(ypU, ypQ, yB)

    # fix negative depths at cell faces
    fix_face_negative_depth_x(xmU, xpU, cH)
    fix_face_negative_depth_y(ymU, ypU, cH)

    # calculate cell faces' velocity at x minus side
    recnstrt_face_velocity(xmU, xmQ, drytol, tol)
    recnstrt_face_velocity(xpU, xpQ, drytol, tol)
    recnstrt_face_velocity(ymU, ymQ, drytol, tol)
    recnstrt_face_velocity(ypU, ypQ, drytol, tol)

    # reconstruct cell faces' conservative quantities at x minus side
    recnstrt_face_conservatives(xmQ, xmU, xB)
    recnstrt_face_conservatives(xpQ, xpU, xB)
    recnstrt_face_conservatives(ymQ, ymU, yB)
    recnstrt_face_conservatives(ypQ, ypU, yB)


def reconstruct(object states, object runtime, object config):
    """Reconstructs quantities at cell interfaces and centers.

    The following quantities in `states` are updated in this function:
        1. discontinuous non-conservative quantities defined at cell interfaces
        2. discontinuous conservative quantities defined at cell interfaces

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

    # aliases
    face = states.face
    x = face.x
    xm = x.minus
    xp = x.plus
    y = face.y
    ym = y.minus
    yp = y.plus
    topo = runtime.topo
    params = config.params
    dtype = states.Q.dtype

    if dtype == numpy.single:
        reconstruct_kernel[cython.float](
            xm.Q, xp.Q, ym.Q, yp.Q, xm.U, xp.U, ym.U, yp.U, states.Q, states.H,
            runtime.topo.centers, topo.xfcenters, topo.yfcenters,
            states.ngh, params.drytol, params.theta, runtime.tol
        )
    elif dtype == numpy.double:
        reconstruct_kernel[cython.double](
            xm.Q, xp.Q, ym.Q, yp.Q, xm.U, xp.U, ym.U, yp.U, states.Q, states.H,
            runtime.topo.centers, topo.xfcenters, topo.yfcenters,
            states.ngh, params.drytol, params.theta, runtime.tol
        )
    else:
        raise RuntimeError(f"Arrays are using an unrecognized dtype: {dtype}.")

    return states


cdef inline void recnstrt_center_depth(
    cython.floating[:, ::1] H,
    cython.floating[:, :, ::1] Q,  # TODO: read-only buffer
    cython.floating[:, ::1] B,  # TODO: read-only buffer
    const Py_ssize_t ngh
) nogil except *:
    """Get the cell-centered depths for non-halo cells.

    Arguments
    ---------
    H : memoryview with shape (ny+2, nx+2)
    Q : memoryview with shape (ny+2*ngh, nx+2*ngh)
    B : memoryview with shape (ny, nx)
    """
    cdef Py_ssize_t ny = H.shape[0] - 2
    cdef Py_ssize_t nx = H.shape[1] - 2
    cdef Py_ssize_t i, j, iq, jq, ih, jh

    jq = ngh; jh = 1
    for j in range(ny):
        iq = ngh; ih = 1
        for i in range(nx):
            H[jh, ih] = Q[0, jq, iq] - B[j, i]
            iq += 1; ih += 1
        jq += 1; jh += 1


def get_cell_center_depth(object states, object runtime):
    """Calculate cell-centered depths for non-halo-ring cells.

    `states.H` will be updated in this function.

    Arguments
    ---------
    states : torchswe.utils.data.States
    runtime : torchswe.utils.misc.DummyDict

    Returns
    -------
    states : torchswe.utils.data.States
        The same object as the input. Returning it just for coding style. The values are actually
        updated in-place.
    """

    dtype = states.Q.dtype

    if dtype == numpy.single:
        recnstrt_center_depth[cython.float](states.H, states.Q, runtime.topo.centers, states.ngh)
    elif dtype == numpy.double:
        recnstrt_center_depth[cython.double](states.H, states.Q, runtime.topo.centers, states.ngh)
    else:
        raise RuntimeError(f"Arrays are using an unrecognized dtype: {dtype}.")

    return states
