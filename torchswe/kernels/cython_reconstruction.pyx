# vim:fenc=utf-8
# vim:ft=pyrex
cimport cython


# mimic CuPy's raw kernel so that we can have a similar code structure
# (somehow this kernel doesn't work well with fused types and nogil, so we do it manually)
cdef inline double _minmod_slope_raw_kernel_double(
    double s1, double s2, double s3, double theta,
) nogil except *:

    cdef double ans;
    cdef double denominator = s3 - s2;

    if denominator == 0.0: # exact zero; no tolerance
        return 0.0;

    ans = s2 - s1;
    if ans == 0.0 or ans == -1.0:  # exact zero or -1; no tolerance
        return 0.0;

    ans /= denominator;
    ans = min(ans*theta, (1.0+ans) / 2.0);
    ans = min(ans, theta);
    ans = max(ans, 0.);
    ans *= denominator;
    ans /= 2.0;
    return ans


# mimic CuPy's raw kernel so that we can have a similar code structure
# (somehow this kernel doesn't work well with fused types and nogil, so we do it manually)
cdef inline float _minmod_slope_raw_kernel_float(
    float s1, float s2, float s3, float theta,
) nogil except *:

    cdef float ans;
    cdef float denominator = s3 - s2;

    if denominator == 0.0: # exact zero; no tolerance
        return 0.0;

    ans = s2 - s1;
    if ans == 0.0 or ans == -1.0:  # exact zero or -1; no tolerance
        return 0.0;

    ans /= denominator;
    ans = min(ans*theta, (1.0+ans) / 2.0);
    ans = min(ans, theta);
    ans = max(ans, 0.);
    ans *= denominator;
    ans /= 2.0;
    return ans


cdef inline void _minmod_slope_kernel(
    const cython.floating[:, :, :] s1,
    const cython.floating[:, :, :] s2,
    const cython.floating[:, :, :] s3,
    const cython.floating theta,
    cython.floating[:, :, ::1] slp
) nogil except *:
    """Uitlity function helping calculate slops.

    Note: the delta (dx or dy) is already eliminated, because (diff/dx) * (dx/2) = diff / 2.
    However, this is based on the assumption that dx (or dy) is a constant, i.e., uniform grid.
    """
    cdef Py_ssize_t nfields = slp.shape[0];
    cdef Py_ssize_t ny = slp.shape[1];
    cdef Py_ssize_t nx = slp.shape[2];
    cdef Py_ssize_t k, j, i;
    cdef cython.floating ans, denominator;

    for k in range(nfields):
        for j in range(ny):
            for i in range(nx):

                # compile time decision; no runtime overhead
                if cython.floating is double:
                    slp[k, j, i] = _minmod_slope_raw_kernel_double(
                        s1[k, j, i], s2[k, j, i], s3[k, j, i], theta)
                elif cython.floating is float:
                    slp[k, j, i] = _minmod_slope_raw_kernel_float(
                        s1[k, j, i], s2[k, j, i], s3[k, j, i], theta)


cdef inline void _add3(
    const cython.floating[:, :, :] x,
    const cython.floating[:, :, :] y,
    cython.floating[:, :, :] out,
) nogil except *:
    """Addition.

    Note sure why, but cython kernel is faster than numpy's.
    """
    cdef Py_ssize_t n1 = out.shape[0];
    cdef Py_ssize_t n2 = out.shape[1];
    cdef Py_ssize_t n3 = out.shape[2];
    cdef Py_ssize_t k, j, i;

    for k in range(n1):
        for j in range(n2):
            for i in range(n3):
                out[k, j, i] = x[k, j, i] + y[k, j, i]


cdef inline void _subtract3(
    const cython.floating[:, :, :] x,
    const cython.floating[:, :, :] y,
    cython.floating[:, :, :] out,
) nogil except *:
    """Subtraction.

    Note sure why, but cython kernel is faster than numpy's.
    """
    cdef Py_ssize_t n1 = out.shape[0];
    cdef Py_ssize_t n2 = out.shape[1];
    cdef Py_ssize_t n3 = out.shape[2];
    cdef Py_ssize_t k, j, i;

    for k in range(n1):
        for j in range(n2):
            for i in range(n3):
                out[k, j, i] = x[k, j, i] - y[k, j, i]


cdef inline void _subtract2(
    const cython.floating[:, :, :] x,
    const cython.floating[:, :] y,
    cython.floating[:, :, :] out,
) nogil except *:
    """Subtraction.

    Note sure why, but cython kernel is faster than numpy's.
    """
    cdef Py_ssize_t n1 = out.shape[0];
    cdef Py_ssize_t n2 = out.shape[1];
    cdef Py_ssize_t n3 = out.shape[2];
    cdef Py_ssize_t k, j, i;

    for k in range(n1):
        for j in range(n2):
            for i in range(n3):
                out[k, j, i] = x[k, j, i] - y[j, i]


cdef inline void _fix_face_depth_internal(
    const cython.floating[:, :] Hc,
    const cython.floating tol,
    cython.floating[:, :] Hl,
    cython.floating[:, :] Hr
) nogil except *:
    cdef Py_ssize_t ny = Hc.shape[0];
    cdef Py_ssize_t nx = Hc.shape[1];
    cdef Py_ssize_t j, i;

    cdef cython.floating hc;

    for j in range(ny):
        for i in range(nx):
            hc = Hc[j, i];  # aliases to reduce calculation of raveled indices
            if (hc < tol):
                Hl[j, i] = 0.0;
                Hr[j, i] = 0.0;
            elif (Hl[j, i] < tol):
                Hl[j, i] = 0.0;
                Hr[j, i] = hc * 2.0;
            elif (Hr[j, i] < tol):
                Hr[j, i] = 0.0;
                Hl[j, i] = hc * 2.0;


cdef inline void _fix_face_depth_edge(
    const cython.floating[:] Hc,
    const cython.floating tol,
    cython.floating[:] H,
) nogil except *:
    cdef Py_ssize_t n = H.shape[0];
    cdef Py_ssize_t i;

    cdef cython.floating h, hc, hc2;

    for i in range(n):
        h = H[i];
        hc = Hc[i];
        hc2 = hc * 2.0;
        if hc < tol:
            H[i] = 0.0;
        elif h < tol:
            H[i] = 0.0;
        elif h > (hc2 - tol):
            H[i] = hc2;


cdef inline void _recnstrt_face_velocity(
    const cython.floating[:, ::1] H,
    const cython.floating[:, ::1] HU,
    const cython.floating[:, ::1] HV,
    const cython.floating drytol,
    cython.floating[:, ::1] U,
    cython.floating[:, ::1] V,
) nogil except *:
    cdef Py_ssize_t n1 = H.shape[0];
    cdef Py_ssize_t n2 = H.shape[1];
    cdef Py_ssize_t j, i;

    for j in range(n1):
        for i in range(n2):
            if H[j, i] <= drytol:
                U[j, i] = 0.0;
                V[j, i] = 0.0;
            else:
                U[j, i] = HU[j, i] / H[j, i];
                V[j, i] = HV[j, i] / H[j, i];


cdef inline void _recnstrt_face_conservatives(
    const cython.floating[:, ::1] H,
    const cython.floating[:, ::1] U,
    const cython.floating[:, ::1] V,
    const cython.floating[:, ::1] B,
    cython.floating[:, ::1] W,
    cython.floating[:, ::1] HU,
    cython.floating[:, ::1] HV,
) nogil except *:
    cdef Py_ssize_t ny = H.shape[0];
    cdef Py_ssize_t nx = H.shape[1];
    cdef Py_ssize_t j, i;
    cdef cython.floating h;

    for j in range(ny):
        for i in range(nx):
            h = H[j, i];
            W[j, i] = h + B[j, i];
            HU[j, i] = h * U[j, i];
            HV[j, i] = h * V[j, i];


cdef inline void _reconstruct(
    const cython.floating[:, :, ::1] Q,
    const cython.floating[:, :, ::1] U,
    const cython.floating[:, ::1] xfcenters,
    const cython.floating[:, ::1] yfcenters,
    const Py_ssize_t ngh,
    const cython.floating theta,
    const cython.floating drytol,
    const cython.floating tol,
    cython.floating[:, :, ::1] slpx,
    cython.floating[:, :, ::1] slpy,
    cython.floating[:, :, ::1] xmQ,
    cython.floating[:, :, ::1] xmU,
    cython.floating[:, :, ::1] xpQ,
    cython.floating[:, :, ::1] xpU,
    cython.floating[:, :, ::1] ymQ,
    cython.floating[:, :, ::1] ymU,
    cython.floating[:, :, ::1] ypQ,
    cython.floating[:, :, ::1] ypU,
) nogil except *:

    cdef Py_ssize_t ny = Q.shape[1] - 2 * ngh
    cdef Py_ssize_t nx = Q.shape[2] - 2 * ngh
    cdef Py_ssize_t xbg = ngh
    cdef Py_ssize_t xed = ngh + nx
    cdef Py_ssize_t ybg = ngh
    cdef Py_ssize_t yed = ngh + ny

    # slopes for w, hu, and hv in x and y
    _minmod_slope_kernel[cython.floating](Q[:, ybg:yed, xbg-2:xed], Q[:, ybg:yed, xbg-1:xed+1], Q[:, ybg:yed, xbg:xed+2], theta, slpx)
    _minmod_slope_kernel[cython.floating](Q[:, ybg-2:yed, xbg:xed], Q[:, ybg-1:yed+1, xbg:xed], Q[:, ybg:yed+2, xbg:xed], theta, slpy)

    # extrapolate discontinuous w, hu, and hv
    _add3[cython.floating](Q[:, ybg:yed, xbg-1:xed], slpx[:, :, :nx+1], xmQ)
    _subtract3[cython.floating](Q[:, ybg:yed, xbg:xed+1], slpx[:, :, 1:], xpQ)
    _add3[cython.floating](Q[:, ybg-1:yed, xbg:xed], slpy[:, :ny+1, :], ymQ)
    _subtract3[cython.floating](Q[:, ybg:yed+1, xbg:xed], slpy[:, 1:, :], ypQ)

    # calculate depth at cell faces
    _subtract2[cython.floating](xmQ[:1], xfcenters, xmU[:1])
    _subtract2[cython.floating](xpQ[:1], xfcenters, xpU[:1])
    _subtract2[cython.floating](ymQ[:1], yfcenters, ymU[:1])
    _subtract2[cython.floating](ypQ[:1], yfcenters, ypU[:1])

    # fix negative depths in x direction
    _fix_face_depth_internal[cython.floating](U[0, ybg:yed, xbg:xed], tol, xpU[0, :, :nx], xmU[0, :, 1:])
    _fix_face_depth_edge[cython.floating](U[0, ybg:yed, xbg-1], tol, xmU[0, :, 0])
    _fix_face_depth_edge[cython.floating](U[0, ybg:yed, xed], tol, xpU[0, :, nx])

    # fix negative depths in y direction
    _fix_face_depth_internal[cython.floating](U[0, ybg:yed, xbg:xed], tol, ypU[0, :ny, :], ymU[0, 1:, :])
    _fix_face_depth_edge[cython.floating](U[0, ybg-1, xbg:xed], tol, ymU[0, 0, :])
    _fix_face_depth_edge[cython.floating](U[0, yed, xbg:xed], tol, ypU[0, ny, :])

    # reconstruct velocity at cell faces in x and y directions
    _recnstrt_face_velocity[cython.floating](xpU[0], xpQ[1], xpQ[2], drytol, xpU[1], xpU[2])
    _recnstrt_face_velocity[cython.floating](xmU[0], xmQ[1], xmQ[2], drytol, xmU[1], xmU[2])
    _recnstrt_face_velocity[cython.floating](ypU[0], ypQ[1], ypQ[2], drytol, ypU[1], ypU[2])
    _recnstrt_face_velocity[cython.floating](ymU[0], ymQ[1], ymQ[2], drytol, ymU[1], ymU[2])

    # reconstruct conservative quantities at cell faces
    _recnstrt_face_conservatives[cython.floating](xmU[0], xmU[1], xmU[2], xfcenters, xmQ[0], xmQ[1], xmQ[2])
    _recnstrt_face_conservatives[cython.floating](xpU[0], xpU[1], xpU[2], xfcenters, xpQ[0], xpQ[1], xpQ[2])
    _recnstrt_face_conservatives[cython.floating](ymU[0], ymU[1], ymU[2], yfcenters, ymQ[0], ymQ[1], ymQ[2])
    _recnstrt_face_conservatives[cython.floating](ypU[0], ypU[1], ypU[2], yfcenters, ypQ[0], ypQ[1], ypQ[2])


def reconstruct(object states, object runtime, object config) -> object:
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

    # aliases to save object look-up time in Python's underlying dictionary
    cdef object Q = states.q
    cdef object U = states.p
    cdef object slpx = states.slpx
    cdef object slpy = states.slpy
    cdef object face = states.face
    cdef object fx = face.x
    cdef object xm = fx.minus
    cdef object xmQ = xm.q
    cdef object xmU = xm.p
    cdef object xp = fx.plus
    cdef object xpQ = xp.q
    cdef object xpU = xp.p
    cdef object fy = face.y
    cdef object ym = fy.minus
    cdef object ymQ = ym.q
    cdef object ymU = ym.p
    cdef object yp = fy.plus
    cdef object ypQ = yp.q
    cdef object ypU = yp.p
    cdef object xfcenters = runtime.topo.xf
    cdef object yfcenters = runtime.topo.yf

    cdef Py_ssize_t ngh = states.domain.nhalo
    cdef double theta = config.params.theta
    cdef double drytol = config.params.drytol
    cdef double tol = runtime.tol

    cdef object dtype = Q.dtype

    if dtype == "float32":
        _reconstruct[float](
            Q, U, xfcenters, yfcenters, ngh, theta, drytol, tol,
            slpx, slpy, xmQ, xmU, xpQ, xpU, ymQ, ymU, ypQ, ypU
        )
    elif dtype == "float64":
        _reconstruct[double](
            Q, U, xfcenters, yfcenters, ngh, theta, drytol, tol,
            slpx, slpy, xmQ, xmU, xpQ, xpU, ymQ, ymU, ypQ, ypU
        )
    else:
        raise TypeError(f"Unacceptable type {dtype}")

    return states


cdef inline void _recnstrt_cell_centers(
    cython.floating[:, :, ::1] Q,
    cython.floating[:, :, ::1] U,
    cython.floating[:, ::1] B,  # TODO: read-only buffer
    const double drytol,
    const double tol
) nogil except *:
    """Get the cell-centered depths and reconstruct w, hu, and hv for non-halo cells.

    Arguments
    ---------
    Q : memoryview with shape (ny+2*ngh, nx+2*ngh)
    U : memoryview with shape (ny+2*ngh, nx+2*ngh)
    B : memoryview with shape (ny+2*ngh, nx+2*ngh)
    """
    cdef Py_ssize_t i, j

    for j in range(Q.shape[1]):
        for i in range(Q.shape[2]):
            U[0, j, i] = Q[0, j, i] - B[j, i]

            if U[0, j, i] < tol:  # completely dry cells
                U[0, j, i] = 0.0
                U[1, j, i] = 0.0
                U[2, j, i] = 0.0
                Q[0, j, i] = B[j, i]
                Q[1, j, i] = 0.0
                Q[2, j, i] = 0.0
            elif U[0, j, i] < drytol:  # wet but still cells
                U[1, j, i] = 0.0
                U[2, j, i] = 0.0
                Q[1, j, i] = 0.0
                Q[2, j, i] = 0.0
            else:
                U[1, j, i] = Q[1, j, i] / U[0, j, i]
                U[2, j, i] = Q[2, j, i] / U[0, j, i]


def reconstruct_cell_centers(object states, object runtime, object config):
    """Calculate cell-centered depths for non-halo-ring cells.

    `states.U` will be updated in this function.

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

    dtype = states.q.dtype

    if dtype == "float32":
        _recnstrt_cell_centers[cython.float](
            states.q, states.p, runtime.topo.c, config.params.drytol, runtime.tol)
    elif dtype == "float64":
        _recnstrt_cell_centers[cython.double](
            states.q, states.p, runtime.topo.c, config.params.drytol, runtime.tol)
    else:
        raise RuntimeError(f"Arrays are using an unrecognized dtype: {dtype}.")

    return states
