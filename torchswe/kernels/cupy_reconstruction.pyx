# vim:fenc=utf-8
# vim:ft=pyrex

"""Linear reconstruction.
"""


cdef _minmod_slope_kernel = cupy.ElementwiseKernel(
    "T s1, T s2, T s3, float64 theta",
    "T slp",
    """
        T denominator = s3 - s2;
        slp = (s2 - s1) / denominator;
        slp = min(slp*theta, (1.0 + slp) / 2.0);
        slp = min(slp, theta);
        slp = max(slp, 0.);
        slp *= denominator;
        slp /= 2.0;
    """,
    "minmod_slope_kernel",
)


cdef _fix_rounding_err_kernel = cupy.ElementwiseKernel(
    "T depth, float64 tol",
    "T ans",
    """
        if (depth < tol) ans = 0.0;
    """,
    "fix_rounding_err_kernel",
)


cdef _recnstrt_face_vel_kernel = cupy.ElementwiseKernel(
    "T depth, T hu, T hv, float64 drytol, float64 tol",
    "T h, T u, T v",
    r"""
        if (depth < tol) {
            h = 0.0;
            u = 0.0;
            v = 0.0;
        } else if (depth < drytol) {
            u = 0.0;
            v = 0.0;
        } else {
            u = hu / depth;
            v = hv / depth;
        }
    """,
    "recnstrt_face_vel_kernel",
)


cdef _recnstrt_face_conservatives = cupy.ElementwiseKernel(
    "T h, T u, T v, T b",
    "T w, T hu, T hv",
    r"""
        w = h + b;
        hu = h * u;
        hv = h * v;
    """,
    "recnstrt_face_conservatives"
)


cdef _correct_neg_depth_internal = cupy.ElementwiseKernel(
    "T hc, T hl, T hr",
    "T nhl, T nhr",
    r"""
        if (hl < 0.0) {
            nhl = 0.;
            nhr = hc * 2.0;
        } else if (hr < 0.0) {
            nhr = 0.0;
            nhl = hc * 2.0;
        }
    """,
    "_correct_neg_depth_internal"
)


cdef _correct_neg_depth_edge = cupy.ElementwiseKernel(
    "T h, T hc",
    "T nh",
    r"""
        T hc2 = 2 * hc;
        if (h < 0.0) {
            nh = 0.0;
        } else if (h > hc2) {
            nh = hc2;
        }
    """,
    "_correct_neg_depth_edge"
)


cpdef reconstruct(object states, object runtime, object config):
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
    cdef object face = states.face
    cdef object x = face.x
    cdef object xm = face.x.minus
    cdef object xp = face.x.plus
    cdef object y = face.y
    cdef object ym = face.y.minus
    cdef object yp = face.y.plus
    cdef object Q = states.Q
    cdef object H = states.H
    cdef object xmQ = xm.Q
    cdef object xpQ = xp.Q
    cdef object ymQ = ym.Q
    cdef object ypQ = yp.Q
    cdef object xmU = xm.U
    cdef object xpU = xp.U
    cdef object ymU = ym.U
    cdef object ypU = yp.U

    cdef object topo = runtime.topo

    cdef Py_ssize_t ngh = states.ngh
    cdef Py_ssize_t ny = Q.shape[1] - 2 * ngh
    cdef Py_ssize_t nx = Q.shape[2] - 2 * ngh
    cdef Py_ssize_t xbg = ngh
    cdef Py_ssize_t xed = nx+ngh
    cdef Py_ssize_t ybg = ngh
    cdef Py_ssize_t yed = ny+ngh

    cdef double theta = config.params.theta
    cdef double drytol = config.params.drytol
    cdef double tol = runtime.tol

    # get slopes
    cdef object slpx = _minmod_slope_kernel(
            Q[:, ybg:yed, xbg-2:xed], Q[:, ybg:yed, xbg-1:xed+1], Q[:, ybg:yed, xbg:xed+2], theta)
    cdef object slpy = _minmod_slope_kernel(
            Q[:, ybg-2:yed, xbg:xed], Q[:, ybg-1:yed+1, xbg:xed], Q[:, ybg:yed+2, xbg:xed], theta)

    # get discontinuous conservative quantities at cell faces
    cupy.add(Q[:, ybg:yed, xbg-1:xed], slpx[:, :, :nx+1], out=xmQ)
    cupy.subtract(Q[:, ybg:yed, xbg:xed+1], slpx[:, :, 1:], out=xpQ)
    cupy.add(Q[:, ybg-1:yed, xbg:xed], slpy[:, :ny+1, :], out=ymQ)
    cupy.subtract(Q[:, ybg:yed+1, xbg:xed], slpy[:, 1:, :], out=ypQ)

    # calculate depth at cell centers and faces
    cupy.subtract(xmQ[0], topo.xfcenters, out=xmU[0])
    cupy.subtract(xpQ[0], topo.xfcenters, out=xpU[0])
    cupy.subtract(ymQ[0], topo.yfcenters, out=ymU[0])
    cupy.subtract(ypQ[0], topo.yfcenters, out=ypU[0])

    # fix negative depths in x direction
    _correct_neg_depth_internal(H[1:ny+1, 1:nx+1], xpU[0, :, :nx], xmU[0, :, 1:], xpU[0, :, :nx], xmU[0, :, 1:])
    _correct_neg_depth_edge(xpU[0, :, nx], H[1:ny+1, nx+1], xpU[0, :, nx])
    _correct_neg_depth_edge(xmU[0, :, 0], H[1:ny+1, 0], xmU[0, :, 0])

    # fix negative depths in x direction
    _correct_neg_depth_internal(H[1:ny+1, 1:nx+1], ypU[0, :ny, :], ymU[0, 1:, :], ypU[0, :ny, :], ymU[0, 1:, :])
    _correct_neg_depth_edge(ypU[0, ny, :], H[ny+1, 1:nx+1], ypU[0, ny, :])
    _correct_neg_depth_edge(ymU[0, 0, :], H[0, 1:nx+1], ymU[0, 0, :])

    # reconstruct velocity at cell faces
    _recnstrt_face_vel_kernel(xmU[0], xmQ[1], xmQ[2], drytol, tol, xmU[0], xmU[1], xmU[2])
    _recnstrt_face_vel_kernel(xpU[0], xpQ[1], xpQ[2], drytol, tol, xpU[0], xpU[1], xpU[2])
    _recnstrt_face_vel_kernel(ymU[0], ymQ[1], ymQ[2], drytol, tol, ymU[0], ymU[1], ymU[2])
    _recnstrt_face_vel_kernel(ypU[0], ypQ[1], ypQ[2], drytol, tol, ypU[0], ypU[1], ypU[2])

    # reconstruct conservative quantities at cell faces
    _recnstrt_face_conservatives(xmU[0], xmU[1], xmU[2], topo.xfcenters, xmQ[0], xmQ[1], xmQ[2])
    _recnstrt_face_conservatives(xpU[0], xpU[1], xpU[2], topo.xfcenters, xpQ[0], xpQ[1], xpQ[2])
    _recnstrt_face_conservatives(ymU[0], ymU[1], ymU[2], topo.yfcenters, ymQ[0], ymQ[1], ymQ[2])
    _recnstrt_face_conservatives(ypU[0], ypU[1], ypU[2], topo.yfcenters, ypQ[0], ypQ[1], ypQ[2])

    return states


cdef _recnstrt_cell_center = cupy.ElementwiseKernel(
    "T ow, T b, float64 drytol, float64 tol",
    "T h, T w, T hu, T hv",
    """
        h = ow - b;
        if (h < tol) {
            h = 0.0;
            w = b;
            hu = 0.0;
            hv = 0.0;
        } else if (h < drytol) {
            hu = 0.0;
            hv = 0.0;
        }
    """,
    "_recnstrt_cell_center"
)


cpdef get_cell_center_depth(object states, object runtime, object config):
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

    cdef Py_ssize_t ngh = states.ngh
    cdef Py_ssize_t ny = states.H.shape[0] - 2
    cdef Py_ssize_t nx = states.H.shape[1] - 2
    cdef double tol = runtime.tol
    cdef double drytol = config.params.drytol

    _recnstrt_cell_center(
        states.Q[0, ngh:ngh+ny, ngh:ngh+nx],
        runtime.topo.centers,
        drytol,
        tol,
        states.H[1:1+ny, 1:1+nx],
        states.Q[0, ngh:ngh+ny, ngh:ngh+nx],
        states.Q[1, ngh:ngh+ny, ngh:ngh+nx],
        states.Q[2, ngh:ngh+ny, ngh:ngh+nx]
    )

    return states
