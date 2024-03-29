# vim:fenc=utf-8
# vim:ft=pyrex

"""Linear reconstruction.
"""


# raw CUDA kernel so we can reuse it in other CUDA kernels
_minmod_slope_raw_kernel = r"""
    template<typename T> __inline__ __device__
    void _minmod_slope_raw_kernel(const T &s1, const T &s2, const T &s3, const T &theta, T &slp) {
        T denominator = s3 - s2;

        if (denominator == 0.0) {
            slp = 0.0;
            return;
        }

        slp = s2 - s1;
        if ((slp == 0.0) or (slp == -1.0)) {
            slp = 0.0;
            return;
        }

        slp /= denominator;
        slp = min(slp*theta, (1.0+slp) / 2.0);
        slp = min(slp, theta);
        slp = max(slp, 0.);
        slp *= denominator;
        slp /= 2.0;
    }
"""


# minmod kernel for cupy.ndarray in Python
cdef _minmod_slope_kernel = cupy.ElementwiseKernel(
    "T s1, T s2, T s3, T theta",
    "T slp",
    "_minmod_slope_raw_kernel(s1, s2, s3, theta, slp);",
    "minmod_slope_kernel",
    preamble=_minmod_slope_raw_kernel
)


cdef _fix_face_depth_internal = cupy.ElementwiseKernel(
    "T hl, T hc, T hr, T tol",
    "T nhl, T nhr",
    r"""
        if (hc < tol) {  // a slightly relaxed confition for hc == 0 (hc definitely >= 0)
            nhl = 0.0;
            nhr = 0.0;
        } else if (hl < tol) {  // a slightly relaxed condition for hl < 0
            nhl = 0.;
            nhr = hc * 2.0;
        } else if (hr < tol) {  // a slightly relaxed condition for hr < 0
            nhr = 0.0;
            nhl = hc * 2.0;
        }
    """,
    "_fix_face_depth_internal"
)


cdef _fix_face_depth_edge = cupy.ElementwiseKernel(
    "T h, T hc, T tol",
    "T nh",
    r"""
        T hc2 = 2 * hc;
        if (hc < tol) {
            nh = 0.0;
        } else if (h < tol) {
            nh = 0.0;
        } else if (h > hc2) {
            nh = hc2;
        }
    """,
    "_fix_face_depth_edge"
)


cdef _recnstrt_face_velocity = cupy.ElementwiseKernel(
    "T h, T hu, T hv, T drytol",
    "T u, T v",
    r"""
        if (h <= drytol) {
            u = 0.0;
            v = 0.0;
        } else {
            u = hu / h;
            v = hv / h;
        }
    """,
    "_recnstrt_face_velocity",
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

    cdef Py_ssize_t ny = states.domain.y.n
    cdef Py_ssize_t nx = states.domain.x.n
    cdef Py_ssize_t ngh = states.domain.nhalo
    cdef Py_ssize_t xbg = ngh
    cdef Py_ssize_t xed = nx + ngh
    cdef Py_ssize_t ybg = ngh
    cdef Py_ssize_t yed = ny + ngh

    cdef double theta = config.params.theta
    cdef double drytol = config.params.drytol
    cdef double tol = runtime.tol

    # slopes for w, hu, and hv in x and y
    _minmod_slope_kernel(Q[:, ybg:yed, xbg-2:xed], Q[:, ybg:yed, xbg-1:xed+1], Q[:, ybg:yed, xbg:xed+2], theta, slpx)
    _minmod_slope_kernel(Q[:, ybg-2:yed, xbg:xed], Q[:, ybg-1:yed+1, xbg:xed], Q[:, ybg:yed+2, xbg:xed], theta, slpy)

    # extrapolate discontinuous w, hu, and hv
    cupy.add(Q[:, ybg:yed, xbg-1:xed], slpx[:, :, :nx+1], out=xmQ)
    cupy.subtract(Q[:, ybg:yed, xbg:xed+1], slpx[:, :, 1:], out=xpQ)
    cupy.add(Q[:, ybg-1:yed, xbg:xed], slpy[:, :ny+1, :], out=ymQ)
    cupy.subtract(Q[:, ybg:yed+1, xbg:xed], slpy[:, 1:, :], out=ypQ)

    # calculate depth at cell faces
    cupy.subtract(xmQ[0], xfcenters, out=xmU[0])
    cupy.subtract(xpQ[0], xfcenters, out=xpU[0])
    cupy.subtract(ymQ[0], yfcenters, out=ymU[0])
    cupy.subtract(ypQ[0], yfcenters, out=ypU[0])

    # fix negative depths in x direction
    _fix_face_depth_internal(xpU[0, :, :nx], U[0, ybg:yed, xbg:xed], xmU[0, :, 1:], tol, xpU[0, :, :nx], xmU[0, :, 1:])
    _fix_face_depth_edge(xmU[0, :, 0], U[0, ybg:yed, xbg-1], tol, xmU[0, :, 0])
    _fix_face_depth_edge(xpU[0, :, nx], U[0, ybg:yed, xed], tol, xpU[0, :, nx])

    # fix negative depths in y direction
    _fix_face_depth_internal(ypU[0, :ny, :], U[0, ybg:yed, xbg:xed], ymU[0, 1:, :], tol, ypU[0, :ny, :], ymU[0, 1:, :])
    _fix_face_depth_edge(ymU[0, 0, :], U[0, ybg-1, xbg:xed], tol, ymU[0, 0, :])
    _fix_face_depth_edge(ypU[0, ny, :], U[0, yed, xbg:xed], tol, ypU[0, ny, :])

    # reconstruct velocity at cell faces in x and y directions
    _recnstrt_face_velocity(xpU[0], xpQ[1], xpQ[2], drytol, xpU[1], xpU[2])
    _recnstrt_face_velocity(xmU[0], xmQ[1], xmQ[2], drytol, xmU[1], xmU[2])
    _recnstrt_face_velocity(ypU[0], ypQ[1], ypQ[2], drytol, ypU[1], ypU[2])
    _recnstrt_face_velocity(ymU[0], ymQ[1], ymQ[2], drytol, ymU[1], ymU[2])

    # reconstruct conservative quantities at cell faces
    _recnstrt_face_conservatives(xmU[0], xmU[1], xmU[2], xfcenters, xmQ[0], xmQ[1], xmQ[2])
    _recnstrt_face_conservatives(xpU[0], xpU[1], xpU[2], xfcenters, xpQ[0], xpQ[1], xpQ[2])
    _recnstrt_face_conservatives(ymU[0], ymU[1], ymU[2], yfcenters, ymQ[0], ymQ[1], ymQ[2])
    _recnstrt_face_conservatives(ypU[0], ypU[1], ypU[2], yfcenters, ypQ[0], ypQ[1], ypQ[2])

    return states


cdef _recnstrt_cell_centers = cupy.ElementwiseKernel(
    "T win, T huin, T hvin, T bin, T drytol, T tol",
    "T wout, T huout, T hvout, T hout, T uout, T vout",
    """
        hout = win - bin;
        uout = huin / hout;
        vout = hvin / hout;

        if (hout < tol) {
            hout = 0.0;
            uout = 0.0;
            vout = 0.0;
            wout = bin;
            huout = 0.0;
            hvout = 0.0;
        } else if (hout < drytol) {
            uout = 0.0;
            vout = 0.0;
            huout = 0.0;
            hvout = 0.0;
        }
    """,
    "_recnstrt_cell_centers"
)


cpdef reconstruct_cell_centers(object states, object runtime, object config):
    """Calculate cell-centered non-conservatives.

    `states.U` will be updated in this function, and `states.Q` may be changed, too.

    Arguments
    ---------
    states : torchswe.utils.data.States
    runtime : torchswe.utils.misc.DummyDict
    config : torchswe.utils.config.Config

    Returns
    -------
    states : torchswe.utils.data.States
    """

    cdef double tol = runtime.tol
    cdef double drytol = config.params.drytol

    _recnstrt_cell_centers(
        states.q[0], states.q[1], states.q[2], runtime.topo.c, drytol, tol,
        states.q[0], states.q[1], states.q[2], states.p[0], states.p[1], states.p[2]
    )

    return states
