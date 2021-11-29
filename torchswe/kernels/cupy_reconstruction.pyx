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
    "T hul, T hur, T hvl, T hvr, T hl, T hr, T uim1, T ui, T uip1, T vim1, T vi, T vip1, T theta, T drytol",
    "T ul, T ur, T vl, T vr",
    r"""
        if ((hl < drytol) || (hr < drytol)) {
            _minmod_slope_raw_kernel(uim1, ui, uip1, theta, du);
            _minmod_slope_raw_kernel(vim1, vi, vip1, theta, dv);
            ul = ui - du;
            ur = ui + du;
            vl = vi - dv;
            vr = vi + dv;
        } else {
            ul = hul / hl;
            ur = hur / hr;
            vl = hvl / hl;
            vr = hvr / hr;
        }
    """,
    "_recnstrt_face_velocity",
    preamble=_minmod_slope_raw_kernel,
    loop_prep="T du; T dv;"
)


cdef _recnstrt_face_velocity_edge_minus = cupy.ElementwiseKernel(
    "T hu, T hv, T h, T hi, T uim1, T ui, T uip1, T vim1, T vi, T vip1, T theta, T drytol",
    "T u, T v",
    r"""
        if ((h < drytol) || ((hi * 2.0 - h) < drytol)) {
            _minmod_slope_raw_kernel(uim1, ui, uip1, theta, du);
            _minmod_slope_raw_kernel(vim1, vi, vip1, theta, dv);
            u = ui + du;
            v = vi + dv;
        } else {
            u = hu / h;
            v = hv / h;
        }
    """,
    "_recnstrt_face_velocity_edge_minus",
    preamble=_minmod_slope_raw_kernel,
    loop_prep="T du; T dv;"
)


cdef _recnstrt_face_velocity_edge_plus = cupy.ElementwiseKernel(
    "T hu, T hv, T h, T hi, T uim1, T ui, T uip1, T vim1, T vi, T vip1, T theta, T drytol",
    "T u, T v",
    r"""
        if ((h < drytol) || ((hi * 2.0 - h) < drytol)) {
            _minmod_slope_raw_kernel(uim1, ui, uip1, theta, du);
            _minmod_slope_raw_kernel(vim1, vi, vip1, theta, dv);
            u = ui - du;
            v = vi - dv;
        } else {
            u = hu / h;
            v = hv / h;
        }
    """,
    "_recnstrt_face_velocity_edge_plus",
    preamble=_minmod_slope_raw_kernel,
    loop_prep="T du; T dv;"
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
    cdef object Q = states.Q
    cdef object U = states.U
    cdef object slpx = states.slpx
    cdef object slpy = states.slpy
    cdef object face = states.face
    cdef object fx = face.x
    cdef object xm = fx.minus
    cdef object xmQ = xm.Q
    cdef object xmU = xm.U
    cdef object xp = fx.plus
    cdef object xpQ = xp.Q
    cdef object xpU = xp.U
    cdef object fy = face.y
    cdef object ym = fy.minus
    cdef object ymQ = ym.Q
    cdef object ymU = ym.U
    cdef object yp = fy.plus
    cdef object ypQ = yp.Q
    cdef object ypU = yp.U
    cdef object xfcenters = runtime.topo.xfcenters
    cdef object yfcenters = runtime.topo.yfcenters

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

    # reconstruct velocity at cell faces in x direction
    _recnstrt_face_velocity(
        xpQ[1, :, :nx], xmQ[1, :, 1:],  # hul, hur
        xpQ[2, :, :nx], xmQ[2, :, 1:],  # hvl, hvr
        xpU[0, :, :nx], xmU[0, :, 1:],  # hl, hr
        U[1, ybg:yed, xbg-1:xed-1], U[1, ybg:yed, xbg:xed], U[1, ybg:yed, xbg+1:xed+1],  # uim1, ui, uip1
        U[2, ybg:yed, xbg-1:xed-1], U[2, ybg:yed, xbg:xed], U[2, ybg:yed, xbg+1:xed+1],  # vim1, vi, vip1
        theta, drytol,
        xpU[1, :, :nx], xmU[1, :, 1:],  # output: ul, ur
        xpU[2, :, :nx], xmU[2, :, 1:],  # output: vl, vr
    )
    _recnstrt_face_velocity_edge_minus(
        xmQ[1, :, 0], xmQ[2, :, 0], xmU[0, :, 0], U[0, ybg:yed, xbg-1],
        U[1, ybg:yed, xbg-2], U[1, ybg:yed, xbg-1], U[1, ybg:yed, xbg],
        U[2, ybg:yed, xbg-2], U[2, ybg:yed, xbg-1], U[2, ybg:yed, xbg],
        theta, drytol, xmU[1, :, 0], xmU[2, :, 0],
    )
    _recnstrt_face_velocity_edge_plus(
        xpQ[1, :, nx], xpQ[2, :, nx], xpU[0, :, nx], U[0, ybg:yed, xed],
        U[1, ybg:yed, xed-1], U[1, ybg:yed, xed], U[1, ybg:yed, xed+1],
        U[2, ybg:yed, xed-1], U[2, ybg:yed, xed], U[2, ybg:yed, xed+1],
        theta, drytol, xpU[1, :, nx], xpU[2, :, nx],
    )

    # reconstruct velocity at cell faces in y direction
    _recnstrt_face_velocity(
        ypQ[1, :ny, :], ymQ[1, 1:, :],  # hul, hur
        ypQ[2, :ny, :], ymQ[2, 1:, :],  # hvl, hvr
        ypU[0, :ny, :], ymU[0, 1:, :],  # hl, hr
        U[1, ybg-1:yed-1, xbg:xed], U[1, ybg:yed, xbg:xed], U[1, ybg+1:yed+1, xbg:xed],  # uim1, ui, uip1
        U[2, ybg-1:yed-1, xbg:xed], U[2, ybg:yed, xbg:xed], U[2, ybg+1:yed+1, xbg:xed],  # vim1, vi, vip1
        theta, drytol,
        ypU[1, :ny, :], ymU[1, 1:, :],  # output: hul, hur
        ypU[2, :ny, :], ymU[2, 1:, :],  # output: hvl, hvr
    )
    _recnstrt_face_velocity_edge_minus(
        ymQ[1, 0, :], ymQ[2, 0, :], ymU[0, 0, :], U[0, ybg-1, xbg:xed],
        U[1, ybg-2, xbg:xed], U[1, ybg-1, xbg:xed], U[1, ybg, xbg:xed],
        U[2, ybg-2, xbg:xed], U[2, ybg-1, xbg:xed], U[2, ybg, xbg:xed],
        theta, drytol, ymU[1, 0, :], ymU[2, 0, :],
    )
    _recnstrt_face_velocity_edge_plus(
        ypQ[1, ny, :], ypQ[2, ny, :], ypU[0, ny, :], U[0, yed, xbg:xed],
        U[1, yed-1, xbg:xed], U[1, yed, xbg:xed], U[1, yed+1, xbg:xed],
        U[2, yed-1, xbg:xed], U[2, yed, xbg:xed], U[2, yed+1, xbg:xed],
        theta, drytol, ypU[1, ny, :], ypU[2, ny, :],
    )

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
        states.Q[0], states.Q[1], states.Q[2], runtime.topo.centers, drytol, tol,
        states.Q[0], states.Q[1], states.Q[2], states.U[0], states.U[1], states.U[2]
    )

    return states
