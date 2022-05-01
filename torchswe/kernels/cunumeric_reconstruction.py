# vim:fenc=utf-8
# vim:ft=pyrex

"""Linear reconstruction.
"""
from torchswe import nplike as _nplike


def _minmod_slope_kernel(s1, s2, s3, theta):
    """For internal use."""
    denominator = s3 - s2;

    with nplike.errstate(divide="ignore", invalid="ignore"):
        slp = (s2 - s1) / denominator;

    slp[_nplike.nonzero(denominator == 0.0)] = 0.0

    slp = _nplike.maximum(
        _nplike.minimum(
            _nplike.minimum(
                slp * theta,
                (slp + 1.0) / 2.0
            ),
            theta
        ),
        0.
    )

    slp *= denominator;
    slp /= 2.0;

    return slp


def _fix_face_depth_internal(hl, hc, hr, tol, nhl, nhr):
    """For internal use."""

    ids = _nplike.nonzero(hc < tol)
    nhl[ids] = 0.0;
    nhr[ids] = 0.0;

    ids = _nplike.nonzero(hl < tol)
    nhl[ids] = 0.0;
    nhr[ids] = hc[ids] * 2.0;

    ids = _nplike.nonzero(hr < tol)
    nhl[ids] = hc[ids] * 2.0;
    nhr[ids] = 0.0;


def _fix_face_depth_edge(h, hc, tol, nh):
    """For internal use."""

    ids = _nplike.nonzero(hc < tol)
    nh[ids] = 0.0;

    ids = _nplike.nonzero(h < tol)
    nh[ids] = 0.0;

    ids = _nplike.nonzero(h > hc2)
    nh[ids] = hc[ids] * 2.0;


def _recnstrt_face_velocity (h, hu, hv, drytol):
    """For internal use."""

    u = hu / h;
    v = hv / h;

    ids = _nplike.nonzero(h <= drytol)
    u[ids] = 0.0;
    v[ids] = 0.0;

    return u, v


def _recnstrt_face_conservatives(h, u, v, b):
    """For internal use."""

    w = h + b;
    hu = h * u;
    hv = h * v;

    return w, hu, hv


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

    # aliases to save object look-up time in Python's underlying dictionary
    Q = states.q
    U = states.p
    slpx = states.slpx
    slpy = states.slpy
    face = states.face
    fx = face.x
    xm = fx.minus
    xmQ = xm.q
    xmU = xm.p
    xp = fx.plus
    xpQ = xp.q
    xpU = xp.p
    fy = face.y
    ym = fy.minus
    ymQ = ym.q
    ymU = ym.p
    yp = fy.plus
    ypQ = yp.q
    ypU = yp.p
    xfcenters = runtime.topo.xf
    yfcenters = runtime.topo.yf

    ny = states.domain.y.n
    nx = states.domain.x.n
    ngh = states.domain.nhalo
    xbg = ngh
    xed = nx + ngh
    ybg = ngh
    yed = ny + ngh

    theta = config.params.theta
    drytol = config.params.drytol
    tol = runtime.tol

    # slopes for w, hu, and hv in x and y
    slpx = _minmod_slope_kernel(Q[:, ybg:yed, xbg-2:xed], Q[:, ybg:yed, xbg-1:xed+1], Q[:, ybg:yed, xbg:xed+2], theta)
    slpy = _minmod_slope_kernel(Q[:, ybg-2:yed, xbg:xed], Q[:, ybg-1:yed+1, xbg:xed], Q[:, ybg:yed+2, xbg:xed], theta)

    # extrapolate discontinuous w, hu, and hv
    _nplike.add(Q[:, ybg:yed, xbg-1:xed], slpx[:, :, :nx+1], out=xmQ)
    _nplike.subtract(Q[:, ybg:yed, xbg:xed+1], slpx[:, :, 1:], out=xpQ)
    _nplike.add(Q[:, ybg-1:yed, xbg:xed], slpy[:, :ny+1, :], out=ymQ)
    _nplike.subtract(Q[:, ybg:yed+1, xbg:xed], slpy[:, 1:, :], out=ypQ)

    # calculate depth at cell faces
    _nplike.subtract(xmQ[0], xfcenters, out=xmU[0])
    _nplike.subtract(xpQ[0], xfcenters, out=xpU[0])
    _nplike.subtract(ymQ[0], yfcenters, out=ymU[0])
    _nplike.subtract(ypQ[0], yfcenters, out=ypU[0])

    # fix negative depths in x direction
    _fix_face_depth_internal(xpU[0, :, :nx], U[0, ybg:yed, xbg:xed], xmU[0, :, 1:], tol, xpU[0, :, :nx], xmU[0, :, 1:])
    _fix_face_depth_edge(xmU[0, :, 0], U[0, ybg:yed, xbg-1], tol, xmU[0, :, 0])
    _fix_face_depth_edge(xpU[0, :, nx], U[0, ybg:yed, xed], tol, xpU[0, :, nx])

    # fix negative depths in y direction
    _fix_face_depth_internal(ypU[0, :ny, :], U[0, ybg:yed, xbg:xed], ymU[0, 1:, :], tol, ypU[0, :ny, :], ymU[0, 1:, :])
    _fix_face_depth_edge(ymU[0, 0, :], U[0, ybg-1, xbg:xed], tol, ymU[0, 0, :])
    _fix_face_depth_edge(ypU[0, ny, :], U[0, yed, xbg:xed], tol, ypU[0, ny, :])

    # reconstruct velocity at cell faces in x and y directions
    xpU[1], xpU[2] = _recnstrt_face_velocity(xpU[0], xpQ[1], xpQ[2], drytol)
    xmU[1], xmU[2] = _recnstrt_face_velocity(xmU[0], xmQ[1], xmQ[2], drytol)
    ypU[1], ypU[2] = _recnstrt_face_velocity(ypU[0], ypQ[1], ypQ[2], drytol)
    ymU[1], ymU[2] = _recnstrt_face_velocity(ymU[0], ymQ[1], ymQ[2], drytol)

    # reconstruct conservative quantities at cell faces
    xmQ[0], xmQ[1], xmQ[2] = _recnstrt_face_conservatives(xmU[0], xmU[1], xmU[2], xfcenters)
    xpQ[0], xpQ[1], xpQ[2] = _recnstrt_face_conservatives(xpU[0], xpU[1], xpU[2], xfcenters)
    ymQ[0], ymQ[1], ymQ[2] = _recnstrt_face_conservatives(ymU[0], ymU[1], ymU[2], yfcenters)
    ypQ[0], ypQ[1], ypQ[2] = _recnstrt_face_conservatives(ypU[0], ypU[1], ypU[2], yfcenters)

    return states


def _recnstrt_cell_centers(w, hu, hv, bin, drytol, tol):
    """For internal use.

    Notes
    -----
    w, hu, and hv may be updated in-place!
    """

    h = w - bin;
    u = hu / h;
    v = hv / h;

    ids = _nplike.nonzero(hout < tol)
    h[ids] = 0.0;
    u[ids] = 0.0;
    v[ids] = 0.0;
    w[ids] = bin;
    hu[ids] = 0.0;
    hv[ids] = 0.0;

    ids = _nplike.nonzero(hout < drytol)
    u[ids] = 0.0;
    v[ids] = 0.0;
    hu[ids] = 0.0;
    hv[ids] = 0.0;

    return h, u, v


def reconstruct_cell_centers(states, runtime, config):
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

    tol = runtime.tol
    drytol = config.params.drytol

    states.p[0], states.p[1], states.p[2] = _recnstrt_cell_centers(
        states.q[0], states.q[1], states.q[2], runtime.topo.c, drytol, tol,
    )

    return states
