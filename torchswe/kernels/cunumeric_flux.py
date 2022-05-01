# vim:fenc=utf-8
# vim:ft=pyrex
from torchswe import nplike as _nplike


def get_discontinuous_flux(states, gravity):
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

    x = states.face.x
    xm = x.minus
    xp = x.plus

    y = states.face.y
    ym = y.minus
    yp = y.plus

    # face normal to x-direction: [hu, hu^2 + g(h^2)/2, huv]
    xm.f[0] = xm.q[1]
    xm.f[1] = xm.q[1] * xm.p[1] + grav2 * xm.p[0] * xm.p[0]
    xm.f[2] = xm.q[1] * xm.p[2]
    xp.f[0] = xp.q[1]
    xp.f[1] = xp.q[1] * xp.p[1] + grav2 * xp.p[0] * xp.p[0]
    xp.f[2] = xp.q[1] * xp.p[2]

    # face normal to y-direction: [hv, huv, hv^2+g(h^2)/2]
    ym.f[0] = ym.q[2]
    ym.f[1] = ym.q[2] * ym.p[1]
    ym.f[2] = ym.q[2] * ym.p[2] + grav2 * ym.p[0] * ym.p[0]
    yp.f[0] = yp.q[2]
    yp.f[1] = yp.q[2] * yp.p[1]
    yp.f[2] = yp.q[2] * yp.p[2] + grav2 * yp.p[0] * yp.p[0]

    return states


def central_scheme_kernel(ma, pa, mf, pf, mq, pq):
    """For internal use"""
    denominator = pa - ma
    coeff = pa * ma

    with _nplike.errstate(divide="ignore", invalid="ignore"):
        flux = (pa * mf - ma * pf + coeff * (pq - mq)) / denominator;

    zero_ji = _nplike.nonzero(denominator == 0.)  # should we deal with small rounding err here???
    flux[zero_ji] = 0.

    return flux


def central_scheme(states):
    """A central scheme to calculate numerical flux at interfaces.

    Arguments
    ---------
    states : torchswe.utils.data.States

    Returns
    -------
    states : torchswe.utils.data.States
        The same object as the input. Updated in-place. Returning it just for coding style.
    """

    x = states.face.x
    xm = x.minus
    xp = x.plus

    y = states.face.y
    ym = y.minus
    yp = y.plus

    x.cf = central_scheme_kernel(xm.a, xp.a, xm.f, xp.f, xm.q, xp.q)
    y.cf = central_scheme_kernel(ym.a, yp.a, ym.f, yp.f, ym.q, yp.q)

    return states


def get_local_speed_kernel(hp, hm, up, um, g):
    """For internal use to mimic CuPy and Cython kernels."""
    ghp = sqrt(g * hp);
    ghm = sqrt(g * hm);
    ap = _nplike.maximum(_nplike.maximum(up+ghp, um+ghm), 0.0);
    am = _nplike.minimum(_nplike.minimum(up-ghp, um-ghm), 0.0);
    return ap, am


def get_local_speed(states, gravity):
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

    # alias to reduce dictionary look-up
    face = states.face;
    fx = face.x;
    fy = face.y;
    fxp = fx.plus;
    fxm = fx.minus;
    fyp = fy.plus;
    fym = fy.minus;
    xpU = fxp.p;
    xmU = fxm.p;
    xpa = fxp.a;
    xma = fxm.a;
    ypU = fyp.p;
    ymU = fym.p;
    ypa = fyp.a;
    yma = fym.a;

    # faces normal to x- and y-directions
    xpa, xma = get_local_speed_kernel(xpU[0], xmU[0], xpU[1], xmU[1], gravity)
    ypa, yma = get_local_speed_kernel(ypU[0], ymU[0], ypU[2], ymU[2], gravity)

    return states
