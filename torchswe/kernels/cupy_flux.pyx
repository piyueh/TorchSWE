# vim:fenc=utf-8
# vim:ft=pyrex
import cupy


cdef get_discontinuous_flux_x = cupy.ElementwiseKernel(
    "T hu, T h, T u, T v, float64 grav2",
    "T f0, T f1, T f2",
    """
        f0 = hu;
        f1 = hu * u + grav2 * h * h;
        f2 = hu * v;
    """,
    "get_discontinuous_flux_x"
)


cdef get_discontinuous_flux_y = cupy.ElementwiseKernel(
    "T hv, T h, T u, T v, float64 grav2",
    "T f0, T f1, T f2",
    """
        f0 = hv;
        f1 = u * hv;
        f2 = hv * v + grav2 * h * h;
    """,
    "get_discontinuous_flux_y"
)


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

    cdef double grav2 = gravity / 2.

    x = states.face.x
    xm = x.minus
    xp = x.plus

    y = states.face.y
    ym = y.minus
    yp = y.plus

    # face normal to x-direction: [hu, hu^2 + g(h^2)/2, huv]
    get_discontinuous_flux_x(xm.q[1], xm.p[0], xm.p[1], xm.p[2], grav2, xm.f[0], xm.f[1], xm.f[2])
    get_discontinuous_flux_x(xp.q[1], xp.p[0], xp.p[1], xp.p[2], grav2, xp.f[0], xp.f[1], xp.f[2])

    # face normal to y-direction: [hv, huv, hv^2+g(h^2)/2]
    get_discontinuous_flux_y(ym.q[2], ym.p[0], ym.p[1], ym.p[2], grav2, ym.f[0], ym.f[1], ym.f[2])
    get_discontinuous_flux_y(yp.q[2], yp.p[0], yp.p[1], yp.p[2], grav2, yp.f[0], yp.f[1], yp.f[2])

    return states


cdef central_scheme_kernel = cupy.ElementwiseKernel(
    "T ma, T pa, T mf, T pf, T mq, T pq",
    "T flux",
    """
        T denominator = pa - ma;
        T coeff = pa * ma;
        flux = (pa * mf - ma * pf + coeff * (pq - mq)) / denominator;
        if (flux != flux) flux = 0.0;
    """,
    "central_scheme_kernel"
)


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

    x = states.face.x
    xm = x.minus
    xp = x.plus

    y = states.face.y
    ym = y.minus
    yp = y.plus

    central_scheme_kernel(xm.a, xp.a, xm.f, xp.f, xm.q, xp.q, x.cf)
    central_scheme_kernel(ym.a, yp.a, ym.f, yp.f, ym.q, yp.q, y.cf)

    return states


cdef get_local_speed_kernel = cupy.ElementwiseKernel(
    "T hp, T hm, T up, T um, T g",
    "T ap, T am",
    r"""
        T ghp = sqrt(g * hp);
        T ghm = sqrt(g * hm);
        ap = max(max(up+ghp, um+ghm), 0.0);
        am = min(min(up-ghp, um-ghm), 0.0);
    """,
    "get_local_speed_kernel"
)


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

    # alias to reduce dictionary look-up
    cdef object face = states.face;
    cdef object fx = face.x;
    cdef object fy = face.y;
    cdef object fxp = fx.plus;
    cdef object fxm = fx.minus;
    cdef object fyp = fy.plus;
    cdef object fym = fy.minus;
    cdef object xpU = fxp.p;
    cdef object xmU = fxm.p;
    cdef object xpa = fxp.a;
    cdef object xma = fxm.a;
    cdef object ypU = fyp.p;
    cdef object ymU = fym.p;
    cdef object ypa = fyp.a;
    cdef object yma = fym.a;

    # faces normal to x- and y-directions
    get_local_speed_kernel(xpU[0], xmU[0], xpU[1], xmU[1], gravity, xpa, xma)
    get_local_speed_kernel(ypU[0], ymU[0], ypU[2], ymU[2], gravity, ypa, yma)

    return states
