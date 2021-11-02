# Cython implementation of flux calculations
import numpy
from torchswe.utils.data import States as _States
cimport numpy
cimport cython
numpy.seterr(divide="ignore", invalid="ignore")


# fused floating point type; uses as the template varaible type in C++
ctypedef fused fptype:
    float
    double


@cython.boundscheck(False)  # deactivate bounds checking
@cython.wraparound(False)  # deactivate negative indexing.
cdef numpy.ndarray[fptype, ndim=3] get_discontinuous_flux_x(
    numpy.ndarray[fptype, ndim=3] F, fptype[:, :, ::1] Q, fptype[:, :, ::1] U, fptype gravity):
    """Kernel of calculating discontinuous flux in x direction (in-place).
    """
    if fptype is float:
        dtype = numpy.float32
    elif fptype is double:
        dtype = numpy.double

    cdef int nx = Q.shape[2]
    cdef int ny = Q.shape[1]
    cdef int k, j, i
    cdef fptype grav2 = gravity / 2.0
    cdef fptype[:, :, ::1] F_view = F  # using a memory view to efficiently access elements

    # F[0] = hu
    for j in range(ny):
        for i in range(nx):
            F_view[0, j, i] = Q[1, j, i]

    # F[1] = hu * u + g/2 * h * h
    for j in range(ny):
        for i in range(nx):
            F_view[1, j, i] = Q[1, j, i] * U[1, j, i] + grav2 * (U[0, j, i] * U[0, j, i])

    # F[2] = hu * v
    for j in range(ny):
        for i in range(nx):
            F_view[2, j, i] = Q[1, j, i] * U[2, j, i]

    return F  # though we already modify it inplace, we return it for coding style


@cython.boundscheck(False)  # deactivate bounds checking
@cython.wraparound(False)  # deactivate negative indexing.
cdef numpy.ndarray[fptype, ndim=3] get_discontinuous_flux_y(
    numpy.ndarray[fptype, ndim=3] F, fptype[:, :, ::1] Q, fptype[:, :, ::1] U, fptype gravity):
    """Kernel of calculating discontinuous flux in y direction.
    """
    if fptype is float:
        dtype = numpy.float32
    elif fptype is double:
        dtype = numpy.double

    cdef int nx = Q.shape[2]
    cdef int ny = Q.shape[1]
    cdef int k, j, i
    cdef fptype grav2 = gravity / 2.0
    cdef fptype[:, :, ::1] F_view = F  # using a memory view to efficiently access elements

    # F[0] = hv
    for j in range(ny):
        for i in range(nx):
            F_view[0, j, i] = Q[2, j, i]

    # F[1] = u * hv
    for j in range(ny):
        for i in range(nx):
            F_view[1, j, i] = U[1, j, i] * Q[2, j, i]

    # F[2] = hv * v + g/2 * h * h
    for j in range(ny):
        for i in range(nx):
            F_view[2, j, i] = Q[2, j, i] * U[2, j, i] + grav2 * U[0, j, i] * U[0, j, i]

    return F  # though we already modify it inplace, we return it for coding style


@cython.boundscheck(False)  # deactivate bounds checking
@cython.wraparound(False)  # deactivate negative indexing.
def get_discontinuous_flux(object states, fptype gravity):
    """Calculting the discontinuous fluxes on the both sides at cell faces.

    Arguments
    ---------
    states : torchswe.utils.data.States
    topo : torchswe.utils.data.Topography
    gravity : float

    Returns
    -------
    states : torchswe.utils.data.States
        The same object as the input. Changed inplace. Returning it just for coding style.

    Notes
    -----
    When calculating (w-z)^2, it seems using w*w-w*z-z*w+z*z has smaller rounding errors. Not sure
    why. But it worth more investigation. This is apparently slower, though with smaller errors.
    """

    # face normal to x-direction: [hu, hu^2 + g(h^2)/2, huv]
    states.face.x.minus.F = get_discontinuous_flux_x[fptype](
        states.face.x.minus.F, states.face.x.minus.Q, states.face.x.minus.U, gravity)
    states.face.x.plus.F = get_discontinuous_flux_x[fptype](
        states.face.x.plus.F, states.face.x.plus.Q, states.face.x.plus.U, gravity)

    # face normal to y-direction: [hv, huv, hv^2+g(h^2)/2]
    states.face.y.minus.F = get_discontinuous_flux_y[fptype](
        states.face.y.minus.F, states.face.y.minus.Q, states.face.y.minus.U, gravity)
    states.face.y.plus.F = get_discontinuous_flux_y[fptype](
        states.face.y.plus.F, states.face.y.plus.Q, states.face.y.plus.U, gravity)

    return states


@cython.boundscheck(False)  # deactivate bounds checking
@cython.wraparound(False)  # deactivate negative indexing.
cdef numpy.ndarray[fptype, ndim=3] central_scheme_kernel(
    numpy.ndarray[fptype, ndim=3] H,
    fptype[:, :, ::1] Qm,
    fptype[:, :, ::1] Qp,
    fptype[:, :, ::1] Fm,
    fptype[:, :, ::1] Fp,
    fptype[:, ::1] Am,
    fptype[:, ::1] Ap
):
    """Kernel calculating common/numerical flux.
    """
    if fptype is float:
        dtype = numpy.float32
    elif fptype is double:
        dtype = numpy.double

    cdef int nx = Qm.shape[2]
    cdef int ny = Qm.shape[1]
    cdef int k, j, i
    cdef fptype[:, :, ::1] H_view = H

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

                H_view[k, j, i] = (
                    Ap[j, i] * Fm[k, j, i] - Am[j, i] * Fp[k, j, i] +
                    coeff * (Qp[k, j, i] - Qm[k, j, i])
                ) / denominator
    return H


@cython.boundscheck(False)  # deactivate bounds checking
@cython.wraparound(False)  # deactivate negative indexing.
def central_scheme(object states, fptype tol=1e-12):
    """A central scheme to calculate numerical flux at interfaces.

    Arguments
    ---------
    states : torchswe.utils.data.States
    tol : float
        The tolerance that can be considered as zero.

    Returns
    -------
    states : torchswe.utils.data.States
        The same object as the input. Updated in-place. Returning it just for coding style.
    """

    states.face.x.H = central_scheme_kernel[fptype](
        states.face.x.H,
        states.face.x.minus.Q, states.face.x.plus.Q,
        states.face.x.minus.F, states.face.x.plus.F,
        states.face.x.minus.a, states.face.x.plus.a
    )

    states.face.y.H = central_scheme_kernel[fptype](
        states.face.y.H,
        states.face.y.minus.Q, states.face.y.plus.Q,
        states.face.y.minus.F, states.face.y.plus.F,
        states.face.y.minus.a, states.face.y.plus.a
    )

    return states
