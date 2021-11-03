# Cython implementation of minmod
import numpy
cimport numpy
cimport cython
numpy.seterr(divide="ignore", invalid="ignore")


@cython.boundscheck(False)  # deactivate bounds checking
@cython.wraparound(False)  # deactivate negative indexing.
cdef void minmod_slope_x(
    fptype[:, :, ::1] slp,
    confptype[:, :, ::1] Q,
    const double theta,
    const double delta,
    const Py_ssize_t ngh
) nogil:
    """Kernel for calculating slope in x direction.
    """
    cdef Py_ssize_t ny = Q.shape[1] - ngh * 2
    cdef Py_ssize_t nx = Q.shape[2] - ngh * 2
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t k

    cdef fptype value
    cdef fptype denominator
    cdef fptype[:, :, ::1] slp_view = slp

    for k in range(3):
        for j in range(ny):
            for i in range(nx+2):
                denominator = Q[k, ngh+j, ngh+i] - Q[k, ngh+j, ngh+i-1]

                if denominator == 0.:  # only care when it is exactly zero
                    continue  # imnplying slp[k, j, i] = 0, which is set when initialization

                value = (Q[k, ngh+j, ngh+i-1] - Q[k, ngh+j, ngh+i-2]) / denominator
                value = min(theta*value, (1.0+value)/2.0)
                value = min(value, theta)
                value = max(value, 0.)
                value *= denominator
                value /= delta
                slp[k, j, i] = value


@cython.boundscheck(False)  # deactivate bounds checking
@cython.wraparound(False)  # deactivate negative indexing.
cdef void minmod_slope_y(
    fptype[:, :, ::1] slp,
    confptype[:, :, ::1] Q,
    const double theta,
    const double delta,
    const Py_ssize_t ngh
) nogil:
    """Kernel for calculating slope in y direction.
    """
    cdef Py_ssize_t ny = Q.shape[1] - ngh * 2
    cdef Py_ssize_t nx = Q.shape[2] - ngh * 2
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t k

    cdef fptype value
    cdef fptype denominator

    for k in range(3):
        for j in range(ny+2):
            for i in range(nx):
                denominator = Q[k, ngh+j, ngh+i] - Q[k, ngh+j-1, ngh+i]

                if denominator == 0.:  # only care when it is exactly zero
                    continue  # imnplying slp[k, j, i] = 0, which is set when initialization

                value = (Q[k, ngh+j-1, ngh+i] - Q[k, ngh+j-2, ngh+i]) / denominator
                value = min(theta*value, (1.0+value)/2.0)
                value = min(value, theta)
                value = max(value, 0.)
                value *= denominator
                value /= delta
                slp[k, j, i] = value


@cython.boundscheck(False)  # deactivate bounds checking
@cython.wraparound(False)  # deactivate negative indexing.
def minmod_slope(object states, double theta):
    """Calculate the slope of using minmod limiter.

    Arguments
    ---------
    states : torchswe.utils.data.State
        The instance of States holding quantities.
    theta : float
        The parameter adjusting the dissipation.

    Returns
    -------
    slpx : numpy.ndarray with shape (3, ny, nx+2)
    slpy : numpy.ndarray with shape (3, ny+2, nx)
    """

    # aliases to reduce calls in the generated c/c++ code
    cdef double dx = states.domain.x.delta
    cdef double dy = states.domain.y.delta
    cdef Py_ssize_t ngh = states.ngh
    dtype = states.slpx.dtype  # what's the type?
    slpx = states.slpx
    slpy = states.slpy
    Q = states.Q

    if dtype == numpy.single:
        minmod_slope_x[cython.float, cython.float](slpx, Q, theta, dx, ngh)
        minmod_slope_y[cython.float, cython.float](slpy, Q, theta, dy, ngh)
    elif dtype == numpy.double:
        minmod_slope_x[cython.double, cython.double](slpx, Q, theta, dx, ngh)
        minmod_slope_y[cython.double, cython.double](slpy, Q, theta, dy, ngh)
    else:
        raise RuntimeError(f"Arrays are using an unrecognized dtype: {dtype}")
    return states
