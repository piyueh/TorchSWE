# Cython implementation of minmod
import numpy
cimport numpy
cimport cython
numpy.seterr(divide="ignore", invalid="ignore")


# fused floating point type; uses as the template varaible type in C++
ctypedef fused fptype:
    float
    double


@cython.boundscheck(False)  # deactivate bounds checking
@cython.wraparound(False)  # deactivate negative indexing.
cdef numpy.ndarray[fptype, ndim=3] minmod_slope_x(
    numpy.ndarray[fptype, ndim=3] slp,
    fptype[:, :, ::1] Q,
    fptype theta,
    fptype delta,
    int ngh
):
    if fptype is float:
        dtype = numpy.float32
    elif fptype is double:
        dtype = numpy.double

    cdef int ny = Q.shape[1] - ngh * 2
    cdef int nx = Q.shape[2] - ngh * 2
    cdef int i
    cdef int j
    cdef int k

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
                slp_view[k, j, i] = value
    return slp


@cython.boundscheck(False)  # deactivate bounds checking
@cython.wraparound(False)  # deactivate negative indexing.
cdef numpy.ndarray[fptype, ndim=3] minmod_slope_y(
    numpy.ndarray[fptype, ndim=3] slp,
    fptype[:, :, ::1] Q,
    fptype theta,
    fptype delta,
    int ngh
):
    if fptype is float:
        dtype = numpy.float32
    elif fptype is double:
        dtype = numpy.double

    cdef int ny = Q.shape[1] - ngh * 2
    cdef int nx = Q.shape[2] - ngh * 2
    cdef int i
    cdef int j
    cdef int k

    cdef fptype value
    cdef fptype denominator
    cdef fptype[:, :, ::1] slp_view = slp

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
                slp_view[k, j, i] = value
    return slp


@cython.boundscheck(False)  # deactivate bounds checking
@cython.wraparound(False)  # deactivate negative indexing.
def minmod_slope(object states, fptype theta):
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

    states.slpx = \
            minmod_slope_x[fptype](states.slpx, states.Q, theta, states.domain.x.delta, states.ngh)
    states.slpy = \
            minmod_slope_y[fptype](states.slpy, states.Q, theta, states.domain.y.delta, states.ngh)
    return states
