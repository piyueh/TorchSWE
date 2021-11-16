# vim:fenc=utf-8
# vim:ft=pyrex


cdef _linear_copy_kernel = cupy.ElementwiseKernel(
    "T q0, T q1, T hg",
    "T qg1, T qg2",
    """
        T dq;

        if (hg > 0.0) {
            dq = q0 - q1;
            qg1 = q0 + dq;
            qg2 = qg1 + dq;
        } else {
            qg1 = 0.0; 
            qg2 = 0.0;
        }
    """,
    "_linear_copy_kernel"
)


cdef _linear_copy_w_h_kernel = cupy.ElementwiseKernel(
    "T q0, T q1, T h0, T h1",
    "T qg1, T qg2, T hg",
    """
        T dq;
        T b0;

        hg = h0 * 2.0 - h1;
        if (hg <= 0.0) {
            b0 = (q0 - h0);
            dq = b0 - q1 + h1;
            qg1 = b0 + dq; 
            qg2 = qg1 + dq;
            hg = 0.0;
        } else {
            dq = q0 - q1;
            qg1 = q0 + dq;
            qg2 = qg1 + dq;
        }
    """,
    "_linear_copy_w_h_kernel"
)


cpdef void _linear_extrap_west_w(q, h, const Py_ssize_t ngh) except *:
    cdef Py_ssize_t qyed = q.shape[1] - ngh
    cdef Py_ssize_t hyed = h.shape[0] - 1
    assert ngh == 2, "Currently only support ngh = 2"

    _linear_copy_w_h_kernel(
        q[0, ngh:qyed, ngh], q[0, ngh:qyed, ngh+1], h[1:hyed, 1], h[1:hyed, 2],
        q[0, ngh:qyed, 1], q[0, ngh:qyed, 0], h[1:hyed, 0])


cpdef void _linear_extrap_west_hu(q, h, const Py_ssize_t ngh) except *:
    cdef Py_ssize_t qyed = q.shape[1] - ngh
    cdef Py_ssize_t hyed = h.shape[0] - 1
    assert ngh == 2, "Currently only support ngh = 2"

    _linear_copy_kernel(
        q[1, ngh:qyed, ngh], q[1, ngh:qyed, ngh+1], h[1:hyed, 0],
        q[1, ngh:qyed, 1], q[1, ngh:qyed, 0])


cpdef void _linear_extrap_west_hv(q, h, const Py_ssize_t ngh) except *:
    cdef Py_ssize_t qyed = q.shape[1] - ngh
    cdef Py_ssize_t hyed = h.shape[0] - 1
    assert ngh == 2, "Currently only support ngh = 2"

    _linear_copy_kernel(
        q[2, ngh:qyed, ngh], q[2, ngh:qyed, ngh+1], h[1:hyed, 0],
        q[2, ngh:qyed, 1], q[2, ngh:qyed, 0])


cpdef void _linear_extrap_east_w(q, h, const Py_ssize_t ngh) except *:
    cdef Py_ssize_t qyed = q.shape[1] - ngh
    cdef Py_ssize_t qxed = q.shape[2] - ngh
    cdef Py_ssize_t hyed = h.shape[0] - 1
    cdef Py_ssize_t hxed = h.shape[1] - 1
    assert ngh == 2, "Currently only support ngh = 2"

    _linear_copy_w_h_kernel(
        q[0, ngh:qyed, qxed-1], q[0, ngh:qyed, qxed-2], h[1:hyed, hxed-1], h[1:hyed, hxed-2],
        q[0, ngh:qyed, qxed], q[0, ngh:qyed, qxed+1], h[1:hyed, hxed])


cpdef void _linear_extrap_east_hu(q, h, const Py_ssize_t ngh) except *:
    cdef Py_ssize_t qyed = q.shape[1] - ngh
    cdef Py_ssize_t qxed = q.shape[2] - ngh
    cdef Py_ssize_t hyed = h.shape[0] - 1
    cdef Py_ssize_t hxed = h.shape[1] - 1
    assert ngh == 2, "Currently only support ngh = 2"

    _linear_copy_kernel(
        q[1, ngh:qyed, qxed-1], q[1, ngh:qyed, qxed-2], h[1:hyed, hxed],
        q[1, ngh:qyed, qxed], q[1, ngh:qyed, qxed+1])


cpdef void _linear_extrap_east_hv(q, h, const Py_ssize_t ngh) except *:
    cdef Py_ssize_t qyed = q.shape[1] - ngh
    cdef Py_ssize_t qxed = q.shape[2] - ngh
    cdef Py_ssize_t hyed = h.shape[0] - 1
    cdef Py_ssize_t hxed = h.shape[1] - 1
    assert ngh == 2, "Currently only support ngh = 2"

    _linear_copy_kernel(
        q[2, ngh:qyed, qxed-1], q[2, ngh:qyed, qxed-2], h[1:hyed, hxed],
        q[2, ngh:qyed, qxed], q[2, ngh:qyed, qxed+1])


cpdef void _linear_extrap_south_w(q, h, const Py_ssize_t ngh) except *:
    cdef Py_ssize_t qxed = q.shape[2] - ngh
    cdef Py_ssize_t hxed = h.shape[1] - 1
    assert ngh == 2, "Currently only support ngh = 2"

    _linear_copy_w_h_kernel(
        q[0, ngh, ngh:qxed], q[0, ngh+1, ngh:qxed], h[1, 1:hxed], h[2, 1:hxed],
        q[0, 1, ngh:qxed], q[0, 0, ngh:qxed], h[0, 1:hxed]) 


cpdef void _linear_extrap_south_hu(q, h, const Py_ssize_t ngh) except *:
    cdef Py_ssize_t qxed = q.shape[2] - ngh
    cdef Py_ssize_t hxed = h.shape[1] - 1
    assert ngh == 2, "Currently only support ngh = 2"

    _linear_copy_kernel(
        q[1, ngh, ngh:qxed], q[1, ngh+1, ngh:qxed], h[0, 1:hxed],
        q[1, 1, ngh:qxed], q[1, 0, ngh:qxed]) 


cpdef void _linear_extrap_south_hv(q, h, const Py_ssize_t ngh) except *:
    cdef Py_ssize_t qxed = q.shape[2] - ngh
    cdef Py_ssize_t hxed = h.shape[1] - 1
    assert ngh == 2, "Currently only support ngh = 2"

    _linear_copy_kernel(
        q[2, ngh, ngh:qxed], q[2, ngh+1, ngh:qxed], h[0, 1:hxed],
        q[2, 1, ngh:qxed], q[2, 0, ngh:qxed]) 


cpdef void _linear_extrap_north_w(q, h, const Py_ssize_t ngh) except *:
    cdef Py_ssize_t qyed = q.shape[1] - ngh
    cdef Py_ssize_t qxed = q.shape[2] - ngh
    cdef Py_ssize_t hyed = h.shape[0] - 1
    cdef Py_ssize_t hxed = h.shape[1] - 1
    assert ngh == 2, "Currently only support ngh = 2"

    _linear_copy_w_h_kernel(
        q[0, qyed-1, ngh:qxed], q[0, qyed-2, ngh:qxed], h[hyed-1, 1:hxed], h[hyed-2, 1:hxed],
        q[0, qyed, ngh:qxed], q[0, qyed+1, ngh:qxed], h[hyed, 1:hxed])


cpdef void _linear_extrap_north_hu(q, h, const Py_ssize_t ngh) except *:
    cdef Py_ssize_t qyed = q.shape[1] - ngh
    cdef Py_ssize_t qxed = q.shape[2] - ngh
    cdef Py_ssize_t hyed = h.shape[0] - 1
    cdef Py_ssize_t hxed = h.shape[1] - 1
    assert ngh == 2, "Currently only support ngh = 2"

    _linear_copy_kernel(
        q[1, qyed-1, ngh:qxed], q[1, qyed-2, ngh:qxed], h[hyed, 1:hxed],
        q[1, qyed, ngh:qxed], q[1, qyed+1, ngh:qxed])


cpdef void _linear_extrap_north_hv(q, h, const Py_ssize_t ngh) except *:
    cdef Py_ssize_t qyed = q.shape[1] - ngh
    cdef Py_ssize_t qxed = q.shape[2] - ngh
    cdef Py_ssize_t hyed = h.shape[0] - 1
    cdef Py_ssize_t hxed = h.shape[1] - 1
    assert ngh == 2, "Currently only support ngh = 2"

    _linear_copy_kernel(
        q[2, qyed-1, ngh:qxed], q[2, qyed-2, ngh:qxed], h[hyed, 1:hxed],
        q[2, qyed, ngh:qxed], q[2, qyed+1, ngh:qxed])
