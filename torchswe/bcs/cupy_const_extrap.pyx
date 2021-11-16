# vim:fenc=utf-8
# vim:ft=pyrex


cdef _const_copy_kernel = cupy.ElementwiseKernel(
    "T qr",
    "T ql1, T ql2",
    """
        ql1 = qr;
        ql2 = qr;
    """,
    "_const_copy_kernel"
)


cdef _const_copy_w_h_kernel = cupy.ElementwiseKernel(
    "T qr, T hr",
    "T ql1, T ql2, T hl",
    """
        ql1 = qr;
        ql2 = qr;
        hl = hr;
    """,
    "_const_copy_w_h_kernel"
)


cpdef void _const_extrap_west_w(q, h, const Py_ssize_t ngh) except *:
    cdef Py_ssize_t qyed = q.shape[1] - ngh
    cdef Py_ssize_t hyed = h.shape[0] - 1
    assert ngh == 2, "Currently only support ngh = 2"

    _const_copy_w_h_kernel(
        q[0, ngh:qyed, ngh], h[1:hyed, 1],
        q[0, ngh:qyed, 0], q[0, ngh:qyed, 1], h[1:hyed, 0]
    )


cpdef void _const_extrap_west_hu(q, h, const Py_ssize_t ngh) except *:
    cdef Py_ssize_t qyed = q.shape[1] - ngh
    assert ngh == 2, "Currently only support ngh = 2"

    _const_copy_kernel(q[1, ngh:qyed, ngh], q[1, ngh:qyed, 0],  q[1, ngh:qyed, 1])


cpdef void _const_extrap_west_hv(q, h, const Py_ssize_t ngh) except *:
    cdef Py_ssize_t qyed = q.shape[1] - ngh
    assert ngh == 2, "Currently only support ngh = 2"

    _const_copy_kernel(q[2, ngh:qyed, ngh], q[2, ngh:qyed, 0],  q[2, ngh:qyed, 1])


cpdef void _const_extrap_east_w(q, h, const Py_ssize_t ngh) except *:
    cdef Py_ssize_t qyed = q.shape[1] - ngh
    cdef Py_ssize_t qxed = q.shape[2] - ngh
    cdef Py_ssize_t hyed = h.shape[0] - 1
    cdef Py_ssize_t hxed = h.shape[1] - 1
    assert ngh == 2, "Currently only support ngh = 2"

    _const_copy_w_h_kernel(
        q[0, ngh:qyed, qxed-1], h[1:hyed, hxed-1],
        q[0, ngh:qyed, qxed], q[0, ngh:qyed, qxed+1], h[1:hyed, hxed]
    )


cpdef void _const_extrap_east_hu(q, h, const Py_ssize_t ngh) except *:
    cdef Py_ssize_t qyed = q.shape[1] - ngh
    cdef Py_ssize_t qxed = q.shape[2] - ngh
    assert ngh == 2, "Currently only support ngh = 2"

    _const_copy_kernel(q[1, ngh:qyed, qxed-1], q[1, ngh:qyed, qxed],  q[1, ngh:qyed, qxed+1])


cpdef void _const_extrap_east_hv(q, h, const Py_ssize_t ngh) except *:
    cdef Py_ssize_t qyed = q.shape[1] - ngh
    cdef Py_ssize_t qxed = q.shape[2] - ngh
    assert ngh == 2, "Currently only support ngh = 2"

    _const_copy_kernel(q[2, ngh:qyed, qxed-1], q[2, ngh:qyed, qxed],  q[2, ngh:qyed, qxed+1])


cpdef void _const_extrap_south_w(q, h, const Py_ssize_t ngh) except *:
    cdef Py_ssize_t qxed = q.shape[2] - ngh
    cdef Py_ssize_t hxed = h.shape[1] - 1
    assert ngh == 2, "Currently only support ngh = 2"

    _const_copy_w_h_kernel(
        q[0, ngh, ngh:qxed], h[1, 1:hxed],
        q[0, 0, ngh:qxed], q[0, 1, ngh:qxed], h[0, 1:hxed]
    )


cpdef void _const_extrap_south_hu(q, h, const Py_ssize_t ngh) except *:
    cdef Py_ssize_t qxed = q.shape[2] - ngh
    assert ngh == 2, "Currently only support ngh = 2"

    _const_copy_kernel(q[1, ngh, ngh:qxed], q[1, 0, ngh:qxed], q[1, 1, ngh:qxed])


cpdef void _const_extrap_south_hv(q, h, const Py_ssize_t ngh) except *:
    cdef Py_ssize_t qxed = q.shape[2] - ngh
    assert ngh == 2, "Currently only support ngh = 2"

    _const_copy_kernel(q[2, ngh, ngh:qxed], q[2, 0, ngh:qxed], q[2, 1, ngh:qxed])


cpdef void _const_extrap_north_w(q, h, const Py_ssize_t ngh) except *:
    cdef Py_ssize_t qyed = q.shape[1] - ngh
    cdef Py_ssize_t qxed = q.shape[2] - ngh
    cdef Py_ssize_t hyed = h.shape[0] - 1
    cdef Py_ssize_t hxed = h.shape[1] - 1
    assert ngh == 2, "Currently only support ngh = 2"

    _const_copy_w_h_kernel(
        q[0, qyed-1, ngh:qxed], h[hyed-1, 1:hxed],
        q[0, qyed, ngh:qxed], q[0, qyed+1, ngh:qxed], h[hyed, 1:hxed]
    )


cpdef void _const_extrap_north_hu(q, h, const Py_ssize_t ngh) except *:
    cdef Py_ssize_t qyed = q.shape[1] - ngh
    cdef Py_ssize_t qxed = q.shape[2] - ngh
    assert ngh == 2, "Currently only support ngh = 2"

    _const_copy_kernel(q[1, qyed-1, ngh:qxed], q[1, qyed, ngh:qxed], q[1, qyed+1, ngh:qxed])


cpdef void _const_extrap_north_hv(q, h, const Py_ssize_t ngh) except *:
    cdef Py_ssize_t qyed = q.shape[1] - ngh
    cdef Py_ssize_t qxed = q.shape[2] - ngh
    assert ngh == 2, "Currently only support ngh = 2"

    _const_copy_kernel(q[2, qyed-1, ngh:qxed], q[2, qyed, ngh:qxed], q[2, qyed+1, ngh:qxed])
