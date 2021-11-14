# vim:fenc=utf-8
# vim:ft=pyrex
cimport numpy


cpdef void _const_extrap_west_w(qtype q, htype h, const Py_ssize_t ngh) nogil except *:
    cdef Py_ssize_t qyed = q.shape[1] - ngh
    cdef Py_ssize_t hyed = h.shape[0] - 1
    cdef Py_ssize_t i, j

    for j in range(ngh, qyed):
        for i in range(ngh):
            q[0, j, i] = q[0, j, ngh]

    for j in range(1, hyed):
        h[j, 0] = h[j, 1]


cpdef void _const_extrap_west_hu(qtype q, htype h, const Py_ssize_t ngh) nogil except *:
    cdef Py_ssize_t qyed = q.shape[1] - ngh
    cdef Py_ssize_t i, j

    for j in range(ngh, qyed):
        for i in range(ngh):
            q[1, j, i] = q[1, j, ngh]


cpdef void _const_extrap_west_hv(qtype q, htype h, const Py_ssize_t ngh) nogil except *:
    cdef Py_ssize_t qyed = q.shape[1] - ngh
    cdef Py_ssize_t i, j

    for j in range(ngh, qyed):
        for i in range(ngh):
            q[2, j, i] = q[2, j, ngh]


cpdef void _const_extrap_east_w(qtype q, htype h, const Py_ssize_t ngh) nogil except *:
    cdef Py_ssize_t qyed = q.shape[1] - ngh
    cdef Py_ssize_t qxed = q.shape[2] - ngh
    cdef Py_ssize_t target = qxed - 1
    cdef Py_ssize_t hyed = h.shape[0] - 1
    cdef Py_ssize_t hxed = h.shape[1] - 1
    cdef Py_ssize_t i, j

    for j in range(ngh, qyed):
        for i in range(qxed, q.shape[2]):
            q[0, j, i] = q[0, j, target]

    target = hxed - 1
    for j in range(1, hyed):
        h[j, hxed] = h[j, target]


cpdef void _const_extrap_east_hu(qtype q, htype h, const Py_ssize_t ngh) nogil except *:
    cdef Py_ssize_t qyed = q.shape[1] - ngh
    cdef Py_ssize_t qxed = q.shape[2] - ngh
    cdef Py_ssize_t target = qxed - 1
    cdef Py_ssize_t i, j

    for j in range(ngh, qyed):
        for i in range(qxed, q.shape[2]):
            q[1, j, i] = q[1, j, target]


cpdef void _const_extrap_east_hv(qtype q, htype h, const Py_ssize_t ngh) nogil except *:
    cdef Py_ssize_t qyed = q.shape[1] - ngh
    cdef Py_ssize_t qxed = q.shape[2] - ngh
    cdef Py_ssize_t target = qxed - 1
    cdef Py_ssize_t i, j

    for j in range(ngh, qyed):
        for i in range(qxed, q.shape[2]):
            q[2, j, i] = q[2, j, target]


cpdef void _const_extrap_south_w(qtype q, htype h, const Py_ssize_t ngh) nogil except *:
    cdef Py_ssize_t qxed = q.shape[2] - ngh
    cdef Py_ssize_t hxed = h.shape[1] - 1
    cdef Py_ssize_t i, j

    for j in range(ngh):
        for i in range(ngh, qxed):
            q[0, j, i] = q[0, ngh, i]

    for i in range(1, hxed):
        h[0, i] = h[1, i]


cpdef void _const_extrap_south_hu(qtype q, htype h, const Py_ssize_t ngh) nogil except *:
    cdef Py_ssize_t qxed = q.shape[2] - ngh
    cdef Py_ssize_t i, j

    for j in range(ngh):
        for i in range(ngh, qxed):
            q[1, j, i] = q[1, ngh, i]


cpdef void _const_extrap_south_hv(qtype q, htype h, const Py_ssize_t ngh) nogil except *:
    cdef Py_ssize_t qxed = q.shape[2] - ngh
    cdef Py_ssize_t i, j

    for j in range(ngh):
        for i in range(ngh, qxed):
            q[2, j, i] = q[2, ngh, i]


cpdef void _const_extrap_north_w(qtype q, htype h, const Py_ssize_t ngh) nogil except *:
    cdef Py_ssize_t qyed = q.shape[1] - ngh
    cdef Py_ssize_t qxed = q.shape[2] - ngh
    cdef Py_ssize_t target = qyed - 1
    cdef Py_ssize_t hyed = h.shape[0] - 1
    cdef Py_ssize_t hxed = h.shape[1] - 1
    cdef Py_ssize_t i, j

    for j in range(qyed, q.shape[1]):
        for i in range(ngh, qxed):
            q[0, j, i] = q[0, target, i]

    target = hyed - 1
    for i in range(1, hxed):
        h[hyed, i] = h[target, i]


cpdef void _const_extrap_north_hu(qtype q, htype h, const Py_ssize_t ngh) nogil except *:
    cdef Py_ssize_t qyed = q.shape[1] - ngh
    cdef Py_ssize_t qxed = q.shape[2] - ngh
    cdef Py_ssize_t target = qyed - 1
    cdef Py_ssize_t i, j

    for j in range(qyed, q.shape[1]):
        for i in range(ngh, qxed):
            q[1, j, i] = q[1, target, i]


cpdef void _const_extrap_north_hv(qtype q, htype h, const Py_ssize_t ngh) nogil except *:
    cdef Py_ssize_t qyed = q.shape[1] - ngh
    cdef Py_ssize_t qxed = q.shape[2] - ngh
    cdef Py_ssize_t target = qyed - 1
    cdef Py_ssize_t i, j

    for j in range(qyed, q.shape[1]):
        for i in range(ngh, qxed):
            q[2, j, i] = q[2, target, i]
