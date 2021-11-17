# vim:fenc=utf-8
# vim:ft=pyrex


cpdef void _linear_extrap_west_w(
    qtype q, htype h, const Py_ssize_t ngh, btype b, btype bx, *args
) nogil except *:
    cdef Py_ssize_t qyed = q.shape[1] - ngh
    cdef Py_ssize_t i, j, jh, jb
    cdef cython.floating delta  # actual FP type aligned with underlying type of qtype/htype

    cdef Py_ssize_t hg = 0
    cdef Py_ssize_t hi1 = hg + 1
    cdef Py_ssize_t hi2 = hi1 + 1

    cdef Py_ssize_t qg1 = ngh - 1
    cdef Py_ssize_t qi1 = qg1 + 1
    cdef Py_ssize_t qi2 = qi1 + 1

    jh = 1; jb = 0
    for j in range(ngh, qyed):
        h[jh, hg] = h[jh, hi1] * 2.0 - h[jh, hi2]

        if h[jh, hg] <= 0.0:
            h[jh, hg] = 0.0
            delta = (bx[jb, 0] - b[jb, 0]) * 2.0  # delta represents db
            q[0, j, qg1] = b[jb, 0] + delta  # interpolate topo elevation
        else:
            delta = q[0, j, qi1] - q[0, j, qi2]
            q[0, j, qg1] = q[0, j, qi1] + delta  # interpolate water elevation

        for i in range(qg1-1, -1, -1):
            q[0, j, i] = q[0, j, i+1] + delta

        jh += 1; jb += 1


cpdef void _linear_extrap_west_hu(qtype q, htype h, const Py_ssize_t ngh, *args) nogil except *:
    cdef Py_ssize_t qyed = q.shape[1] - ngh
    cdef Py_ssize_t i, j, jh
    cdef cython.floating delta  # actual FP type aligned with underlying type of qtype/htype

    cdef Py_ssize_t hg = 0
    cdef Py_ssize_t qg1 = ngh - 1
    cdef Py_ssize_t qi1 = qg1 + 1
    cdef Py_ssize_t qi2 = qi1 + 1

    jh = 1
    for j in range(ngh, qyed):
        if h[jh, hg] > 0.0:
            delta = q[1, j, qi1] - q[1, j, qi2]
            for i in range(qg1, -1, -1):
                q[1, j, i] = q[1, j, i+1] + delta
        else:
            for i in range(qg1, -1, -1):
                q[1, j, i] = 0.0

        jh += 1


cpdef void _linear_extrap_west_hv(qtype q, htype h, const Py_ssize_t ngh, *args) nogil except *:
    cdef Py_ssize_t qyed = q.shape[1] - ngh
    cdef Py_ssize_t i, j, jh
    cdef cython.floating delta  # actual FP type aligned with underlying type of qtype/htype

    cdef Py_ssize_t hg = 0
    cdef Py_ssize_t qg1 = ngh - 1
    cdef Py_ssize_t qi1 = qg1 + 1
    cdef Py_ssize_t qi2 = qi1 + 1

    jh = 1
    for j in range(ngh, qyed):
        if h[jh, hg] > 0.0:
            delta = q[2, j, qi1] - q[2, j, qi2]
            for i in range(qg1, -1, -1):
                q[2, j, i] = q[2, j, i+1] + delta
        else:
            for i in range(qg1, -1, -1):
                q[2, j, i] = 0.0

        jh += 1


cpdef void _linear_extrap_east_w(
    qtype q, htype h, const Py_ssize_t ngh, btype b, btype bx, *args
) nogil except *:
    cdef Py_ssize_t qyed = q.shape[1] - ngh
    cdef Py_ssize_t bxed = bx.shape[1] - 1
    cdef Py_ssize_t bed = b.shape[1] - 1
    cdef Py_ssize_t i, j, jh, jb
    cdef cython.floating delta  # actual FP type aligned with underlying type of qtype/htype

    cdef Py_ssize_t hg = h.shape[1] - 1
    cdef Py_ssize_t hi1 = hg - 1
    cdef Py_ssize_t hi2 = hi1 - 1

    cdef Py_ssize_t qg1 = q.shape[2] - ngh
    cdef Py_ssize_t qi1 = qg1 - 1
    cdef Py_ssize_t qi2 = qi1 - 1

    jh = 1; jb = 0
    for j in range(ngh, qyed):
        h[jh, hg] = h[jh, hi1] * 2.0 - h[jh, hi2]

        if h[jh, hg] <= 0.0:
            h[jh, hg] = 0.0
            delta = (bx[jb, bxed] - b[jb, bed]) * 2.0  # delta represents db
            q[0, j, qg1] = b[jb, bed] + delta  # interpolate topo elevation
        else:
            delta = q[0, j, qi1] - q[0, j, qi2]
            q[0, j, qg1] = q[0, j, qi1] + delta  # interpolate water elevation

        for i in range(qg1+1, q.shape[2]):  # cythong works well with `range` and an implicit step
            q[0, j, i] = q[0, j, i-1] + delta

        jh += 1; jb += 1


cpdef void _linear_extrap_east_hu(qtype q, htype h, const Py_ssize_t ngh, *args) nogil except *:
    cdef Py_ssize_t qyed = q.shape[1] - ngh
    cdef Py_ssize_t i, j, jh
    cdef cython.floating delta  # actual FP type aligned with underlying type of qtype/htype

    cdef Py_ssize_t hg = h.shape[1] - 1
    cdef Py_ssize_t qg1 = q.shape[2] - ngh
    cdef Py_ssize_t qi1 = qg1 - 1
    cdef Py_ssize_t qi2 = qi1 - 1

    jh = 1
    for j in range(ngh, qyed):
        if h[jh, hg] > 0.0:
            delta = q[1, j, qi1] - q[1, j, qi2]
            for i in range(qg1, q.shape[2]):
                q[1, j, i] = q[1, j, i-1] + delta
        else:
            for i in range(qg1, q.shape[2]):
                q[1, j, i] = 0.0

        jh += 1


cpdef void _linear_extrap_east_hv(qtype q, htype h, const Py_ssize_t ngh, *args) nogil except *:
    cdef Py_ssize_t qyed = q.shape[1] - ngh
    cdef Py_ssize_t i, j, jh
    cdef cython.floating delta  # actual FP type aligned with underlying type of qtype/htype

    cdef Py_ssize_t hg = h.shape[1] - 1
    cdef Py_ssize_t qg1 = q.shape[2] - ngh
    cdef Py_ssize_t qi1 = qg1 - 1
    cdef Py_ssize_t qi2 = qi1 - 1

    jh = 1
    for j in range(ngh, qyed):
        if h[jh, hg] > 0.0:
            delta = q[2, j, qi1] - q[2, j, qi2]
            for i in range(qg1, q.shape[2]):
                q[2, j, i] = q[2, j, i-1] + delta
        else:
            for i in range(qg1, q.shape[2]):
                q[2, j, i] = 0.0

        jh += 1


cpdef void _linear_extrap_south_w(
    qtype q, htype h, const Py_ssize_t ngh, btype b, btype by, *args
) nogil except *:
    cdef Py_ssize_t qxed = q.shape[2] - ngh
    cdef Py_ssize_t i, j, ih, ib
    cdef cython.floating delta  # actual FP type aligned with underlying type of qtype/htype

    cdef Py_ssize_t hg = 0
    cdef Py_ssize_t hi1 = hg + 1
    cdef Py_ssize_t hi2 = hi1 + 1

    cdef Py_ssize_t qg1 = ngh - 1
    cdef Py_ssize_t qi1 = qg1 + 1
    cdef Py_ssize_t qi2 = qi1 + 1

    ih = 1; ib = 0
    for i in range(ngh, qxed):  # not an efficient loop; cache miss
        h[hg, ih] = h[hi1, ih] * 2.0 - h[hi2, ih]

        if h[hg, ih] <= 0.0:
            h[hg, ih] = 0.0
            delta = (by[0, ib] - b[0, ib]) * 2.0
            q[0, qg1, i] = b[0, ib] + delta
        else:
            delta = q[0, qi1, i] - q[0, qi2, i]
            q[0, qg1, i] = q[0, qi1, i] + delta

        for j in range(qg1-1, -1, -1):
            q[0, j, i] = q[0, j+1, i] + delta
        
        ih += 1; ib += 1


cpdef void _linear_extrap_south_hu(qtype q, htype h, const Py_ssize_t ngh, *args) nogil except *:
    cdef Py_ssize_t qxed = q.shape[2] - ngh
    cdef Py_ssize_t i, j, ih
    cdef cython.floating delta  # actual FP type aligned with underlying type of qtype/htype

    cdef Py_ssize_t hg = 0
    cdef Py_ssize_t qg1 = ngh - 1
    cdef Py_ssize_t qi1 = qg1 + 1
    cdef Py_ssize_t qi2 = qi1 + 1

    ih = 1
    for i in range(ngh, qxed):  # not an efficient loop; cache miss
        if h[hg, ih] > 0.0:
            delta = q[1, qi1, i] - q[1, qi2, i]
            for j in range(qg1, -1, -1):
                q[1, j, i] = q[1, j+1, i] + delta
        else:
            for j in range(qg1, -1, -1):
                q[1, j, i] = 0.0

        ih += 1


cpdef void _linear_extrap_south_hv(qtype q, htype h, const Py_ssize_t ngh, *args) nogil except *:
    cdef Py_ssize_t qxed = q.shape[2] - ngh
    cdef Py_ssize_t i, j, ih
    cdef cython.floating delta  # actual FP type aligned with underlying type of qtype/htype

    cdef Py_ssize_t hg = 0
    cdef Py_ssize_t qg1 = ngh - 1
    cdef Py_ssize_t qi1 = qg1 + 1
    cdef Py_ssize_t qi2 = qi1 + 1

    ih = 1
    for i in range(ngh, qxed):  # not an efficient loop; cache miss
        if h[hg, ih] > 0.0:
            delta = q[2, qi1, i] - q[2, qi2, i]
            for j in range(qg1, -1, -1):
                q[2, j, i] = q[2, j+1, i] + delta
        else:
            for j in range(qg1, -1, -1):
                q[2, j, i] = 0.0

        ih += 1


cpdef void _linear_extrap_north_w(
    qtype q, htype h, const Py_ssize_t ngh, btype b, btype by, *args
) nogil except *:
    cdef Py_ssize_t qxed = q.shape[2] - ngh
    cdef Py_ssize_t byed = by.shape[0] - 1
    cdef Py_ssize_t bed = b.shape[0] - 1
    cdef Py_ssize_t i, j, ih, ib
    cdef cython.floating delta  # actual FP type aligned with underlying type of qtype/htype

    cdef Py_ssize_t hg = h.shape[0] - 1
    cdef Py_ssize_t hi1 = hg - 1
    cdef Py_ssize_t hi2 = hi1 - 1

    cdef Py_ssize_t qg1 = q.shape[1] - ngh
    cdef Py_ssize_t qi1 = qg1 - 1
    cdef Py_ssize_t qi2 = qi1 - 1

    ih = 1; ib = 0
    for i in range(ngh, qxed):
        h[hg, ih] = h[hi1, ih] * 2.0 - h[hi2, ih]

        if h[hg, ih] <= 0.0:
            h[hg, ih] = 0.0
            delta = (by[byed, ib] - b[bed, ib]) * 2.0
            q[0, qg1, i] = b[bed, ib] + delta
        else:
            delta = q[0, qi1, i] - q[0, qi2, i]
            q[0, qg1, i] = q[0, qi1, i] + delta

        for j in range(qg1+1, q.shape[1]):
            q[0, j, i] = q[0, j-1, i] + delta
        
        ih += 1; ib += 1


cpdef void _linear_extrap_north_hu(qtype q, htype h, const Py_ssize_t ngh, *args) nogil except *:
    cdef Py_ssize_t qxed = q.shape[2] - ngh
    cdef Py_ssize_t i, j, ih
    cdef cython.floating delta  # actual FP type aligned with underlying type of qtype/htype

    cdef Py_ssize_t hg = h.shape[0] - 1
    cdef Py_ssize_t qg1 = q.shape[1] - ngh
    cdef Py_ssize_t qi1 = qg1 - 1
    cdef Py_ssize_t qi2 = qi1 - 1

    ih = 1
    for i in range(ngh, qxed):
        if h[hg, ih] > 0.0:
            delta = q[1, qi1, i] - q[1, qi2, i]
            for j in range(qg1, q.shape[1]):
                q[1, j, i] = q[1, j-1, i] + delta
        else:
            for j in range(qg1, q.shape[1]):
                q[1, j, i] = 0.0
        
        ih += 1


cpdef void _linear_extrap_north_hv(qtype q, htype h, const Py_ssize_t ngh, *args) nogil except *:
    cdef Py_ssize_t qxed = q.shape[2] - ngh
    cdef Py_ssize_t i, j, ih
    cdef cython.floating delta  # actual FP type aligned with underlying type of qtype/htype

    cdef Py_ssize_t hg = h.shape[0] - 1
    cdef Py_ssize_t qg1 = q.shape[1] - ngh
    cdef Py_ssize_t qi1 = qg1 - 1
    cdef Py_ssize_t qi2 = qi1 - 1

    ih = 1
    for i in range(ngh, qxed):
        if h[hg, ih] > 0.0:
            delta = q[2, qi1, i] - q[2, qi2, i]
            for j in range(qg1, q.shape[1]):
                q[2, j, i] = q[2, j-1, i] + delta
        else:
            for j in range(qg1, q.shape[1]):
                q[2, j, i] = 0.0
        
        ih += 1
