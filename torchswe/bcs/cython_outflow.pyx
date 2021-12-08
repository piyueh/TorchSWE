# vim:fenc=utf-8
# vim:ft=pyrex
cimport cython


ctypedef fused OutflowBC:
    OutflowFloat
    OutflowDouble


@cython.auto_pickle(False)  # meaningless to pickle a BC instance as everything is a memoryview
cdef class OutflowDouble:  # cython doesn't support templated class yet, so...
    """Outflow (constant extraption) boundary coditions with 64-bit floating points.
    """
    # number of elements
    cdef Py_ssize_t n

    # conservatives
    cdef const double[:] qc0  # w/hu/hv at the cell centers of the 1st internal cell layer
    cdef double[:] qbcm1  # w/hu/hv at the 1st layer of ghost cells
    cdef double[:] qbcm2  # w/hu/hv at the 2nd layer of ghost cells

    def __call__(self):
        _outflow_bc_kernel[OutflowDouble](self)


@cython.auto_pickle(False)  # meaningless to pickle a BC instance as everything is a memoryview
cdef class OutflowFloat:  # cython doesn't support templated class yet, so...
    """Outflow (constant extraption) boundary coditions with 32-bit floating points.
    """
    # number of elements
    cdef Py_ssize_t n

    # conservatives
    cdef const float[:] qc0  # w at the cell centers of the 1st internal cell layer
    cdef float[:] qbcm1  # w/hu/hv at the 1st layer of ghost cells
    cdef float[:] qbcm2  # w/hu/hv at the 2nd layer of ghost cells

    def __call__(self):
        _outflow_bc_kernel[OutflowFloat](self)


cdef inline void _outflow_bc_kernel(OutflowBC bc) nogil:
    cdef Py_ssize_t i
    for i in range(bc.n):
        bc.qbcm1[i] = bc.qc0[i]
        bc.qbcm2[i] = bc.qc0[i]


cdef inline void _outflow_bc_set_west(
    OutflowBC bc,
    cython.floating[:, :, ::1] Q,
    cython.floating[:, ::1] B,
    const Py_ssize_t ngh, const Py_ssize_t comp,
) nogil except *:

    if cython.floating is float and OutflowBC is OutflowDouble:
        raise TypeError("Mismatched types")
    elif cython.floating is double and OutflowBC is OutflowFloat:
        raise TypeError("Mismatched types")
    else:
        bc.n = Q.shape[1] - 2 * ngh  # ny
        bc.qc0      = Q[comp, ngh:Q.shape[1]-ngh, ngh]
        bc.qbcm1    = Q[comp, ngh:Q.shape[1]-ngh, ngh-1]
        bc.qbcm2    = Q[comp, ngh:Q.shape[1]-ngh, ngh-2]

        # modify the topography elevation in ghost cells
        B[ngh:B.shape[0]-ngh, ngh-1] = B[ngh:B.shape[0]-ngh, ngh]
        B[ngh:B.shape[0]-ngh, ngh-2] = B[ngh:B.shape[0]-ngh, ngh]


cdef inline void _outflow_bc_set_east(
    OutflowBC bc,
    cython.floating[:, :, ::1] Q,
    cython.floating[:, ::1] B,
    const Py_ssize_t ngh, const Py_ssize_t comp,
) nogil except *:

    # the first two combinations will be pruned by cython when translating to C/C++
    if cython.floating is float and OutflowBC is OutflowDouble:
        raise TypeError("Mismatched types")
    elif cython.floating is double and OutflowBC is OutflowFloat:
        raise TypeError("Mismatched types")
    else:
        bc.n = Q.shape[1] - 2 * ngh  # ny
        bc.qc0      = Q[comp, ngh:Q.shape[1]-ngh, Q.shape[2]-ngh-1]
        bc.qbcm1    = Q[comp, ngh:Q.shape[1]-ngh, Q.shape[2]-ngh]
        bc.qbcm2    = Q[comp, ngh:Q.shape[1]-ngh, Q.shape[2]-ngh+1]

        # modify the topography elevation in ghost cells
        B[ngh:B.shape[0]-ngh, B.shape[1]-ngh]   = B[ngh:B.shape[0]-ngh, B.shape[1]-ngh-1]
        B[ngh:B.shape[0]-ngh, B.shape[1]-ngh+1] = B[ngh:B.shape[0]-ngh, B.shape[1]-ngh-1]


cdef inline void _outflow_bc_set_south(
    OutflowBC bc,
    cython.floating[:, :, ::1] Q,
    cython.floating[:, ::1] B,
    const Py_ssize_t ngh, const Py_ssize_t comp,
) nogil except *:

    # the first two combinations will be pruned by cython when translating to C/C++
    if cython.floating is float and OutflowBC is OutflowDouble:
        raise TypeError("Mismatched types")
    elif cython.floating is double and OutflowBC is OutflowFloat:
        raise TypeError("Mismatched types")
    else:
        bc.n = Q.shape[2] - 2 * ngh  # nx
        bc.qc0      = Q[comp, ngh,      ngh:Q.shape[2]-ngh]
        bc.qbcm1    = Q[comp, ngh-1,    ngh:Q.shape[2]-ngh]
        bc.qbcm2    = Q[comp, ngh-2,    ngh:Q.shape[2]-ngh]

        # modify the topography elevation in ghost cells
        B[ngh-1, ngh:B.shape[1]-ngh] = B[ngh, ngh:B.shape[1]-ngh]
        B[ngh-2, ngh:B.shape[1]-ngh] = B[ngh, ngh:B.shape[1]-ngh]


cdef inline void _outflow_bc_set_north(
    OutflowBC bc,
    cython.floating[:, :, ::1] Q,
    cython.floating[:, ::1] B,
    const Py_ssize_t ngh, const Py_ssize_t comp,
) nogil except *:

    # the first two combinations will be pruned by cython when translating to C/C++
    if cython.floating is float and OutflowBC is OutflowDouble:
        raise TypeError("Mismatched types")
    elif cython.floating is double and OutflowBC is OutflowFloat:
        raise TypeError("Mismatched types")
    else:
        bc.n = Q.shape[2] - 2 * ngh  # nx
        bc.qc0      = Q[comp, Q.shape[1]-ngh-1,     ngh:Q.shape[2]-ngh]
        bc.qbcm1    = Q[comp, Q.shape[1]-ngh,       ngh:Q.shape[2]-ngh]
        bc.qbcm2    = Q[comp, Q.shape[1]-ngh+1,     ngh:Q.shape[2]-ngh]

        # modify the topography elevation in ghost cells
        B[B.shape[0]-ngh,   ngh:B.shape[1]-ngh] = B[B.shape[0]-ngh-1, ngh:B.shape[1]-ngh]
        B[B.shape[0]-ngh+1, ngh:B.shape[1]-ngh] = B[B.shape[0]-ngh-1, ngh:B.shape[1]-ngh]


cdef inline void _outflow_bc_factory(
    OutflowBC bc,
    cython.floating[:, :, ::1] Q,
    cython.floating[:, ::1] B,
    const Py_ssize_t ngh, const unsigned comp, const unsigned ornt,
) nogil except *:

    # the first two combinations will be pruned by cython when translating to C/C++
    if cython.floating is float and OutflowBC is OutflowDouble:
        raise TypeError("Mismatched types")
    elif cython.floating is double and OutflowBC is OutflowFloat:
        raise TypeError("Mismatched types")
    else:

        assert Q.shape[1] == B.shape[0]
        assert Q.shape[2] == B.shape[1]

        if ornt == 0:  # west
            _outflow_bc_set_west[OutflowBC, cython.floating](bc, Q, B, ngh, comp)
        elif ornt == 1:  # east
            _outflow_bc_set_east[OutflowBC, cython.floating](bc, Q, B, ngh, comp)
        elif ornt == 2:  # south
            _outflow_bc_set_south[OutflowBC, cython.floating](bc, Q, B, ngh, comp)
        elif ornt == 3:  # north
            _outflow_bc_set_north[OutflowBC, cython.floating](bc, Q, B, ngh, comp)
        else:
            raise ValueError(f"orientation id {ornt} not accepted.")


def outflow_bc_factory(ornt, comp, states, topo, *args, **kwargs):
    """Factory to create a outflow (constant extrapolation) boundary condition callable object.
    """

    # aliases
    cdef object Q = states.Q
    cdef object B = topo.centers
    cdef Py_ssize_t ngh = states.domain.nhalo
    cdef str dtype = str(Q.dtype)

    if isinstance(ornt, str):
        if ornt in ["w", "west"]:
            ornt = 0
        elif ornt in ["e", "east"]:
            ornt = 1
        elif ornt in ["s", "south"]:
            ornt = 2
        elif ornt in ["n", "north"]:
            ornt = 3

    if dtype == "float64":
        bc = OutflowDouble()
        _outflow_bc_factory[OutflowDouble, double](bc, Q, B, ngh, comp, ornt)
    elif dtype == "float32":
        bc = OutflowFloat()
        _outflow_bc_factory[OutflowFloat, float](bc, Q, B, ngh, comp, ornt)

    return bc
