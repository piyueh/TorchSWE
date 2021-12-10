# vim:fenc=utf-8
# vim:ft=pyrex
cimport cython


ctypedef fused LinearExtrapBC:
    LinearExtrapFloat
    LinearExtrapDouble


@cython.auto_pickle(False)  # meaningless to pickle a BC instance as everything is a memoryview
cdef class LinearExtrapDouble:
    """Linear extraption boundary coditions wiht 64-bit floating points.
    """
    # number of elements
    cdef Py_ssize_t n

    # conservatives
    cdef const double[:] qc0  # w/hu/hv at the cell centers of the 1st internal cell layer
    cdef const double[:] qc1  # w/hu/hv at the cell centers of the 2nd internal cell layer
    cdef double[:] qbcm1  # w/hu/hv at the 1st layer of ghost cells
    cdef double[:] qbcm2  # w/hu/hv at the 2nd layer of ghost cells

    def __call__(self):
        _linear_extrap_bc_kernel[LinearExtrapDouble, double](self, 0.0)


@cython.auto_pickle(False)  # meaningless to pickle a BC instance as everything is a memoryview
cdef class LinearExtrapFloat:
    """Linear extraption boundary coditions wiht 32-bit floating points.
    """
    # number of elements
    cdef Py_ssize_t n

    # conservatives
    cdef const float[:] qc0  # w/hu/hv at the cell centers of the 1st internal cell layer
    cdef const float[:] qc1  # w/hu/hv at the cell centers of the 2nd internal cell layer
    cdef float[:] qbcm1  # w/hu/hv at the 1st layer of ghost cells
    cdef float[:] qbcm2  # w/hu/hv at the 2nd layer of ghost cells

    def __call__(self):
        _linear_extrap_bc_kernel[LinearExtrapFloat, float](self, 0.0)


cdef void _linear_extrap_bc_kernel(LinearExtrapBC bc, cython.floating delta) nogil:
    cdef Py_ssize_t i
    for i in range(bc.n):
        delta = bc.qc0[i] - bc.qc1[i];
        bc.qbcm1[i] = bc.qc0[i] + delta;
        bc.qbcm2[i] = bc.qbcm1[i] + delta;


cdef inline void _linear_extrap_bc_set_west(
    LinearExtrapBC bc,
    cython.floating[:, :, ::1] Q,
    cython.floating[:, ::1] B,
    const cython.floating[:, ::1] Bx,
    const Py_ssize_t ngh, const Py_ssize_t comp,
) nogil except *:

    cdef Py_ssize_t i;

    # the first two combinations will be pruned by cython when translating to C/C++
    if cython.floating is float and LinearExtrapBC is LinearExtrapDouble:
        raise TypeError("Mismatched types")
    elif cython.floating is double and LinearExtrapBC is LinearExtrapFloat:
        raise TypeError("Mismatched types")
    else:
        bc.n = Q.shape[1] - 2 * ngh  # ny
        bc.qc0      = Q[comp, ngh:Q.shape[1]-ngh, ngh]
        bc.qc1      = Q[comp, ngh:Q.shape[1]-ngh, ngh+1]
        bc.qbcm1    = Q[comp, ngh:Q.shape[1]-ngh, ngh-1]
        bc.qbcm2    = Q[comp, ngh:Q.shape[1]-ngh, ngh-2]

        # modify the topography elevation in ghost cells
        for i in range(ngh, B.shape[0]-ngh):
            B[i, ngh-1] = Bx[i-ngh, 0] * 2.0 - B[i, ngh]
            B[i, ngh-2] = Bx[i-ngh, 0] * 4.0 - B[i, ngh] * 3.0


cdef inline void _linear_extrap_bc_set_east(
    LinearExtrapBC bc,
    cython.floating[:, :, ::1] Q,
    cython.floating[:, ::1] B,
    const cython.floating[:, ::1] Bx,
    const Py_ssize_t ngh, const Py_ssize_t comp,
) nogil except *:

    cdef Py_ssize_t i;

    # the first two combinations will be pruned by cython when translating to C/C++
    if cython.floating is float and LinearExtrapBC is LinearExtrapDouble:
        raise TypeError("Mismatched types")
    elif cython.floating is double and LinearExtrapBC is LinearExtrapFloat:
        raise TypeError("Mismatched types")
    else:
        bc.n = Q.shape[1] - 2 * ngh  # ny
        bc.qc0      = Q[comp, ngh:Q.shape[1]-ngh, Q.shape[2]-ngh-1]
        bc.qc1      = Q[comp, ngh:Q.shape[1]-ngh, Q.shape[2]-ngh-2]
        bc.qbcm1    = Q[comp, ngh:Q.shape[1]-ngh, Q.shape[2]-ngh]
        bc.qbcm2    = Q[comp, ngh:Q.shape[1]-ngh, Q.shape[2]-ngh+1]

        # modify the topography elevation in ghost cells
        for i in range(ngh, B.shape[0]-ngh):
            B[i, B.shape[1]-ngh]   = Bx[i-ngh, Bx.shape[1]-1] * 2.0 - B[i, B.shape[1]-ngh-1]
            B[i, B.shape[1]-ngh+1] = Bx[i-ngh, Bx.shape[1]-1] * 4.0 - B[i, B.shape[1]-ngh-1] * 3.0


cdef inline void _linear_extrap_bc_set_south(
    LinearExtrapBC bc,
    cython.floating[:, :, ::1] Q,
    cython.floating[:, ::1] B,
    const cython.floating[:, ::1] By,
    const Py_ssize_t ngh, const Py_ssize_t comp,
) nogil except *:

    cdef Py_ssize_t i;

    # the first two combinations will be pruned by cython when translating to C/C++
    if cython.floating is float and LinearExtrapBC is LinearExtrapDouble:
        raise TypeError("Mismatched types")
    elif cython.floating is double and LinearExtrapBC is LinearExtrapFloat:
        raise TypeError("Mismatched types")
    else:
        bc.n = Q.shape[2] - 2 * ngh  # nx
        bc.qc0      = Q[comp, ngh,      ngh:Q.shape[2]-ngh]
        bc.qc1      = Q[comp, ngh+1,    ngh:Q.shape[2]-ngh]
        bc.qbcm1    = Q[comp, ngh-1,    ngh:Q.shape[2]-ngh]
        bc.qbcm2    = Q[comp, ngh-2,    ngh:Q.shape[2]-ngh]

        # modify the topography elevation in ghost cells
        for i in range(ngh, B.shape[1]-ngh):
            B[ngh-1, i] = By[0, i-ngh] * 2.0 - B[ngh, i]
            B[ngh-2, i] = By[0, i-ngh] * 4.0 - B[ngh, i] * 3.0


cdef inline void _linear_extrap_bc_set_north(
    LinearExtrapBC bc,
    cython.floating[:, :, ::1] Q,
    cython.floating[:, ::1] B,
    const cython.floating[:, ::1] By,
    const Py_ssize_t ngh, const Py_ssize_t comp,
) nogil except *:

    cdef Py_ssize_t i;

    # the first two combinations will be pruned by cython when translating to C/C++
    if cython.floating is float and LinearExtrapBC is LinearExtrapDouble:
        raise TypeError("Mismatched types")
    elif cython.floating is double and LinearExtrapBC is LinearExtrapFloat:
        raise TypeError("Mismatched types")
    else:
        bc.n = Q.shape[2] - 2 * ngh  # nx
        bc.qc0      = Q[comp, Q.shape[1]-ngh-1,     ngh:Q.shape[2]-ngh]
        bc.qc1      = Q[comp, Q.shape[1]-ngh-2,     ngh:Q.shape[2]-ngh]
        bc.qbcm1    = Q[comp, Q.shape[1]-ngh,       ngh:Q.shape[2]-ngh]
        bc.qbcm2    = Q[comp, Q.shape[1]-ngh+1,     ngh:Q.shape[2]-ngh]

        # modify the topography elevation in ghost cells
        for i in range(ngh, B.shape[1]-ngh):
            B[B.shape[0]-ngh,   i] = By[By.shape[0]-1, i-ngh] * 2.0 - B[B.shape[0]-ngh-1, i]
            B[B.shape[0]-ngh+1, i] = By[By.shape[0]-1, i-ngh] * 4.0 - B[B.shape[0]-ngh-1, i] * 3.0


cdef inline void _linear_extrap_bc_factory(
    LinearExtrapBC bc,
    cython.floating[:, :, ::1] Q,
    cython.floating[:, ::1] B,
    const cython.floating[:, ::1] Bx,
    const cython.floating[:, ::1] By,
    const Py_ssize_t ngh, const unsigned comp, const unsigned ornt,
) nogil except *:

    # the first two combinations will be pruned by cython when translating to C/C++
    if cython.floating is float and LinearExtrapBC is LinearExtrapDouble:
        raise TypeError("Mismatched types")
    elif cython.floating is double and LinearExtrapBC is LinearExtrapFloat:
        raise TypeError("Mismatched types")
    else:

        assert Q.shape[1] == B.shape[0]
        assert Q.shape[2] == B.shape[1]
        assert Q.shape[1] == Bx.shape[0] + 2 * ngh
        assert Q.shape[2] == Bx.shape[1] + 2 * ngh - 1
        assert Q.shape[1] == By.shape[0] + 2 * ngh - 1
        assert Q.shape[2] == By.shape[1] + 2 * ngh

        if ornt == 0:  # west
            _linear_extrap_bc_set_west[LinearExtrapBC, cython.floating](bc, Q, B, Bx, ngh, comp)
        elif ornt == 1:  # east
            _linear_extrap_bc_set_east[LinearExtrapBC, cython.floating](bc, Q, B, Bx, ngh, comp)
        elif ornt == 2:  # south
            _linear_extrap_bc_set_south[LinearExtrapBC, cython.floating](bc, Q, B, By, ngh, comp)
        elif ornt == 3:  # north
            _linear_extrap_bc_set_north[LinearExtrapBC, cython.floating](bc, Q, B, By, ngh, comp)
        else:
            raise ValueError(f"orientation id {ornt} not accepted.")


def linear_extrap_bc_factory(ornt, comp, states, topo, *args, **kwargs):
    """Factory to create a linear extrapolation boundary condition callable object.
    """

    # aliases
    cdef object Q = states.q
    cdef object B = topo.c
    cdef object Bx = topo.xf
    cdef object By = topo.yf
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
        bc = LinearExtrapDouble()
        _linear_extrap_bc_factory[LinearExtrapDouble, double](bc, Q, B, Bx, By, ngh, comp, ornt)
    elif dtype == "float32":
        bc = LinearExtrapFloat()
        _linear_extrap_bc_factory[LinearExtrapFloat, float](bc, Q, B, Bx, By, ngh, comp, ornt)

    return bc
