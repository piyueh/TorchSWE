# vim:fenc=utf-8
# vim:ft=pyrex
cimport cython


ctypedef fused InflowBC:
    InflowFloat
    InflowDouble


ctypedef void (*_kernel_double_t)(InflowDouble) nogil;
ctypedef void (*_kernel_float_t)(InflowFloat) nogil;


@cython.auto_pickle(False)  # meaningless to pickle a BC instance as everything is a memoryview
cdef class InflowDouble:
    """Base class of inflow boundary coditions wiht 64-bit floating points.
    """

    # number of elements to be updated on this boundary
    cdef Py_ssize_t n

    # read-only auxiliary data
    cdef const double[:] w  # water elevation
    cdef const double[:] b  # topography elevation

    # conservatives
    cdef double[:] qbcm1  # w/hu/hv at the 1st layer of ghost cells
    cdef double[:] qbcm2  # w/hu/hv at the 2nd layer of ghost cells

    # read-only values (boundary target value)
    cdef double val

    # kernel
    cdef _kernel_double_t kernel;

    def __call__(self):
        self.kernel(self)


@cython.auto_pickle(False)  # meaningless to pickle a BC instance as everything is a memoryview
cdef class InflowFloat:
    """Base class of inflow boundary coditions wiht 32-bit floating points.
    """

    # number of elements to be updated on this boundary
    cdef Py_ssize_t n

    # read-only auxiliary data
    cdef const float[:] w  # water elevation
    cdef const float[:] b  # topography elevation

    # conservatives
    cdef float[:] qbcm1  # w/hu/hv at the 1st layer of ghost cells
    cdef float[:] qbcm2  # w/hu/hv at the 2nd layer of ghost cells

    # read-only values (boundary target value)
    cdef float val

    # kernel
    cdef _kernel_float_t kernel;

    def __call__(self):
        self.kernel(self)


cdef void _inflow_bc_w_kernel(InflowBC bc) nogil:
    cdef Py_ssize_t i
    for i in range(bc.n):
        bc.qbcm1[i] = bc.b[i] + bc.val;  # b + h
        bc.qbcm2[i] = bc.qbcm1[i]


cdef void _inflow_bc_kernel(InflowBC bc) nogil:
    cdef Py_ssize_t i
    for i in range(bc.n):
        bc.qbcm1[i] = (bc.w[i] - bc.b[i]) * bc.val;  # h * u or h * v
        bc.qbcm2[i] = bc.qbcm1[i];


cdef void _inflow_bc_set_west(
    InflowBC bc,
    cython.floating[:, :, ::1] Q,
    cython.floating[:, ::1] B,
    const cython.floating[:, ::1] Bx,
    const cython.floating val,
    const Py_ssize_t ngh, const Py_ssize_t comp
) nogil except *:

    cdef Py_ssize_t i

    # the first two combinations will be pruned by cython when translating to C/C++
    if cython.floating is float and InflowBC is InflowDouble:
        raise RuntimeError("Mismatched types")
    elif cython.floating is double and InflowBC is InflowFloat:
        raise RuntimeError("Mismatched types")
    else:
        bc.n        = Q.shape[1] - 2 * ngh  # ny
        bc.val      = val
        bc.kernel   = _inflow_bc_w_kernel[InflowBC] if comp == 0 else _inflow_bc_kernel[InflowBC]

        bc.w        = Q[0,    ngh:Q.shape[1]-ngh, ngh-1]
        bc.b        = B[      ngh:B.shape[0]-ngh, ngh-1]
        bc.qbcm1    = Q[comp, ngh:Q.shape[1]-ngh, ngh-1]
        bc.qbcm2    = Q[comp, ngh:Q.shape[1]-ngh, ngh-2]

        # modify the topography elevation in ghost cells
        for i in range(ngh, B.shape[0]-ngh):
            B[i, ngh-1] = Bx[i-ngh, 0]
            B[i, ngh-2] = Bx[i-ngh, 0]


cdef void _inflow_bc_set_east(
    InflowBC bc,
    cython.floating[:, :, ::1] Q,
    cython.floating[:, ::1] B,
    const cython.floating[:, ::1] Bx,
    const cython.floating val,
    const Py_ssize_t ngh, const Py_ssize_t comp
) nogil except *:

    cdef Py_ssize_t i

    # the first two combinations will be pruned by cython when translating to C/C++
    if cython.floating is float and InflowBC is InflowDouble:
        raise RuntimeError("Mismatched types")
    elif cython.floating is double and InflowBC is InflowFloat:
        raise RuntimeError("Mismatched types")
    else:
        bc.n        = Q.shape[1] - 2 * ngh  # ny
        bc.val      = val
        bc.kernel   = _inflow_bc_w_kernel[InflowBC] if comp == 0 else _inflow_bc_kernel[InflowBC]

        bc.w        = Q[0,    ngh:Q.shape[1]-ngh, Q.shape[2]-ngh]
        bc.b        = B[      ngh:B.shape[0]-ngh, B.shape[1]-ngh]
        bc.qbcm1    = Q[comp, ngh:Q.shape[1]-ngh, Q.shape[2]-ngh]
        bc.qbcm2    = Q[comp, ngh:Q.shape[1]-ngh, Q.shape[2]-ngh+1]

        # modify the topography elevation in ghost cells
        for i in range(ngh, B.shape[0]-ngh):
            B[i, B.shape[1]-ngh]   = Bx[i-ngh, Bx.shape[1]-1]
            B[i, B.shape[1]-ngh+1] = Bx[i-ngh, Bx.shape[1]-1]


cdef void _inflow_bc_set_south(
    InflowBC bc,
    cython.floating[:, :, ::1] Q,
    cython.floating[:, ::1] B,
    const cython.floating[:, ::1] By,
    const cython.floating val,
    const Py_ssize_t ngh, const Py_ssize_t comp
) nogil except *:

    cdef Py_ssize_t i

    # the first two combinations will be pruned by cython when translating to C/C++
    if cython.floating is float and InflowBC is InflowDouble:
        raise RuntimeError("Mismatched types")
    elif cython.floating is double and InflowBC is InflowFloat:
        raise RuntimeError("Mismatched types")
    else:
        bc.n        = Q.shape[2] - 2 * ngh  # nx
        bc.val      = val
        bc.kernel   = _inflow_bc_w_kernel[InflowBC] if comp == 0 else _inflow_bc_kernel[InflowBC]

        bc.w        = Q[0,    ngh-1,    ngh:Q.shape[2]-ngh]
        bc.b        = B[      ngh-1,    ngh:B.shape[1]-ngh]
        bc.qbcm1    = Q[comp, ngh-1,    ngh:Q.shape[2]-ngh]
        bc.qbcm2    = Q[comp, ngh-2,    ngh:Q.shape[2]-ngh]

        # modify the topography elevation in ghost cells
        for i in range(ngh, B.shape[1]-ngh):
            B[ngh-1, i] = By[0, i-ngh]
            B[ngh-2, i] = By[0, i-ngh]


cdef void _inflow_bc_set_north(
    InflowBC bc,
    cython.floating[:, :, ::1] Q,
    cython.floating[:, ::1] B,
    const cython.floating[:, ::1] By,
    const cython.floating val,
    const Py_ssize_t ngh, const Py_ssize_t comp
) nogil except *:

    cdef Py_ssize_t i

    # the first two combinations will be pruned by cython when translating to C/C++
    if cython.floating is float and InflowBC is InflowDouble:
        raise RuntimeError("Mismatched types")
    elif cython.floating is double and InflowBC is InflowFloat:
        raise RuntimeError("Mismatched types")
    else:
        bc.n        = Q.shape[2] - 2 * ngh  # nx
        bc.val      = val
        bc.kernel   = _inflow_bc_w_kernel[InflowBC] if comp == 0 else _inflow_bc_kernel[InflowBC]

        bc.w        = Q[0,    Q.shape[1]-ngh,       ngh:Q.shape[2]-ngh]
        bc.b        = B[      B.shape[0]-ngh,       ngh:B.shape[1]-ngh]
        bc.qbcm1    = Q[comp, Q.shape[1]-ngh,       ngh:Q.shape[2]-ngh]
        bc.qbcm2    = Q[comp, Q.shape[1]-ngh+1,     ngh:Q.shape[2]-ngh]

        # modify the topography elevation in ghost cells
        for i in range(ngh, B.shape[1]-ngh):
            B[B.shape[0]-ngh,   i] = By[By.shape[0]-1, i-ngh]
            B[B.shape[0]-ngh+1, i] = By[By.shape[0]-1, i-ngh]


cdef inline void _inflow_bc_factory(
    InflowBC bc,
    cython.floating[:, :, ::1] Q,
    cython.floating[:, ::1] B,
    const cython.floating[:, ::1] Bx,
    const cython.floating[:, ::1] By,
    const cython.floating val,
    const Py_ssize_t ngh, const unsigned comp, const unsigned ornt,
) nogil except *:

    # the first two combinations will be pruned by cython when translating to C/C++
    if cython.floating is float and InflowBC is InflowDouble:
        raise RuntimeError("Mismatched types")
    elif cython.floating is double and InflowBC is InflowFloat:
        raise RuntimeError("Mismatched types")
    else:

        assert Q.shape[1] == B.shape[0]
        assert Q.shape[2] == B.shape[1]
        assert Q.shape[1] == Bx.shape[0] + 2 * ngh
        assert Q.shape[2] == Bx.shape[1] + 2 * ngh - 1
        assert Q.shape[1] == By.shape[0] + 2 * ngh - 1
        assert Q.shape[2] == By.shape[1] + 2 * ngh

        if ornt == 0:  # west
            _inflow_bc_set_west[InflowBC, cython.floating](bc, Q, B, Bx, val, ngh, comp)
        elif ornt == 1:  # east
            _inflow_bc_set_east[InflowBC, cython.floating](bc, Q, B, Bx, val, ngh, comp)
        elif ornt == 2:  # south
            _inflow_bc_set_south[InflowBC, cython.floating](bc, Q, B, By, val, ngh, comp)
        elif ornt == 3:  # north
            _inflow_bc_set_north[InflowBC, cython.floating](bc, Q, B, By, val, ngh, comp)
        else:
            raise ValueError(f"orientation id {ornt} not accepted.")


def inflow_bc_factory(ornt, comp, states, topo, val, *args, **kwargs):
    """Factory to create an inflow boundary condition callable object.
    """

    # aliases
    cdef object Q = states.q
    cdef object B = topo.c
    cdef object Bx = topo.xf
    cdef object By = topo.yf
    cdef Py_ssize_t ngh = states.domain.nhalo
    cdef object dtype = Q.dtype

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
        bc = InflowDouble()
        _inflow_bc_factory[InflowDouble, double](bc, Q, B, Bx, By, val, ngh, comp, ornt)
    elif dtype == "float32":
        bc = InflowFloat()
        _inflow_bc_factory[InflowFloat, float](bc, Q, B, Bx, By, val, ngh, comp, ornt)
    else:
        raise ValueError(f"Unrecognized data type: {dtype}")

    return bc
