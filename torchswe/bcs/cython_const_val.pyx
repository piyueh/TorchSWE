# vim:fenc=utf-8
# vim:ft=pyrex
cimport cython


ctypedef fused ConstValBC:
    ConstValFloat
    ConstValDouble


@cython.auto_pickle(False)  # meaningless to pickle a BC instance as everything is a memoryview
cdef class ConstValDouble:
    """Constant-value boundary coditions wiht 64-bit floating points.
    """

    # number of elements to be updated on this boundary
    cdef Py_ssize_t n

    # conservatives
    cdef double[:] qbcm1  # w/hu/hv at the 1st layer of ghost cells
    cdef double[:] qbcm2  # w/hu/hv at the 2nd layer of ghost cells

    # read-only values (boundary target value)
    cdef double val

    def __call__(self):
        _const_val_bc_kernel[ConstValDouble](self)


@cython.auto_pickle(False)  # meaningless to pickle a BC instance as everything is a memoryview
cdef class ConstValFloat:
    """Constant-value boundary coditions wiht 32-bit floating points.
    """

    # number of elements to be updated on this boundary
    cdef Py_ssize_t n

    # conservatives
    cdef float[:] qbcm1  # w/hu/hv at the 1st layer of ghost cells
    cdef float[:] qbcm2  # w/hu/hv at the 2nd layer of ghost cells

    # read-only values (boundary target value)
    cdef float val

    def __call__(self):
        _const_val_bc_kernel[ConstValFloat](self)


cdef void _const_val_bc_kernel(ConstValBC bc) nogil:
    cdef Py_ssize_t i
    for i in range(bc.n):
        bc.qbcm1[i] = bc.val;
        bc.qbcm2[i] = bc.val;


cdef void _const_val_bc_set_west(
    ConstValBC bc,
    cython.floating[:, :, ::1] Q,
    cython.floating[:, ::1] B,
    const cython.floating[:, ::1] Bx,
    const cython.floating val,
    const Py_ssize_t ngh, const Py_ssize_t comp,
) nogil except *:

    cdef Py_ssize_t i;

    # the first two combinations will be pruned by cython when translating to C/C++
    if cython.floating is float and ConstValBC is ConstValDouble:
        raise RuntimeError("Mismatched types")
    elif cython.floating is double and ConstValBC is ConstValFloat:
        raise RuntimeError("Mismatched types")
    else:
        bc.n = Q.shape[1] - 2 * ngh  # ny
        bc.val = val

        bc.qbcm1    = Q[comp, ngh:Q.shape[1]-ngh, ngh-1]
        bc.qbcm2    = Q[comp, ngh:Q.shape[1]-ngh, ngh-2]

        # modify the topography elevation in ghost cells
        for i in range(ngh, B.shape[0]-ngh):
            B[i, ngh-1] = Bx[i-ngh, 0]
            B[i, ngh-2] = Bx[i-ngh, 0]


cdef void _const_val_bc_set_east(
    ConstValBC bc,
    cython.floating[:, :, ::1] Q,
    cython.floating[:, ::1] B,
    const cython.floating[:, ::1] Bx,
    const cython.floating val,
    const Py_ssize_t ngh, const Py_ssize_t comp,
) nogil except *:

    cdef Py_ssize_t i;

    # the first two combinations will be pruned by cython when translating to C/C++
    if cython.floating is float and ConstValBC is ConstValDouble:
        raise RuntimeError("Mismatched types")
    elif cython.floating is double and ConstValBC is ConstValFloat:
        raise RuntimeError("Mismatched types")
    else:
        bc.n = Q.shape[1] - 2 * ngh  # ny
        bc.val = val

        bc.qbcm1    = Q[comp, ngh:Q.shape[1]-ngh, Q.shape[2]-ngh]
        bc.qbcm2    = Q[comp, ngh:Q.shape[1]-ngh, Q.shape[2]-ngh+1]

        # modify the topography elevation in ghost cells
        for i in range(ngh, B.shape[0]-ngh):
            B[i, B.shape[1]-ngh]   = Bx[i-ngh, Bx.shape[1]-1]
            B[i, B.shape[1]-ngh+1] = Bx[i-ngh, Bx.shape[1]-1]


cdef void _const_val_bc_set_south(
    ConstValBC bc,
    cython.floating[:, :, ::1] Q,
    cython.floating[:, ::1] B,
    const cython.floating[:, ::1] By,
    const cython.floating val,
    const Py_ssize_t ngh, const Py_ssize_t comp,
) nogil except *:

    cdef Py_ssize_t i;

    # the first two combinations will be pruned by cython when translating to C/C++
    if cython.floating is float and ConstValBC is ConstValDouble:
        raise RuntimeError("Mismatched types")
    elif cython.floating is double and ConstValBC is ConstValFloat:
        raise RuntimeError("Mismatched types")
    else:
        bc.n = Q.shape[2] - 2 * ngh  # nx
        bc.val = val

        bc.qbcm1    = Q[comp, ngh-1,    ngh:Q.shape[2]-ngh]
        bc.qbcm2    = Q[comp, ngh-2,    ngh:Q.shape[2]-ngh]

        # modify the topography elevation in ghost cells
        for i in range(ngh, B.shape[1]-ngh):
            B[ngh-1, i] = By[0, i-ngh]
            B[ngh-2, i] = By[0, i-ngh]


cdef void _const_val_bc_set_north(
    ConstValBC bc,
    cython.floating[:, :, ::1] Q,
    cython.floating[:, ::1] B,
    const cython.floating[:, ::1] By,
    const cython.floating val,
    const Py_ssize_t ngh, const Py_ssize_t comp,
) nogil except *:

    cdef Py_ssize_t i;

    # the first two combinations will be pruned by cython when translating to C/C++
    if cython.floating is float and ConstValBC is ConstValDouble:
        raise RuntimeError("Mismatched types")
    elif cython.floating is double and ConstValBC is ConstValFloat:
        raise RuntimeError("Mismatched types")
    else:
        bc.n = Q.shape[2] - 2 * ngh  # nx
        bc.val = val

        bc.qbcm1    = Q[comp, Q.shape[1]-ngh,       ngh:Q.shape[2]-ngh]
        bc.qbcm2    = Q[comp, Q.shape[1]-ngh+1,     ngh:Q.shape[2]-ngh]

        # modify the topography elevation in ghost cells
        for i in range(ngh, B.shape[1]-ngh):
            B[B.shape[0]-ngh,   i] = By[By.shape[0]-1, i-ngh]
            B[B.shape[0]-ngh+1, i] = By[By.shape[0]-1, i-ngh]


cdef inline void _const_val_bc_factory(
    ConstValBC bc,
    cython.floating[:, :, ::1] Q,
    cython.floating[:, ::1] B,
    const cython.floating[:, ::1] Bx,
    const cython.floating[:, ::1] By,
    const cython.floating val,
    const Py_ssize_t ngh, const unsigned comp, const unsigned ornt,
) nogil except *:

    # the first two combinations will be pruned by cython when translating to C/C++
    if cython.floating is float and ConstValBC is ConstValDouble:
        raise RuntimeError("Mismatched types")
    elif cython.floating is double and ConstValBC is ConstValFloat:
        raise RuntimeError("Mismatched types")
    else:

        assert Q.shape[1] == B.shape[0]
        assert Q.shape[2] == B.shape[1]
        assert Q.shape[1] == Bx.shape[0] + 2 * ngh
        assert Q.shape[2] == Bx.shape[1] + 2 * ngh - 1
        assert Q.shape[1] == By.shape[0] + 2 * ngh - 1
        assert Q.shape[2] == By.shape[1] + 2 * ngh

        if ornt == 0:  # west
            _const_val_bc_set_west[ConstValBC, cython.floating](bc, Q, B, Bx, val, ngh, comp)
        elif ornt == 1:  # east
            _const_val_bc_set_east[ConstValBC, cython.floating](bc, Q, B, Bx, val, ngh, comp)
        elif ornt == 2:  # south
            _const_val_bc_set_south[ConstValBC, cython.floating](bc, Q, B, By, val, ngh, comp)
        elif ornt == 3:  # north
            _const_val_bc_set_north[ConstValBC, cython.floating](bc, Q, B, By, val, ngh, comp)
        else:
            raise ValueError(f"orientation id {ornt} not accepted.")


def const_val_bc_factory(ornt, comp, states, topo, val, *args, **kwargs):
    """Factory to create a constant-valued boundary condition callable object.
    """

    # aliases
    cdef object Q = states.Q
    cdef object B = topo.centers
    cdef object Bx = topo.xfcenters
    cdef object By = topo.yfcenters
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
        bc = ConstValDouble()
        _const_val_bc_factory[ConstValDouble, double](bc, Q, B, Bx, By, val, ngh, comp, ornt)
    elif dtype == "float32":
        bc = ConstValFloat()
        _const_val_bc_factory[ConstValFloat, float](bc, Q, B, Bx, By, val, ngh, comp, ornt)
    else:
        raise ValueError(f"Unrecognized data type: {dtype}")

    return bc
