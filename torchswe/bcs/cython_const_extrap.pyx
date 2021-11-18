# vim:fenc=utf-8
# vim:ft=pyrex
import numpy
cimport numpy
cimport cython


cdef fused ConstExtrapBC:
    ConstExtrapFloat
    ConstExtrapDouble

ctypedef void (*const_extrap_kernel_double_t)(ConstExtrapDouble) nogil except *;
ctypedef void (*const_extrap_kernel_float_t)(ConstExtrapFloat) nogil except *;


@cython.auto_pickle(False)  # meaningless to pickle a BC instance as everything is a memoryview
cdef class ConstExtrapDouble:
    """Base class of constant extraption boundary coditions with 64-bit floating points.
    """

    cdef Py_ssize_t n
    cdef readonly const double[:] qp1
    cdef readonly const double[:] hp1
    cdef readonly double[:] qm1
    cdef readonly double[:] qm2
    cdef readonly double[:] hm1
    cdef const_extrap_kernel_double_t kernel

    def __init__(
        self, double[:, :, ::1] Q, double[:, ::1] H,
        const Py_ssize_t ngh, const unsigned comp, const unsigned ornt
    ):
        _const_extrap_bc_init[ConstExtrapDouble, double](self, Q, H, ngh, comp, ornt)

    def __call__(self):
        self.kernel(self)


@cython.auto_pickle(False)  # meaningless to pickle a BC instance as everything is a memoryview
cdef class ConstExtrapFloat:
    """Base class of constant extraption boundary coditions with 32-bit floating points.
    """

    cdef Py_ssize_t n
    cdef readonly const float[:] qp1
    cdef readonly const float[:] hp1
    cdef readonly float[:] qm1
    cdef readonly float[:] qm2
    cdef readonly float[:] hm1
    cdef const_extrap_kernel_float_t kernel

    def __init__(
        self, float[:, :, ::1] Q, float[:, ::1] H,
        const Py_ssize_t ngh, const unsigned comp, const unsigned ornt
    ):
        _const_extrap_bc_init[ConstExtrapFloat, float](self, Q, H, ngh, comp, ornt)

    def __call__(self):
        self.kernel(self)


cdef void _const_extrap_bc_init(
    ConstExtrapBC bc, cython.floating[:, :, ::1] Q, cython.floating[:, ::1] H,
    const Py_ssize_t ngh, const Py_ssize_t comp, const unsigned ornt,
) nogil except *:

    # the first two combinations will be pruned by cython when translating to C/C++
    if cython.floating is float and ConstExtrapBC is ConstExtrapDouble:
        raise RuntimeError("Mismatched types")
    elif cython.floating is double and ConstExtrapBC is ConstExtrapFloat:
        raise RuntimeError("Mismatched types")
    else:
        # runtime check for the shapes
        assert ngh == 2, "Currently only support ngh = 2"
        assert 0 <= comp <= 2, "comp should be 0 <= comp <= 2"
        assert 0 <= ornt <= 3, "ornt should be 0 <= ornt <= 3"
        assert Q.shape[0] == 3, f"{Q.shape}"
        assert H.shape[0] == Q.shape[1] - 2, f"{H.shape[0]}, {Q.shape[1]-2}"
        assert H.shape[1] == Q.shape[2] - 2, f"{H.shape[1]}, {Q.shape[2]-2}"

        # note, cython doesn't use * to de-reference pointers, so we use `[0]` instead
        if ornt == 0 or ornt == 1:  # west or east
            bc.n = Q.shape[1] - 2 * ngh  # ny
        elif ornt == 2 or ornt == 3:  # west or east
            bc.n = Q.shape[2] - 2 * ngh  # nx
        else:
            raise ValueError(f"`ornt` should be >= 0 and <= 3: {ornt}")

        if ornt == 0:  # west
            _const_extrap_bc_set_west[ConstExtrapBC, cython.floating](bc, Q, H, ngh, comp)
        elif ornt == 1:  # east
            _const_extrap_bc_set_east[ConstExtrapBC, cython.floating](bc, Q, H, ngh, comp)
        elif ornt == 2:  # south
            _const_extrap_bc_set_south[ConstExtrapBC, cython.floating](bc, Q, H, ngh, comp)
        elif ornt == 3:  # north
            _const_extrap_bc_set_north[ConstExtrapBC, cython.floating](bc, Q, H, ngh, comp)
        else:
            raise ValueError(f"orientation id {ornt} not accepted.")

        if comp == 0:
            bc.kernel = _const_extrap_w_h_kernel[ConstExtrapBC]
        elif comp <= 2:
            bc.kernel = _const_extrap_kernel[ConstExtrapBC]
        else:
            raise ValueError(f"component id {comp} not accepted.")


cdef void _const_extrap_w_h_kernel(ConstExtrapBC bc) nogil except *:
    cdef Py_ssize_t i
    for i in range(bc.n):
        bc.hm1[i] = bc.hp1[i]
        bc.qm1[i] = bc.qp1[i]
        bc.qm2[i] = bc.qp1[i]


cdef void _const_extrap_kernel(ConstExtrapBC bc) nogil except *:
    cdef Py_ssize_t i
    for i in range(bc.n):
        bc.qm1[i] = bc.qp1[i]
        bc.qm2[i] = bc.qp1[i]


cdef void _const_extrap_bc_set_west(
    ConstExtrapBC bc, cython.floating[:, :, ::1] Q, cython.floating[:, ::1] H,
    const Py_ssize_t ngh, const Py_ssize_t comp
) nogil except *:

    # the first two combinations will be pruned by cython when translating to C/C++
    if cython.floating is float and ConstExtrapBC is ConstExtrapDouble:
        raise RuntimeError("Mismatched types")
    elif cython.floating is double and ConstExtrapBC is ConstExtrapFloat:
        raise RuntimeError("Mismatched types")
    else:
        bc.qp1 = Q[comp, ngh:Q.shape[1]-ngh, ngh]
        bc.qm1 = Q[comp, ngh:Q.shape[1]-ngh, ngh-1]
        bc.qm2 = Q[comp, ngh:Q.shape[1]-ngh, ngh-2]
        bc.hp1 = H[1:H.shape[0]-1, 1]
        bc.hm1 = H[1:H.shape[0]-1, 0]


cdef void _const_extrap_bc_set_east(
    ConstExtrapBC bc, cython.floating[:, :, ::1] Q, cython.floating[:, ::1] H,
    const Py_ssize_t ngh, const Py_ssize_t comp
) nogil except *:

    # the first two combinations will be pruned by cython when translating to C/C++
    if cython.floating is float and ConstExtrapBC is ConstExtrapDouble:
        raise RuntimeError("Mismatched types")
    elif cython.floating is double and ConstExtrapBC is ConstExtrapFloat:
        raise RuntimeError("Mismatched types")
    else:
        bc.qp1 = Q[comp, ngh:Q.shape[1]-ngh, Q.shape[2]-ngh-1]
        bc.qm1 = Q[comp, ngh:Q.shape[1]-ngh, Q.shape[2]-ngh]
        bc.qm2 = Q[comp, ngh:Q.shape[1]-ngh, Q.shape[2]-ngh+1]
        bc.hp1 = H[1:H.shape[0]-1, H.shape[1]-2]
        bc.hm1 = H[1:H.shape[0]-1, H.shape[1]-1]


cdef void _const_extrap_bc_set_south(
    ConstExtrapBC bc, cython.floating[:, :, ::1] Q, cython.floating[:, ::1] H,
    const Py_ssize_t ngh, const Py_ssize_t comp
) nogil except *:

    # the first two combinations will be pruned by cython when translating to C/C++
    if cython.floating is float and ConstExtrapBC is ConstExtrapDouble:
        raise RuntimeError("Mismatched types")
    elif cython.floating is double and ConstExtrapBC is ConstExtrapFloat:
        raise RuntimeError("Mismatched types")
    else:
        bc.qp1 = Q[comp, ngh, ngh:Q.shape[2]-ngh]
        bc.qm1 = Q[comp, ngh-1, ngh:Q.shape[2]-ngh]
        bc.qm2 = Q[comp, ngh-2, ngh:Q.shape[2]-ngh]
        bc.hp1 = H[1, 1:H.shape[1]-1]
        bc.hm1 = H[0, 1:H.shape[1]-1]


cdef void _const_extrap_bc_set_north(
    ConstExtrapBC bc, cython.floating[:, :, ::1] Q, cython.floating[:, ::1] H,
    const Py_ssize_t ngh, const Py_ssize_t comp
) nogil except *:

    # the first two combinations will be pruned by cython when translating to C/C++
    if cython.floating is float and ConstExtrapBC is ConstExtrapDouble:
        raise RuntimeError("Mismatched types")
    elif cython.floating is double and ConstExtrapBC is ConstExtrapFloat:
        raise RuntimeError("Mismatched types")
    else:
        bc.qp1 = Q[comp, Q.shape[1]-ngh-1, ngh:Q.shape[2]-ngh]
        bc.qm1 = Q[comp, Q.shape[1]-ngh, ngh:Q.shape[2]-ngh]
        bc.qm2 = Q[comp, Q.shape[1]-ngh+1, ngh:Q.shape[2]-ngh]
        bc.hp1 = H[H.shape[0]-2, 1:H.shape[1]-1]
        bc.hm1 = H[H.shape[0]-1, 1:H.shape[1]-1]


def const_extrap_factory(ornt, comp, states, *args, **kwargs):
    """Factory to create a constant extrapolation boundary condition callable object.
    """

    # aliases
    cdef object dtype = states.domain.dtype
    cdef numpy.ndarray Q = states.Q
    cdef numpy.ndarray H = states.H
    cdef Py_ssize_t ngh = states.ngh

    if isinstance(ornt, str):
        if ornt in ["w", "west"]:
            ornt = 0
        elif ornt in ["e", "east"]:
            ornt = 1
        elif ornt in ["s", "south"]:
            ornt = 2
        elif ornt in ["n", "north"]:
            ornt = 3

    if dtype == numpy.double:
        bc = ConstExtrapDouble(Q, H, ngh, comp, ornt)
    elif dtype == numpy.single:
        bc = ConstExtrapFloat(Q, H, ngh, comp, ornt)
    else:
        raise ValueError(f"Unrecognized data type: {dtype}")

    return bc
