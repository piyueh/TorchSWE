# vim:fenc=utf-8
# vim:ft=pyrex
import numpy
cimport numpy
cimport cython
from cpython.mem cimport PyMem_Malloc, PyMem_Free


cdef fused ConstValBC:
    ConstValFloat
    ConstValDouble

ctypedef void (*const_val_kernel_double_t)(ConstValDouble, const double) nogil except *;
ctypedef void (*const_val_kernel_float_t)(ConstValFloat, const float) nogil except *;


@cython.auto_pickle(False)  # meaningless to pickle a BC instance as everything is a memoryview
cdef class ConstValDouble:
    """Base class of constant-value boundary coditions wiht 64-bit floating points.
    """

    cdef Py_ssize_t n
    cdef const double[:] qp1
    cdef const double[:] hp1
    cdef const double[:] bb
    cdef double[:] qm1
    cdef double[:] qm2
    cdef double[:] hm1
    cdef double val
    cdef double* _hvals
    cdef double[::1] hvals
    cdef const_val_kernel_double_t kernel

    def __cinit__(
        self, double[:, :, ::1] Q, double[:, ::1] H,
        const double[:, ::1] Bx, const double[:, ::1] By, const double val,
        const Py_ssize_t ngh, const unsigned comp, const unsigned ornt
    ):
        _const_val_bc_cinit[double](
            Q, H, Bx, By, ngh, comp, ornt,
            &self.n, &self._hvals
        )
        if comp == 0:
            self.hvals = <double[:self.n]>self._hvals  # we prefer to access values using MemoryView

    def __init__(
        self, double[:, :, ::1] Q, double[:, ::1] H,
        const double[:, ::1] Bx, const double[:, ::1] By, const double val,
        const Py_ssize_t ngh, const unsigned comp, const unsigned ornt
    ):
        _const_val_bc_init[ConstValDouble, double](self, Q, H, Bx, By, val, ngh, comp, ornt)

    def __dealloc__(self):
        PyMem_Free(self._hvals)

    def __call__(self):
        self.kernel(self, 0.0)


@cython.auto_pickle(False)  # meaningless to pickle a BC instance as everything is a memoryview
cdef class ConstValFloat:
    """Base class of constant-value boundary coditions wiht 32-bit floating points.
    """

    cdef Py_ssize_t n
    cdef const float[:] qp1
    cdef const float[:] hp1
    cdef const float[:] bb
    cdef float[:] qm1
    cdef float[:] qm2
    cdef float[:] hm1
    cdef float val
    cdef float* _hvals
    cdef float[::1] hvals
    cdef const_val_kernel_float_t kernel

    def __cinit__(
        self, float[:, :, ::1] Q, float[:, ::1] H,
        const float[:, ::1] Bx, const float[:, ::1] By, const float val,
        const Py_ssize_t ngh, const unsigned comp, const unsigned ornt
    ):
        _const_val_bc_cinit[float](
            Q, H, Bx, By, ngh, comp, ornt,
            &self.n, &self._hvals
        )

        if comp == 0:
            self.hvals = <float[:self.n]>self._hvals  # we prefer to access values using MemoryView

    def __init__(
        self, float[:, :, ::1] Q, float[:, ::1] H,
        const float[:, ::1] Bx, const float[:, ::1] By, const float val,
        const Py_ssize_t ngh, const unsigned comp, const unsigned ornt
    ):
        _const_val_bc_init[ConstValFloat, float](self, Q, H, Bx, By, val, ngh, comp, ornt)

    def __dealloc__(self):
        PyMem_Free(self._hvals)

    def __call__(self):
        self.kernel(self, 0.0)


cdef void _const_val_bc_cinit(
    const cython.floating[:, :, ::1] Q,
    const cython.floating[:, ::1] H,
    const cython.floating[:, ::1] Bx,
    const cython.floating[:, ::1] By,
    const Py_ssize_t ngh,
    const unsigned comp,
    const unsigned ornt,
    Py_ssize_t* n,
    cython.floating** _hvals,
):

    # runtime check for the shapes
    assert ngh == 2, "Currently only support ngh = 2"
    assert 0 <= comp <= 2, "comp should be 0 <= comp <= 2"
    assert 0 <= ornt <= 3, "ornt should be 0 <= ornt <= 3"
    assert Q.shape[0] == 3, f"{Q.shape}"
    assert H.shape[0] == Q.shape[1] - 2, f"{H.shape[0]}, {Q.shape[1]-2}"
    assert H.shape[1] == Q.shape[2] - 2, f"{H.shape[1]}, {Q.shape[2]-2}"
    assert Bx.shape[0] == H.shape[0] - 2, f"{Bx.shape[0]}, {H.shape[0]-2}"
    assert Bx.shape[1] == H.shape[1] - 1, f"{Bx.shape[1]}, {H.shape[1]-1}"
    assert By.shape[0] == H.shape[0] - 1, f"{By.shape[0]}, {H.shape[0]-1}"
    assert By.shape[1] == H.shape[1] - 2, f"{By.shape[1]}, {H.shape[1]-2}"

    # note, cython doesn't use * to de-reference pointers, so we use `[0]` instead
    if ornt == 0 or ornt == 1:  # west or east
        n[0] = Q.shape[1] - 2 * ngh  # ny
    elif ornt == 2 or ornt == 3:  # west or east
        n[0] = Q.shape[2] - 2 * ngh  # nx
    else:
        raise ValueError(f"`ornt` should be >= 0 and <= 3: {ornt}")

    if comp == 0:
        _hvals[0] = <cython.floating*>PyMem_Malloc(n[0]*sizeof(cython.floating))
        if not _hvals[0]: raise MemoryError()
    else:
        _hvals[0] = NULL


cdef void _const_val_bc_init(
    ConstValBC bc,
    cython.floating[:, :, ::1] Q,
    cython.floating[:, ::1] H,
    const cython.floating[:, ::1] Bx,
    const cython.floating[:, ::1] By,
    const cython.floating val,
    const Py_ssize_t ngh,
    const Py_ssize_t comp,
    const unsigned ornt,
) nogil except *:
    cdef Py_ssize_t i

    # the first two combinations will be pruned by cython when translating to C/C++
    if cython.floating is float and ConstValBC is ConstValDouble:
        raise RuntimeError("Mismatched types")
    elif cython.floating is double and ConstValBC is ConstValFloat:
        raise RuntimeError("Mismatched types")
    else:
        if ornt == 0:  # west
            _const_val_bc_set_west(bc, Q, H, Bx, ngh, comp)
        elif ornt == 1:  # east
            _const_val_bc_set_east(bc, Q, H, Bx, ngh, comp)
        elif ornt == 2:  # south
            _const_val_bc_set_south(bc, Q, H, By, ngh, comp)
        elif ornt == 3:  # north
            _const_val_bc_set_north(bc, Q, H, By, ngh, comp)
        else:
            raise ValueError(f"orientation id {ornt} not accepted.")

        bc.val = val

        if comp == 0:
            bc.kernel = _const_val_w_h_kernel[ConstValBC, cython.floating]
            for i in range(bc.n):
                bc.hvals[i] = bc.val - bc.bb[i]
        elif comp <= 2:
            bc.kernel = _const_val_kernel[ConstValBC, cython.floating]
        else:
            raise ValueError(f"component id {comp} not accepted.")


cdef void _const_val_bc_set_west(
    ConstValBC bc,
    cython.floating[:, :, ::1] Q, cython.floating[:, ::1] H, const cython.floating[:, ::1] Bx,
    const Py_ssize_t ngh, const Py_ssize_t comp
) nogil except *:

    # the first two combinations will be pruned by cython when translating to C/C++
    if cython.floating is float and ConstValBC is ConstValDouble:
        raise RuntimeError("Mismatched types")
    elif cython.floating is double and ConstValBC is ConstValFloat:
        raise RuntimeError("Mismatched types")
    else:
        # these should be views into original data buffer
        bc.qp1 = Q[comp, ngh:Q.shape[1]-ngh, ngh]
        bc.qm1 = Q[comp, ngh:Q.shape[1]-ngh, ngh-1]
        bc.qm2 = Q[comp, ngh:Q.shape[1]-ngh, ngh-2]
        bc.hp1 = H[1:H.shape[0]-1, 1]
        bc.hm1 = H[1:H.shape[0]-1, 0]
        bc.bb = Bx[:, 0]


cdef void _const_val_bc_set_east(
    ConstValBC bc,
    cython.floating[:, :, ::1] Q, cython.floating[:, ::1] H, const cython.floating[:, ::1] Bx,
    const Py_ssize_t ngh, const Py_ssize_t comp
) nogil except *:

    # the first two combinations will be pruned by cython when translating to C/C++
    if cython.floating is float and ConstValBC is ConstValDouble:
        raise RuntimeError("Mismatched types")
    elif cython.floating is double and ConstValBC is ConstValFloat:
        raise RuntimeError("Mismatched types")
    else:
        # these should be views into original data buffer
        bc.qp1 = Q[comp, ngh:Q.shape[1]-ngh, Q.shape[2]-ngh-1]
        bc.qm1 = Q[comp, ngh:Q.shape[1]-ngh, Q.shape[2]-ngh]
        bc.qm2 = Q[comp, ngh:Q.shape[1]-ngh, Q.shape[2]-ngh+1]
        bc.hp1 = H[1:H.shape[0]-1, H.shape[1]-2]
        bc.hm1 = H[1:H.shape[0]-1, H.shape[1]-1]
        bc.bb = Bx[:, Bx.shape[1]-1]


cdef void _const_val_bc_set_south(
    ConstValBC bc,
    cython.floating[:, :, ::1] Q, cython.floating[:, ::1] H, const cython.floating[:, ::1] By,
    const Py_ssize_t ngh, const Py_ssize_t comp
) nogil except *:

    # the first two combinations will be pruned by cython when translating to C/C++
    if cython.floating is float and ConstValBC is ConstValDouble:
        raise RuntimeError("Mismatched types")
    elif cython.floating is double and ConstValBC is ConstValFloat:
        raise RuntimeError("Mismatched types")
    else:
        # these should be views into original data buffer
        bc.qp1 = Q[comp, ngh, ngh:Q.shape[2]-ngh]
        bc.qm1 = Q[comp, ngh-1, ngh:Q.shape[2]-ngh]
        bc.qm2 = Q[comp, ngh-2, ngh:Q.shape[2]-ngh]
        bc.hp1 = H[1, 1:H.shape[1]-1]
        bc.hm1 = H[0, 1:H.shape[1]-1]
        bc.bb = By[0, :]


cdef void _const_val_bc_set_north(
    ConstValBC bc,
    cython.floating[:, :, ::1] Q, cython.floating[:, ::1] H, const cython.floating[:, ::1] By,
    const Py_ssize_t ngh, const Py_ssize_t comp
) nogil except *:

    # the first two combinations will be pruned by cython when translating to C/C++
    if cython.floating is float and ConstValBC is ConstValDouble:
        raise RuntimeError("Mismatched types")
    elif cython.floating is double and ConstValBC is ConstValFloat:
        raise RuntimeError("Mismatched types")
    else:
        # these should be views into original data buffer
        bc.qp1 = Q[comp, Q.shape[1]-ngh-1, ngh:Q.shape[2]-ngh]
        bc.qm1 = Q[comp, Q.shape[1]-ngh, ngh:Q.shape[2]-ngh]
        bc.qm2 = Q[comp, Q.shape[1]-ngh+1, ngh:Q.shape[2]-ngh]
        bc.hp1 = H[H.shape[0]-2, 1:H.shape[1]-1]
        bc.hm1 = H[H.shape[0]-1, 1:H.shape[1]-1]
        bc.bb = By[By.shape[0]-1, :]


cdef void _const_val_w_h_kernel(
    ConstValBC bc,
    const cython.floating dummy  # dummy is used just for specializing templates/fused types
) nogil except *:
    cdef Py_ssize_t i
    cdef cython.floating delta

    for i in range(bc.n):
        delta = (bc.val - bc.qp1[i]) * 2.0
        bc.qm1[i] = bc.qp1[i] + delta
        bc.qm2[i] = bc.qm1[i] + delta
        bc.hm1[i] = bc.hvals[i] * 2.0 - bc.hp1[i]


cdef void _const_val_kernel(
    ConstValBC bc,
    const cython.floating dummy  # dummy is used just for specializing templates/fused types
) nogil except *:
    cdef Py_ssize_t i
    cdef cython.floating delta

    for i in range(bc.n):
        delta = (bc.val - bc.qp1[i]) * 2.0
        bc.qm1[i] = bc.qp1[i] + delta
        bc.qm2[i] = bc.qm1[i] + delta


def const_val_factory(ornt, comp, states, topo, const double val, *args, **kwargs):
    """Factory to create a linear extrapolation boundary condition callable object.
    """

    # aliases
    cdef object dtype = states.domain.dtype
    cdef numpy.ndarray Q = states.Q
    cdef numpy.ndarray H = states.H
    cdef numpy.ndarray Bx = topo.xfcenters
    cdef numpy.ndarray By = topo.yfcenters
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
        bc = ConstValDouble(Q, H, Bx, By, val, ngh, comp, ornt)
    elif dtype == numpy.single:
        bc = ConstValFloat(Q, H, Bx, By, val, ngh, comp, ornt)
    else:
        raise ValueError(f"Unrecognized data type: {dtype}")

    return bc
