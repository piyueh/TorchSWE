# vim:fenc=utf-8
# vim:ft=pyrex
import numpy
cimport numpy
cimport cython
from cpython.mem cimport PyMem_Malloc, PyMem_Free


cdef fused LinearExtrapBC:
    LinearExtrapFloat
    LinearExtrapDouble

ctypedef void (*linear_extrap_kernel_double_t)(LinearExtrapDouble, const double) nogil except *;
ctypedef void (*linear_extrap_kernel_float_t)(LinearExtrapFloat, const float) nogil except *;


@cython.auto_pickle(False)  # meaningless to pickle a BC instance as everything is a memoryview
cdef class LinearExtrapDouble:
    """Base class of linear extraption boundary coditions wiht 64-bit floating points.
    """

    cdef Py_ssize_t n
    cdef readonly const double[:] qp1
    cdef readonly const double[:] qp2
    cdef readonly const double[:] hp1
    cdef readonly const double[:] hp2
    cdef readonly const double[:] bb
    cdef readonly const double[:] bcp1
    cdef readonly double[:] qm1
    cdef readonly double[:] qm2
    cdef readonly double[:] hm1
    cdef double* _bcm1
    cdef double* _bcm2
    cdef readonly double[::1] bcm1
    cdef readonly double[::1] bcm2
    cdef linear_extrap_kernel_double_t kernel

    def __cinit__(
        self, double[:, :, ::1] Q, double[:, ::1] H,
        const double[:, ::1] Bc, const double[:, ::1] Bx, const double[:, ::1] By,
        const Py_ssize_t ngh, const unsigned comp, const unsigned ornt
    ):
        _linear_extrap_bc_cinit[double](
            Q, H, Bc, Bx, By, ngh, comp, ornt,
            &self.n, &self._bcm1, &self._bcm2
        )
        self.bcm1 = <double[:self.n]>self._bcm1  # we prefer to access values using MemoryView
        self.bcm2 = <double[:self.n]>self._bcm2  # we prefer to access values using MemoryView

    def __init__(
        self, double[:, :, ::1] Q, double[:, ::1] H,
        const double[:, ::1] Bc, const double[:, ::1] Bx, const double[:, ::1] By,
        const Py_ssize_t ngh, const unsigned comp, const unsigned ornt
    ):
        _linear_extrap_bc_init[LinearExtrapDouble, double](self, Q, H, Bc, Bx, By, ngh, comp, ornt)

    def __dealloc__(self):
        PyMem_Free(self._bcm1)
        PyMem_Free(self._bcm2)

    def __call__(self):
        self.kernel(self, 0.0)


@cython.auto_pickle(False)  # meaningless to pickle a BC instance as everything is a memoryview
cdef class LinearExtrapFloat:
    """Base class of linear extraption boundary coditions wiht 32-bit floating points.
    """

    cdef Py_ssize_t n
    cdef readonly const float[:] qp1
    cdef readonly const float[:] qp2
    cdef readonly const float[:] hp1
    cdef readonly const float[:] hp2
    cdef readonly const float[:] bb
    cdef readonly const float[:] bcp1
    cdef readonly float[:] qm1
    cdef readonly float[:] qm2
    cdef readonly float[:] hm1
    cdef float* _bcm1
    cdef float* _bcm2
    cdef readonly float[::1] bcm1
    cdef readonly float[::1] bcm2
    cdef linear_extrap_kernel_float_t kernel

    def __cinit__(
        self, float[:, :, ::1] Q, float[:, ::1] H,
        const float[:, ::1] Bc, const float[:, ::1] Bx, const float[:, ::1] By,
        const Py_ssize_t ngh, const unsigned comp, const unsigned ornt
    ):
        _linear_extrap_bc_cinit[float](
            Q, H, Bc, Bx, By, ngh, comp, ornt,
            &self.n, &self._bcm1, &self._bcm2
        )
        self.bcm1 = <float[:self.n]>self._bcm1  # we prefer to access values using MemoryView
        self.bcm2 = <float[:self.n]>self._bcm2  # we prefer to access values using MemoryView

    def __init__(
        self, float[:, :, ::1] Q, float[:, ::1] H,
        const float[:, ::1] Bc, const float[:, ::1] Bx, const float[:, ::1] By,
        const Py_ssize_t ngh, const unsigned comp, const unsigned ornt
    ):
        _linear_extrap_bc_init[LinearExtrapFloat, float](self, Q, H, Bc, Bx, By, ngh, comp, ornt)

    def __dealloc__(self):
        PyMem_Free(self._bcm1)
        PyMem_Free(self._bcm2)

    def __call__(self):
        self.kernel(self, 0.0)


cdef void _linear_extrap_bc_cinit(
    const cython.floating[:, :, ::1] Q,
    const cython.floating[:, ::1] H,
    const cython.floating[:, ::1] Bc,
    const cython.floating[:, ::1] Bx,
    const cython.floating[:, ::1] By,
    const Py_ssize_t ngh,
    const unsigned comp,
    const unsigned ornt,
    Py_ssize_t* n,
    cython.floating** _bcm1,
    cython.floating** _bcm2,
):

    # runtime check for the shapes
    assert ngh == 2, "Currently only support ngh = 2"
    assert 0 <= comp <= 2, "comp should be 0 <= comp <= 2"
    assert 0 <= ornt <= 3, "ornt should be 0 <= ornt <= 3"
    assert Q.shape[0] == 3, f"{Q.shape}"
    assert H.shape[0] == Q.shape[1] - 2, f"{H.shape[0]}, {Q.shape[1]-2}"
    assert H.shape[1] == Q.shape[2] - 2, f"{H.shape[1]}, {Q.shape[2]-2}"
    assert Bc.shape[0] == H.shape[0] - 2, f"{Bc.shape[0]}, {H.shape[0]-2}"
    assert Bc.shape[1] == H.shape[1] - 2, f"{Bc.shape[1]}, {H.shape[1]-2}"
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

    _bcm1[0] = <cython.floating*>PyMem_Malloc(n[0]*sizeof(cython.floating))
    _bcm2[0] = <cython.floating*>PyMem_Malloc(n[0]*sizeof(cython.floating))
    if not _bcm1[0]: raise MemoryError()
    if not _bcm2[0]: raise MemoryError()


cdef void _linear_extrap_bc_init(
    LinearExtrapBC bc,
    cython.floating[:, :, ::1] Q,
    cython.floating[:, ::1] H,
    const cython.floating[:, ::1] Bc,
    const cython.floating[:, ::1] Bx,
    const cython.floating[:, ::1] By,
    const Py_ssize_t ngh,
    const Py_ssize_t comp,
    const unsigned ornt,
) nogil except *:

    # the first two combinations will be pruned by cython when translating to C/C++
    if cython.floating is float and LinearExtrapBC is LinearExtrapDouble:
        raise RuntimeError("Mismatched types")
    elif cython.floating is double and LinearExtrapBC is LinearExtrapFloat:
        raise RuntimeError("Mismatched types")
    else:
        if ornt == 0:  # west
            _linear_extrap_bc_set_west(bc, Q, H, Bc, Bx, ngh, comp)
        elif ornt == 1:  # east
            _linear_extrap_bc_set_east(bc, Q, H, Bc, Bx, ngh, comp)
        elif ornt == 2:  # south
            _linear_extrap_bc_set_south(bc, Q, H, Bc, By, ngh, comp)
        elif ornt == 3:  # north
            _linear_extrap_bc_set_north(bc, Q, H, Bc, By, ngh, comp)
        else:
            raise ValueError(f"orientation id {ornt} not accepted.")

        _init_ghost_topo(bc, 0.0)

        if comp == 0:
            bc.kernel = _linear_extrap_w_h_kernel[LinearExtrapBC, cython.floating]
        elif comp <= 2:
            bc.kernel = _linear_extrap_kernel[LinearExtrapBC, cython.floating]
        else:
            raise ValueError(f"component id {comp} not accepted.")


cdef void _linear_extrap_w_h_kernel(
    LinearExtrapBC bc,
    const cython.floating dummy  # dummy is used just for specializing templates/fused types
) nogil except *:
    cdef Py_ssize_t i
    cdef cython.floating delta

    for i in range(bc.n):
        bc.hm1[i] = bc.hp1[i] * 2.0 - bc.hp2[i]
        if bc.hm1[i] <= 0.0:
            bc.hm1[i] = 0.0
            bc.qm1[i] = bc.bcm1[i]
            bc.qm2[i] = bc.bcm2[i]
        else:
            delta = bc.qp1[i] - bc.qp2[i]
            bc.qm1[i] = bc.qp1[i] + delta
            bc.qm2[i] = bc.qm1[i] + delta


cdef void _linear_extrap_kernel(
    LinearExtrapBC bc,
    const cython.floating dummy  # dummy is used just for specializing templates/fused types
) nogil except *:
    cdef Py_ssize_t i
    cdef cython.floating delta

    for i in range(bc.n):
        if bc.hm1[i] <= 0.0:
            bc.qm1[i] = 0.0
            bc.qm2[i] = 0.0
        else:
            delta = bc.qp1[i] - bc.qp2[i]
            bc.qm1[i] = bc.qp1[i] + delta
            bc.qm2[i] = bc.qm1[i] + delta


cdef void _init_ghost_topo(
    LinearExtrapBC bc,
    const cython.floating dummy  # dummy is used just for specializing templates/fused types
) nogil except *:
    cdef Py_ssize_t i
    cdef cython.floating db

    for i in range(bc.n):
        db = (bc.bb[i] - bc.bcp1[i]) * 2.0
        bc.bcm1[i] = bc.bcp1[i] + db
        bc.bcm2[i] = bc.bcm1[i] + db


cdef void _linear_extrap_bc_set_west(
    LinearExtrapBC bc,
    cython.floating[:, :, ::1] Q,
    cython.floating[:, ::1] H,
    const cython.floating[:, ::1] Bc,
    const cython.floating[:, ::1] Bx,
    const Py_ssize_t ngh,
    const Py_ssize_t comp
) nogil except *:

    # the first two combinations will be pruned by cython when translating to C/C++
    if cython.floating is float and LinearExtrapBC is LinearExtrapDouble:
        raise RuntimeError("Mismatched types")
    elif cython.floating is double and LinearExtrapBC is LinearExtrapFloat:
        raise RuntimeError("Mismatched types")
    else:
        bc.qp1 = Q[comp, ngh:Q.shape[1]-ngh, ngh]
        bc.qp2 = Q[comp, ngh:Q.shape[1]-ngh, ngh+1]
        bc.qm1 = Q[comp, ngh:Q.shape[1]-ngh, ngh-1]
        bc.qm2 = Q[comp, ngh:Q.shape[1]-ngh, ngh-2]
        bc.hp1 = H[1:H.shape[0]-1, 1]
        bc.hp2 = H[1:H.shape[0]-1, 2]
        bc.hm1 = H[1:H.shape[0]-1, 0]
        bc.bb = Bx[:, 0]
        bc.bcp1 = Bc[:, 0]


cdef void _linear_extrap_bc_set_east(
    LinearExtrapBC bc,
    cython.floating[:, :, ::1] Q,
    cython.floating[:, ::1] H,
    const cython.floating[:, ::1] Bc,
    const cython.floating[:, ::1] Bx,
    const Py_ssize_t ngh,
    const Py_ssize_t comp
) nogil except *:

    # the first two combinations will be pruned by cython when translating to C/C++
    if cython.floating is float and LinearExtrapBC is LinearExtrapDouble:
        raise RuntimeError("Mismatched types")
    elif cython.floating is double and LinearExtrapBC is LinearExtrapFloat:
        raise RuntimeError("Mismatched types")
    else:
        bc.qp1 = Q[comp, ngh:Q.shape[1]-ngh, Q.shape[2]-ngh-1]
        bc.qp2 = Q[comp, ngh:Q.shape[1]-ngh, Q.shape[2]-ngh-2]
        bc.qm1 = Q[comp, ngh:Q.shape[1]-ngh, Q.shape[2]-ngh]
        bc.qm2 = Q[comp, ngh:Q.shape[1]-ngh, Q.shape[2]-ngh+1]
        bc.hp1 = H[1:H.shape[0]-1, H.shape[1]-2]
        bc.hp2 = H[1:H.shape[0]-1, H.shape[1]-3]
        bc.hm1 = H[1:H.shape[0]-1, H.shape[1]-1]
        bc.bb = Bx[:, Bx.shape[1]-1]
        bc.bcp1 = Bc[:, Bc.shape[1]-1]


cdef void _linear_extrap_bc_set_south(
    LinearExtrapBC bc,
    cython.floating[:, :, ::1] Q,
    cython.floating[:, ::1] H,
    const cython.floating[:, ::1] Bc,
    const cython.floating[:, ::1] By,
    const Py_ssize_t ngh,
    const Py_ssize_t comp
) nogil except *:

    # the first two combinations will be pruned by cython when translating to C/C++
    if cython.floating is float and LinearExtrapBC is LinearExtrapDouble:
        raise RuntimeError("Mismatched types")
    elif cython.floating is double and LinearExtrapBC is LinearExtrapFloat:
        raise RuntimeError("Mismatched types")
    else:
        bc.qp1 = Q[comp, ngh, ngh:Q.shape[2]-ngh]
        bc.qp2 = Q[comp, ngh+1, ngh:Q.shape[2]-ngh]
        bc.qm1 = Q[comp, ngh-1, ngh:Q.shape[2]-ngh]
        bc.qm2 = Q[comp, ngh-2, ngh:Q.shape[2]-ngh]
        bc.hp1 = H[1, 1:H.shape[1]-1]
        bc.hp2 = H[2, 1:H.shape[1]-1]
        bc.hm1 = H[0, 1:H.shape[1]-1]
        bc.bb = By[0, :]
        bc.bcp1 = Bc[0, :]


cdef void _linear_extrap_bc_set_north(
    LinearExtrapBC bc,
    cython.floating[:, :, ::1] Q,
    cython.floating[:, ::1] H,
    const cython.floating[:, ::1] Bc,
    const cython.floating[:, ::1] By,
    const Py_ssize_t ngh,
    const Py_ssize_t comp
) nogil except *:

    # the first two combinations will be pruned by cython when translating to C/C++
    if cython.floating is float and LinearExtrapBC is LinearExtrapDouble:
        raise RuntimeError("Mismatched types")
    elif cython.floating is double and LinearExtrapBC is LinearExtrapFloat:
        raise RuntimeError("Mismatched types")
    else:
        bc.qp1 = Q[comp, Q.shape[1]-ngh-1, ngh:Q.shape[2]-ngh]
        bc.qp2 = Q[comp, Q.shape[1]-ngh-2, ngh:Q.shape[2]-ngh]
        bc.qm1 = Q[comp, Q.shape[1]-ngh, ngh:Q.shape[2]-ngh]
        bc.qm2 = Q[comp, Q.shape[1]-ngh+1, ngh:Q.shape[2]-ngh]
        bc.hp1 = H[H.shape[0]-2, 1:H.shape[1]-1]
        bc.hp2 = H[H.shape[0]-3, 1:H.shape[1]-1]
        bc.hm1 = H[H.shape[0]-1, 1:H.shape[1]-1]
        bc.bb = By[By.shape[0]-1, :]
        bc.bcp1 = Bc[Bc.shape[0]-1, :]


def linear_extrap_factory(ornt, comp, states, topo, *args, **kwargs):
    """Factory to create a linear extrapolation boundary condition callable object.
    """

    # aliases
    cdef object dtype = states.domain.dtype
    cdef numpy.ndarray Q = states.Q
    cdef numpy.ndarray H = states.H
    cdef numpy.ndarray Bc = topo.centers
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
        bc = LinearExtrapDouble(Q, H, Bc, Bx, By, ngh, comp, ornt)
    elif dtype == numpy.single:
        bc = LinearExtrapFloat(Q, H, Bc, Bx, By, ngh, comp, ornt)
    else:
        raise ValueError(f"Unrecognized data type: {dtype}")

    return bc
