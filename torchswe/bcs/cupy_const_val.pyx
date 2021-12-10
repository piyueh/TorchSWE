# vim:fenc=utf-8
# vim:ft=pyrex
import cupy
cimport cython


@cython.auto_pickle(False)  # meaningless to pickle a BC instance as everything is a memoryview
cdef class ConstValBC:
    """Constant-value (conservative quantities) boundary condition for the CuPy backend.
    """

    # conservatives
    cdef object qbcm1  # w/hu/hv at the 1st layer of ghost cells
    cdef object qbcm2  # w/hu/hv at the 2nd layer of ghost cells

    # read-only values (boundary target value; broadcasted)
    cdef object val

    def __call__(self):
        _const_val_bc_kernel(self.val, self.qbcm1, self.qbcm2)


cdef _const_val_bc_kernel = cupy.ElementwiseKernel(
    "T val",
    "T qbcm1, T qbcm2",
    """
        qbcm1 = val;
        qbcm2 = val;
    """,
    "_const_val_bc_kernel"
)


cdef void _const_val_bc_set_west(
    ConstValBC bc,
    object Q, object B, object Bx,
    const double val,
    const Py_ssize_t ngh, const Py_ssize_t comp
) except *:

    bc.qbcm1    = Q[comp, ngh:Q.shape[1]-ngh, ngh-1]
    bc.qbcm2    = Q[comp, ngh:Q.shape[1]-ngh, ngh-2]
    bc.val = cupy.full_like(bc.qbcm1, val, dtype=Q.dtype)

    # modify the topography elevation in ghost cells
    for i in range(ngh, B.shape[0]-ngh):
        B[i, ngh-1] = Bx[i-ngh, 0]
        B[i, ngh-2] = Bx[i-ngh, 0]


cdef void _const_val_bc_set_east(
    ConstValBC bc,
    object Q, object B, object Bx,
    const double val,
    const Py_ssize_t ngh, const Py_ssize_t comp
) except *:

    bc.qbcm1    = Q[comp, ngh:Q.shape[1]-ngh, Q.shape[2]-ngh]
    bc.qbcm2    = Q[comp, ngh:Q.shape[1]-ngh, Q.shape[2]-ngh+1]
    bc.val = cupy.full_like(bc.qbcm1, val, dtype=Q.dtype)

    # modify the topography elevation in ghost cells
    for i in range(ngh, B.shape[0]-ngh):
        B[i, B.shape[1]-ngh]   = Bx[i-ngh, Bx.shape[1]-1]
        B[i, B.shape[1]-ngh+1] = Bx[i-ngh, Bx.shape[1]-1]


cdef void _const_val_bc_set_south(
    ConstValBC bc,
    object Q, object B, object By,
    const double val,
    const Py_ssize_t ngh, const Py_ssize_t comp
) except *:

    bc.qbcm1    = Q[comp, ngh-1,    ngh:Q.shape[2]-ngh]
    bc.qbcm2    = Q[comp, ngh-2,    ngh:Q.shape[2]-ngh]
    bc.val = cupy.full_like(bc.qbcm1, val, dtype=Q.dtype)

    # modify the topography elevation in ghost cells
    for i in range(ngh, B.shape[1]-ngh):
        B[ngh-1, i] = By[0, i-ngh]
        B[ngh-2, i] = By[0, i-ngh]


cdef void _const_val_bc_set_north(
    ConstValBC bc,
    object Q, object B, object By,
    const double val,
    const Py_ssize_t ngh, const Py_ssize_t comp
) except *:

    bc.qbcm1    = Q[comp, Q.shape[1]-ngh,       ngh:Q.shape[2]-ngh]
    bc.qbcm2    = Q[comp, Q.shape[1]-ngh+1,     ngh:Q.shape[2]-ngh]
    bc.val = cupy.full_like(bc.qbcm1, val, dtype=Q.dtype)

    # modify the topography elevation in ghost cells
    for i in range(ngh, B.shape[1]-ngh):
        B[B.shape[0]-ngh,   i] = By[By.shape[0]-1, i-ngh]
        B[B.shape[0]-ngh+1, i] = By[By.shape[0]-1, i-ngh]


cdef inline void _const_val_bc_factory(
    ConstValBC bc,
    object Q, object B, object Bx, object By,
    const double val,
    const Py_ssize_t ngh, const unsigned comp, const unsigned ornt,
) except *:

    assert Q.shape[1] == B.shape[0]
    assert Q.shape[2] == B.shape[1]
    assert Q.shape[1] == Bx.shape[0] + 2 * ngh
    assert Q.shape[2] == Bx.shape[1] + 2 * ngh - 1
    assert Q.shape[1] == By.shape[0] + 2 * ngh - 1
    assert Q.shape[2] == By.shape[1] + 2 * ngh

    if ornt == 0:  # west
        _const_val_bc_set_west(bc, Q, B, Bx, val, ngh, comp)
    elif ornt == 1:  # east
        _const_val_bc_set_east(bc, Q, B, Bx, val, ngh, comp)
    elif ornt == 2:  # south
        _const_val_bc_set_south(bc, Q, B, By, val, ngh, comp)
    elif ornt == 3:  # north
        _const_val_bc_set_north(bc, Q, B, By, val, ngh, comp)
    else:
        raise ValueError(f"orientation id {ornt} not accepted.")


def const_val_bc_factory(ornt, comp, states, topo, val, *args, **kwargs):
    """Factory to create a constant-valued boundary condition callable object.
    """

    # aliases
    cdef object Q = states.q
    cdef object B = topo.c
    cdef object Bx = topo.xf
    cdef object By = topo.yf
    cdef Py_ssize_t ngh = states.domain.nhalo

    if isinstance(ornt, str):
        if ornt in ["w", "west"]:
            ornt = 0
        elif ornt in ["e", "east"]:
            ornt = 1
        elif ornt in ["s", "south"]:
            ornt = 2
        elif ornt in ["n", "north"]:
            ornt = 3

    bc = ConstValBC()
    _const_val_bc_factory(bc, Q, B, Bx, By, val, ngh, comp, ornt)

    return bc
