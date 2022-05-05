# vim:fenc=utf-8
# vim:ft=pyrex
cimport cython
from torchswe import nplike as _nplike


@cython.auto_pickle(False)  # meaningless to pickle a BC instance as everything is a memoryview
cdef class LinearExtrapBC:
    """Linear extraption boundary coditions for the CuPy backend.
    """

    # conservatives
    cdef object qc0  # w/hu/hv at the cell centers of the 1st internal cell layer
    cdef object qc1  # w/hu/hv at the cell centers of the 2nd internal cell layer
    cdef object qbcm1  # w/hu/hv at the 1st layer of ghost cells
    cdef object qbcm2  # w/hu/hv at the 2nd layer of ghost cells

    # depth
    cdef object hbci  # depth at the inner side of the boundary cell faces
    cdef object hbco  # depth at the outer side of the boundary cell faces
    cdef object hother  # depth at the inner side of the another face of the 1st internal cell

    def __call__(self):
        _linear_extrap_bc_kernel(self.qc0, self.qc1, self.qbcm1, self.qbcm2)


cdef _linear_extrap_bc_kernel(qc0, qc1, qbcm1, qbcm2):
    """For internal use"""
    delta = qc0 - qc1;
    qbcm1[...] = qc0 + delta;
    qbcm2[...] = qbcm1 + delta;


cdef void _linear_extrap_bc_set_west(
    LinearExtrapBC bc,
    object Q, object B, object Bx,
    const Py_ssize_t ngh, const Py_ssize_t comp,
) except *:

    bc.qc0      = Q[comp, ngh:Q.shape[1]-ngh, ngh]
    bc.qc1      = Q[comp, ngh:Q.shape[1]-ngh, ngh+1]
    bc.qbcm1    = Q[comp, ngh:Q.shape[1]-ngh, ngh-1]
    bc.qbcm2    = Q[comp, ngh:Q.shape[1]-ngh, ngh-2]

    # modify the topography elevation in ghost cells
    for i in range(ngh, B.shape[0]-ngh):
        B[i, ngh-1] = Bx[i-ngh, 0] * 2.0 - B[i, ngh]
        B[i, ngh-2] = Bx[i-ngh, 0] * 4.0 - B[i, ngh] * 3.0


cdef void _linear_extrap_bc_set_east(
    LinearExtrapBC bc,
    object Q, object B, object Bx,
    const Py_ssize_t ngh, const Py_ssize_t comp,
) except *:

    bc.qc0      = Q[comp, ngh:Q.shape[1]-ngh, Q.shape[2]-ngh-1]
    bc.qc1      = Q[comp, ngh:Q.shape[1]-ngh, Q.shape[2]-ngh-2]
    bc.qbcm1    = Q[comp, ngh:Q.shape[1]-ngh, Q.shape[2]-ngh]
    bc.qbcm2    = Q[comp, ngh:Q.shape[1]-ngh, Q.shape[2]-ngh+1]

    # modify the topography elevation in ghost cells
    for i in range(ngh, B.shape[0]-ngh):
        B[i, B.shape[1]-ngh]   = Bx[i-ngh, Bx.shape[1]-1] * 2.0 - B[i, B.shape[1]-ngh-1]
        B[i, B.shape[1]-ngh+1] = Bx[i-ngh, Bx.shape[1]-1] * 4.0 - B[i, B.shape[1]-ngh-1] * 3.0


cdef void _linear_extrap_bc_set_south(
    LinearExtrapBC bc,
    object Q, object B, object By,
    const Py_ssize_t ngh, const Py_ssize_t comp,
) except *:

    bc.qc0      = Q[comp, ngh,      ngh:Q.shape[2]-ngh]
    bc.qc1      = Q[comp, ngh+1,    ngh:Q.shape[2]-ngh]
    bc.qbcm1    = Q[comp, ngh-1,    ngh:Q.shape[2]-ngh]
    bc.qbcm2    = Q[comp, ngh-2,    ngh:Q.shape[2]-ngh]

    # modify the topography elevation in ghost cells
    for i in range(ngh, B.shape[1]-ngh):
        B[ngh-1, i] = By[0, i-ngh] * 2.0 - B[ngh, i]
        B[ngh-2, i] = By[0, i-ngh] * 4.0 - B[ngh, i] * 3.0


cdef void _linear_extrap_bc_set_north(
    LinearExtrapBC bc,
    object Q, object B, object By,
    const Py_ssize_t ngh, const Py_ssize_t comp,
) except *:

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
    object Q, object B, object Bx, object By,
    const Py_ssize_t ngh, const unsigned comp, const unsigned ornt,
) except *:

    assert Q.shape[1] == B.shape[0]
    assert Q.shape[2] == B.shape[1]
    assert Q.shape[1] == Bx.shape[0] + 2 * ngh
    assert Q.shape[2] == Bx.shape[1] + 2 * ngh - 1
    assert Q.shape[1] == By.shape[0] + 2 * ngh - 1
    assert Q.shape[2] == By.shape[1] + 2 * ngh

    if ornt == 0:  # west
        _linear_extrap_bc_set_west(bc, Q, B, Bx, ngh, comp)
    elif ornt == 1:  # east
        _linear_extrap_bc_set_east(bc, Q, B, Bx, ngh, comp)
    elif ornt == 2:  # south
        _linear_extrap_bc_set_south(bc, Q, B, By, ngh, comp)
    elif ornt == 3:  # north
        _linear_extrap_bc_set_north(bc, Q, B, By, ngh, comp)
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

    if isinstance(ornt, str):
        if ornt in ["w", "west"]:
            ornt = 0
        elif ornt in ["e", "east"]:
            ornt = 1
        elif ornt in ["s", "south"]:
            ornt = 2
        elif ornt in ["n", "north"]:
            ornt = 3

    bc = LinearExtrapBC()
    _linear_extrap_bc_factory(bc, Q, B, Bx, By, ngh, comp, ornt)

    return bc
