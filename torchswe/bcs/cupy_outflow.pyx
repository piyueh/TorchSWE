# vim:fenc=utf-8
# vim:ft=pyrex
cimport cython
import cupy


@cython.auto_pickle(False)  # meaningless to pickle a BC instance as everything is a memoryview
cdef class OutflowBC:
    """Outflow (constant extraption) boundary coditions for the CuPy backend.
    """

    # conservatives
    cdef object qc0  # w/hu/hv at the cell centers of the 1st internal cell layer
    cdef object qbcm1  # w/hu/hv at the 1st layer of ghost cells
    cdef object qbcm2  # w/hu/hv at the 2nd layer of ghost cells

    def __call__(self):
        _outflow_bc_kernel(self.qc0, self.qbcm1, self.qbcm2)


cdef _outflow_bc_kernel = cupy.ElementwiseKernel(
    "T qc0",
    "T qbcm1, T qbcm2",
    """
        qbcm1 = qc0;
        qbcm2 = qc0;
    """,
    "_outflow_bc_kernel"
)


cdef void _outflow_bc_set_west(
    OutflowBC bc,
    object Q, object B,
    const Py_ssize_t ngh, const Py_ssize_t comp,
) except *:

    bc.qc0      = Q[comp, ngh:Q.shape[1]-ngh, ngh]
    bc.qbcm1    = Q[comp, ngh:Q.shape[1]-ngh, ngh-1]
    bc.qbcm2    = Q[comp, ngh:Q.shape[1]-ngh, ngh-2]

    # modify the topography elevation in ghost cells
    B[ngh:B.shape[0]-ngh, ngh-1] = B[ngh:B.shape[0]-ngh, ngh]
    B[ngh:B.shape[0]-ngh, ngh-2] = B[ngh:B.shape[0]-ngh, ngh]


cdef void _outflow_bc_set_east(
    OutflowBC bc,
    object Q, object B,
    const Py_ssize_t ngh, const Py_ssize_t comp,
) except *:

    bc.qc0      = Q[comp, ngh:Q.shape[1]-ngh, Q.shape[2]-ngh-1]
    bc.qbcm1    = Q[comp, ngh:Q.shape[1]-ngh, Q.shape[2]-ngh]
    bc.qbcm2    = Q[comp, ngh:Q.shape[1]-ngh, Q.shape[2]-ngh+1]

    # modify the topography elevation in ghost cells
    B[ngh:B.shape[0]-ngh, B.shape[1]-ngh]   = B[ngh:B.shape[0]-ngh, B.shape[1]-ngh-1]
    B[ngh:B.shape[0]-ngh, B.shape[1]-ngh+1] = B[ngh:B.shape[0]-ngh, B.shape[1]-ngh-1]


cdef void _outflow_bc_set_south(
    OutflowBC bc,
    object Q, object B,
    const Py_ssize_t ngh, const Py_ssize_t comp,
) except *:

    bc.qc0      = Q[comp, ngh,      ngh:Q.shape[2]-ngh]
    bc.qbcm1    = Q[comp, ngh-1,    ngh:Q.shape[2]-ngh]
    bc.qbcm2    = Q[comp, ngh-2,    ngh:Q.shape[2]-ngh]

    # modify the topography elevation in ghost cells
    B[ngh-1, ngh:B.shape[1]-ngh] = B[ngh, ngh:B.shape[1]-ngh]
    B[ngh-2, ngh:B.shape[1]-ngh] = B[ngh, ngh:B.shape[1]-ngh]


cdef void _outflow_bc_set_north(
    OutflowBC bc,
    object Q, object B,
    const Py_ssize_t ngh, const Py_ssize_t comp,
) except *:

    bc.qc0      = Q[comp, Q.shape[1]-ngh-1,     ngh:Q.shape[2]-ngh]
    bc.qbcm1    = Q[comp, Q.shape[1]-ngh,       ngh:Q.shape[2]-ngh]
    bc.qbcm2    = Q[comp, Q.shape[1]-ngh+1,     ngh:Q.shape[2]-ngh]

    # modify the topography elevation in ghost cells
    B[B.shape[0]-ngh,   ngh:B.shape[1]-ngh] = B[B.shape[0]-ngh-1, ngh:B.shape[1]-ngh]
    B[B.shape[0]-ngh+1, ngh:B.shape[1]-ngh] = B[B.shape[0]-ngh-1, ngh:B.shape[1]-ngh]


cdef inline void _outflow_bc_factory(
    OutflowBC bc,
    object Q, object B,
    const Py_ssize_t ngh, const unsigned comp, const unsigned ornt,
) except *:

    assert Q.shape[1] == B.shape[0]
    assert Q.shape[2] == B.shape[1]

    if ornt == 0:  # west
        _outflow_bc_set_west(bc, Q, B, ngh, comp)
    elif ornt == 1:  # east
        _outflow_bc_set_east(bc, Q, B, ngh, comp)
    elif ornt == 2:  # south
        _outflow_bc_set_south(bc, Q, B, ngh, comp)
    elif ornt == 3:  # north
        _outflow_bc_set_north(bc, Q, B, ngh, comp)
    else:
        raise ValueError(f"orientation id {ornt} not accepted.")


def outflow_bc_factory(ornt, comp, states, topo, *args, **kwargs):
    """Factory to create a outflow (constant extrapolation) boundary condition callable object.
    """

    # aliases
    cdef object Q = states.Q
    cdef object B = topo.centers
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

    bc = OutflowBC()
    _outflow_bc_factory(bc, Q, B, ngh, comp, ornt)

    return bc
