# vim:fenc=utf-8
# vim:ft=pyrex
import cupy


cdef class ConstExtrapBC:
    cdef object qp1
    cdef object qm1
    cdef object qm2
    cdef object hp1
    cdef object hm1

    def __init__(
        self, object Q, object H,
        const Py_ssize_t ngh, const unsigned comp, const unsigned ornt
    ):

        # runtime check for the shapes
        assert ngh == 2, "Currently only support ngh = 2"
        assert 0 <= comp <= 2, "comp should be 0 <= comp <= 2"
        assert 0 <= ornt <= 3, "ornt should be 0 <= ornt <= 3"
        assert Q.shape[0] == 3, f"{Q.shape}"
        assert H.shape[0] == Q.shape[1] - 2, f"{H.shape[0]}, {Q.shape[1]-2}"
        assert H.shape[1] == Q.shape[2] - 2, f"{H.shape[1]}, {Q.shape[2]-2}"

        if ornt == 0:  # west
            _const_extrap_bc_set_west(self, Q, H, ngh, comp)
        elif ornt == 1:  # east
            _const_extrap_bc_set_east(self, Q, H, ngh, comp)
        elif ornt == 2:  # south
            _const_extrap_bc_set_south(self, Q, H, ngh, comp)
        elif ornt == 3:  # north
            _const_extrap_bc_set_north(self, Q, H, ngh, comp)
        else:
            raise ValueError(f"orientation id {ornt} not accepted.")


cdef class ConstExtrapWH(ConstExtrapBC):

    def __call__(self):
        _const_extrap_w_h_kernel(self.qp1, self.hp1, self.qm1, self.qm2, self.hm1)


cdef class ConstExtrapOther(ConstExtrapBC):

    def __call__(self):
        _const_extrap_kernel(self.qp1, self.qm1, self.qm2)


cdef _const_extrap_kernel = cupy.ElementwiseKernel(
    "T qr",
    "T ql1, T ql2",
    """
        ql1 = qr;
        ql2 = qr;
    """,
    "_const_extrap_kernel"
)


cdef _const_extrap_w_h_kernel = cupy.ElementwiseKernel(
    "T qr, T hr",
    "T ql1, T ql2, T hl",
    """
        ql1 = qr;
        ql2 = qr;
        hl = hr;
    """,
    "_const_copy_w_h_kernel"
)


cdef void _const_extrap_bc_set_west(
    ConstExtrapBC bc, object Q, object H,
    const Py_ssize_t ngh, const Py_ssize_t comp
) except *:

    # these should be views into original data buffer
    bc.qp1 = Q[comp, ngh:Q.shape[1]-ngh, ngh]
    bc.qm1 = Q[comp, ngh:Q.shape[1]-ngh, ngh-1]
    bc.qm2 = Q[comp, ngh:Q.shape[1]-ngh, ngh-2]
    bc.hp1 = H[1:H.shape[0]-1, 1]
    bc.hm1 = H[1:H.shape[0]-1, 0]


cdef void _const_extrap_bc_set_east(
    ConstExtrapBC bc, object Q, object H,
    const Py_ssize_t ngh, const Py_ssize_t comp
) except *:

    # these should be views into original data buffer
    bc.qp1 = Q[comp, ngh:Q.shape[1]-ngh, Q.shape[2]-ngh-1]
    bc.qm1 = Q[comp, ngh:Q.shape[1]-ngh, Q.shape[2]-ngh]
    bc.qm2 = Q[comp, ngh:Q.shape[1]-ngh, Q.shape[2]-ngh+1]
    bc.hp1 = H[1:H.shape[0]-1, H.shape[1]-2]
    bc.hm1 = H[1:H.shape[0]-1, H.shape[1]-1]


cdef void _const_extrap_bc_set_south(
    ConstExtrapBC bc, object Q, object H,
    const Py_ssize_t ngh, const Py_ssize_t comp
) except *:

    # these should be views into original data buffer
    bc.qp1 = Q[comp, ngh, ngh:Q.shape[2]-ngh]
    bc.qm1 = Q[comp, ngh-1, ngh:Q.shape[2]-ngh]
    bc.qm2 = Q[comp, ngh-2, ngh:Q.shape[2]-ngh]
    bc.hp1 = H[1, 1:H.shape[1]-1]
    bc.hm1 = H[0, 1:H.shape[1]-1]


cdef void _const_extrap_bc_set_north(
    ConstExtrapBC bc, object Q, object H,
    const Py_ssize_t ngh, const Py_ssize_t comp
) except *:

    # these should be views into original data buffer
    bc.qp1 = Q[comp, Q.shape[1]-ngh-1, ngh:Q.shape[2]-ngh]
    bc.qm1 = Q[comp, Q.shape[1]-ngh, ngh:Q.shape[2]-ngh]
    bc.qm2 = Q[comp, Q.shape[1]-ngh+1, ngh:Q.shape[2]-ngh]
    bc.hp1 = H[H.shape[0]-2, 1:H.shape[1]-1]
    bc.hm1 = H[H.shape[0]-1, 1:H.shape[1]-1]


def const_extrap_factory(ornt, comp, states, *args, **kwargs):
    """Factory to create a constant extrapolation boundary condition callable object.
    """

    # aliases
    cdef object Q = states.Q
    cdef object H = states.H
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

    if comp == 0:
        bc = ConstExtrapWH(Q, H, ngh, comp, ornt)
    elif comp == 1 or comp == 2:
        bc = ConstExtrapOther(Q, H, ngh, comp, ornt)
    else:
        raise ValueError(f"Unrecognized component: {comp}")

    return bc
