# vim:fenc=utf-8
# vim:ft=pyrex
import cupy


cdef class ConstValCuPy:
    """Constant-value (conservative quantities) boundary condition.
    """
    # read-only views
    cdef object qp1
    cdef object hp1
    cdef object bb

    # target & editable views
    cdef object qm1
    cdef object qm2
    cdef object hm1

    # read-only values (boundary target value)
    cdef double val

    # real ndarrays owned by this instance
    cdef object hvals

    def __init__(
            self, object Q, object H, object Bx, object By, const double val,
            const Py_ssize_t ngh, const unsigned comp, const unsigned ornt
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

        if ornt == 0:  # west
            _const_val_bc_set_west(self, Q, H, Bx, ngh, comp)
        elif ornt == 1:  # east
            _const_val_bc_set_east(self, Q, H, Bx, ngh, comp)
        elif ornt == 2:  # south
            _const_val_bc_set_south(self, Q, H, By, ngh, comp)
        elif ornt == 3:  # north
            _const_val_bc_set_north(self, Q, H, By, ngh, comp)
        else:
            raise ValueError(f"orientation id {ornt} not accepted.")

        # store boundary value
        self.val = val

        if comp == 0:
            # store the constant depths
            self.hvals = val - self.bb
            # make sure all depths right on the boundary are non-negative
            assert cupy.all(self.hval >= 0.0), \
                f"Not all depths are non-negative on the boundary ({ornt}, {comp})"
        else:
            self.hvals = None  # so accidentallly using this variable should raise errors


cdef class ConstValCuPyWH:
    """Constant-value (conservative quantities) boundary condition for updating w and h.
    """

    def __call__(self):
        _const_val_bc_w_h_kernel(
            self.qp1, self.hp1, self.val, self.hvals, self.qm1, self.qm2, self.hm1)


cdef class ConstValCuPyOther:
    """Constant-value (conservative quantities) boundary condition for updating hu or hv.
    """

    def __call__(self):
        _const_val_bc_kernel(self.qp1, self.val, self.qm1, self.qm2)


cdef _const_val_bc_w_h_kernel = cupy.ElementwiseKernel(
    "T qp1, T hp1, T qbc, T hbc",
    "T qm1, T qm2, T hm1",
    """
        T dq = (qbc - qp1) * 2.0
        qm1 = qp1 + dq
        qm2 = qm1 + dq
        hm1 = hbc * 2.0 - hp1
    """,
    "_const_val_bc_w_h_kernel"
)


cdef _const_val_bc_kernel = cupy.ElementwiseKernel(
    "T qp1, T qbc",
    "T qm1, T qm2",
    """
        T dq = (qbc - qp1) * 2.0
        qm1 = qp1 + dq
        qm2 = qm1 + dq
    """,
    "_const_val_bc_w_h_kernel"
)


cdef void _const_val_bc_set_west(
    ConstValCuPy bc, object Q, object H, object Bx,
    const Py_ssize_t ngh, const Py_ssize_t comp
) except *:

    # these should be views into original data buffer
    bc.qp1 = Q[comp, ngh:Q.shape[1]-ngh, ngh]
    bc.qm1 = Q[comp, ngh:Q.shape[1]-ngh, ngh-1]
    bc.qm2 = Q[comp, ngh:Q.shape[1]-ngh, ngh-2]
    bc.hp1 = H[1:H.shape[0]-1, 1]
    bc.hm1 = H[1:H.shape[0]-1, 0]
    bc.bb = Bx[:, 0]


cdef void _const_val_bc_set_east(
    ConstValCuPy bc, object Q, object H, object Bx,
    const Py_ssize_t ngh, const Py_ssize_t comp
) except *:

    # these should be views into original data buffer
    bc.qp1 = Q[comp, ngh:Q.shape[1]-ngh, Q.shape[2]-ngh-1]
    bc.qm1 = Q[comp, ngh:Q.shape[1]-ngh, Q.shape[2]-ngh]
    bc.qm2 = Q[comp, ngh:Q.shape[1]-ngh, Q.shape[2]-ngh+1]
    bc.hp1 = H[1:H.shape[0]-1, H.shape[1]-2]
    bc.hm1 = H[1:H.shape[0]-1, H.shape[1]-1]
    bc.bb = Bx[:, Bx.shape[1]-1]


cdef void _const_val_bc_set_south(
    ConstValCuPy bc, object Q, object H, object By,
    const Py_ssize_t ngh, const Py_ssize_t comp
) except *:

    # these should be views into original data buffer
    bc.qp1 = Q[comp, ngh, ngh:Q.shape[2]-ngh]
    bc.qm1 = Q[comp, ngh-1, ngh:Q.shape[2]-ngh]
    bc.qm2 = Q[comp, ngh-2, ngh:Q.shape[2]-ngh]
    bc.hp1 = H[1, 1:H.shape[1]-1]
    bc.hm1 = H[0, 1:H.shape[1]-1]
    bc.bb = By[0, :]


cdef void _const_val_bc_set_north(
    ConstValCuPy bc, object Q, object H, object By,
    const Py_ssize_t ngh, const Py_ssize_t comp
) except *:

    # these should be views into original data buffer
    bc.qp1 = Q[comp, Q.shape[1]-ngh-1, ngh:Q.shape[2]-ngh]
    bc.qm1 = Q[comp, Q.shape[1]-ngh, ngh:Q.shape[2]-ngh]
    bc.qm2 = Q[comp, Q.shape[1]-ngh+1, ngh:Q.shape[2]-ngh]
    bc.hp1 = H[H.shape[0]-2, 1:H.shape[1]-1]
    bc.hm1 = H[H.shape[0]-1, 1:H.shape[1]-1]
    bc.bb = By[By.shape[0]-1, :]


def const_val_factory(ornt, comp, states, topo, const double val, *args, **kwargs):
    """Factory to create a constant extrapolation boundary condition callable object.
    """

    # aliases
    cdef object Q = states.Q
    cdef object H = states.H
    cdef object Bx = topo.xfcenters
    cdef object By = topo.yfcenters
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
        bc = ConstValCuPyWH(Q, H, Bx, By, val, ngh, comp, ornt)
    elif comp == 1 or comp == 2:
        bc = ConstValCuPyOther(Q, H, Bx, By, val, ngh, comp, ornt)
    else:
        raise ValueError(f"Unrecognized component: {comp}")

    return bc
