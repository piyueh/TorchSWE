# vim:fenc=utf-8
# vim:ft=pyrex
import cupy


cdef class LinearExtrapBC:
    # read-only views
    cdef object qp1
    cdef object qp2
    cdef object hp1
    cdef object hp2
    cdef object bb
    cdef object bcp1

    # editable views
    cdef object qm1
    cdef object qm2
    cdef object hm1

    # solid cupy.ndarrays
    cdef readonly object bcm1
    cdef readonly object bcm2

    def __init__(
        self, object Q, object H, object Bc, object Bx, object By,
        const Py_ssize_t ngh, const unsigned comp, const unsigned ornt
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

        cdef Py_ssize_t n = (Q.shape[1] - 2 * ngh) if (ornt < 2) else (Q.shape[2] - 2 * ngh)
        self.bcm1 = cupy.zeros(n, dtype=Q.dtype)
        self.bcm2 = cupy.zeros(n, dtype=Q.dtype)

        if ornt == 0:  # west
            _linear_extrap_bc_set_west(self, Q, H, Bc, Bx, ngh, comp)
        elif ornt == 1:  # east
            _linear_extrap_bc_set_east(self, Q, H, Bc, Bx, ngh, comp)
        elif ornt == 2:  # south
            _linear_extrap_bc_set_south(self, Q, H, Bc, By, ngh, comp)
        elif ornt == 3:  # north
            _linear_extrap_bc_set_north(self, Q, H, Bc, By, ngh, comp)
        else:
            raise ValueError(f"orientation id {ornt} not accepted.")

        # calculate the topo elevation of the ghost cells
        db = (self.bb - self.bcp1) * 2.0
        cupy.add(self.bcp1, db, out=self.bcm1)
        cupy.add(self.bcm1, db, out=self.bcm2)


cdef class LinearExtrapWH(LinearExtrapBC):

    def __call__(self):
        _linear_extrap_w_h_kernel(
            self.qp1, self.qp2, self.hp1, self.hp2, self.bcm1, self.bcm2,
            self.qm1, self.qm2, self.hm1
        )


cdef class LinearExtrapOther(LinearExtrapBC):

    def __call__(self):
        _linear_extrap_kernel(self.qp1, self.qp2, self.hm1, self.qm1, self.qm2)


cdef _linear_extrap_kernel = cupy.ElementwiseKernel(
    "T qp1, T qp2, T hm1",
    "T qm1, T qm2",
    """
        T dq;

        if (hm1 > 0.0) {
            dq = qp1 - qp2;
            qm1 = qp1 + dq;
            qm2 = qm1 + dq;
        } else {
            qm1 = 0.0;
            qm2 = 0.0;
        }
    """,
    "_linear_extrap_kernel"
)



cdef _linear_extrap_w_h_kernel = cupy.ElementwiseKernel(
    "T qp1, T qp2, T hp1, T hp2, T bcm1, T bcm2",
    "T qm1, T qm2, T hm1",
    """
        T dq;

        hm1 = hp1 * 2.0 - hp2;

        if (hm1 <= 0.0) {
            hm1 = 0.0;
            qm1 = bcm1;
            qm2 = bcm2;
        } else {
            dq = qp1 - qp2;
            qm1 = qp1 + dq;
            qm2 = qm1 + dq;
        }
    """,
    "_linear_extrap_w_h_kernel"
)



cdef void _linear_extrap_bc_set_west(
    LinearExtrapBC bc, object Q, object H, object Bc, object Bx,
    const Py_ssize_t ngh, const Py_ssize_t comp
) except *:

    # these should be views into original data buffer
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
    LinearExtrapBC bc, object Q, object H, object Bc, object Bx,
    const Py_ssize_t ngh, const Py_ssize_t comp
) except *:

    # these should be views into original data buffer
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
    LinearExtrapBC bc, object Q, object H, object Bc, object By,
    const Py_ssize_t ngh, const Py_ssize_t comp
) except *:

    # these should be views into original data buffer
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
    LinearExtrapBC bc, object Q, object H, object Bc, object By,
    const Py_ssize_t ngh, const Py_ssize_t comp
) except *:

    # these should be views into original data buffer
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
    """Factory to create a constant extrapolation boundary condition callable object.
    """

    # aliases
    cdef object Q = states.Q
    cdef object H = states.H
    cdef object Bc = topo.centers
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
        bc = LinearExtrapWH(Q, H, Bc, Bx, By, ngh, comp, ornt)
    elif comp == 1 or comp == 2:
        bc = LinearExtrapOther(Q, H, Bc, Bx, By, ngh, comp, ornt)
    else:
        raise ValueError(f"Unrecognized component: {comp}")

    return bc
