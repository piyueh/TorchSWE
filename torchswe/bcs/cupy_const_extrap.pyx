# vim:fenc=utf-8
# vim:ft=pyrex
import cupy


cdef class ConstExtrapBC:

    # conservatives
    cdef object qc0  # q at the cell centers of the 1st internal cell layer
    cdef object qbci  # q at the inner side of the boundary cell faces
    cdef object qbco  # q at the outer side of the boundary cell faces
    cdef object qother  # q at the inner side of the another face of the 1st internal cell

    # topography elevation
    cdef object bbc  # topo elevations at the boundary cell faces
    cdef object bother  # topo elevation at cell faces between the 1st and 2nd internal cell layers

    # depth
    cdef object hc0  # depth at the cell centers of the 1st internal cell layer
    cdef object hbci  # depth at the inner side of the boundary cell faces
    cdef object hbco  # depth at the outer side of the boundary cell faces
    cdef object hother  # depth at the inner side of the another face of the 1st internal cell

    # velocities
    cdef object ubci  # u or v at the inner side of the boundary cell faces
    cdef object ubco  # u or v at the outer side of the boundary cell faces
    cdef object uother  # u or v at the inner side of the another face of the 1st internal cell

    # tolerance
    cdef double tol  # depths under this tolerance are considered dry cells
    cdef double drytol  # depths under this values are considered wet but still cells

    def __init__(
        self,
        object Q, object Qmx, object Qpx, object Qmy, object Qpy,
        object H, object Hmx, object Hpx, object Hmy, object Hpy,
        object Bx, object By,
        const Py_ssize_t ngh, const unsigned comp, const unsigned ornt,
        const double tol, const double drytol
    ):

        # runtime check for the shapes
        _shape_checker(Q, Qmx, Qpx, Qmy, Qpy, Hmx, Hpx, Hmy, Hpy, ngh, comp, ornt)

        if ornt == 0:  # west
            _const_extrap_bc_set_west(self, Q, Bx, Qmx, Qpx, H, Hmx, Hpx, ngh, comp)
        elif ornt == 1:  # east
            _const_extrap_bc_set_east(self, Q, Bx, Qmx, Qpx, H, Hmx, Hpx, ngh, comp)
        elif ornt == 2:  # south
            _const_extrap_bc_set_south(self, Q, By, Qmy, Qpy, H, Hmy, Hpy, ngh, comp)
        elif ornt == 3:  # north
            _const_extrap_bc_set_north(self, Q, By, Qmy, Qpy, H, Hmy, Hpy, ngh, comp)
        else:
            raise ValueError(f"orientation id {ornt} not accepted.")

        self.tol = tol
        self.drytol = drytol


cdef class ConstExtrapWH(ConstExtrapBC):

    def __call__(self):
        _const_extrap_bc_w_h_kernel(
            self.qc0, self.hc0, self.bbc, self.bother, self.tol,
            self.qbci, self.qbco, self.qother, self.hbci, self.hbco, self.hother
        )


cdef class ConstExtrapOther(ConstExtrapBC):

    def __call__(self):
        _const_extrap_bc_kernel(
            self.qc0, self.hbci, self.hbco, self.hother, self.drytol,
            self.qbci, self.qbco, self.qother, self.ubci, self.ubco, self.uother
        )


cdef _const_extrap_bc_w_h_kernel = cupy.ElementwiseKernel(
    "T wc0, T hc0, T bbc, T bother, T tol",
    "T wbci, T wbco, T wother, T hbci, T hbco, T hother",
    """
        wbci = wc0;
        wother = wc0;

        hbci = wbci - bbc;
        hother = wother - bother;

        // fix negative depth
        if (hc0 < tol) {
            wbci = bbc;
            wother = bother;
            hbci = 0.0;
            hother = 0.0;
        } else if (hbci < tol) {
            wbci = bbc;
            wother = wc0 * 2.0 - bbc;
            hbci = 0.0;
            hother = wother - bother;
        } else if (hother < tol) {
            wbci = wc0 * 2.0 - bother;
            wother = bother;
            hbci = wbci - bbc;
            hother = 0.0;
        }

        // reconstruct to eliminate rounding error-edffect in further calculations
        wbci = hbci + bbc;
        wother = hother + bother;

        // outer side of the bc face
        wbco = wbci;
        hbco = hbci;
    """,
    "_const_copy_bc_w_h_kernel"
)


cdef _const_extrap_bc_kernel = cupy.ElementwiseKernel(
    "T qc0, T hbci, T hbco, T hother, T drytol",
    "T qbci, T qbco, T qother, T ubci, T ubco, T uother",
    """
        qbci = qc0;
        qother = qc0;

        ubci = qbci / hbci;
        uother = qother / hother;

        // reconstruct to eliminate rounding error-edffect in further calculations
        qbci = hbci * ubci;
        qother = hother * uother;

        // outer side of the bc face
        qbco = qbci;
        ubco = ubci;

        if (hbci < drytol) {
            qbci = 0.0;
            ubci = 0.0;
            qbco = 0.0;
            ubco = 0.0;
        }

        if (hother < drytol) {
            qother = 0.0;
            uother = 0.0;
        }
    """,
    "_const_extrap_bc_kernel"
)


cdef void _const_extrap_bc_set_west(
    ConstExtrapBC bc,
    object Q, object Bx, object Qmx, object Qpx,
    object H, object Hmx, object Hpx,
    const Py_ssize_t ngh, const Py_ssize_t comp
) except *:

    # these should be views into original data buffer
    bc.qc0 = Q[comp, ngh:Q.shape[1]-ngh, ngh]
    bc.qbci = Qpx[comp, :, 0]
    bc.qbco = Qmx[comp, :, 0]
    bc.qother = Qmx[comp, :, 1]

    bc.hc0 = H[1:H.shape[0]-1, 1]
    bc.hbci = Hpx[0, :, 0]
    bc.hbco = Hmx[0, :, 0]
    bc.hother = Hmx[0, :, 1]

    bc.bbc = Bx[:, 0]
    bc.bother = Bx[:, 1]

    if comp != 0:
        bc.ubci = Hpx[comp, :, 0]
        bc.ubco = Hmx[comp, :, 0]
        bc.uother = Hmx[comp, :, 1]
    else:
        bc.ubci = None
        bc.ubco = None
        bc.uother = None


cdef void _const_extrap_bc_set_east(
    ConstExtrapBC bc,
    object Q, object Bx, object Qmx, object Qpx,
    object H, object Hmx, object Hpx,
    const Py_ssize_t ngh, const Py_ssize_t comp
) except *:

    # these should be views into original data buffer
    bc.qc0 = Q[comp, ngh:Q.shape[1]-ngh, Q.shape[2]-ngh-1]
    bc.qbci = Qmx[comp, :, Qmx.shape[2]-1]
    bc.qbco = Qpx[comp, :, Qpx.shape[2]-1]
    bc.qother = Qpx[comp, :, Qpx.shape[2]-2]

    bc.hc0 = H[1:H.shape[0]-1, H.shape[1]-2]
    bc.hbci = Hmx[0, :, Hmx.shape[2]-1]
    bc.hbco = Hpx[0, :, Hpx.shape[2]-1]
    bc.hother = Hpx[0, :, Hpx.shape[2]-2]

    bc.bbc = Bx[:, Bx.shape[1]-1]
    bc.bother = Bx[:, Bx.shape[1]-2]

    if comp != 0:
        bc.ubci = Hmx[comp, :, Hmx.shape[2]-1]
        bc.ubco = Hpx[comp, :, Hpx.shape[2]-1]
        bc.uother = Hpx[comp, :, Hpx.shape[2]-2]
    else:
        bc.ubci = None
        bc.ubco = None
        bc.uother = None


cdef void _const_extrap_bc_set_south(
    ConstExtrapBC bc,
    object Q, object By, object Qmy, object Qpy,
    object H, object Hmy, object Hpy,
    const Py_ssize_t ngh, const Py_ssize_t comp
) except *:

    # these should be views into original data buffer
    bc.qc0 = Q[comp, ngh, ngh:Q.shape[2]-ngh]
    bc.qbci = Qpy[comp, 0, :]
    bc.qbco = Qmy[comp, 0, :]
    bc.qother = Qmy[comp, 1, :]

    bc.hc0 = H[1, 1:H.shape[1]-1]
    bc.hbci = Hpy[0, 0, :]
    bc.hbco = Hmy[0, 0, :]
    bc.hother = Hmy[0, 1, :]

    bc.bbc = By[0, :]
    bc.bother = By[1, :]

    if comp != 0:
        bc.ubci = Hpy[comp, 0, :]
        bc.ubco = Hmy[comp, 0, :]
        bc.uother = Hmy[comp, 1, :]
    else:
        bc.ubci = None
        bc.ubco = None
        bc.uother = None


cdef void _const_extrap_bc_set_north(
    ConstExtrapBC bc,
    object Q, object By, object Qmy, object Qpy,
    object H, object Hmy, object Hpy,
    const Py_ssize_t ngh, const Py_ssize_t comp
) except *:

    # these should be views into original data buffer
    bc.qc0 = Q[comp, Q.shape[1]-ngh-1, ngh:Q.shape[2]-ngh]
    bc.qbci = Qmy[comp, Qmy.shape[1]-1, :]
    bc.qbco = Qpy[comp, Qpy.shape[1]-1, :]
    bc.qother = Qpy[comp, Qpy.shape[1]-2, :]

    bc.hc0 = H[H.shape[0]-2, 1:H.shape[1]-1]
    bc.hbci = Hmy[0, Hmy.shape[1]-1, :]
    bc.hbco = Hpy[0, Hpy.shape[1]-1, :]
    bc.hother = Hpy[0, Hpy.shape[1]-2, :]

    bc.bbc = By[By.shape[0]-1, :]
    bc.bother = By[By.shape[0]-2, :]

    if comp != 0:
        bc.ubci = Hmy[comp, Hmy.shape[1]-1, :]
        bc.ubco = Hpy[comp, Hpy.shape[1]-1, :]
        bc.uother = Hpy[comp, Hpy.shape[1]-2, :]
    else:
        bc.ubci = None
        bc.ubco = None
        bc.uother = None


cdef _shape_checker(
    object Q, object Qmx, object Qpx, object Qmy, object Qpy,
    object Hmx, object Hpx, object Hmy, object Hpy,
    const Py_ssize_t ngh, const Py_ssize_t comp, const unsigned ornt
):
    assert ngh == 2, "Currently only support ngh = 2"
    assert 0 <= comp <= 2, "comp should be 0 <= comp <= 2"
    assert 0 <= ornt <= 3, "ornt should be 0 <= ornt <= 3"

    assert Q.shape[0] == 3, f"{Q.shape}"

    assert Qmx.shape[0] == 3, f"{Qmx.shape}"
    assert Qmx.shape[1] == Q.shape[1] - 2 * ngh, f"{Qmx.shape}"
    assert Qmx.shape[2] == Q.shape[2] - 2 * ngh + 1, f"{Qmx.shape}"

    assert Qpx.shape[0] == 3, f"{Qpx.shape}"
    assert Qpx.shape[1] == Q.shape[1] - 2 * ngh, f"{Qpx.shape}"
    assert Qpx.shape[2] == Q.shape[2] - 2 * ngh + 1, f"{Qpx.shape}"

    assert Qmy.shape[0] == 3, f"{Qmy.shape}"
    assert Qmy.shape[1] == Q.shape[1] - 2 * ngh + 1, f"{Qmy.shape}"
    assert Qmy.shape[2] == Q.shape[2] - 2 * ngh, f"{Qmy.shape}"

    assert Qpy.shape[0] == 3, f"{Qpy.shape}"
    assert Qpy.shape[1] == Q.shape[1] - 2 * ngh + 1, f"{Qpy.shape}"
    assert Qpy.shape[2] == Q.shape[2] - 2 * ngh, f"{Qpy.shape}"

    assert Hmx.shape[0] == 3, f"{Hmx.shape}"
    assert Hmx.shape[1] == Q.shape[1] - 2 * ngh, f"{Hmx.shape}"
    assert Hmx.shape[2] == Q.shape[2] - 2 * ngh + 1, f"{Hmx.shape}"

    assert Hpx.shape[0] == 3, f"{Hpx.shape}"
    assert Hpx.shape[1] == Q.shape[1] - 2 * ngh, f"{Hpx.shape}"
    assert Hpx.shape[2] == Q.shape[2] - 2 * ngh + 1, f"{Hpx.shape}"

    assert Hmy.shape[0] == 3, f"{Hmy.shape}"
    assert Hmy.shape[1] == Q.shape[1] - 2 * ngh + 1, f"{Hmy.shape}"
    assert Hmy.shape[2] == Q.shape[2] - 2 * ngh, f"{Hmy.shape}"

    assert Hpy.shape[0] == 3, f"{Hpy.shape}"
    assert Hpy.shape[1] == Q.shape[1] - 2 * ngh + 1, f"{Hpy.shape}"
    assert Hpy.shape[2] == Q.shape[2] - 2 * ngh, f"{Hpy.shape}"


def const_extrap_factory(ornt, comp, states, topo, tol, drytol, *args, **kwargs):
    """Factory to create a constant extrapolation boundary condition callable object.
    """

    # aliases
    cdef object Q = states.Q
    cdef object Qmx = states.face.x.minus.Q
    cdef object Qpx = states.face.x.plus.Q
    cdef object Qmy = states.face.y.minus.Q
    cdef object Qpy = states.face.y.plus.Q
    cdef object H = states.H
    cdef object Hmx = states.face.x.minus.U
    cdef object Hpx = states.face.x.plus.U
    cdef object Hmy = states.face.y.minus.U
    cdef object Hpy = states.face.y.plus.U
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
        bc = ConstExtrapWH(
            Q, Qmx, Qpx, Qmy, Qpy, H, Hmx, Hpx, Hmy, Hpy, Bx, By, ngh, comp, ornt, tol, drytol)
    elif comp == 1 or comp == 2:
        bc = ConstExtrapOther(
            Q, Qmx, Qpx, Qmy, Qpy, H, Hmx, Hpx, Hmy, Hpy, Bx, By, ngh, comp, ornt, tol, drytol)
    else:
        raise ValueError(f"Unrecognized component: {comp}")

    return bc
