# vim:fenc=utf-8
# vim:ft=pyrex
import cupy
cimport _checker
cimport cython


ctypedef fused ConstValBC:
    ConstValWH
    ConstValOther


@cython.auto_pickle(False)  # meaningless to pickle a BC instance as everything is a memoryview
cdef class ConstValBase:
    """Constant-value (conservative quantities) boundary condition.
    """

    # conservatives
    cdef object qc0  # q at the cell centers of the 1st internal cell layer
    cdef object qc1  # q at the cell centers of the 2nd internal cell layer
    cdef object qbci  # q at the inner side of the boundary cell faces
    cdef object qbco  # q at the outer side of the boundary cell faces
    cdef object qother  # q at the inner side of the another face of the 1st internal cell

    # depth
    cdef object hbci  # depth at the inner side of the boundary cell faces
    cdef object hbco  # depth at the outer side of the boundary cell faces
    cdef object hother  # depth at the inner side of the another face of the 1st internal cell

    # read-only values (boundary target value & theta)
    cdef double val
    cdef double theta


@cython.auto_pickle(False)  # meaningless to pickle a BC instance as everything is a memoryview
cdef class ConstValWH(ConstValBase):
    """Constant-value (conservative quantities) boundary condition for updating w and h.
    """

    # depth
    cdef object hc0  # depth at the cell centers of the 1st internal cell layer

    # topography elevation
    cdef object bbc  # topo elevations at the boundary cell faces
    cdef object bother  # topo elevation at cell faces between the 1st and 2nd internal cell layers

    # saved for convenience if the bc is for w and h, as topography doesn't change
    cdef object wbcval
    cdef object hbcval

    # tolerance
    cdef double tol  # depths under this tolerance are considered dry cells

    def __init__(
        self,
        object Q, object xmQ, object xpQ, object ymQ, object ypQ,
        object U, object xmU, object xpU, object ymU, object ypU,
        object Bx, object By,
        const Py_ssize_t ngh, const unsigned comp, const unsigned ornt,
        const double tol, const double drytol,
        const double theta, const double val,
    ):
        _const_val_bc_init[ConstValWH](
            self, Q, xmQ, xpQ, ymQ, ypQ, U, xmU, xpU, ymU, ypU, Bx, By,
            ngh, comp, ornt, tol, drytol, theta, val
        )

    def __call__(self):
        _const_val_bc_w_h_kernel(
            self.qc0, self.qc1, self.hc0, self.val, self.wbcval, self.hbcval,
            self.bbc, self.bother, self.theta, self.tol,
            self.qbci, self.qbco, self.qother, self.hbci, self.hbco, self.hother
        )


@cython.auto_pickle(False)  # meaningless to pickle a BC instance as everything is a memoryview
cdef class ConstValOther(ConstValBase):
    """Constant-value (conservative quantities) boundary condition for updating hu or hv.
    """

    # velocities
    cdef object ubci  # u or v at the inner side of the boundary cell faces
    cdef object ubco  # u or v at the outer side of the boundary cell faces
    cdef object uother  # u or v at the inner side of the another face of the 1st internal cell

    # tolerance
    cdef double drytol  # depths under this values are considered wet but still cells

    def __init__(
        self,
        object Q, object xmQ, object xpQ, object ymQ, object ypQ,
        object U, object xmU, object xpU, object ymU, object ypU,
        object Bx, object By,
        const Py_ssize_t ngh, const unsigned comp, const unsigned ornt,
        const double tol, const double drytol,
        const double theta, const double val,
    ):
        _const_val_bc_init[ConstValOther](
            self, Q, xmQ, xpQ, ymQ, ypQ, U, xmU, xpU, ymU, ypU, Bx, By,
            ngh, comp, ornt, tol, drytol, theta, val
        )

    def __call__(self):
        _const_val_bc_kernel(
            self.qc0, self.qc1, self.val, self.hbci, self.hbco, self.hother, self.theta, self.drytol,
            self.qbci, self.qbco, self.qother, self.ubci, self.ubco, self.uother
        )


cdef _const_val_bc_w_h_kernel = cupy.ElementwiseKernel(
    "T wc0, T wc1, T hc0, T val, T wval, T hval, T bbc, T bother, T theta, T tol",
    "T wbci, T wbco, T wother, T hbci, T hbco, T hother",
    """
        T denominator;
        T slp;

        // outer side follows desired values (but can't be negative depth)
        wbco = wval;
        hbco = hval;

        // a completely dry cell
        if (hc0 < tol) {
            hbci = 0.0; 
            hother = 0.0;
            wbci = bbc;
            wother = bother;
            continue;
        }

        // inner side depends on minmod reconstruction as if there's a ghost cell
        denominator = wc1 - wc0;
        slp = (wc0 - val) / denominator;
        slp = min(slp*theta, (slp+1.0)/2.0);
        slp = min(slp, theta);
        slp = max(slp, 0.0);
        slp *= denominator;
        slp /= 2.0;
        hbci = wc0 - slp - bbc;
        hother = wc0 + slp - bother;

        // fix negative depth
        if (hbci < tol) {
            hbci = 0.0;
            hother = hc0 * 2.0;
        } else if (hother < tol) {
            hbci = hc0 * 2.0;
            hother = 0.0;
        }

        // reconstruct to eliminate rounding error-edffect in further calculations
        wbci = hbci + bbc;
        wother = hother + bother;
    """,
    "_const_val_bc_w_h_kernel"
)


cdef _const_val_bc_kernel = cupy.ElementwiseKernel(
    "T qc0, T qc1, T val, T hbci, T hbco, T hother, T theta, T drytol",
    "T qbci, T qbco, T qother, T ubci, T ubco, T uother",
    """
        T denominator;
        T slp;

        // outer side follows desired values
        qbco = val;
        ubco = val / hbci;

        // inner side depends on minmod reconstruction as if there's a ghost cell
        denominator = qc1 - qc0;
        slp = (qc0 - val) / denominator;
        slp = min(slp*theta, (slp+1.0)/2.0);
        slp = min(slp, theta);
        slp = max(slp, 0.0);
        slp *= denominator;
        slp /= 2.0;

        // we don't fix values at the outer side -> they follow desired BC values

        if (hbci < drytol) {
            qbci = 0.0;
            ubci = 0.0;
        } else {
            ubci = (qc0 - slp) / hbci;
            qbci = hbci * ubci;
        }

        if (hother < drytol) {
            qother = 0.0;
            uother = 0.0;
        } else {
            uother = (qc0 + slp) / hother;
            qother = hother * uother;
        }
    """,
    "_const_val_bc_kernel"
)


cdef void _const_val_bc_init(
    ConstValBC bc,
    object Q, object xmQ, object xpQ, object ymQ, object ypQ,
    object U, object xmU, object xpU, object ymU, object ypU,
    object Bx, object By,
    const Py_ssize_t ngh, const unsigned comp, const unsigned ornt,
    const double tol, const double drytol,
    const double theta, const double val,
) except *:

    # runtime check for the shapes
    _checker.shape_checker(Q, xmQ, xpQ, ymQ, ypQ, U, xmU, xpU, ymU, ypU, Bx, By, ngh, comp, ornt)

    if ornt == 0:  # west
        _const_val_bc_set_west(bc, Q, xmQ, xpQ, U, xmU, xpU, Bx, ngh, comp, val)
    elif ornt == 1:  # east
        _const_val_bc_set_east(bc, Q, xmQ, xpQ, U, xmU, xpU, Bx, ngh, comp, val)
    elif ornt == 2:  # south
        _const_val_bc_set_south(bc, Q, ymQ, ypQ, U, ymU, ypU, By, ngh, comp, val)
    elif ornt == 3:  # north
        _const_val_bc_set_north(bc, Q, ymQ, ypQ, U, ymU, ypU, By, ngh, comp, val)
    else:
        raise ValueError(f"orientation id {ornt} not accepted.")

    # store other values
    if ConstValBC is ConstValWH:
        bc.tol = tol
    elif ConstValBC is ConstValOther:
        bc.drytol = drytol

    bc.theta = theta
    bc.val = val


cdef void _const_val_bc_set_west(
    ConstValBC bc,
    object Q, object xmQ, object xpQ,
    object U, object xmU, object xpU,
    object Bx, 
    const Py_ssize_t ngh, const Py_ssize_t comp, const double val
) except *:

    # these should be views into original data buffer
    bc.qc0 = Q[comp, ngh:Q.shape[1]-ngh, ngh]
    bc.qc1 = Q[comp, ngh:Q.shape[1]-ngh, ngh+1]
    bc.qbci = xpQ[comp, :, 0]
    bc.qbco = xmQ[comp, :, 0]
    bc.qother = xmQ[comp, :, 1]

    bc.hbci = xpU[0, :, 0]
    bc.hbco = xmU[0, :, 0]
    bc.hother = xmU[0, :, 1]

    if ConstValBC is ConstValOther:
        bc.ubci = xpU[comp, :, 0]
        bc.ubco = xmU[comp, :, 0]
        bc.uother = xmU[comp, :, 1]
    elif ConstValBC is ConstValWH:
        bc.bbc = Bx[:, 0]
        bc.bother = Bx[:, 1]
        bc.hc0 = U[0, ngh:U.shape[1]-ngh, ngh]
        bc.wbcval = cupy.maximum(bc.bbc, val)
        bc.hbcval = cupy.maximum(bc.wbcval-bc.bbc, 0.0)


cdef void _const_val_bc_set_east(
    ConstValBC bc,
    object Q, object xmQ, object xpQ,
    object U, object xmU, object xpU,
    object Bx, 
    const Py_ssize_t ngh, const Py_ssize_t comp, const double val
) except *:

    # these should be views into original data buffer
    bc.qc0 = Q[comp, ngh:Q.shape[1]-ngh, Q.shape[2]-ngh-1]
    bc.qc1 = Q[comp, ngh:Q.shape[1]-ngh, Q.shape[2]-ngh-2]
    bc.qbci = xmQ[comp, :, xmQ.shape[2]-1]
    bc.qbco = xpQ[comp, :, xpQ.shape[2]-1]
    bc.qother = xpQ[comp, :, xpQ.shape[2]-2]

    bc.hbci = xmU[0, :, xmU.shape[2]-1]
    bc.hbco = xpU[0, :, xpU.shape[2]-1]
    bc.hother = xpU[0, :, xpU.shape[2]-2]

    if ConstValBC is ConstValOther:
        bc.ubci = xmU[comp, :, xmU.shape[2]-1]
        bc.ubco = xpU[comp, :, xpU.shape[2]-1]
        bc.uother = xpU[comp, :, xpU.shape[2]-2]
    elif ConstValBC is ConstValWH:
        bc.bbc = Bx[:, Bx.shape[1]-1]
        bc.bother = Bx[:, Bx.shape[1]-2]
        bc.hc0 = U[0, ngh:U.shape[1]-ngh, U.shape[2]-ngh-1]
        bc.wbcval = cupy.maximum(bc.bbc, val)
        bc.hbcval = cupy.maximum(bc.wbcval-bc.bbc, 0.0)


cdef void _const_val_bc_set_south(
    ConstValBC bc,
    object Q, object ymQ, object ypQ,
    object U, object ymU, object ypU,
    object By, 
    const Py_ssize_t ngh, const Py_ssize_t comp, const double val
) except *:

    # these should be views into original data buffer
    bc.qc0 = Q[comp, ngh, ngh:Q.shape[2]-ngh]
    bc.qc1 = Q[comp, ngh+1, ngh:Q.shape[2]-ngh]
    bc.qbci = ypQ[comp, 0, :]
    bc.qbco = ymQ[comp, 0, :]
    bc.qother = ymQ[comp, 1, :]

    bc.hbci = ypU[0, 0, :]
    bc.hbco = ymU[0, 0, :]
    bc.hother = ymU[0, 1, :]

    if ConstValBC is ConstValOther:
        bc.ubci = ypU[comp, 0, :]
        bc.ubco = ymU[comp, 0, :]
        bc.uother = ymU[comp, 1, :]
    elif ConstValBC is ConstValWH:
        bc.bbc = By[0, :]
        bc.bother = By[1, :]
        bc.hc0 = U[0, ngh, ngh:U.shape[2]-ngh]
        bc.wbcval = cupy.maximum(bc.bbc, val)
        bc.hbcval = cupy.maximum(bc.wbcval-bc.bbc, 0.0)


cdef void _const_val_bc_set_north(
    ConstValBC bc,
    object Q, object ymQ, object ypQ,
    object U, object ymU, object ypU,
    object By, 
    const Py_ssize_t ngh, const Py_ssize_t comp, const double val
) except *:

    # these should be views into original data buffer
    bc.qc0 = Q[comp, Q.shape[1]-ngh-1, ngh:Q.shape[2]-ngh]
    bc.qc1 = Q[comp, Q.shape[1]-ngh-2, ngh:Q.shape[2]-ngh]
    bc.qbci = ymQ[comp, ymQ.shape[1]-1, :]
    bc.qbco = ypQ[comp, ypQ.shape[1]-1, :]
    bc.qother = ypQ[comp, ypQ.shape[1]-2, :]

    bc.hbci = ymU[0, ymU.shape[1]-1, :]
    bc.hbco = ypU[0, ypU.shape[1]-1, :]
    bc.hother = ypU[0, ypU.shape[1]-2, :]

    if ConstValBC is ConstValOther:
        bc.ubci = ymU[comp, ymU.shape[1]-1, :]
        bc.ubco = ypU[comp, ypU.shape[1]-1, :]
        bc.uother = ypU[comp, ypU.shape[1]-2, :]
    elif ConstValBC is ConstValWH:
        bc.bbc = By[By.shape[0]-1, :]
        bc.bother = By[By.shape[0]-2, :]
        bc.hc0 = U[0, U.shape[1]-ngh-1, ngh:U.shape[2]-ngh]
        bc.wbcval = cupy.maximum(bc.bbc, val)
        bc.hbcval = cupy.maximum(bc.wbcval-bc.bbc, 0.0)


def const_val_bc_factory(ornt, comp, states, topo, tol, drytol, theta, val, *args, **kwargs):
    """Factory to create a constant-valued boundary condition callable object.
    """

    # aliases
    cdef object Q = states.Q
    cdef object xmQ = states.face.x.minus.Q
    cdef object xpQ = states.face.x.plus.Q
    cdef object ymQ = states.face.y.minus.Q
    cdef object ypQ = states.face.y.plus.Q

    cdef object U = states.U
    cdef object xmU = states.face.x.minus.U
    cdef object xpU = states.face.x.plus.U
    cdef object ymU = states.face.y.minus.U
    cdef object ypU = states.face.y.plus.U

    cdef object Bx = topo.xfcenters
    cdef object By = topo.yfcenters

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

    if comp == 0:
        bc = ConstValWH(
            Q, xmQ, xpQ, ymQ, ypQ, U, xmU, xpU, ymU, ypU, Bx, By,
            ngh, comp, ornt, tol, drytol, theta, val
        )
    elif comp == 1 or comp == 2:
        bc = ConstValOther(
            Q, xmQ, xpQ, ymQ, ypQ, U, xmU, xpU, ymU, ypU, Bx, By,
            ngh, comp, ornt, tol, drytol, theta, val
        )
    else:
        raise ValueError(f"Unrecognized component: {comp}")

    return bc
