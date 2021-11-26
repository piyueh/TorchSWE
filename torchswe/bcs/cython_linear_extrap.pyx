# vim:fenc=utf-8
# vim:ft=pyrex
cimport _checker
cimport cython


cdef fused LinearExtrapBC:
    LinearExtrapFloat
    LinearExtrapDouble

ctypedef void (*linear_extrap_kernel_double_t)(LinearExtrapDouble, double) nogil except *;
ctypedef void (*linear_extrap_kernel_float_t)(LinearExtrapFloat, float) nogil except *;


@cython.auto_pickle(False)  # meaningless to pickle a BC instance as everything is a memoryview
cdef class LinearExtrapDouble:
    """Base class of linear extraption boundary coditions wiht 64-bit floating points.
    """
    # number of elements
    cdef Py_ssize_t n

    # conservatives
    cdef const double[:] qc0  # q at the cell centers of the 1st internal cell layer
    cdef const double[:] qc1  # q at the cell centers of the 2nd internal cell layer
    cdef double[:] qbci  # q at the inner side of the boundary cell faces
    cdef double[:] qbco  # q at the outer side of the boundary cell faces
    cdef double[:] qother  # q at the inner side of the another face of the 1st internal cell

    # topography elevation
    cdef const double[:] bbc  # topo elevations at the boundary cell faces
    cdef const double[:] bother  # topo elevation at cell faces between the 1st and 2nd internal cell layers

    # depth
    cdef const double[:] hc0  # depth at the cell centers of the 1st internal cell layer
    cdef double[:] hbci  # depth at the inner side of the boundary cell faces
    cdef double[:] hbco  # depth at the outer side of the boundary cell faces
    cdef double[:] hother  # depth at the inner side of the another face of the 1st internal cell

    # velocities
    cdef double[:] ubci  # u or v at the inner side of the boundary cell faces
    cdef double[:] ubco  # u or v at the outer side of the boundary cell faces
    cdef double[:] uother  # u or v at the inner side of the another face of the 1st internal cell

    # tolerance
    cdef double tol  # depths under this tolerance are considered dry cells
    cdef double drytol  # depths under this values are considered wet but still cells

    # kernel
    cdef linear_extrap_kernel_double_t kernel

    def __init__(
        self,
        const double[:, :, ::1] Q,
        double[:, :, ::1] xmQ, double[:, :, ::1] xpQ, double[:, :, ::1] ymQ, double[:, :, ::1] ypQ,
        const double[:, :, ::1] U,
        double[:, :, ::1] xmU, double[:, :, ::1] xpU, double[:, :, ::1] ymU, double[:, :, ::1] ypU,
        const double[:, ::1] Bx, const double[:, ::1] By,
        const Py_ssize_t ngh, const unsigned comp, const unsigned ornt,
        const double tol, const double drytol
    ):
        _linear_extrap_bc_init[LinearExtrapDouble, double](
            self, Q, xmQ, xpQ, ymQ, ypQ, U, xmU, xpU, ymU, ypU,
            Bx, By, ngh, comp, ornt, tol, drytol
        )

    def __call__(self):
        self.kernel(self, 0.0)


@cython.auto_pickle(False)  # meaningless to pickle a BC instance as everything is a memoryview
cdef class LinearExtrapFloat:
    """Base class of linear extraption boundary coditions wiht 32-bit floating points.
    """
    # number of elements
    cdef Py_ssize_t n

    # conservatives
    cdef const float[:] qc0  # q at the cell centers of the 1st internal cell layer
    cdef const float[:] qc1  # q at the cell centers of the 2nd internal cell layer
    cdef float[:] qbci  # q at the inner side of the boundary cell faces
    cdef float[:] qbco  # q at the outer side of the boundary cell faces
    cdef float[:] qother  # q at the inner side of the another face of the 1st internal cell

    # topography elevation
    cdef const float[:] bbc  # topo elevations at the boundary cell faces
    cdef const float[:] bother  # topo elevation at cell faces between the 1st and 2nd internal cell layers

    # depth
    cdef const float[:] hc0  # depth at the cell centers of the 1st internal cell layer
    cdef float[:] hbci  # depth at the inner side of the boundary cell faces
    cdef float[:] hbco  # depth at the outer side of the boundary cell faces
    cdef float[:] hother  # depth at the inner side of the another face of the 1st internal cell

    # velocities
    cdef float[:] ubci  # u or v at the inner side of the boundary cell faces
    cdef float[:] ubco  # u or v at the outer side of the boundary cell faces
    cdef float[:] uother  # u or v at the inner side of the another face of the 1st internal cell

    # tolerance
    cdef float tol  # depths under this tolerance are considered dry cells
    cdef float drytol  # depths under this values are considered wet but still cells

    # kernel
    cdef linear_extrap_kernel_float_t kernel

    def __init__(
        self,
        const float[:, :, ::1] Q,
        float[:, :, ::1] xmQ, float[:, :, ::1] xpQ, float[:, :, ::1] ymQ, float[:, :, ::1] ypQ,
        const float[:, :, ::1] U,
        float[:, :, ::1] xmU, float[:, :, ::1] xpU, float[:, :, ::1] ymU, float[:, :, ::1] ypU,
        const float[:, ::1] Bx, const float[:, ::1] By,
        const Py_ssize_t ngh, const unsigned comp, const unsigned ornt,
        const float tol, const float drytol
    ):
        _linear_extrap_bc_init[LinearExtrapFloat, float](
            self, Q, xmQ, xpQ, ymQ, ypQ, U, xmU, xpU, ymU, ypU,
            Bx, By, ngh, comp, ornt, tol, drytol
        )

    def __call__(self):
        self.kernel(self, 0.0)


cdef void _linear_extrap_bc_init(
    LinearExtrapBC bc,
    const cython.floating[:, :, ::1] Q,
    cython.floating[:, :, ::1] xmQ, cython.floating[:, :, ::1] xpQ,
    cython.floating[:, :, ::1] ymQ, cython.floating[:, :, ::1] ypQ,
    const cython.floating[:, :, ::1] U,
    cython.floating[:, :, ::1] xmU, cython.floating[:, :, ::1] xpU,
    cython.floating[:, :, ::1] ymU, cython.floating[:, :, ::1] ypU,
    const cython.floating[:, ::1] Bx, const cython.floating[:, ::1] By,
    const Py_ssize_t ngh, const unsigned comp, const unsigned ornt,
    const cython.floating tol, const cython.floating drytol
) nogil except *:

    # the first two combinations will be pruned by cython when translating to C/C++
    if cython.floating is float and LinearExtrapBC is LinearExtrapDouble:
        raise RuntimeError("Mismatched types")
    elif cython.floating is double and LinearExtrapBC is LinearExtrapFloat:
        raise RuntimeError("Mismatched types")
    else:

        # runtime check for the shapes
        _checker.shape_checker_memoryview[cython.floating](
            Q, xmQ, xpQ, ymQ, ypQ, U, xmU, xpU, ymU, ypU, Bx, By, ngh, comp, ornt)

        if ornt == 0 or ornt == 1:  # west or east
            bc.n = Q.shape[1] - 2 * ngh  # ny
        elif ornt == 2 or ornt == 3:  # west or east
            bc.n = Q.shape[2] - 2 * ngh  # nx
        else:
            raise ValueError(f"`ornt` should be >= 0 and <= 3: {ornt}")

        if ornt == 0:  # west
            _linear_extrap_bc_set_west[LinearExtrapBC, cython.floating](
                bc, Q, xmQ, xpQ, U, xmU, xpU, Bx, ngh, comp)
        elif ornt == 1:  # east
            _linear_extrap_bc_set_east[LinearExtrapBC, cython.floating](
                bc, Q, xmQ, xpQ, U, xmU, xpU, Bx, ngh, comp)
        elif ornt == 2:  # south
            _linear_extrap_bc_set_south[LinearExtrapBC, cython.floating](
                bc, Q, ymQ, ypQ, U, ymU, ypU, By, ngh, comp)
        elif ornt == 3:  # north
            _linear_extrap_bc_set_north[LinearExtrapBC, cython.floating](
                bc, Q, ymQ, ypQ, U, ymU, ypU, By, ngh, comp)
        else:
            raise ValueError(f"orientation id {ornt} not accepted.")

        if comp == 0:
            bc.kernel = _linear_extrap_bc_w_h_kernel[LinearExtrapBC, cython.floating]
        elif comp <= 2:
            bc.kernel = _linear_extrap_bc_kernel[LinearExtrapBC, cython.floating]
        else:
            raise ValueError(f"component id {comp} not accepted.")

        bc.tol = tol
        bc.drytol = drytol


cdef void _linear_extrap_bc_w_h_kernel(LinearExtrapBC bc, cython.floating delta) nogil except *:
    cdef Py_ssize_t i
    for i in range(bc.n):

        if bc.hc0[i] < bc.tol:
            bc.hbci[i] = 0.0;
            bc.hbco[i] = 0.0;
            bc.hother[i] = 0.0;
            bc.qbci[i] = bc.bbc[i];
            bc.qbco[i] = bc.bbc[i];
            bc.qother[i] = bc.bother[i];
            continue

        delta = (bc.qc0[i] - bc.qc1[i]) / 2.0;  # dw
        bc.hbci[i] = bc.qc0[i] + delta - bc.bbc[i];
        bc.hother[i] = bc.qc0[i] - delta - bc.bother[i];

        if bc.hbci[i] < bc.tol:
            bc.hbci[i] = 0.0;
            bc.hother[i] = bc.hc0[i] * 2.0;
        elif bc.hother[i] < bc.tol:
            bc.hbci[i] = bc.hc0[i] * 2.0;
            bc.hother[i] = 0.0;

        bc.hbco[i] = bc.hbci[i]

        #reconstruct to eliminate rounding error-edffect in further calculations
        bc.qbci[i] = bc.hbci[i] + bc.bbc[i];
        bc.qbco[i] = bc.hbco[i] + bc.bbc[i];
        bc.qother[i] = bc.hother[i] + bc.bother[i];


cdef void _linear_extrap_bc_kernel(LinearExtrapBC bc, cython.floating delta) nogil except *:
    cdef Py_ssize_t i
    for i in range(bc.n):

        delta = (bc.qc0[i] - bc.qc1[i]) / 2.0;

        if bc.hbco[i] < bc.drytol:
            bc.ubco[i] = 0.0;
            bc.qbco[i] = 0.0;
        else:
            bc.ubco[i] = (bc.qc0[i] + delta) / bc.hbco[i];
            bc.qbco[i] = bc.hbco[i] * bc.ubco[i];

        if bc.hbci[i] < bc.drytol:
            bc.ubci[i] = 0.0;
            bc.qbci[i] = 0.0;
        else:
            bc.ubci[i] = (bc.qc0[i] + delta) / bc.hbci[i];
            bc.qbci[i] = bc.hbci[i] * bc.ubci[i];

        if bc.hother[i] < bc.drytol:
            bc.uother[i] = 0.0;
            bc.qother[i] = 0.0;
        else:
            bc.uother[i] = (bc.qc0[i] - delta) / bc.hother[i];
            bc.qother[i] = bc.hother[i] * bc.uother[i];


cdef void _linear_extrap_bc_set_west(
    LinearExtrapBC bc,
    const cython.floating[:, :, ::1] Q, cython.floating[:, :, ::1] xmQ, cython.floating[:, :, ::1] xpQ,
    const cython.floating[:, :, ::1] U, cython.floating[:, :, ::1] xmU, cython.floating[:, :, ::1] xpU,
    const cython.floating[:, ::1] Bx,
    const Py_ssize_t ngh, const Py_ssize_t comp
) nogil except *:

    # the first two combinations will be pruned by cython when translating to C/C++
    if cython.floating is float and LinearExtrapBC is LinearExtrapDouble:
        raise RuntimeError("Mismatched types")
    elif cython.floating is double and LinearExtrapBC is LinearExtrapFloat:
        raise RuntimeError("Mismatched types")
    else:

        # these should be views into original data buffer
        bc.qc0 = Q[comp, ngh:Q.shape[1]-ngh, ngh]
        bc.qc1 = Q[comp, ngh:Q.shape[1]-ngh, ngh+1]
        bc.qbci = xpQ[comp, :, 0]
        bc.qbco = xmQ[comp, :, 0]
        bc.qother = xmQ[comp, :, 1]

        bc.hc0 = U[0, ngh:U.shape[1]-ngh, ngh]
        bc.hbci = xpU[0, :, 0]
        bc.hbco = xmU[0, :, 0]
        bc.hother = xmU[0, :, 1]

        bc.bbc = Bx[:, 0]
        bc.bother = Bx[:, 1]

        if comp != 0:
            bc.ubci = xpU[comp, :, 0]
            bc.ubco = xmU[comp, :, 0]
            bc.uother = xmU[comp, :, 1]
        else:
            bc.ubci = None
            bc.ubco = None
            bc.uother = None


cdef void _linear_extrap_bc_set_east(
    LinearExtrapBC bc,
    const cython.floating[:, :, ::1] Q, cython.floating[:, :, ::1] xmQ, cython.floating[:, :, ::1] xpQ,
    const cython.floating[:, :, ::1] U, cython.floating[:, :, ::1] xmU, cython.floating[:, :, ::1] xpU,
    const cython.floating[:, ::1] Bx,
    const Py_ssize_t ngh, const Py_ssize_t comp
) nogil except *:

    # the first two combinations will be pruned by cython when translating to C/C++
    if cython.floating is float and LinearExtrapBC is LinearExtrapDouble:
        raise RuntimeError("Mismatched types")
    elif cython.floating is double and LinearExtrapBC is LinearExtrapFloat:
        raise RuntimeError("Mismatched types")
    else:

        # these should be views into original data buffer
        bc.qc0 = Q[comp, ngh:Q.shape[1]-ngh, Q.shape[2]-ngh-1]
        bc.qc1 = Q[comp, ngh:Q.shape[1]-ngh, Q.shape[2]-ngh-2]
        bc.qbci = xmQ[comp, :, xmQ.shape[2]-1]
        bc.qbco = xpQ[comp, :, xpQ.shape[2]-1]
        bc.qother = xpQ[comp, :, xpQ.shape[2]-2]

        bc.hc0 = U[0, ngh:U.shape[1]-ngh, U.shape[2]-ngh-1]
        bc.hbci = xmU[0, :, xmU.shape[2]-1]
        bc.hbco = xpU[0, :, xpU.shape[2]-1]
        bc.hother = xpU[0, :, xpU.shape[2]-2]

        bc.bbc = Bx[:, Bx.shape[1]-1]
        bc.bother = Bx[:, Bx.shape[1]-2]

        if comp != 0:
            bc.ubci = xmU[comp, :, xmU.shape[2]-1]
            bc.ubco = xpU[comp, :, xpU.shape[2]-1]
            bc.uother = xpU[comp, :, xpU.shape[2]-2]
        else:
            bc.ubci = None
            bc.ubco = None
            bc.uother = None


cdef void _linear_extrap_bc_set_south(
    LinearExtrapBC bc,
    const cython.floating[:, :, ::1] Q, cython.floating[:, :, ::1] ymQ, cython.floating[:, :, ::1] ypQ,
    const cython.floating[:, :, ::1] U, cython.floating[:, :, ::1] ymU, cython.floating[:, :, ::1] ypU,
    const cython.floating[:, ::1] By,
    const Py_ssize_t ngh, const Py_ssize_t comp
) nogil except *:

    # the first two combinations will be pruned by cython when translating to C/C++
    if cython.floating is float and LinearExtrapBC is LinearExtrapDouble:
        raise RuntimeError("Mismatched types")
    elif cython.floating is double and LinearExtrapBC is LinearExtrapFloat:
        raise RuntimeError("Mismatched types")
    else:

        # these should be views into original data buffer
        bc.qc0 = Q[comp, ngh, ngh:Q.shape[2]-ngh]
        bc.qc1 = Q[comp, ngh+1, ngh:Q.shape[2]-ngh]
        bc.qbci = ypQ[comp, 0, :]
        bc.qbco = ymQ[comp, 0, :]
        bc.qother = ymQ[comp, 1, :]

        bc.hc0 = U[0, ngh, ngh:U.shape[2]-ngh]
        bc.hbci = ypU[0, 0, :]
        bc.hbco = ymU[0, 0, :]
        bc.hother = ymU[0, 1, :]

        bc.bbc = By[0, :]
        bc.bother = By[1, :]

        if comp != 0:
            bc.ubci = ypU[comp, 0, :]
            bc.ubco = ymU[comp, 0, :]
            bc.uother = ymU[comp, 1, :]
        else:
            bc.ubci = None
            bc.ubco = None
            bc.uother = None


cdef void _linear_extrap_bc_set_north(
    LinearExtrapBC bc,
    const cython.floating[:, :, ::1] Q, cython.floating[:, :, ::1] ymQ, cython.floating[:, :, ::1] ypQ,
    const cython.floating[:, :, ::1] U, cython.floating[:, :, ::1] ymU, cython.floating[:, :, ::1] ypU,
    const cython.floating[:, ::1] By,
    const Py_ssize_t ngh, const Py_ssize_t comp
) nogil except *:

    # the first two combinations will be pruned by cython when translating to C/C++
    if cython.floating is float and LinearExtrapBC is LinearExtrapDouble:
        raise RuntimeError("Mismatched types")
    elif cython.floating is double and LinearExtrapBC is LinearExtrapFloat:
        raise RuntimeError("Mismatched types")
    else:

        # these should be views into original data buffer
        bc.qc0 = Q[comp, Q.shape[1]-ngh-1, ngh:Q.shape[2]-ngh]
        bc.qc1 = Q[comp, Q.shape[1]-ngh-2, ngh:Q.shape[2]-ngh]
        bc.qbci = ymQ[comp, ymQ.shape[1]-1, :]
        bc.qbco = ypQ[comp, ypQ.shape[1]-1, :]
        bc.qother = ypQ[comp, ypQ.shape[1]-2, :]

        bc.hc0 = U[0, U.shape[1]-ngh-1, ngh:U.shape[2]-ngh]
        bc.hbci = ymU[0, ymU.shape[1]-1, :]
        bc.hbco = ypU[0, ypU.shape[1]-1, :]
        bc.hother = ypU[0, ypU.shape[1]-2, :]

        bc.bbc = By[By.shape[0]-1, :]
        bc.bother = By[By.shape[0]-2, :]

        if comp != 0:
            bc.ubci = ymU[comp, ymU.shape[1]-1, :]
            bc.ubco = ypU[comp, ypU.shape[1]-1, :]
            bc.uother = ypU[comp, ypU.shape[1]-2, :]
        else:
            bc.ubci = None
            bc.ubco = None
            bc.uother = None


def linear_extrap_bc_factory(ornt, comp, states, topo, tol, drytol, *args, **kwargs):
    """Factory to create a linear extrapolation boundary condition callable object.
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
    cdef object dtype = Q.dtype

    if isinstance(ornt, str):
        if ornt in ["w", "west"]:
            ornt = 0
        elif ornt in ["e", "east"]:
            ornt = 1
        elif ornt in ["s", "south"]:
            ornt = 2
        elif ornt in ["n", "north"]:
            ornt = 3

    if dtype == "float64":
        bc = LinearExtrapDouble(
            Q, xmQ, xpQ, ymQ, ypQ, U, xmU, xpU, ymU, ypU, Bx, By, ngh, comp, ornt, tol, drytol)
    elif dtype == "float32":
        bc = LinearExtrapFloat(
            Q, xmQ, xpQ, ymQ, ypQ, U, xmU, xpU, ymU, ypU, Bx, By, ngh, comp, ornt, tol, drytol)
    else:
        raise ValueError(f"Unrecognized data type: {dtype}")

    return bc
