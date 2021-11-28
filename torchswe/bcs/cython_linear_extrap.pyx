# vim:fenc=utf-8
# vim:ft=pyrex
cimport _checker
cimport cython


cdef fused LinearExtrapBC:
    LinearExtrapFloatWH
    LinearExtrapDoubleWH
    LinearExtrapFloatOther
    LinearExtrapDoubleOther


ctypedef fused LinearExtrapWHBC:
    LinearExtrapFloatWH
    LinearExtrapDoubleWH


ctypedef fused LinearExtrapOtherBC:
    LinearExtrapFloatOther
    LinearExtrapDoubleOther


@cython.auto_pickle(False)  # meaningless to pickle a BC instance as everything is a memoryview
cdef class LinearExtrapDoubleWH:
    """Linear extraption boundary coditions wiht 64-bit floating points for w and h.
    """
    # number of elements
    cdef Py_ssize_t n

    # conservatives
    cdef const double[:] wc0  # q at the cell centers of the 1st internal cell layer
    cdef const double[:] wc1  # q at the cell centers of the 2nd internal cell layer
    cdef double[:] wbci  # q at the inner side of the boundary cell faces
    cdef double[:] wbco  # q at the outer side of the boundary cell faces
    cdef double[:] wother  # q at the inner side of the another face of the 1st internal cell

    # topography elevation
    cdef const double[:] bbc  # topo elevations at the boundary cell faces
    cdef const double[:] bother  # topo elevation at cell faces between the 1st and 2nd internal cell layers

    # depth
    cdef const double[:] hc0  # depth at the cell centers of the 1st internal cell layer
    cdef double[:] hbci  # depth at the inner side of the boundary cell faces
    cdef double[:] hbco  # depth at the outer side of the boundary cell faces
    cdef double[:] hother  # depth at the inner side of the another face of the 1st internal cell

    # tolerance
    cdef double tol  # depths under this tolerance are considered dry cells

    def __call__(self):
        _linear_extrap_bc_w_h_kernel[LinearExtrapDoubleWH, double](self, 0.0)


@cython.auto_pickle(False)  # meaningless to pickle a BC instance as everything is a memoryview
cdef class LinearExtrapDoubleOther:
    """Linear extraption boundary coditions wiht 64-bit floating points for hu or hv.
    """
    # number of elements
    cdef Py_ssize_t n

    # conservatives
    cdef double[:] qbci  # hu or hv at the inner side of the boundary cell faces
    cdef double[:] qbco  # hu or hv at the outer side of the boundary cell faces
    cdef double[:] qother  # hu or hv at the inner side of the another face of the 1st internal cell

    # depth
    cdef const double[:] hbci  # depth at the inner side of the boundary cell faces
    cdef const double[:] hbco  # depth at the outer side of the boundary cell faces
    cdef const double[:] hother  # depth at the inner side of the another face of the 1st internal cell

    # velocities
    cdef const double[:] uc0  # u or v at the cell centers of the 1st internal cell layer
    cdef const double[:] uc1  # u or v at the cell centers of the 2nd internal cell layer
    cdef double[:] ubci  # u or v at the inner side of the boundary cell faces
    cdef double[:] ubco  # u or v at the outer side of the boundary cell faces
    cdef double[:] uother  # u or v at the inner side of the another face of the 1st internal cell

    # tolerance
    cdef double drytol  # depths under this values are considered wet but still cells

    def __call__(self):
        _linear_extrap_bc_kernel[LinearExtrapDoubleOther, double](self, 0.0)


@cython.auto_pickle(False)  # meaningless to pickle a BC instance as everything is a memoryview
cdef class LinearExtrapFloatWH:
    """Linear extraption boundary coditions wiht 64-bit floating points for w and h.
    """
    # number of elements
    cdef Py_ssize_t n

    # conservatives
    cdef const float[:] wc0  # q at the cell centers of the 1st internal cell layer
    cdef const float[:] wc1  # q at the cell centers of the 2nd internal cell layer
    cdef float[:] wbci  # q at the inner side of the boundary cell faces
    cdef float[:] wbco  # q at the outer side of the boundary cell faces
    cdef float[:] wother  # q at the inner side of the another face of the 1st internal cell

    # topography elevation
    cdef const float[:] bbc  # topo elevations at the boundary cell faces
    cdef const float[:] bother  # topo elevation at cell faces between the 1st and 2nd internal cell layers

    # depth
    cdef const float[:] hc0  # depth at the cell centers of the 1st internal cell layer
    cdef float[:] hbci  # depth at the inner side of the boundary cell faces
    cdef float[:] hbco  # depth at the outer side of the boundary cell faces
    cdef float[:] hother  # depth at the inner side of the another face of the 1st internal cell

    # tolerance
    cdef float tol  # depths under this tolerance are considered dry cells

    def __call__(self):
        _linear_extrap_bc_w_h_kernel[LinearExtrapFloatWH, float](self, 0.0)


@cython.auto_pickle(False)  # meaningless to pickle a BC instance as everything is a memoryview
cdef class LinearExtrapFloatOther:
    """Linear extraption boundary coditions wiht 64-bit floating points for hu or hv.
    """
    # number of elements
    cdef Py_ssize_t n

    # conservatives
    cdef float[:] qbci  # hu or hv at the inner side of the boundary cell faces
    cdef float[:] qbco  # hu or hv at the outer side of the boundary cell faces
    cdef float[:] qother  # hu or hv at the inner side of the another face of the 1st internal cell

    # depth
    cdef const float[:] hbci  # depth at the inner side of the boundary cell faces
    cdef const float[:] hbco  # depth at the outer side of the boundary cell faces
    cdef const float[:] hother  # depth at the inner side of the another face of the 1st internal cell

    # velocities
    cdef const float[:] uc0  # u or v at the cell centers of the 1st internal cell layer
    cdef const float[:] uc1  # u or v at the cell centers of the 2nd internal cell layer
    cdef float[:] ubci  # u or v at the inner side of the boundary cell faces
    cdef float[:] ubco  # u or v at the outer side of the boundary cell faces
    cdef float[:] uother  # u or v at the inner side of the another face of the 1st internal cell

    # tolerance
    cdef float drytol  # depths under this values are considered wet but still cells

    def __call__(self):
        _linear_extrap_bc_kernel[LinearExtrapFloatOther, float](self, 0.0)


cdef void _linear_extrap_bc_w_h_kernel(LinearExtrapWHBC bc, cython.floating delta) nogil except *:
    cdef Py_ssize_t i
    for i in range(bc.n):

        if bc.hc0[i] < bc.tol:
            bc.hbci[i] = 0.0;
            bc.hbco[i] = 0.0;
            bc.hother[i] = 0.0;
            bc.wbci[i] = bc.bbc[i];
            bc.wbco[i] = bc.bbc[i];
            bc.wother[i] = bc.bother[i];
            continue

        delta = (bc.wc0[i] - bc.wc1[i]) / 2.0;  # dw
        bc.hbci[i] = bc.wc0[i] + delta - bc.bbc[i];
        bc.hother[i] = bc.wc0[i] - delta - bc.bother[i];

        if bc.hbci[i] < bc.tol:
            bc.hbci[i] = 0.0;
            bc.hother[i] = bc.hc0[i] * 2.0;
        elif bc.hother[i] < bc.tol:
            bc.hbci[i] = bc.hc0[i] * 2.0;
            bc.hother[i] = 0.0;

        bc.hbco[i] = bc.hbci[i]

        #reconstruct to eliminate rounding error-edffect in further calculations
        bc.wbci[i] = bc.hbci[i] + bc.bbc[i];
        bc.wbco[i] = bc.hbco[i] + bc.bbc[i];
        bc.wother[i] = bc.hother[i] + bc.bother[i];


cdef void _linear_extrap_bc_kernel(LinearExtrapOtherBC bc, cython.floating delta) nogil except *:
    cdef Py_ssize_t i
    for i in range(bc.n):

        delta = (bc.uc0[i] - bc.uc1[i]) / 2.0;

        if bc.hbco[i] < bc.drytol:
            bc.ubco[i] = 0.0;
            bc.qbco[i] = 0.0;
        else:
            bc.ubco[i] = bc.uc0[i] + delta;
            bc.qbco[i] = bc.hbco[i] * bc.ubco[i];

        if bc.hbci[i] < bc.drytol:
            bc.ubci[i] = 0.0;
            bc.qbci[i] = 0.0;
        else:
            bc.ubci[i] = bc.uc0[i] + delta;
            bc.qbci[i] = bc.hbci[i] * bc.ubci[i];

        if bc.hother[i] < bc.drytol:
            bc.uother[i] = 0.0;
            bc.qother[i] = 0.0;
        else:
            bc.uother[i] = bc.uc0[i] - delta;
            bc.qother[i] = bc.hother[i] * bc.uother[i];


cdef inline void _linear_extrap_bc_set_west(
    LinearExtrapBC bc,
    const cython.floating[:, :, ::1] Q, cython.floating[:, :, ::1] xmQ, cython.floating[:, :, ::1] xpQ,
    const cython.floating[:, :, ::1] U, cython.floating[:, :, ::1] xmU, cython.floating[:, :, ::1] xpU,
    const cython.floating[:, ::1] Bx,
    const Py_ssize_t ngh, const Py_ssize_t comp,
    const cython.floating tol, const cython.floating drytol
) nogil except *:

    # the first two combinations will be pruned by cython when translating to C/C++
    if (
        cython.floating is float and (
            LinearExtrapBC is LinearExtrapDoubleWH or LinearExtrapBC is LinearExtrapDoubleOther)
    ):
        raise TypeError("Mismatched types")
    elif (
        cython.floating is double and (
            LinearExtrapBC is LinearExtrapFloatWH or LinearExtrapBC is LinearExtrapFloatOther)
    ):
        raise TypeError("Mismatched types")
    else:

        bc.hbci = xpU[0, :, 0]
        bc.hbco = xmU[0, :, 0]
        bc.hother = xmU[0, :, 1]

        if LinearExtrapBC is LinearExtrapDoubleWH or LinearExtrapBC is LinearExtrapFloatWH:
            bc.wc0 = Q[0, ngh:Q.shape[1]-ngh, ngh]
            bc.wc1 = Q[0, ngh:Q.shape[1]-ngh, ngh+1]
            bc.wbci = xpQ[0, :, 0]
            bc.wbco = xmQ[0, :, 0]
            bc.wother = xmQ[0, :, 1]

            bc.hc0 = U[0, ngh:U.shape[1]-ngh, ngh]

            bc.bbc = Bx[:, 0]
            bc.bother = Bx[:, 1]

            bc.tol = tol
        else:
            bc.qbci = xpQ[comp, :, 0]
            bc.qbco = xmQ[comp, :, 0]
            bc.qother = xmQ[comp, :, 1]

            bc.uc0 = U[comp, ngh:Q.shape[1]-ngh, ngh]
            bc.uc1 = U[comp, ngh:Q.shape[1]-ngh, ngh+1]
            bc.ubci = xpU[comp, :, 0]
            bc.ubco = xmU[comp, :, 0]
            bc.uother = xmU[comp, :, 1]

            bc.drytol = drytol


cdef inline void _linear_extrap_bc_set_east(
    LinearExtrapBC bc,
    const cython.floating[:, :, ::1] Q, cython.floating[:, :, ::1] xmQ, cython.floating[:, :, ::1] xpQ,
    const cython.floating[:, :, ::1] U, cython.floating[:, :, ::1] xmU, cython.floating[:, :, ::1] xpU,
    const cython.floating[:, ::1] Bx,
    const Py_ssize_t ngh, const Py_ssize_t comp,
    const cython.floating tol, const cython.floating drytol
) nogil except *:

    # the first two combinations will be pruned by cython when translating to C/C++
    if (
        cython.floating is float and (
            LinearExtrapBC is LinearExtrapDoubleWH or LinearExtrapBC is LinearExtrapDoubleOther)
    ):
        raise TypeError("Mismatched types")
    elif (
        cython.floating is double and (
            LinearExtrapBC is LinearExtrapFloatWH or LinearExtrapBC is LinearExtrapFloatOther)
    ):
        raise TypeError("Mismatched types")
    else:

        bc.hbci = xmU[0, :, xmU.shape[2]-1]
        bc.hbco = xpU[0, :, xpU.shape[2]-1]
        bc.hother = xpU[0, :, xpU.shape[2]-2]

        if LinearExtrapBC is LinearExtrapDoubleWH or LinearExtrapBC is LinearExtrapFloatWH:
            bc.wc0 = Q[0, ngh:Q.shape[1]-ngh, Q.shape[2]-ngh-1]
            bc.wc1 = Q[0, ngh:Q.shape[1]-ngh, Q.shape[2]-ngh-2]
            bc.wbci = xmQ[0, :, xmQ.shape[2]-1]
            bc.wbco = xpQ[0, :, xpQ.shape[2]-1]
            bc.wother = xpQ[0, :, xpQ.shape[2]-2]

            bc.hc0 = U[0, ngh:U.shape[1]-ngh, U.shape[2]-ngh-1]

            bc.bbc = Bx[:, Bx.shape[1]-1]
            bc.bother = Bx[:, Bx.shape[1]-2]

            bc.tol = tol
        else:
            bc.qbci = xmQ[comp, :, xmQ.shape[2]-1]
            bc.qbco = xpQ[comp, :, xpQ.shape[2]-1]
            bc.qother = xpQ[comp, :, xpQ.shape[2]-2]

            bc.uc0 = U[comp, ngh:Q.shape[1]-ngh, Q.shape[2]-ngh-1]
            bc.uc1 = U[comp, ngh:Q.shape[1]-ngh, Q.shape[2]-ngh-2]
            bc.ubci = xmU[comp, :, xmU.shape[2]-1]
            bc.ubco = xpU[comp, :, xpU.shape[2]-1]
            bc.uother = xpU[comp, :, xpU.shape[2]-2]

            bc.drytol = drytol


cdef inline void _linear_extrap_bc_set_south(
    LinearExtrapBC bc,
    const cython.floating[:, :, ::1] Q, cython.floating[:, :, ::1] ymQ, cython.floating[:, :, ::1] ypQ,
    const cython.floating[:, :, ::1] U, cython.floating[:, :, ::1] ymU, cython.floating[:, :, ::1] ypU,
    const cython.floating[:, ::1] By,
    const Py_ssize_t ngh, const Py_ssize_t comp,
    const cython.floating tol, const cython.floating drytol
) nogil except *:

    # the first two combinations will be pruned by cython when translating to C/C++
    if (
        cython.floating is float and (
            LinearExtrapBC is LinearExtrapDoubleWH or LinearExtrapBC is LinearExtrapDoubleOther)
    ):
        raise TypeError("Mismatched types")
    elif (
        cython.floating is double and (
            LinearExtrapBC is LinearExtrapFloatWH or LinearExtrapBC is LinearExtrapFloatOther)
    ):
        raise TypeError("Mismatched types")
    else:
        bc.hbci = ypU[0, 0, :]
        bc.hbco = ymU[0, 0, :]
        bc.hother = ymU[0, 1, :]

        if LinearExtrapBC is LinearExtrapDoubleWH or LinearExtrapBC is LinearExtrapFloatWH:
            bc.wc0 = Q[0, ngh, ngh:Q.shape[2]-ngh]
            bc.wc1 = Q[0, ngh+1, ngh:Q.shape[2]-ngh]
            bc.wbci = ypQ[0, 0, :]
            bc.wbco = ymQ[0, 0, :]
            bc.wother = ymQ[0, 1, :]

            bc.hc0 = U[0, ngh, ngh:U.shape[2]-ngh]

            bc.bbc = By[0, :]
            bc.bother = By[1, :]

            bc.tol = tol
        else:
            bc.qbci = ypQ[comp, 0, :]
            bc.qbco = ymQ[comp, 0, :]
            bc.qother = ymQ[comp, 1, :]

            bc.uc0 = U[comp, ngh, ngh:Q.shape[2]-ngh]
            bc.uc1 = U[comp, ngh+1, ngh:Q.shape[2]-ngh]
            bc.ubci = ypU[comp, 0, :]
            bc.ubco = ymU[comp, 0, :]
            bc.uother = ymU[comp, 1, :]

            bc.drytol = drytol


cdef inline void _linear_extrap_bc_set_north(
    LinearExtrapBC bc,
    const cython.floating[:, :, ::1] Q, cython.floating[:, :, ::1] ymQ, cython.floating[:, :, ::1] ypQ,
    const cython.floating[:, :, ::1] U, cython.floating[:, :, ::1] ymU, cython.floating[:, :, ::1] ypU,
    const cython.floating[:, ::1] By,
    const Py_ssize_t ngh, const Py_ssize_t comp,
    const cython.floating tol, const cython.floating drytol
) nogil except *:

    # the first two combinations will be pruned by cython when translating to C/C++
    if (
        cython.floating is float and (
            LinearExtrapBC is LinearExtrapDoubleWH or LinearExtrapBC is LinearExtrapDoubleOther)
    ):
        raise TypeError("Mismatched types")
    elif (
        cython.floating is double and (
            LinearExtrapBC is LinearExtrapFloatWH or LinearExtrapBC is LinearExtrapFloatOther)
    ):
        raise TypeError("Mismatched types")
    else:

        bc.hbci = ymU[0, ymU.shape[1]-1, :]
        bc.hbco = ypU[0, ypU.shape[1]-1, :]
        bc.hother = ypU[0, ypU.shape[1]-2, :]

        if LinearExtrapBC is LinearExtrapDoubleWH or LinearExtrapBC is LinearExtrapFloatWH:
            bc.wc0 = Q[0, Q.shape[1]-ngh-1, ngh:Q.shape[2]-ngh]
            bc.wc1 = Q[0, Q.shape[1]-ngh-2, ngh:Q.shape[2]-ngh]
            bc.wbci = ymQ[0, ymQ.shape[1]-1, :]
            bc.wbco = ypQ[0, ypQ.shape[1]-1, :]
            bc.wother = ypQ[0, ypQ.shape[1]-2, :]

            bc.hc0 = U[0, U.shape[1]-ngh-1, ngh:U.shape[2]-ngh]

            bc.bbc = By[By.shape[0]-1, :]
            bc.bother = By[By.shape[0]-2, :]

            bc.tol = tol
        else:
            bc.qbci = ymQ[comp, ymQ.shape[1]-1, :]
            bc.qbco = ypQ[comp, ypQ.shape[1]-1, :]
            bc.qother = ypQ[comp, ypQ.shape[1]-2, :]

            bc.uc0 = U[comp, Q.shape[1]-ngh-1, ngh:Q.shape[2]-ngh]
            bc.uc1 = U[comp, Q.shape[1]-ngh-2, ngh:Q.shape[2]-ngh]
            bc.ubci = ymU[comp, ymU.shape[1]-1, :]
            bc.ubco = ypU[comp, ypU.shape[1]-1, :]
            bc.uother = ypU[comp, ypU.shape[1]-2, :]

            bc.drytol = drytol


cdef inline void _linear_extrap_bc_factory(
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
    if (
        cython.floating is float and (
            LinearExtrapBC is LinearExtrapDoubleWH or LinearExtrapBC is LinearExtrapDoubleOther)
    ):
        raise TypeError("Mismatched types")
    elif (
        cython.floating is double and (
            LinearExtrapBC is LinearExtrapFloatWH or LinearExtrapBC is LinearExtrapFloatOther)
    ):
        raise TypeError("Mismatched types")
    else:

        # runtime check for the shapes
        _checker.shape_checker_memoryview[cython.floating](
            Q, xmQ, xpQ, ymQ, ypQ, U, xmU, xpU, ymU, ypU, Bx, By, ngh, comp, ornt)

        if ornt == 0:  # west
            bc.n = Q.shape[1] - 2 * ngh  # ny
            _linear_extrap_bc_set_west[LinearExtrapBC, cython.floating](
                bc, Q, xmQ, xpQ, U, xmU, xpU, Bx, ngh, comp, tol, drytol)
        elif ornt == 1:  # east
            bc.n = Q.shape[1] - 2 * ngh  # ny
            _linear_extrap_bc_set_east[LinearExtrapBC, cython.floating](
                bc, Q, xmQ, xpQ, U, xmU, xpU, Bx, ngh, comp, tol, drytol)
        elif ornt == 2:  # south
            bc.n = Q.shape[2] - 2 * ngh  # nx
            _linear_extrap_bc_set_south[LinearExtrapBC, cython.floating](
                bc, Q, ymQ, ypQ, U, ymU, ypU, By, ngh, comp, tol, drytol)
        elif ornt == 3:  # north
            bc.n = Q.shape[2] - 2 * ngh  # nx
            _linear_extrap_bc_set_north[LinearExtrapBC, cython.floating](
                bc, Q, ymQ, ypQ, U, ymU, ypU, By, ngh, comp, tol, drytol)
        else:
            raise ValueError(f"orientation id {ornt} not accepted.")


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
    cdef str dtype = str(Q.dtype)

    if isinstance(ornt, str):
        if ornt in ["w", "west"]:
            ornt = 0
        elif ornt in ["e", "east"]:
            ornt = 1
        elif ornt in ["s", "south"]:
            ornt = 2
        elif ornt in ["n", "north"]:
            ornt = 3

    options = {
        (0, "float64"): (LinearExtrapDoubleWH, _linear_extrap_bc_factory[LinearExtrapDoubleWH, double]),
        (1, "float64"): (LinearExtrapDoubleOther, _linear_extrap_bc_factory[LinearExtrapDoubleOther, double]),
        (2, "float64"): (LinearExtrapDoubleOther, _linear_extrap_bc_factory[LinearExtrapDoubleOther, double]),
        (0, "float32"): (LinearExtrapFloatWH, _linear_extrap_bc_factory[LinearExtrapFloatWH, float]),
        (1, "float32"): (LinearExtrapFloatOther, _linear_extrap_bc_factory[LinearExtrapFloatOther, float]),
        (2, "float32"): (LinearExtrapFloatOther, _linear_extrap_bc_factory[LinearExtrapFloatOther, float]),
    }

    bc = options[comp, dtype][0]()

    options[comp, dtype][1](
        bc, Q, xmQ, xpQ, ymQ, ypQ, U, xmU, xpU, ymU, ypU, Bx, By,
        ngh, comp, ornt, tol, drytol
    )

    return bc
