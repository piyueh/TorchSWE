# vim:fenc=utf-8
# vim:ft=pyrex
import numpy
cimport numpy
cimport cython
from cpython.mem cimport PyMem_Malloc, PyMem_Free


cdef fused InflowBC:
    InflowFloatWH
    InflowFloatOther
    InflowDoubleWH
    InflowDoubleOther


ctypedef fused InflowWHBC:
    InflowFloatWH
    InflowDoubleWH


ctypedef fused InflowOtherBC:
    InflowFloatOther
    InflowDoubleOther


@cython.auto_pickle(False)  # meaningless to pickle a BC instance as everything is a memoryview
cdef class InflowDoubleBase:
    """Base class of constant-value boundary coditions wiht 64-bit floating points.
    """
    
    # number of elements to be updated on this boundary
    cdef Py_ssize_t n

    # conservatives
    cdef const double[:] qc0  # q at the cell centers of the 1st internal cell layer
    cdef const double[:] qc1  # q at the cell centers of the 2nd internal cell layer
    cdef double[:] qbci  # q at the inner side of the boundary cell faces
    cdef double[:] qbco  # q at the outer side of the boundary cell faces
    cdef double[:] qother  # q at the inner side of the another face of the 1st internal cell

    # read-only values (boundary target value & theta)
    cdef double val
    cdef double theta


@cython.auto_pickle(False)  # meaningless to pickle a BC instance as everything is a memoryview
cdef class InflowFloatBase:
    """Base class of constant-value boundary coditions wiht 32-bit floating points.
    """
    
    # number of elements to be updated on this boundary
    cdef Py_ssize_t n

    # conservatives
    cdef const float[:] qc0  # q at the cell centers of the 1st internal cell layer
    cdef const float[:] qc1  # q at the cell centers of the 2nd internal cell layer
    cdef float[:] qbci  # q at the inner side of the boundary cell faces
    cdef float[:] qbco  # q at the outer side of the boundary cell faces
    cdef float[:] qother  # q at the inner side of the another face of the 1st internal cell

    # read-only values (boundary target value & theta)
    cdef float val
    cdef float theta


@cython.auto_pickle(False)  # meaningless to pickle a BC instance as everything is a memoryview
cdef class InflowDoubleWH(InflowDoubleBase):
    """Constant-value boundary coditions wiht 64-bit floating points for w and h.
    """

    # depth
    cdef const double[:] hc0  # depth at the cell centers of the 1st internal cell layer
    cdef double[:] hbci  # depth at the inner side of the boundary cell faces
    cdef double[:] hbco  # depth at the outer side of the boundary cell faces
    cdef double[:] hother  # depth at the inner side of the another face of the 1st internal cell

    # topography elevation
    cdef const double[:] bbc  # topo elevations at the boundary cell faces
    cdef const double[:] bother  # topo elevation at cell faces between the 1st and 2nd internal cell layers

    # saved for convenience if the bc is for w and h, as topography doesn't change
    cdef double* _wbcval
    cdef double* _hbcval
    cdef double[::1] wbcval
    cdef double[::1] hbcval

    # tolerance
    cdef double tol  # depths under this tolerance are considered dry cells

    def __cinit__(
        self,
        object Q, object xmQ, object xpQ, object ymQ, object ypQ,
        object U, object xmU, object xpU, object ymU, object ypU,
        object Bx, object By,
        const Py_ssize_t ngh, const unsigned comp, const unsigned ornt,
        const double tol, const double drytol,
        const double theta, const double val,
    ):
        _inflow_bc_w_h_cinit[InflowDoubleWH](self, Q.shape[2], Q.shape[1], ngh, ornt)

    def __dealloc__(self):
        PyMem_Free(self._hbcval)
        PyMem_Free(self._wbcval)

    def __init__(
        self,
        object Q, object xmQ, object xpQ, object ymQ, object ypQ,
        object U, object xmU, object xpU, object ymU, object ypU,
        object Bx, object By,
        const Py_ssize_t ngh, const unsigned comp, const unsigned ornt,
        const double tol, const double drytol,
        const double theta, const double val,
    ):
        _inflow_bc_init[InflowDoubleWH, double](
            self, Q, xmQ, xpQ, ymQ, ypQ, U, xmU, xpU, ymU, ypU, Bx, By,
            ngh, comp, ornt, tol, drytol, theta, val
        )

    def __call__(self):
        _inflow_bc_w_h_kernel[InflowDoubleWH, double](self, 0.0, 0.0);


@cython.auto_pickle(False)  # meaningless to pickle a BC instance as everything is a memoryview
cdef class InflowFloatWH(InflowFloatBase):
    """Constant-value boundary coditions wiht 32-bit floating points for w and h.
    """

    # depth
    cdef const float[:] hc0  # depth at the cell centers of the 1st internal cell layer
    cdef float[:] hbci  # depth at the inner side of the boundary cell faces
    cdef float[:] hbco  # depth at the outer side of the boundary cell faces
    cdef float[:] hother  # depth at the inner side of the another face of the 1st internal cell

    # topography elevation
    cdef const float[:] bbc  # topo elevations at the boundary cell faces
    cdef const float[:] bother  # topo elevation at cell faces between the 1st and 2nd internal cell layers

    # saved for convenience if the bc is for w and h, as topography doesn't change
    cdef float* _wbcval
    cdef float* _hbcval
    cdef float[::1] wbcval
    cdef float[::1] hbcval

    # tolerance
    cdef float tol  # depths under this tolerance are considered dry cells

    def __cinit__(
        self,
        object Q, object xmQ, object xpQ, object ymQ, object ypQ,
        object U, object xmU, object xpU, object ymU, object ypU,
        object Bx, object By,
        const Py_ssize_t ngh, const unsigned comp, const unsigned ornt,
        const float tol, const float drytol,
        const float theta, const float val,
    ):
        _inflow_bc_w_h_cinit[InflowFloatWH](self, Q.shape[2], Q.shape[1], ngh, ornt)

    def __dealloc__(self):
        PyMem_Free(self._hbcval)
        PyMem_Free(self._wbcval)

    def __init__(
        self,
        object Q, object xmQ, object xpQ, object ymQ, object ypQ,
        object U, object xmU, object xpU, object ymU, object ypU,
        object Bx, object By,
        const Py_ssize_t ngh, const unsigned comp, const unsigned ornt,
        const double tol, const double drytol,
        const double theta, const double val,
    ):
        _inflow_bc_init[InflowFloatWH, float](
            self, Q, xmQ, xpQ, ymQ, ypQ, U, xmU, xpU, ymU, ypU, Bx, By,
            ngh, comp, ornt, tol, drytol, theta, val
        )

    def __call__(self):
        _inflow_bc_w_h_kernel[InflowFloatWH, float](self, 0.0, 0.0);


@cython.auto_pickle(False)  # meaningless to pickle a BC instance as everything is a memoryview
cdef class InflowDoubleOther(InflowDoubleBase):
    """Constant-value boundary coditions wiht 64-bit floating points for hu or hv.
    """

    # read-only depth
    cdef const double[:] hbci  # depth at the inner side of the boundary cell faces
    cdef const double[:] hbco  # depth at the outer side of the boundary cell faces
    cdef const double[:] hother  # depth at the inner side of the another face of the 1st internal cell

    # velocities
    cdef double[:] ubci  # u or v at the inner side of the boundary cell faces
    cdef double[:] ubco  # u or v at the outer side of the boundary cell faces
    cdef double[:] uother  # u or v at the inner side of the another face of the 1st internal cell

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
        _inflow_bc_init[InflowDoubleOther, double](
            self, Q, xmQ, xpQ, ymQ, ypQ, U, xmU, xpU, ymU, ypU, Bx, By,
            ngh, comp, ornt, tol, drytol, theta, val
        )

    def __call__(self):
        _inflow_bc_kernel[InflowDoubleOther, double](self, 0.0, 0.0);


@cython.auto_pickle(False)  # meaningless to pickle a BC instance as everything is a memoryview
cdef class InflowFloatOther(InflowFloatBase):
    """Constant-value boundary coditions wiht 32-bit floating points for hu or hv.
    """

    # read-only depth
    cdef const float[:] hbci  # depth at the inner side of the boundary cell faces
    cdef const float[:] hbco  # depth at the outer side of the boundary cell faces
    cdef const float[:] hother  # depth at the inner side of the another face of the 1st internal cell

    # velocities
    cdef float[:] ubci  # u or v at the inner side of the boundary cell faces
    cdef float[:] ubco  # u or v at the outer side of the boundary cell faces
    cdef float[:] uother  # u or v at the inner side of the another face of the 1st internal cell

    # tolerance
    cdef float drytol  # depths under this values are considered wet but still cells

    def __init__(
        self,
        object Q, object xmQ, object xpQ, object ymQ, object ypQ,
        object U, object xmU, object xpU, object ymU, object ypU,
        object Bx, object By,
        const Py_ssize_t ngh, const unsigned comp, const unsigned ornt,
        const float tol, const float drytol,
        const float theta, const float val,
    ):
        _inflow_bc_init[InflowFloatOther, float](
            self, Q, xmQ, xpQ, ymQ, ypQ, U, xmU, xpU, ymU, ypU, Bx, By,
            ngh, comp, ornt, tol, drytol, theta, val
        )

    def __call__(self):
        _inflow_bc_kernel[InflowFloatOther, float](self, 0.0, 0.0);


cdef void _inflow_bc_w_h_kernel(
    InflowWHBC bc,
    cython.floating denominator, cython.floating slp
) nogil except *:

    cdef Py_ssize_t i

    if InflowWHBC is InflowDoubleWH and cython.floating is float:
        raise TypeError("InflowDoubleWH does not work with float")
    elif InflowWHBC is InflowFloatWH and cython.floating is double:
        raise TypeError("InflowFloatWH does not work with double")
    else:

        for i in range(bc.n):
            # outer side follows desired values (but can't be negative depth)
            bc.hbco[i] = bc.hbcval[i];
            bc.qbco[i] = bc.wbcval[i];

            # a completely dry cell
            if bc.hc0[i] < bc.tol:
                bc.hbci[i] = 0.0;
                bc.hother[i] = 0.0;
                bc.qbci[i] = bc.bbc[i];
                bc.qother[i] = bc.bother[i];
                continue;

            # inner side depends on minmod reconstruction as if there's a ghost cell
            denominator = bc.qc1[i] - bc.qc0[i];
            slp = (bc.qc0[i] - bc.qbco[i]) / denominator;
            slp = min(slp*bc.theta, (slp+1.0)/2.0);
            slp = min(slp, bc.theta);
            slp = max(slp, 0.0);
            slp *= denominator;
            slp /= 2.0;
            bc.hbci[i] = bc.qc0[i] - slp - bc.bbc[i];
            bc.hother[i] = bc.qc0[i] + slp - bc.bother[i];

            # fix negative depth
            if (bc.hbci[i] < bc.tol):
                bc.hbci[i] = 0.0;
                bc.hother[i] = bc.hc0[i] * 2.0;
            elif (bc.hother[i] < bc.tol):
                bc.hbci[i] = bc.hc0[i] * 2.0;
                bc.hother[i] = 0.0;

            # reconstruct to eliminate rounding error-edffect in further calculations
            bc.qbci[i] = bc.hbci[i] + bc.bbc[i];
            bc.qother[i] = bc.hother[i] + bc.bother[i];


cdef void _inflow_bc_kernel(
    InflowOtherBC bc,
    cython.floating denominator, cython.floating slp
) nogil except *:

    cdef Py_ssize_t i

    if InflowOtherBC is InflowDoubleOther and cython.floating is float:
        raise TypeError("InflowDoubleOther does not work with float")
    elif InflowOtherBC is InflowFloatOther and cython.floating is double:
        raise TypeError("InflowFloatOther does not work with double")
    else:

        for i in range(bc.n):
            # outer side follows desired values
            bc.ubco[i] = bc.val;
            bc.qbco[i] = bc.val * bc.hbco[i];

            # inner side depends on minmod reconstruction as if there's a ghost cell
            denominator = bc.qc1[i] - bc.qc0[i];
            slp = (bc.qc0[i] - bc.qbco[i]) / denominator;
            slp = min(slp*bc.theta, (slp+1.0)/2.0);
            slp = min(slp, bc.theta);
            slp = max(slp, 0.0);
            slp *= denominator;
            slp /= 2.0;

            # we don't fix values at the outer side -> they follow desired BC values

            if (bc.hbci[i] < bc.drytol):
                bc.qbci[i] = 0.0;
                bc.ubci[i] = 0.0;
            else:
                bc.ubci[i] = (bc.qc0[i] - slp) / bc.hbci[i];
                bc.qbci[i] = bc.hbci[i] * bc.ubci[i];

            if (bc.hother[i] < bc.drytol):
                bc.qother[i] = 0.0;
                bc.uother[i] = 0.0;
            else:
                bc.uother[i] = (bc.qc0[i] + slp) / bc.hother[i];
                bc.qother[i] = bc.hother[i] * bc.uother[i];


cdef void _inflow_bc_w_h_cinit(
    InflowBC bc,
    const Py_ssize_t nxgh,
    const Py_ssize_t nygh,
    const Py_ssize_t ngh,
    const unsigned ornt,
) except *:
    if ornt == 0 or ornt == 1:  # west or east
        bc.n = nygh - 2 * ngh  # ny
    elif ornt == 2 or ornt == 3:  # west or east
        bc.n = nxgh - 2 * ngh  # nx
    else:
        raise ValueError(f"`ornt` should be >= 0 and <= 3: {ornt}")

    if InflowBC is InflowFloatWH:
        bc._hbcval = <cython.float*>PyMem_Malloc(bc.n*sizeof(cython.float))
        bc._wbcval = <cython.float*>PyMem_Malloc(bc.n*sizeof(cython.float))
        if not bc._hbcval: raise MemoryError()
        if not bc._wbcval: raise MemoryError()
        bc.hbcval = <cython.float[:bc.n]>bc._hbcval
        bc.wbcval = <cython.float[:bc.n]>bc._wbcval
    elif InflowBC is InflowDoubleWH:
        bc._hbcval = <cython.double*>PyMem_Malloc(bc.n*sizeof(cython.double))
        bc._wbcval = <cython.double*>PyMem_Malloc(bc.n*sizeof(cython.double))
        if not bc._hbcval: raise MemoryError()
        if not bc._wbcval: raise MemoryError()
        bc.hbcval = <cython.double[:bc.n]>bc._hbcval
        bc.wbcval = <cython.double[:bc.n]>bc._wbcval


cdef void _inflow_bc_init(
    InflowBC bc,
    const cython.floating[:, :, ::1] Q,
    cython.floating[:, :, ::1] xmQ, cython.floating[:, :, ::1] xpQ,
    cython.floating[:, :, ::1] ymQ, cython.floating[:, :, ::1] ypQ,
    const cython.floating[:, :, ::1] U,
    cython.floating[:, :, ::1] xmU, cython.floating[:, :, ::1] xpU,
    cython.floating[:, :, ::1] ymU, cython.floating[:, :, ::1] ypU,
    const cython.floating[:, ::1] Bx, const cython.floating[:, ::1] By,
    const Py_ssize_t ngh, const unsigned comp, const unsigned ornt,
    const cython.floating tol, const cython.floating drytol,
    const cython.floating theta, const cython.floating val,
) nogil except *:

    cdef Py_ssize_t i

    # the first two combinations will be pruned by cython when translating to C/C++
    if cython.floating is float and (InflowBC is InflowDoubleWH or InflowBC is InflowDoubleOther):
        raise RuntimeError("Mismatched types")
    elif cython.floating is double and (InflowBC is InflowFloatWH or InflowBC is InflowFloatOther):
        raise RuntimeError("Mismatched types")
    else:

        if ornt == 0:  # west
            _inflow_bc_set_west[InflowBC, cython.floating](
                bc, Q, xmQ, xpQ, U, xmU, xpU, Bx, ngh, comp, val)
        elif ornt == 1:  # east
            _inflow_bc_set_east[InflowBC, cython.floating](
                bc, Q, xmQ, xpQ, U, xmU, xpU, Bx, ngh, comp, val)
        elif ornt == 2:  # south
            _inflow_bc_set_south[InflowBC, cython.floating](
                bc, Q, ymQ, ypQ, U, ymU, ypU, By, ngh, comp, val)
        elif ornt == 3:  # north
            _inflow_bc_set_north[InflowBC, cython.floating](
                bc, Q, ymQ, ypQ, U, ymU, ypU, By, ngh, comp, val)
        else:
            raise ValueError(f"orientation id {ornt} not accepted.")

        if InflowBC is InflowDoubleWH:
            bc.tol = tol
        elif InflowBC is InflowFloatWH:
            bc.tol = tol
        elif InflowBC is InflowDoubleOther:
            bc.drytol = drytol
        elif InflowBC is InflowFloatOther:
            bc.drytol = drytol

        bc.theta = theta
        bc.val = val


cdef void _inflow_bc_set_west(
    InflowBC bc,
    const cython.floating[:, :, ::1] Q, cython.floating[:, :, ::1] xmQ, cython.floating[:, :, ::1] xpQ,
    const cython.floating[:, :, ::1] U, cython.floating[:, :, ::1] xmU, cython.floating[:, :, ::1] xpU,
    const cython.floating[:, ::1] Bx,
    const Py_ssize_t ngh, const Py_ssize_t comp, const cython.floating val
) nogil except *:

    cdef Py_ssize_t i

    # the first two combinations will be pruned by cython when translating to C/C++
    if cython.floating is float and (InflowBC is InflowDoubleWH or InflowBC is InflowDoubleOther):
        raise RuntimeError("Mismatched types")
    elif cython.floating is double and (InflowBC is InflowFloatWH or InflowBC is InflowFloatOther):
        raise RuntimeError("Mismatched types")
    else:

        # these should be views into original data buffer
        bc.qc0 = Q[comp, ngh:Q.shape[1]-ngh, ngh]
        bc.qc1 = Q[comp, ngh:Q.shape[1]-ngh, ngh+1]
        bc.qbci = xpQ[comp, :, 0]
        bc.qbco = xmQ[comp, :, 0]
        bc.qother = xmQ[comp, :, 1]

        bc.hbci = xpU[0, :, 0]
        bc.hbco = xmU[0, :, 0]
        bc.hother = xmU[0, :, 1]

        if InflowBC is InflowFloatOther or InflowBC is InflowDoubleOther:
            bc.ubci = xpU[comp, :, 0]
            bc.ubco = xmU[comp, :, 0]
            bc.uother = xmU[comp, :, 1]
        elif InflowBC is InflowFloatWH or InflowBC is InflowDoubleWH:
            bc.bbc = Bx[:, 0]
            bc.bother = Bx[:, 1]
            bc.hc0 = U[0, ngh:U.shape[1]-ngh, ngh]
            assert val >= 0.0;
            for i in range(bc.n):
                bc.wbcval[i] = bc.bbc[i] + val
                bc.hbcval[i] = val


cdef void _inflow_bc_set_east(
    InflowBC bc,
    const cython.floating[:, :, ::1] Q, cython.floating[:, :, ::1] xmQ, cython.floating[:, :, ::1] xpQ,
    const cython.floating[:, :, ::1] U, cython.floating[:, :, ::1] xmU, cython.floating[:, :, ::1] xpU,
    const cython.floating[:, ::1] Bx,
    const Py_ssize_t ngh, const Py_ssize_t comp, const cython.floating val
) nogil except *:

    cdef Py_ssize_t i

    # the first two combinations will be pruned by cython when translating to C/C++
    if cython.floating is float and (InflowBC is InflowDoubleWH or InflowBC is InflowDoubleOther):
        raise RuntimeError("Mismatched types")
    elif cython.floating is double and (InflowBC is InflowFloatWH or InflowBC is InflowFloatOther):
        raise RuntimeError("Mismatched types")
    else:

        # these should be views into original data buffer
        bc.qc0 = Q[comp, ngh:Q.shape[1]-ngh, Q.shape[2]-ngh-1]
        bc.qc1 = Q[comp, ngh:Q.shape[1]-ngh, Q.shape[2]-ngh-2]
        bc.qbci = xmQ[comp, :, xmQ.shape[2]-1]
        bc.qbco = xpQ[comp, :, xpQ.shape[2]-1]
        bc.qother = xpQ[comp, :, xpQ.shape[2]-2]

        bc.hbci = xmU[0, :, xmU.shape[2]-1]
        bc.hbco = xpU[0, :, xpU.shape[2]-1]
        bc.hother = xpU[0, :, xpU.shape[2]-2]

        if InflowBC is InflowFloatOther or InflowBC is InflowDoubleOther:
            bc.ubci = xmU[comp, :, xmU.shape[2]-1]
            bc.ubco = xpU[comp, :, xpU.shape[2]-1]
            bc.uother = xpU[comp, :, xpU.shape[2]-2]
        elif InflowBC is InflowFloatWH or InflowBC is InflowDoubleWH:
            bc.bbc = Bx[:, Bx.shape[1]-1]
            bc.bother = Bx[:, Bx.shape[1]-2]
            bc.hc0 = U[0, ngh:U.shape[1]-ngh, U.shape[2]-ngh-1]
            assert val >= 0.0;
            for i in range(bc.n):
                bc.wbcval[i] = bc.bbc[i] + val
                bc.hbcval[i] = val


cdef void _inflow_bc_set_south(
    InflowBC bc,
    const cython.floating[:, :, ::1] Q, cython.floating[:, :, ::1] ymQ, cython.floating[:, :, ::1] ypQ,
    const cython.floating[:, :, ::1] U, cython.floating[:, :, ::1] ymU, cython.floating[:, :, ::1] ypU,
    const cython.floating[:, ::1] By,
    const Py_ssize_t ngh, const Py_ssize_t comp, const cython.floating val
) nogil except *:

    cdef Py_ssize_t i

    # the first two combinations will be pruned by cython when translating to C/C++
    if cython.floating is float and (InflowBC is InflowDoubleWH or InflowBC is InflowDoubleOther):
        raise RuntimeError("Mismatched types")
    elif cython.floating is double and (InflowBC is InflowFloatWH or InflowBC is InflowFloatOther):
        raise RuntimeError("Mismatched types")
    else:

        # these should be views into original data buffer
        bc.qc0 = Q[comp, ngh, ngh:Q.shape[2]-ngh]
        bc.qc1 = Q[comp, ngh+1, ngh:Q.shape[2]-ngh]
        bc.qbci = ypQ[comp, 0, :]
        bc.qbco = ymQ[comp, 0, :]
        bc.qother = ymQ[comp, 1, :]

        bc.hbci = ypU[0, 0, :]
        bc.hbco = ymU[0, 0, :]
        bc.hother = ymU[0, 1, :]

        if InflowBC is InflowFloatOther or InflowBC is InflowDoubleOther:
            bc.ubci = ypU[comp, 0, :]
            bc.ubco = ymU[comp, 0, :]
            bc.uother = ymU[comp, 1, :]
        elif InflowBC is InflowFloatWH or InflowBC is InflowDoubleWH:
            bc.bbc = By[0, :]
            bc.bother = By[1, :]
            bc.hc0 = U[0, ngh, ngh:U.shape[2]-ngh]
            assert val >= 0.0;
            for i in range(bc.n):
                bc.wbcval[i] = bc.bbc[i] + val
                bc.hbcval[i] = val


cdef void _inflow_bc_set_north(
    InflowBC bc,
    const cython.floating[:, :, ::1] Q, cython.floating[:, :, ::1] ymQ, cython.floating[:, :, ::1] ypQ,
    const cython.floating[:, :, ::1] U, cython.floating[:, :, ::1] ymU, cython.floating[:, :, ::1] ypU,
    const cython.floating[:, ::1] By,
    const Py_ssize_t ngh, const Py_ssize_t comp, const cython.floating val
) nogil except *:

    cdef Py_ssize_t i

    # the first two combinations will be pruned by cython when translating to C/C++
    if cython.floating is float and (InflowBC is InflowDoubleWH or InflowBC is InflowDoubleOther):
        raise RuntimeError("Mismatched types")
    elif cython.floating is double and (InflowBC is InflowFloatWH or InflowBC is InflowFloatOther):
        raise RuntimeError("Mismatched types")
    else:

        # these should be views into original data buffer
        bc.qc0 = Q[comp, Q.shape[1]-ngh-1, ngh:Q.shape[2]-ngh]
        bc.qc1 = Q[comp, Q.shape[1]-ngh-2, ngh:Q.shape[2]-ngh]
        bc.qbci = ymQ[comp, ymQ.shape[1]-1, :]
        bc.qbco = ypQ[comp, ypQ.shape[1]-1, :]
        bc.qother = ypQ[comp, ypQ.shape[1]-2, :]

        bc.hbci = ymU[0, ymU.shape[1]-1, :]
        bc.hbco = ypU[0, ypU.shape[1]-1, :]
        bc.hother = ypU[0, ypU.shape[1]-2, :]

        if InflowBC is InflowFloatOther or InflowBC is InflowDoubleOther:
            bc.ubci = ymU[comp, ymU.shape[1]-1, :]
            bc.ubco = ypU[comp, ypU.shape[1]-1, :]
            bc.uother = ypU[comp, ypU.shape[1]-2, :]
        elif InflowBC is InflowFloatWH or InflowBC is InflowDoubleWH:
            bc.bbc = By[By.shape[0]-1, :]
            bc.bother = By[By.shape[0]-2, :]
            bc.hc0 = U[0, U.shape[1]-ngh-1, ngh:U.shape[2]-ngh]
            assert val >= 0.0;
            for i in range(bc.n):
                bc.wbcval[i] = bc.bbc[i] + val
                bc.hbcval[i] = val


def inflow_bc_factory(ornt, comp, states, topo, tol, drytol, theta, val, *args, **kwargs):
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
        if comp == 0:
            bc = InflowDoubleWH(
                Q, xmQ, xpQ, ymQ, ypQ, U, xmU, xpU, ymU, ypU, Bx, By,
                ngh, comp, ornt, tol, drytol, theta, val
            )
        else:
            bc = InflowDoubleOther(
                Q, xmQ, xpQ, ymQ, ypQ, U, xmU, xpU, ymU, ypU, Bx, By,
                ngh, comp, ornt, tol, drytol, theta, val
            )
    elif dtype == "float32":
        if comp == 0:
            bc = InflowFloatWH(
                Q, xmQ, xpQ, ymQ, ypQ, U, xmU, xpU, ymU, ypU, Bx, By,
                ngh, comp, ornt, tol, drytol, theta, val
            )
        else:
            bc = InflowFloatOther(
                Q, xmQ, xpQ, ymQ, ypQ, U, xmU, xpU, ymU, ypU, Bx, By,
                ngh, comp, ornt, tol, drytol, theta, val
            )
    else:
        raise ValueError(f"Unrecognized data type: {dtype}")

    return bc
