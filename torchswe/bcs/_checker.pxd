cimport cython

cdef inline void shape_checker(
    object Q, object xmQ, object xpQ, object ymQ, object ypQ,
    object U, object xmU, object xpU, object ymU, object ypU,
    object Bx, object By,
    const Py_ssize_t ngh, const Py_ssize_t comp, const unsigned ornt
) except *:
    assert ngh == 2, "Currently only support ngh = 2"
    assert 0 <= comp <= 2, "comp should be 0 <= comp <= 2"
    assert 0 <= ornt <= 3, "ornt should be 0 <= ornt <= 3"

    assert Q.shape[0] == 3, f"{Q.shape}"

    assert xmQ.shape[0] == 3, f"{xmQ.shape}"
    assert xmQ.shape[1] == Q.shape[1] - 2 * ngh, f"{xmQ.shape}"
    assert xmQ.shape[2] == Q.shape[2] - 2 * ngh + 1, f"{xmQ.shape}"

    assert xpQ.shape[0] == 3, f"{xpQ.shape}"
    assert xpQ.shape[1] == Q.shape[1] - 2 * ngh, f"{xpQ.shape}"
    assert xpQ.shape[2] == Q.shape[2] - 2 * ngh + 1, f"{xpQ.shape}"

    assert ymQ.shape[0] == 3, f"{ymQ.shape}"
    assert ymQ.shape[1] == Q.shape[1] - 2 * ngh + 1, f"{ymQ.shape}"
    assert ymQ.shape[2] == Q.shape[2] - 2 * ngh, f"{ymQ.shape}"

    assert ypQ.shape[0] == 3, f"{ypQ.shape}"
    assert ypQ.shape[1] == Q.shape[1] - 2 * ngh + 1, f"{ypQ.shape}"
    assert ypQ.shape[2] == Q.shape[2] - 2 * ngh, f"{ypQ.shape}"

    assert U.shape == Q.shape, f"{U.shape}"

    assert xmU.shape[0] == 3, f"{xmU.shape}"
    assert xmU.shape[1] == Q.shape[1] - 2 * ngh, f"{xmU.shape}"
    assert xmU.shape[2] == Q.shape[2] - 2 * ngh + 1, f"{xmU.shape}"

    assert xpU.shape[0] == 3, f"{xpU.shape}"
    assert xpU.shape[1] == Q.shape[1] - 2 * ngh, f"{xpU.shape}"
    assert xpU.shape[2] == Q.shape[2] - 2 * ngh + 1, f"{xpU.shape}"

    assert ymU.shape[0] == 3, f"{ymU.shape}"
    assert ymU.shape[1] == Q.shape[1] - 2 * ngh + 1, f"{ymU.shape}"
    assert ymU.shape[2] == Q.shape[2] - 2 * ngh, f"{ymU.shape}"

    assert ypU.shape[0] == 3, f"{ypU.shape}"
    assert ypU.shape[1] == Q.shape[1] - 2 * ngh + 1, f"{ypU.shape}"
    assert ypU.shape[2] == Q.shape[2] - 2 * ngh, f"{ypU.shape}"

    assert Bx.shape == xmQ.shape[1:], f"{Bx.shape}"
    assert By.shape == ymQ.shape[1:], f"{By.shape}"


cdef inline void shape_checker_memoryview(
    const cython.floating[:, :, ::1] Q,
    const cython.floating[:, :, ::1] xmQ, const cython.floating[:, :, ::1] xpQ,
    const cython.floating[:, :, ::1] ymQ, const cython.floating[:, :, ::1] ypQ,
    const cython.floating[:, :, ::1] U,
    const cython.floating[:, :, ::1] xmU, const cython.floating[:, :, ::1] xpU,
    const cython.floating[:, :, ::1] ymU, const cython.floating[:, :, ::1] ypU,
    const cython.floating[:, ::1] Bx, const cython.floating[:, ::1] By,
    const Py_ssize_t ngh, const Py_ssize_t comp, const unsigned ornt
) nogil except *:
    assert ngh == 2, "Currently only support ngh = 2"
    assert 0 <= comp <= 2, "comp should be 0 <= comp <= 2"
    assert 0 <= ornt <= 3, "ornt should be 0 <= ornt <= 3"

    assert Q.shape[0] == 3, f"{Q.shape}"

    assert xmQ.shape[0] == 3, f"{xmQ.shape}"
    assert xmQ.shape[1] == Q.shape[1] - 2 * ngh, f"{xmQ.shape}"
    assert xmQ.shape[2] == Q.shape[2] - 2 * ngh + 1, f"{xmQ.shape}"

    assert xpQ.shape[0] == 3, f"{xpQ.shape}"
    assert xpQ.shape[1] == Q.shape[1] - 2 * ngh, f"{xpQ.shape}"
    assert xpQ.shape[2] == Q.shape[2] - 2 * ngh + 1, f"{xpQ.shape}"

    assert ymQ.shape[0] == 3, f"{ymQ.shape}"
    assert ymQ.shape[1] == Q.shape[1] - 2 * ngh + 1, f"{ymQ.shape}"
    assert ymQ.shape[2] == Q.shape[2] - 2 * ngh, f"{ymQ.shape}"

    assert ypQ.shape[0] == 3, f"{ypQ.shape}"
    assert ypQ.shape[1] == Q.shape[1] - 2 * ngh + 1, f"{ypQ.shape}"
    assert ypQ.shape[2] == Q.shape[2] - 2 * ngh, f"{ypQ.shape}"

    assert U.shape[0] == Q.shape[0], f"{U.shape}"
    assert U.shape[1] == Q.shape[1], f"{U.shape}"
    assert U.shape[2] == Q.shape[2], f"{U.shape}"

    assert xmU.shape[0] == 3, f"{xmU.shape}"
    assert xmU.shape[1] == Q.shape[1] - 2 * ngh, f"{xmU.shape}"
    assert xmU.shape[2] == Q.shape[2] - 2 * ngh + 1, f"{xmU.shape}"

    assert xpU.shape[0] == 3, f"{xpU.shape}"
    assert xpU.shape[1] == Q.shape[1] - 2 * ngh, f"{xpU.shape}"
    assert xpU.shape[2] == Q.shape[2] - 2 * ngh + 1, f"{xpU.shape}"

    assert ymU.shape[0] == 3, f"{ymU.shape}"
    assert ymU.shape[1] == Q.shape[1] - 2 * ngh + 1, f"{ymU.shape}"
    assert ymU.shape[2] == Q.shape[2] - 2 * ngh, f"{ymU.shape}"

    assert ypU.shape[0] == 3, f"{ypU.shape}"
    assert ypU.shape[1] == Q.shape[1] - 2 * ngh + 1, f"{ypU.shape}"
    assert ypU.shape[2] == Q.shape[2] - 2 * ngh, f"{ypU.shape}"

    assert Bx.shape[0] == xmQ.shape[1], f"{Bx.shape}"
    assert Bx.shape[1] == xmQ.shape[2], f"{Bx.shape}"
    assert By.shape[0] == ymQ.shape[1], f"{By.shape}"
    assert By.shape[1] == ymQ.shape[2], f"{By.shape}"
