cimport cython

ctypedef fused fptype:
    cython.float
    cython.double

ctypedef fused confptype:
    const cython.float
    const cython.double

include "cython_flux.pyx"
include "cython_minmod.pyx"
