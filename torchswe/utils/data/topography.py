#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Data model for topography.
"""
# imports related to type hinting
from __future__ import annotations as _annotations  # allows us not using quotation marks for hints
from typing import TYPE_CHECKING as _TYPE_CHECKING  # indicates if we have type checking right now
if _TYPE_CHECKING:  # if we are having type checking, then we import corresponding classes/types
    from torchswe.nplike import ndarray
    from torchswe.utils.config import Config
    from torchswe.utils.data.grid import Domain
    from mpi4py import MPI

# pylint: disable=wrong-import-position, ungrouped-imports
from logging import getLogger as _getLogger
from mpi4py import MPI as _MPI
from mpi4py.util.dtlib import from_numpy_dtype as _from_numpy_dtype
from pydantic import root_validator as _root_validator
from torchswe import nplike as _nplike
from torchswe.utils.config import BaseConfig as _BaseConfig
from torchswe.utils.io import read_block as _read_block
from torchswe.utils.misc import interpolate as _interpolate
from torchswe.utils.misc import DummyDict as _DummyDict
from torchswe.utils.data.grid import Domain as _Domain
from torchswe.utils.data.grid import get_domain as _get_domain


_logger = _getLogger("torchswe.utils.data.topography")


class Topography(_BaseConfig):
    """Data model for digital elevation.

    Attributes
    ----------
    domain : torchswe.utils.data.Domain
        The Domain instance associated with this Topography instance.
    v : (ny+2*nhalo+1, nx+2*nhalo+1) array
        Elevation at vertices.
    c : (ny+2*nhalo+, nx+2*nhalo+) array
        Elevation at cell centers.
    xf : (ny, nx+1) array
        Elevation at cell faces normal to x-axis.
    yf : (ny+1, nx) array
        Elevation at cell faces normal to y-axis.
    grad : (2, ny, nx) array
        Derivatives w.r.t. x and y at cell centers.
    """
    # pylint: disable=invalid-name

    # associated domain
    domain: _Domain

    # elevations
    v: _nplike.ndarray  # w/ halo rings
    c: _nplike.ndarray  # w/ halo rings
    xf: _nplike.ndarray  # w/o halo rings
    yf: _nplike.ndarray  # w/o halo rings

    # cell-centered gradients
    grad: _nplike.ndarray  # w/o halo rings

    @_root_validator(pre=False, skip_on_failure=True)
    def _val_arrays(cls, values):  # pylint: disable=no-self-argument, no-self-use
        """Validations that rely on other fields' correctness."""

        # aliases
        domain = values["domain"]
        v = values["v"]
        c = values["c"]
        xf = values["xf"]
        yf = values["yf"]
        grad = values["grad"]

        # check dtype
        assert v.dtype == domain.dtype, "vertices: dtype does not match"
        assert c.dtype == domain.dtype, "centers: dtype does not match"
        assert xf.dtype == domain.dtype, "xfcenters: dtype does not match"
        assert yf.dtype == domain.dtype, "yfcenters: dtype does not match"
        assert grad.dtype == domain.dtype, "grad: dtype does not match"

        # check shapes
        ny, nx = domain.hshape
        ngh = domain.nhalo
        assert v.shape == (ny+1, nx+1), "vertices: shape does not match."
        assert c.shape == (ny, nx), "centers: shape does not match."
        assert xf.shape == (ny-2*ngh, nx-2*ngh+1), "xfcenters: shape does not match."
        assert yf.shape == (ny-2*ngh+1, nx-2*ngh), "yfcenters: shape does not match."
        assert grad.shape == (2, ny-2*ngh, nx-2*ngh), "grad: shape does not match."

        # check linear interpolation
        assert _nplike.allclose(xf, (v[ngh+1:-ngh, ngh:-ngh]+v[ngh:-ngh-1, ngh:-ngh])/2.), \
            "The solver requires xfcenters to be linearly interpolated from vertices"
        assert _nplike.allclose(yf, (v[ngh:-ngh, ngh+1:-ngh]+v[ngh:-ngh, ngh:-ngh-1])/2.), \
            "The solver requires yfcenters to be linearly interpolated from vertices"
        assert _nplike.allclose(c, (v[:-1, :-1]+v[:-1, 1:]+v[1:, :-1]+v[1:, 1:])/4.), \
            "The solver requires centers to be linearly interpolated from vertices"
        assert _nplike.allclose(c[ngh:-ngh, ngh:-ngh], (xf[:, 1:]+xf[:, :-1])/2.), \
            "The solver requires centers to be linearly interpolated from xfcenters"
        assert _nplike.allclose(c[ngh:-ngh, ngh:-ngh], (yf[1:, :]+yf[:-1, :])/2.), \
            "The solver requires centers to be linearly interpolated from yfcenters"

        # check central difference
        dy, dx = domain.delta
        assert _nplike.allclose(grad[0], (xf[:, 1:]-xf[:, :-1])/dx), \
            "grad[0] must be obtained from central differences of xfcenters"
        assert _nplike.allclose(grad[1], (yf[1:, :]-yf[:-1, :])/dy), \
            "grad[1] must be obtained from central differences of yfcenters"

        return values


def get_topography(config: Config, domain: Domain = None, comm: MPI.Comm = None):
    """Read local topography elevation data from a file.
    """

    # alias
    topocfg = config.topo

    # if domain is not provided, get a new one
    if domain is None:
        comm = _MPI.COMM_WORLD if comm is None else comm
        domain = _get_domain(comm, config)

    # get dem (digital elevation model); assume dem values defined at cell centers
    dem = _read_block(topocfg.file, topocfg.xykeys, topocfg.key, domain.lextent_v, domain)
    assert dem[topocfg.key].shape == (len(dem.y), len(dem.x))

    topo = _setup_topography(domain, dem[topocfg.key], dem.x, dem.y)
    return topo


def _setup_topography(domain, elev, demx, demy):
    """Set up a Topography object.
    """

    # alias
    dtype = domain.dtype
    ngh = domain.nhalo

    # see if we need to do interpolation
    try:
        interp = not (
            _nplike.allclose(domain.x.v, demx) and
            _nplike.allclose(domain.y.v, demy)
        )
    except ValueError:  # assume this excpetion means a shape mismatch
        interp = True

    # initialize vert
    vert = _nplike.zeros(tuple(i+1 for i in domain.hshape), dtype=dtype)

    if interp:  # unfortunately, we need to do interpolation in such a situation
        _logger.warning("Grids do not match. Doing spline interpolation.")
        vert[domain.nonhalo_v] = _nplike.array(
            _interpolate(demx, demy, elev.T, domain.x.v, domain.y.v).T
        ).astype(domain.dtype)
    else:  # no need for interpolation
        vert[domain.nonhalo_v] = elev.astype(dtype)

    # exchange vertices' elevations in halo rings
    vert = _exchange_topo_vertices(domain, vert)

    # topography elevation at cell centers through linear interpolation
    cntr = (vert[:-1, :-1] + vert[:-1, 1:] + vert[1:, :-1] + vert[1:, 1:]) / 4.

    # topography elevation at cell faces' midpoints through linear interpolation
    xface = (vert[ngh:-ngh-1, ngh:-ngh] + vert[ngh+1:-ngh, ngh:-ngh]) / 2.
    yface = (vert[ngh:-ngh, ngh:-ngh-1] + vert[ngh:-ngh, ngh+1:-ngh]) / 2.

    # gradient at cell centers through central difference; here allows nonuniform grids
    dy, dx = domain.delta
    grad = _nplike.zeros((2,)+domain.shape, dtype=dtype)
    grad[0, ...] = (xface[:, 1:] - xface[:, :-1]) / dx
    grad[1, ...] = (yface[1:, :] - yface[:-1, :]) / dy

    # initialize DataModel and let pydantic validates data
    return Topography(domain=domain, v=vert, c=cntr, xf=xface, yf=yface, grad=grad)


def _exchange_topo_vertices(domain: Domain, vertices: ndarray):
    """Exchange the halo ring information of vertices.
    """

    assert len(vertices.shape) == 2
    assert vertices.shape == tuple(i+1 for i in domain.hshape)

    # aliases
    nhalo = domain.nhalo
    hshape = vertices.shape  # vertices has one more element in both x and y
    ny, nx = hshape[0] - 2 * nhalo, hshape[1] - 2 * nhalo
    ybg, yed = nhalo, nhalo + ny
    xbg, xed = nhalo, nhalo + nx
    fp_t: _MPI.Datatype = _from_numpy_dtype(domain.dtype)

    # make sure all GPU calculations are done
    _nplike.sync()

    # sending buffer datatypes
    send_t = _DummyDict()
    send_t.s = fp_t.Create_subarray(hshape, (nhalo, nx), (ybg+1, xbg)).Commit()
    send_t.n = fp_t.Create_subarray(hshape, (nhalo, nx), (yed-nhalo-1, xbg)).Commit()
    send_t.w = fp_t.Create_subarray(hshape, (ny, nhalo), (ybg, xbg+1)).Commit()
    send_t.e = fp_t.Create_subarray(hshape, (ny, nhalo), (ybg, xed-nhalo-1)).Commit()

    # receiving buffer datatypes
    recv_t = _DummyDict()
    recv_t.s = fp_t.Create_subarray(hshape, (nhalo, nx), (0, xbg)).Commit()
    recv_t.n = fp_t.Create_subarray(hshape, (nhalo, nx), (yed, xbg)).Commit()
    recv_t.w = fp_t.Create_subarray(hshape, (ny, nhalo), (ybg, 0)).Commit()
    recv_t.e = fp_t.Create_subarray(hshape, (ny, nhalo), (ybg, xed)).Commit()

    # send & receive
    reqs = []
    reqs.append(domain.comm.Isend([vertices, 1, send_t.s], domain.s, 10))
    reqs.append(domain.comm.Isend([vertices, 1, send_t.n], domain.n, 11))
    reqs.append(domain.comm.Isend([vertices, 1, send_t.w], domain.w, 12))
    reqs.append(domain.comm.Isend([vertices, 1, send_t.e], domain.e, 13))
    reqs.append(domain.comm.Irecv([vertices, 1, recv_t.s], domain.s, 11))
    reqs.append(domain.comm.Irecv([vertices, 1, recv_t.n], domain.n, 10))
    reqs.append(domain.comm.Irecv([vertices, 1, recv_t.w], domain.w, 13))
    reqs.append(domain.comm.Irecv([vertices, 1, recv_t.e], domain.e, 12))
    _MPI.Request.Waitall(reqs)

    return vertices
