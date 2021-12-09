#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Data model for topography.
"""
from logging import getLogger as _getLogger
from mpi4py import MPI as _MPI
from mpi4py.util.dtlib import from_numpy_dtype as _from_numpy_dtype
from pydantic import root_validator as _root_validator
from torchswe import nplike as _nplike
from torchswe.utils.config import BaseConfig as _BaseConfig
from torchswe.utils.config import Config as _Config
from torchswe.utils.netcdf import read as _ncread
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

    @property
    def nonhalo_slc_v(self):
        """The slice of non-halo nor non-ghost cells in local vertex data."""
        return (
            slice(self.domain.nhalo, self.v.shape[0]-self.domain.nhalo),
            slice(self.domain.nhalo, self.v.shape[1]-self.domain.nhalo)
        )

    @property
    def nonhalo_slc_c(self):
        """The slice of non-halo nor non-ghost cells in local cell-centered data."""
        return (
            slice(self.domain.nhalo, self.c.shape[0]-self.domain.nhalo),
            slice(self.domain.nhalo, self.c.shape[1]-self.domain.nhalo)
        )

    @property
    def global_slc_v(self):
        """The slice of the local vertex elevation in the global topo array"""
        return (
            slice(self.domain.y.ibegin, self.domain.y.iend+1),
            slice(self.domain.x.ibegin, self.domain.x.iend+1)
        )

    @property
    def global_slc_c(self):
        """The slice of the local cell-centerd elevation in the global topo array"""
        return (
            slice(self.domain.y.ibegin, self.domain.y.iend),
            slice(self.domain.x.ibegin, self.domain.x.iend)
        )

    @property
    def global_slc_xf(self):
        """The slice of the local x faces elevation in the global topo array"""
        return (
            slice(self.domain.y.ibegin, self.domain.y.iend),
            slice(self.domain.x.ibegin, self.domain.x.iend+1)
        )

    @property
    def global_slc_yf(self):
        """The slice of the local y faces elevation in the global topo array"""
        return (
            slice(self.domain.y.ibegin, self.domain.y.iend+1),
            slice(self.domain.x.ibegin, self.domain.x.iend)
        )

    @property
    def global_slc_grad(self):
        """The slice of the local cell-centered gradients in the global topo array"""
        return (
            slice(None),
            slice(self.domain.y.ibegin, self.domain.y.iend),
            slice(self.domain.x.ibegin, self.domain.x.iend)
        )

    def write_to_h5group(self, grp):
        """Write parallel data to a given HDF5 group.
        """
        domain = self.domain
        dtype = domain.dtype
        grp.create_dataset("v", (domain.y.gn+1, domain.x.gn+1), dtype)
        grp.create_dataset("c", (domain.y.gn, domain.x.gn), dtype)
        grp.create_dataset("xf", (domain.y.gn, domain.x.gn+1), dtype)
        grp.create_dataset("yf", (domain.y.gn+1, domain.x.gn), dtype)
        grp.create_dataset("grad", (2, domain.y.gn, domain.x.gn), dtype)
        grp["v"][self.global_slc_v] = self.v[self.nonhalo_slc_v]
        grp["c"][self.global_slc_c] = self.c[self.nonhalo_slc_c]
        grp["xf"][self.global_slc_xf] = self.xf
        grp["yf"][self.global_slc_yf] = self.yf
        grp["grad"][self.global_slc_grad] = self.grad


def get_topography(config: _Config, domain: _Domain = None, comm: _MPI.Comm = None):
    """Read in CF-compliant NetCDF file for topography.
    """

    # if domain is not provided, get a new one
    if domain is None:
        comm = _MPI.COMM_WORLD if comm is None else comm
        domain = _get_domain(comm, config)

    # get dem (digital elevation model); assume dem values defined at cell centers
    dem, _ = _ncread(
        fpath=config.topo.file, data_keys=[config.topo.key],
        extent=(domain.x.lower, domain.x.upper, domain.y.lower, domain.y.upper),
        parallel=True, comm=domain.comm
    )

    assert dem[config.topo.key].shape == (len(dem["y"]), len(dem["x"]))

    topo = _setup_topography(domain, dem[config.topo.key], dem["x"], dem["y"])
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
        vert[domain.effybg:domain.effyed+1, domain.effxbg:domain.effxed+1] = _nplike.array(
            _interpolate(
                demx, demy, elev.T,
                domain.x.v, domain.y.v
            ).T
        ).astype(domain.dtype)
    else:  # no need for interpolation
        vert[domain.effybg:domain.effyed+1, domain.effxbg:domain.effxed+1] = elev.astype(dtype)

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


def _exchange_topo_vertices(domain: _Domain, vertices: _nplike.ndarray):
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
