#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Data model for topography.
"""
from pydantic import root_validator as _root_validator
from torchswe import nplike as _nplike
from torchswe.utils.config import BaseConfig as _BaseConfig
from torchswe.utils.data.grid import Domain as _Domain


class Topography(_BaseConfig):
    """Data model for digital elevation.

    Attributes
    ----------
    domain : torchswe.utils.data.Domain
        The Domain instance associated with this Topography instance.
    vertices : (ny+2*nhalo+1, nx+2*nhalo+1) array
        Elevation at vertices.
    centers : (ny+2*nhalo+, nx+2*nhalo+) array
        Elevation at cell centers.
    xfcenters : (ny, nx+1) array
        Elevation at cell faces normal to x-axis.
    yfcenters : (ny+1, nx) array
        Elevation at cell faces normal to y-axis.
    grad : (2, ny, nx) array
        Derivatives w.r.t. x and y at cell centers.
    """

    # associated domain
    domain: _Domain

    # elevations
    vertices: _nplike.ndarray  # w/ halo rings
    centers: _nplike.ndarray  # w/ halo rings
    xfcenters: _nplike.ndarray  # w/o halo rings
    yfcenters: _nplike.ndarray  # w/o halo rings

    # cell-centered gradients
    grad: _nplike.ndarray  # w/o halo rings

    @_root_validator(pre=False, skip_on_failure=True)
    def _val_arrays(cls, values):  # pylint: disable=no-self-argument, no-self-use
        """Validations that rely on other fields' correctness."""

        # aliases
        domain = values["domain"]
        vertices = values["vertices"]
        centers = values["centers"]
        xfcenters = values["xfcenters"]
        yfcenters = values["yfcenters"]
        grad = values["grad"]

        # check dtype
        assert vertices.dtype == domain.dtype, "vertices: dtype does not match"
        assert centers.dtype == domain.dtype, "centers: dtype does not match"
        assert xfcenters.dtype == domain.dtype, "xfcenters: dtype does not match"
        assert yfcenters.dtype == domain.dtype, "yfcenters: dtype does not match"
        assert grad.dtype == domain.dtype, "grad: dtype does not match"

        # check shapes
        ny, nx = domain.hshape
        ngh = domain.nhalo
        assert vertices.shape == (ny+1, nx+1), "vertices: shape does not match."
        assert centers.shape == (ny, nx), "centers: shape does not match."
        assert xfcenters.shape == (ny-2*ngh, nx-2*ngh+1), "xfcenters: shape does not match."
        assert yfcenters.shape == (ny-2*ngh+1, nx-2*ngh), "yfcenters: shape does not match."
        assert grad.shape == (2, ny-2*ngh, nx-2*ngh), "grad: shape does not match."

        # check linear interpolation
        assert _nplike.allclose(
            xfcenters, (vertices[ngh+1:-ngh, ngh:-ngh]+vertices[ngh:-ngh-1, ngh:-ngh])/2.), \
            "The solver requires xfcenters to be linearly interpolated from vertices"
        assert _nplike.allclose(
            yfcenters, (vertices[ngh:-ngh, ngh+1:-ngh]+vertices[ngh:-ngh, ngh:-ngh-1])/2.), \
            "The solver requires yfcenters to be linearly interpolated from vertices"
        assert _nplike.allclose(centers,
            (vertices[:-1, :-1]+vertices[:-1, 1:]+vertices[1:, :-1]+vertices[1:, 1:])/4.), \
            "The solver requires centers to be linearly interpolated from vertices"
        assert _nplike.allclose(
            centers[ngh:-ngh, ngh:-ngh], (xfcenters[:, 1:]+xfcenters[:, :-1])/2.), \
            "The solver requires centers to be linearly interpolated from xfcenters"
        assert _nplike.allclose(
            centers[ngh:-ngh, ngh:-ngh], (yfcenters[1:, :]+yfcenters[:-1, :])/2.), \
            "The solver requires centers to be linearly interpolated from yfcenters"

        # check central difference
        dy, dx = domain.delta
        assert _nplike.allclose(grad[0], (xfcenters[:, 1:]-xfcenters[:, :-1])/dx), \
            "grad[0] must be obtained from central differences of xfcenters"
        assert _nplike.allclose(grad[1], (yfcenters[1:, :]-yfcenters[:-1, :])/dy), \
            "grad[1] must be obtained from central differences of yfcenters"

        return values

    @property
    def local_slice_vertices(self):
        """The slice of non-halo nor non-ghost cells in local vertex data."""
        return (
            slice(self.domain.nhalo, self.vertices.shape[0]-self.domain.nhalo),
            slice(self.domain.nhalo, self.vertices.shape[1]-self.domain.nhalo)
        )

    @property
    def local_slice_centers(self):
        """The slice of non-halo nor non-ghost cells in local cell-centered data."""
        return (
            slice(self.domain.nhalo, self.centers.shape[0]-self.domain.nhalo),
            slice(self.domain.nhalo, self.centers.shape[1]-self.domain.nhalo)
        )

    @property
    def global_slice_vertices(self):
        """The slice of the local vertex elevation in the global topo array"""
        return (
            slice(self.domain.y.ibegin+self.domain.nhalo, self.domain.y.iend+1+self.domain.nhalo),
            slice(self.domain.x.ibegin+self.domain.nhalo, self.domain.x.iend+1+self.domain.nhalo)
        )

    @property
    def global_slice_centers(self):
        """The slice of the local cell-centerd elevation in the global topo array"""
        return (
            slice(self.domain.y.ibegin+self.domain.nhalo, self.domain.y.iend+self.domain.nhalo),
            slice(self.domain.x.ibegin+self.domain.nhalo, self.domain.x.iend+self.domain.nhalo)
        )

    @property
    def global_slice_xfcenters(self):
        """The slice of the local x faces elevation in the global topo array"""
        return (
            slice(self.domain.y.ibegin, self.domain.y.iend),
            slice(self.domain.x.ibegin, self.domain.x.iend+1)
        )

    @property
    def global_slice_yfcenters(self):
        """The slice of the local y faces elevation in the global topo array"""
        return (
            slice(self.domain.y.ibegin, self.domain.y.iend+1),
            slice(self.domain.x.ibegin, self.domain.x.iend)
        )

    @property
    def global_slice_grad(self):
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
        grp.create_dataset("vertices", (domain.y.gn+1, domain.x.gn+1), dtype)
        grp.create_dataset("centers", (domain.y.gn, domain.x.gn), dtype)
        grp.create_dataset("xfcenters", (domain.y.gn, domain.x.gn+1), dtype)
        grp.create_dataset("yfcenters", (domain.y.gn+1, domain.x.gn), dtype)
        grp.create_dataset("grad", (2, domain.y.gn, domain.x.gn), dtype)
        grp["vertices"][self.global_slice_vertices] = self.vertices[self.local_slice_vertices]
        grp["centers"][self.global_slice_centers] = self.centers[self.local_slice_centers]
        grp["xfcenters"][self.global_slice_xfcenters] = self.xfcenters
        grp["yfcenters"][self.global_slice_yfcenters] = self.yfcenters
        grp["grad"][self.global_slice_grad] = self.grad
