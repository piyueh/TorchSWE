#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Data models.
"""
# imports related to type hinting
from __future__ import annotations as _annotations  # allows us not using quotation marks for hints
from typing import TYPE_CHECKING as _TYPE_CHECKING  # indicates if we have type checking right now
if _TYPE_CHECKING:  # if we are having type checking, then we import corresponding classes/types
    from mpi4py import MPI
    from torchswe.nplike import ndarray
    from torchswe.utils.config import Config
    from torchswe.utils.data.grid import Domain

# pylint: disable=wrong-import-position, ungrouped-imports
from logging import getLogger as _getLogger
from typing import Optional as _Optional
from typing import Tuple as _Tuple
from typing import Union as _Union

from mpi4py import MPI as _MPI
from mpi4py.util.dtlib import from_numpy_dtype as _from_numpy_dtype
from pydantic import validator as _validator
from pydantic import root_validator as _root_validator
from torchswe import nplike as _nplike
from torchswe.utils.config import BaseConfig as _BaseConfig
from torchswe.utils.io import read_block as _read_block
from torchswe.utils.misc import DummyDict as _DummyDict
from torchswe.utils.misc import interpolate as _interpolate
from torchswe.utils.data.grid import Domain as _Domain
from torchswe.utils.data.grid import get_domain as _get_domain


_logger = _getLogger("torchswe.utils.data.topography")


def _pydantic_val_nan_inf(val, field):
    """Validates if any elements are NaN or inf."""
    assert not _nplike.any(_nplike.isnan(val)), f"Got NaN in {field.name}"
    assert not _nplike.any(_nplike.isinf(val)), f"Got Inf in {field.name}"
    return val


def _shape_val_factory(shift: _Union[_Tuple[int, int], int]):
    """A function factory creating a function to validate shapes of arrays."""

    def _core_func(val, values):
        """A function to validate the shape."""
        try:
            if isinstance(shift, int):
                target = (values["n"]+shift,)
            else:
                target = (values["ny"]+shift[0], values["nx"]+shift[1])
        except KeyError as err:
            raise AssertionError("Validation failed due to other validation failures.") from err

        assert val.shape == target, f"Shape mismatch. Should be {target}, got {val.shape}"
        return val

    return _core_func


class FaceOneSideModel(_BaseConfig):
    """Data model holding quantities on one side of cell faces normal to one direction.

    Attributes
    ----------
    q : nplike.ndarray of shape (3, ny+1, nx) or (3, ny, nx+1)
        The fluid elevation (i.e., h + b), h * u, and h * v.
    p : nplike.ndarray of shape (3, ny+1, nx) or (3, ny, nx+1)
        The fluid depth, u-velocity, and depth-v-velocity.
    a : nplike.ndarray of shape (ny+1, nx) or (3, ny, nx+1)
        The local speed.
    f : nplike.ndarray of shape (3, ny+1, nx) or (3, ny, nx+1)
        An array holding discontinuous fluxes.
    """

    q: _nplike.ndarray
    p: _nplike.ndarray
    a: _nplike.ndarray
    f: _nplike.ndarray

    # validator
    _val_valid_numbers = _validator("q", "p", "a", "f", allow_reuse=True)(_pydantic_val_nan_inf)

    @_root_validator(pre=False, skip_on_failure=True)
    def _val_arrays(cls, values):  # pylint: disable=no-self-argument, no-self-use
        """Validate the consistency of the arrays shapes and dtypes."""
        # pylint: disable=invalid-name

        try:
            q, p, a, f = values["q"], values["p"], values["a"], values["f"]
        except KeyError as err:
            raise AssertionError("Fix other fields first.") from err

        n1, n2 = a.shape
        assert q.shape == (3, n1, n2), f"q shape mismatch. Should be {(3, n1, n2)}. Got {q.shape}."
        assert p.shape == (3, n1, n2), f"p shape mismatch. Should be {(3, n1, n2)}. Got {p.shape}."
        assert f.shape == (3, n1, n2), f"f shape mismatch. Should be {(3, n1, n2)}. Got {f.shape}."
        assert q.dtype == a.dtype, f"dtype mismatch. Should be {a.dtype}. Got {q.dtype}."
        assert p.dtype == a.dtype, f"dtype mismatch. Should be {a.dtype}. Got {p.dtype}."
        assert f.dtype == a.dtype, f"dtype mismatch. Should be {a.dtype}. Got {f.dtype}."
        return values


class FaceTwoSideModel(_BaseConfig):
    """Date model holding quantities on both sides of cell faces normal to one direction.

    Attributes
    ----------
    plus, minus : FaceOneSideModel
        Objects holding data on one side of each face.
    cf : nplike.ndarray of shape (3, ny+1, nx) or (3, ny, nx+1)
        An object holding common flux (i.e., continuous or numerical flux)
    """

    plus: FaceOneSideModel
    minus: FaceOneSideModel
    cf: _nplike.ndarray

    # validator
    _val_valid_numbers = _validator("cf", allow_reuse=True)(_pydantic_val_nan_inf)

    @_root_validator(pre=False, skip_on_failure=True)
    def _val_arrays(cls, values):  # pylint: disable=no-self-argument, no-self-use
        """Validate shapes and dtypes."""
        # pylint: disable=invalid-name

        try:
            plus, minus, cf = values["plus"], values["minus"], values["cf"]
        except KeyError as err:
            raise AssertionError("Fix other fields first.") from err

        assert plus.q.shape == minus.q.shape, f"Shape mismatch: {plus.q.shape} and {minus.q.shape}."
        assert plus.q.dtype == minus.q.dtype, f"dtype mismatch: {plus.q.dtype} and {minus.q.dtype}."
        assert plus.q.shape == cf.shape, f"Shape mismatch: {plus.q.shape} and {cf.shape}."
        assert plus.q.dtype == cf.dtype, f"dtype mismatch: {plus.q.dtype} and {cf.dtype}."
        return values


class FaceQuantityModel(_BaseConfig):
    """Data model holding quantities on both sides of cell faces in both x and y directions.

    Attributes
    ----------
    x, y : FaceTwoSideModel
        Objects holding data on faces facing x and y directions.
    """

    x: FaceTwoSideModel
    y: FaceTwoSideModel

    @_root_validator(pre=False, skip_on_failure=True)
    def _val_arrays(cls, values):  # pylint: disable=no-self-argument, no-self-use
        """Validate shapes and dtypes."""
        # pylint: disable=invalid-name

        try:
            x, y = values["x"], values["y"]
        except KeyError as err:
            raise AssertionError("Fix other fields first.") from err

        assert (x.plus.a.shape[1] - y.plus.a.shape[1]) == 1, "Incorrect nx size."
        assert (y.plus.a.shape[0] - x.plus.a.shape[0]) == 1, "Incorrect ny size."
        assert x.plus.a.dtype == y.plus.a.dtype, "Mismatched dtype."
        return values


class HaloRingOSC(_BaseConfig):
    """A data holder for MPI datatypes of halo rings for convenience.

    Attributes
    ----------
    win : _MPI.Win
    ss, ns, we, es : mpi4py.MPI.datatype
    sr, nr, wr, er : mpi4py.MPI.datatype
    """

    # one-sided communication window
    win: _MPI.Win

    # send datatype for conservative quantities
    ss: _MPI.Datatype
    ns: _MPI.Datatype
    ws: _MPI.Datatype
    es: _MPI.Datatype

    # recv datatype for conservative quantities
    sr: _MPI.Datatype
    nr: _MPI.Datatype
    wr: _MPI.Datatype
    er: _MPI.Datatype

    @_root_validator(pre=False, skip_on_failure=True)
    def _val_range(cls, vals):  # pylint: disable=no-self-argument, no-self-use
        """Validate subarrays."""

        sends = {k: vals[f"{k}s"].Get_contents()[0] for k in ("w", "e", "s", "n")}
        recvs = {k: vals[f"{k}r"].Get_contents()[0] for k in ("w", "e", "s", "n")}

        ndim = sends["w"][0]
        assert ndim == sends["e"][0], "ws and es have different dimensions."
        assert ndim == sends["s"][0], "ws and ss have different dimensions."
        assert ndim == sends["n"][0], "ws and ns have different dimensions."
        assert ndim == recvs["w"][0], "ws and wr have different dimensions."
        assert ndim == recvs["e"][0], "ws and er have different dimensions."
        assert ndim == recvs["s"][0], "ws and sr have different dimensions."
        assert ndim == recvs["n"][0], "ws and nr have different dimensions."

        # only check the sending array shapes because receivers may have different global shapes
        gshape = sends["w"][1:1+ndim]
        assert gshape == sends["e"][1:1+ndim], "ws and es have different global shapes."
        assert gshape == sends["s"][1:1+ndim], "ws and ss have different global shapes."
        assert gshape == sends["n"][1:1+ndim], "ws and ns have different global shapes."

        bg, ed = 1 + ndim, 1 + 2 * ndim  # pylint: disable=invalid-name
        assert sends["w"][bg:ed] == recvs["w"][bg:ed], "ws and wr have different subarray shapes."
        assert sends["e"][bg:ed] == recvs["e"][bg:ed], "es and er have different subarray shapes."
        assert sends["s"][bg:ed] == recvs["s"][bg:ed], "ss and sr have different subarray shapes."
        assert sends["n"][bg:ed] == recvs["n"][bg:ed], "ns and nr have different subarray shapes."

        return vals


class States(_BaseConfig):
    """A jumbo data model of all arrays on a mesh patch.

    A brief overview of the structure in this jumbo model (ignoring scalars):
    State: {
        q: ndarray                                          # shape: (3, ny+2*ngh, nx+2*ngh)
        p: ndarray                                          # shape: (3, ny+2*ngh, nx+2*ngh)
        s: ndarray                                          # shape: (3, ny, nx)
        ss: ndarray                                         # shape: (3, ny, nx)
        face: {
            x: {
                plus: {
                    q: ndarray                              # shape: (3, ny, nx+1)
                    p: ndarray                              # shape: (3, ny, nx+1)
                    a: ndarray                              # shape: (ny, nx+1)
                    f: ndarray                              # shape: (3, ny, nx+1)
                },
                minus: {
                    q: ndarray                              # shape: (3, ny, nx+1)
                    p: ndarray                              # shape: (3, ny, nx+1)
                    a: ndarray                              # shape: (ny, nx+1)
                    f: ndarray                              # shape: (3, ny, nx+1)
                },
                cf: ndarray                                  # shape: (3, ny, nx+1)
            },
            y: {                                            # shape: (ny+1, nx)
                plus: {
                    q: ndarray                              # shape: (3, ny+1, nx)
                    p: ndarray                              # shape: (3, ny+1, nx)
                    a: ndarray                              # shape: (ny+1, nx)
                    f: ndarray                              # shape: (3, ny+1, nx)
                },
                minus: {
                    q: ndarray                              # shape: (3, ny+1, nx)
                    p: ndarray                              # shape: (3, ny+1, nx)
                    a: ndarray                              # shape: (ny+1, nx)
                    f: ndarray                              # shape: (3, ny+1, nx)
                },
                cf: ndarray                                  # shape: (3, ny+1, nx)
            }
        },
        slpx: ndarray                                       # shape: (3, ny, nx+2)
        slpy: ndarray                                       # shape: (3, ny+2, nx)
    }

    Attributes
    ----------
    domain : torchswe.utils.data.Domain
        The domain associated to this state object.
    q : nplike.ndarray of shape (3, ny+2*nhalo, nx+2*nhalo)
        The conservative quantities defined at cell centers.
    p : nplike.ndarray of shape (3, ny+2*nhalo, nx+2*nhalo)
        The non-conservative quantities defined at cell centers.
    s : nplike.ndarray of shape (3, ny, nx)
        The explicit right-hand-side terms when during time integration. Defined at cell centers.
    ss : nplike.ndarray of shape (3, ny, nx)
        The stiff right-hand-side term that require semi-implicit handling. Defined at cell centers.
    face : torchswe.utils.data.FaceQuantityModel
        Holding quantites defined at cell faces, including continuous and discontinuous ones.
    osc : torchswe.utils.misc.DummyDict
        An object holding MPI datatypes for one-sided communications of halo rings.
    """

    # associated domain
    domain: _Domain

    # one-sided communication windows and datatypes
    osc: _DummyDict

    # quantities defined at cell centers and faces
    q: _nplike.ndarray
    p: _nplike.ndarray
    s: _nplike.ndarray
    ss: _Optional[_nplike.ndarray]
    face: FaceQuantityModel

    # intermediate quantities that we want to pre-allocate memory to save time allocating memory
    slpx: _nplike.ndarray
    slpy: _nplike.ndarray

    @_validator("osc")
    def _val_osc(cls, val):  # pylint: disable=no-self-argument, no-self-use
        """Manually validate each item in the osc field.
        """

        for name in ["q"]:
            assert name in val, f"The solver expected \"{name}\" in the osc field."

            if not isinstance(val[name], HaloRingOSC):
                raise TypeError(f"osc.{name} is not a HaloRingOSC (got {val[name].__class__})")

            val[name].check()  # triger HaloRingOSC's validation

        return val

    @_root_validator(pre=False, skip_on_failure=True)
    def _val_all(cls, values):  # pylint: disable=no-self-argument, no-self-use
        """Validate shapes and dtypes.
        """

        # aliases
        ny, nx = values["domain"].shape
        ngh = values["domain"].nhalo
        dtype = values["domain"].dtype

        assert values["q"].shape == (3, ny+2*ngh, nx+2*ngh), "q: incorrect shape"
        assert values["q"].dtype == dtype, "q: incorrect dtype"

        assert values["p"].shape == (3, ny+2*ngh, nx+2*ngh), "p: incorrect shape"
        assert values["p"].dtype == dtype, "p: incorrect dtype"

        assert values["s"].shape == (3, ny, nx), "s: incorrect shape"
        assert values["s"].dtype == dtype, "s: incorrect dtype"

        assert values["face"].x.plus.q.shape == (3, ny, nx+1), "face.x: incorrect shape"
        assert values["face"].x.plus.q.dtype == dtype, "face.x: incorrect dtype"

        assert values["face"].y.plus.q.shape == (3, ny+1, nx), "face.y: incorrect shape"
        assert values["face"].y.plus.q.dtype == dtype, "face.y: incorrect dtype"

        if values["ss"] is not None:
            assert values["ss"].shape == (3, ny, nx), "ss: incorrect shape"
            assert values["ss"].dtype == dtype, "ss: incorrect dtype"

        assert values["slpx"].shape == (3, ny, nx+2), "slpx: incorrect shape"
        assert values["slpx"].dtype == dtype, "slpx: incorrect dtype"

        assert values["slpy"].shape == (3, ny+2, nx), "slpy: incorrect shape"
        assert values["slpy"].dtype == dtype, "slpy: incorrect dtype"

        return values

    @_root_validator(pre=False, skip_on_failure=True)
    def _val_q_subarray_types(cls, values):  # pylint: disable=no-self-argument, no-self-use
        """Validate the exchanging mechanism of data.
        """

        # aliases
        domain = values["domain"]
        comm = domain.comm
        ny, nx = domain.shape
        ngh = values["domain"].nhalo
        osc = values["osc"].q

        data = _nplike.zeros((3, ny+2*ngh, nx+2*ngh), dtype=values["q"].dtype)
        data[0, ngh:-ngh, ngh:-ngh] = comm.rank * 1000
        data[1, ngh:-ngh, ngh:-ngh] = comm.rank * 1000 + 100
        data[2, ngh:-ngh, ngh:-ngh] = comm.rank * 1000 + 200
        _nplike.sync()

        win = _MPI.Win.Create(data, comm=comm)
        win.Fence()
        for k in ("s", "n", "w", "e"):
            win.Put([data, osc[f"{k}s"]], domain[k], [0, 1, osc[f"{k}r"]])
        win.Fence()

        # check if correct neighbors put correct data to this rank
        if domain.s != _MPI.PROC_NULL:
            assert all(data[0, :ngh, ngh:-ngh].flatten() == domain.s*1000)
            assert all(data[1, :ngh, ngh:-ngh].flatten() == domain.s*1000+100)
            assert all(data[2, :ngh, ngh:-ngh].flatten() == domain.s*1000+200)

        if domain.n != _MPI.PROC_NULL:
            assert all(data[0, -ngh:, ngh:-ngh].flatten() == domain.n*1000)
            assert all(data[1, -ngh:, ngh:-ngh].flatten() == domain.n*1000+100)
            assert all(data[2, -ngh:, ngh:-ngh].flatten() == domain.n*1000+200)

        if domain.w != _MPI.PROC_NULL:
            assert all(data[0, ngh:-ngh, :ngh].flatten() == domain.w*1000)
            assert all(data[1, ngh:-ngh, :ngh].flatten() == domain.w*1000+100)
            assert all(data[2, ngh:-ngh, :ngh].flatten() == domain.w*1000+200)

        if domain.e != _MPI.PROC_NULL:
            assert all(data[0, ngh:-ngh, -ngh:].flatten() == domain.e*1000)
            assert all(data[1, ngh:-ngh, -ngh:].flatten() == domain.e*1000+100)
            assert all(data[2, ngh:-ngh, -ngh:].flatten() == domain.e*1000+200)
        win.Free()

        return values


def _get_osc_conservative_mpi_datatype(comm: MPI.Cartcomm, arry: ndarray, ngh: int):
    """Get the halo ring MPI datatypes for conservative quantities for one-sided communications.
    """

    # make sure GPU has done the calculations
    _nplike.sync()

    # shape and dtype
    ny, nx = arry.shape[1:]
    ny, nx = ny - 2 * ngh, nx - 2 * ngh
    mtype: _MPI.Datatype = _from_numpy_dtype(arry.dtype)

    # data holder
    data = _DummyDict()

    # the window for one-sided communication for Q
    data.win = _MPI.Win.Create(arry, comm=comm)
    data.win.Fence()

    # set up custom MPI datatype for exchangine data in Q (Q's shape: (3, ny+2*ngh, nx+2*ngh))
    data.ss = mtype.Create_subarray(arry.shape, (3, ngh, nx), (0, ngh, ngh)).Commit()
    data.ns = mtype.Create_subarray(arry.shape, (3, ngh, nx), (0, ny, ngh)).Commit()
    data.ws = mtype.Create_subarray(arry.shape, (3, ny, ngh), (0, ngh, ngh)).Commit()
    data.es = mtype.Create_subarray(arry.shape, (3, ny, ngh), (0, ngh, nx)).Commit()

    # get neighbors shapes of Q
    nshps = _nplike.tile(_nplike.array(arry.shape), (4,))
    temp = _nplike.tile(_nplike.array(arry.shape), (4,))
    int_t: _MPI.Datatype = _from_numpy_dtype(temp.dtype)

    _nplike.sync()
    comm.Neighbor_alltoall([temp, int_t], [nshps, int_t])
    nshps = nshps.reshape(4, len(arry.shape))

    # get neighbors y.n and x.n
    nys = nshps[:, 1].flatten() - 2 * ngh
    nxs = nshps[:, 2].flatten() - 2 * ngh
    _nplike.sync()

    # the receiving buffer's datatype should be defined wiht neighbors' shapes
    data.sr = mtype.Create_subarray(nshps[0], (3, ngh, nxs[0]), (0, ngh+nys[0], ngh)).Commit()
    data.nr = mtype.Create_subarray(nshps[1], (3, ngh, nxs[1]), (0, 0, ngh)).Commit()
    data.wr = mtype.Create_subarray(nshps[2], (3, nys[2], ngh), (0, ngh, ngh+nxs[2])).Commit()
    data.er = mtype.Create_subarray(nshps[3], (3, nys[3], ngh), (0, ngh, 0)).Commit()

    return HaloRingOSC(**data)


def get_empty_states(config: Config, domain: Domain = None, comm: MPI.Comm = None):
    """Get an empty (i.e., zero arrays) States.

    Arguments
    ---------
    domain : torchswe.utils.data.Domain
    ngh : int

    Returns
    -------
    A States with zero arrays.
    """

    # to hold data for initializing a Domain instance
    data = _DummyDict()

    # if domain is not provided, get a new one
    if domain is None:
        comm = _MPI.COMM_WORLD if comm is None else comm
        data.domain = domain = _get_domain(comm, config)
    else:
        data.domain = domain

    # aliases
    ny, nx = data.domain.shape
    dtype = data.domain.dtype
    ngh = data.domain.nhalo

    # cell-centered arrays
    data.q = _nplike.zeros((3, ny+2*ngh, nx+2*ngh), dtype=dtype)
    data.p = _nplike.zeros((3, ny+2*ngh, nx+2*ngh), dtype=dtype)
    data.s = _nplike.zeros((3, ny, nx), dtype=dtype)
    data.ss = _nplike.zeros((3, ny, nx), dtype=dtype) if config.friction is not None else None
    data.slpx = _nplike.zeros((3, ny, nx+2), dtype=dtype)
    data.slpy = _nplike.zeros((3, ny+2, nx), dtype=dtype)

    # quantities on faces
    data.face = FaceQuantityModel(
        x=FaceTwoSideModel(
            plus=FaceOneSideModel(
                q=_nplike.zeros((3, ny, nx+1), dtype=dtype),
                p=_nplike.zeros((3, ny, nx+1), dtype=dtype),
                a=_nplike.zeros((ny, nx+1), dtype=dtype),
                f=_nplike.zeros((3, ny, nx+1), dtype)
            ),
            minus=FaceOneSideModel(
                q=_nplike.zeros((3, ny, nx+1), dtype=dtype),
                p=_nplike.zeros((3, ny, nx+1), dtype=dtype),
                a=_nplike.zeros((ny, nx+1), dtype=dtype),
                f=_nplike.zeros((3, ny, nx+1), dtype)
            ),
            cf=_nplike.zeros((3, ny, nx+1), dtype)
        ),
        y=FaceTwoSideModel(
            plus=FaceOneSideModel(
                q=_nplike.zeros((3, ny+1, nx), dtype=dtype),
                p=_nplike.zeros((3, ny+1, nx), dtype=dtype),
                a=_nplike.zeros((ny+1, nx), dtype=dtype),
                f=_nplike.zeros((3, ny+1, nx), dtype)
            ),
            minus=FaceOneSideModel(
                q=_nplike.zeros((3, ny+1, nx), dtype=dtype),
                p=_nplike.zeros((3, ny+1, nx), dtype=dtype),
                a=_nplike.zeros((ny+1, nx), dtype=dtype),
                f=_nplike.zeros((3, ny+1, nx), dtype)
            ),
            cf=_nplike.zeros((3, ny+1, nx), dtype)
        ),
    )

    # get one-sided communication windows and datatypes
    data.osc = _DummyDict()
    data.osc.q = _get_osc_conservative_mpi_datatype(data.domain.comm, data.q, ngh)

    return States(**data)


def get_initial_states(config: Config, domain: Domain = None, comm: MPI.Comm = None):
    """Get a States instance filled with initial conditions.

    Arguments
    ---------

    Returns
    -------
    torchswe.utils.data.States

    Notes
    -----
    When x and y axes have different resolutions from the x and y in the NetCDF file, an bi-cubic
    spline interpolation will take place.
    """

    # get an empty states
    states = get_empty_states(config, domain, comm)

    # rebind; aliases
    domain = states.domain

    # special case: constant I.C.
    if config.ic.values is not None:
        states.q[(slice(None),)+domain.nonhalo_c] = _nplike.array(config.ic.values).reshape(3, 1, 1)
        states.check()
        return states

    # otherwise, read data from a NetCDF file
    data = _read_block(config.ic.file, config.ic.xykeys, config.ic.keys, domain.lextent_c, domain)

    # see if we need to do interpolation
    try:
        interp = not (_nplike.allclose(domain.x.c, data.x) and _nplike.allclose(domain.y.c, data.y))
    except ValueError:  # assume thie excpetion means a shape mismatch
        interp = True

    # unfortunately, we need to do interpolation in such a situation
    if interp:
        _logger.warning("Grids do not match. Doing spline interpolation.")
        for i in range(3):
            states.q[(i,)+domain.nonhalo_c] = _nplike.array(
                _interpolate(data.x, data.y, data[config.ic.keys[i]].T, domain.x.c, domain.y.c).T
            )
    else:
        for i in range(3):
            states.q[(i,)+domain.nonhalo_c] = data[config.ic.keys[i]]

    states.check()
    return states
