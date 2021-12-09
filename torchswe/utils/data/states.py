#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Data models.
"""
from typing import Optional as _Optional
from typing import Tuple as _Tuple
from typing import Union as _Union

from mpi4py import MPI as _MPI
from pydantic import validator as _validator
from pydantic import root_validator as _root_validator
from torchswe import nplike as _nplike
from torchswe.utils.config import BaseConfig as _BaseConfig
from torchswe.utils.misc import DummyDict as _DummyDict
from torchswe.utils.data.grid import Domain as _Domain


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
    U : nplike.ndarray of shape (3, ny+1, nx) or (3, ny, nx+1)
        The fluid depth, u-velocity, and depth-v-velocity.
    a : nplike.ndarray of shape (ny+1, nx) or (3, ny, nx+1)
        The local speed.
    F : nplike.ndarray of shape (3, ny+1, nx) or (3, ny, nx+1)
        An array holding discontinuous fluxes.
    """

    Q: _nplike.ndarray
    U: _nplike.ndarray
    a: _nplike.ndarray
    F: _nplike.ndarray

    # validator
    _val_valid_numbers = _validator("Q", "U", "a", "F", allow_reuse=True)(_pydantic_val_nan_inf)

    @_root_validator(pre=False, skip_on_failure=True)
    def _val_arrays(cls, values):  # pylint: disable=no-self-argument, no-self-use
        """Validate the consistency of the arrays shapes and dtypes."""
        # pylint: disable=invalid-name

        try:
            Q, U, a, F = values["Q"], values["U"], values["a"], values["F"]
        except KeyError as err:
            raise AssertionError("Fix other fields first.") from err

        n1, n2 = a.shape
        assert Q.shape == (3, n1, n2), f"U shape mismatch. Should be {(3, n1, n2)}. Got {U.shape}."
        assert U.shape == (3, n1, n2), f"U shape mismatch. Should be {(3, n1, n2)}. Got {U.shape}."
        assert F.shape == (3, n1, n2), f"F shape mismatch. Should be {(3, n1, n2)}. Got {F.shape}."
        assert Q.dtype == a.dtype, f"dtype mismatch. Should be {a.dtype}. Got {U.dtype}."
        assert U.dtype == a.dtype, f"dtype mismatch. Should be {a.dtype}. Got {U.dtype}."
        assert F.dtype == a.dtype, f"dtype mismatch. Should be {a.dtype}. Got {U.dtype}."
        return values


class FaceTwoSideModel(_BaseConfig):
    """Date model holding quantities on both sides of cell faces normal to one direction.

    Attributes
    ----------
    plus, minus : FaceOneSideModel
        Objects holding data on one side of each face.
    H : nplike.ndarray of shape (3, ny+1, nx) or (3, ny, nx+1)
        An object holding common (i.e., continuous or numerical) flux
    """

    plus: FaceOneSideModel
    minus: FaceOneSideModel
    H: _nplike.ndarray

    # validator
    _val_valid_numbers = _validator("H", allow_reuse=True)(_pydantic_val_nan_inf)

    @_root_validator(pre=False, skip_on_failure=True)
    def _val_arrays(cls, values):  # pylint: disable=no-self-argument, no-self-use
        """Validate shapes and dtypes."""
        # pylint: disable=invalid-name

        try:
            plus, minus, H = values["plus"], values["minus"], values["H"]
        except KeyError as err:
            raise AssertionError("Fix other fields first.") from err

        assert plus.U.shape == minus.U.shape, f"Shape mismatch: {plus.U.shape} and {minus.U.shape}."
        assert plus.U.dtype == minus.U.dtype, f"dtype mismatch: {plus.U.dtype} and {minus.U.dtype}."
        assert plus.U.shape == H.shape, f"Shape mismatch: {plus.U.shape} and {H.shape}."
        assert plus.U.dtype == H.dtype, f"dtype mismatch: {plus.U.dtype} and {H.dtype}."
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
        Q: ndarray                                          # shape: (3, ny+2*ngh, nx+2*ngh)
        U: ndarray                                          # shape: (3, ny+2*ngh, nx+2*ngh)
        S: ndarray                                          # shape: (3, ny, nx)
        SS: ndarray                                         # shape: (3, ny, nx)
        face: {
            x: {
                plus: {
                    Q: ndarray                              # shape: (3, ny, nx+1)
                    U: ndarray                              # shape: (3, ny, nx+1)
                    a: ndarray                              # shape: (ny, nx+1)
                    F: ndarray                              # shape: (3, ny, nx+1)
                },
                minus: {
                    Q: ndarray                              # shape: (3, ny, nx+1)
                    U: ndarray                              # shape: (3, ny, nx+1)
                    a: ndarray                              # shape: (ny, nx+1)
                    F: ndarray                              # shape: (3, ny, nx+1)
                },
                H: ndarray                                  # shape: (3, ny, nx+1)
            },
            y: {                                            # shape: (ny+1, nx)
                plus: {
                    Q: ndarray                              # shape: (3, ny+1, nx)
                    U: ndarray                              # shape: (3, ny+1, nx)
                    a: ndarray                              # shape: (ny+1, nx)
                    F: ndarray                              # shape: (3, ny+1, nx)
                },
                minus: {
                    U: ndarray                              # shape: (3, ny+1, nx)
                    U: ndarray                              # shape: (3, ny+1, nx)
                    a: ndarray                              # shape: (ny+1, nx)
                    F: ndarray                              # shape: (3, ny+1, nx)
                },
                H: ndarray                                  # shape: (3, ny+1, nx)
            }
        },
        slpx: ndarray                                       # shape: (3, ny, nx+2)
        slpy: ndarray                                       # shape: (3, ny+2, nx)
    }

    Attributes
    ----------
    domain : torchswe.utils.data.Domain
        The domain associated to this state object.
    Q : nplike.ndarray of shape (3, ny+2*nhalo, nx+2*nhalo)
        The conservative quantities defined at cell centers.
    U : nplike.ndarray of shape (3, ny+2*nhalo, nx+2*nhalo)
        The non-conservative quantities defined at cell centers.
    S : nplike.ndarray of shape (3, ny, nx)
        The explicit right-hand-side terms when during time integration. Defined at cell centers.
    SS : nplike.ndarray of shape (3, ny, nx)
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
    Q: _nplike.ndarray
    U: _nplike.ndarray
    S: _nplike.ndarray
    SS: _Optional[_nplike.ndarray]
    face: FaceQuantityModel

    # intermediate quantities that we want to pre-allocate memory to save time allocating memory
    slpx: _nplike.ndarray
    slpy: _nplike.ndarray

    @_validator("osc")
    def _val_osc(cls, val):  # pylint: disable=no-self-argument, no-self-use
        """Manually validate each item in the osc field.
        """

        for name in ["Q"]:
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

        assert values["Q"].shape == (3, ny+2*ngh, nx+2*ngh), "Q: incorrect shape"
        assert values["Q"].dtype == dtype, "Q: incorrect dtype"

        assert values["U"].shape == (3, ny+2*ngh, nx+2*ngh), "U: incorrect shape"
        assert values["U"].dtype == dtype, "U: incorrect dtype"

        assert values["S"].shape == (3, ny, nx), "S: incorrect shape"
        assert values["S"].dtype == dtype, "S: incorrect dtype"

        assert values["face"].x.plus.U.shape == (3, ny, nx+1), "face.x: incorrect shape"
        assert values["face"].x.plus.U.dtype == dtype, "face.x: incorrect dtype"

        assert values["face"].y.plus.U.shape == (3, ny+1, nx), "face.y: incorrect shape"
        assert values["face"].y.plus.U.dtype == dtype, "face.y: incorrect dtype"

        if values["SS"] is not None:
            assert values["SS"].shape == (3, ny, nx), "SS: incorrect shape"
            assert values["SS"].dtype == dtype, "SS: incorrect dtype"

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
        osc = values["osc"].Q

        data = _nplike.zeros((3, ny+2*ngh, nx+2*ngh), dtype=values["Q"].dtype)
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
