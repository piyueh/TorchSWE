#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Data models.
"""
# pylint: disable=too-few-public-methods, no-self-argument, invalid-name, no-self-use
import copy
from typing import Literal, Tuple, List, Union

from pydantic import validator, conint, confloat
from scipy.interpolate import RectBivariateSpline
from torchswe import nplike
from torchswe.utils.config import BaseConfig, TemporalConfig, SpatialConfig, TopoConfig
from torchswe.utils.netcdf import read_cf
from torchswe.utils.dummy import DummyDtype


def _pydantic_val_dtype(val: nplike.ndarray, values: dict) -> nplike.ndarray:
    """Validating a given ndarray has dtype matching dtype; used by pydantic."""
    try:
        assert val.dtype == values["dtype"], \
            "float number type mismatch. Should be {}, got {}".format(values["dtype"], val.dtype)
    except KeyError as err:
        raise AssertionError("Validation failed due to other validation failures.") from err
    return val


def _pydantic_val_arrays(val, values):
    """Validate arrays under the same data model, i.e., sharing the same shape and dtype."""
    try:
        shape = (values["ny"], values["nx"])
        dtype = values["dtype"]
    except KeyError as err:
        raise AssertionError("Validation failed due to other validation failures.") from err

    assert val.dtype == dtype, "Dtype mismatch. Should be {}, got {}".format(dtype, val.dtype)
    assert val.shape == shape, "Shape mismatch. Should be {}, got {}".format(shape, val.shape)

    return val


def _pydantic_val_nan_inf(val, field):
    assert not nplike.any(nplike.isnan(val)), "Got NaN in {}".format(field.name)
    assert not nplike.any(nplike.isinf(val)), "Got Inf in {}".format(field.name)
    return val


def _shape_val_factory(shift: Union[Tuple[int, int], Literal["ghost"], int]):
    def _core_func(val, values):
        try:
            if isinstance(shift, int):
                target = (values["n"]+shift,)
            elif shift == "ghost":
                target = (values["ny"]+2*values["n_ghost"], values["nx"]+2*values["n_ghost"])
            else:
                target = (values["ny"]+shift[0], values["nx"]+shift[1])
        except KeyError as err:
            raise AssertionError("Validation failed due to other validation failures.") from err

        assert val.shape == target, "Shape mismatch. Should be {}, got {}".format(target, val.shape)
        return val

    return _core_func


class Gridline(BaseConfig):
    """Gridline data model.

    Attributes
    ----------
    direction: either "x" or "y"
    n: number of cells
    start: lower bound
    end: higher bound
    delta: float; cell size in the corresponding direction.
    dtype: "float32", "float64", nplike.float32, nplike64.
    vert: 1D array of length n+1; coordinates at vertices.
    cntr: 1D array of length n; coordinates at cell centers.
    xface: 1D array of langth n+1 or n; coordinates at the cell faces normal to x-axis.
    yface: 1D array of langth n or n+1; coordinates at the cell faces normal to y-axis.

    Notes
    -----
    The lengths of xface and yface depend on the direction.
    """
    direction: Literal["x", "y"]
    n: conint(gt=0)
    start: float
    end: float
    delta: confloat(gt=0.)
    dtype: DummyDtype
    vert: nplike.ndarray
    cntr: nplike.ndarray
    xface: nplike.ndarray
    yface: nplike.ndarray

    # common validators
    _val_vert = validator('vert', allow_reuse=True)(_shape_val_factory(1))
    _val_cntr = validator('cntr', allow_reuse=True)(_shape_val_factory(0))
    _val_dtypes = validator("vert", "cntr", "xface", "yface", allow_reuse=True)(_pydantic_val_dtype)

    @validator("vert", "cntr", "xface", "yface")
    def _val_linspace(cls, v, values):
        """Make sure the linspace is working correctly."""
        diff = v[1:] - v[:-1]
        assert nplike.all(diff > 0), "Not in monotonically increasing order."
        assert nplike.allclose(diff, values["delta"], atol=1e-10), "Delta does not match."
        return v

    @validator("xface")
    def _val_xface(cls, v, values):
        if values["direction"] == "x":
            return _shape_val_factory(1)(v, values)
        return _shape_val_factory(0)(v, values)

    @validator("yface")
    def _val_yface(cls, v, values):
        if values["direction"] == "x":
            return _shape_val_factory(0)(v, values)
        return _shape_val_factory(1)(v, values)


class Gridlines(BaseConfig):
    """Gridlines.

    Attributes
    ----------
    x, y: Gridline object; representing x and y grindline coordinates.
    t: a list; time values for outputing solutions.
    """
    x: Gridline
    y: Gridline
    t: List[float]


class Topography(BaseConfig):
    """Data model for topography elevation.

    Attributes
    ----------
    vert : (Ny+1, Nx+1) array; elevation at vertices.
    cntr : (Ny, Nx) array; elevation at cell centers.
    xface : (Ny, Nx+1) array; elevation at cell faces normal to x-axis.
    yface : (Ny+1, Nx) array; elevation at cell faces normal to y-axis.
    xgrad : (Ny, Nx) array; derivatives w.r.t. x at cell centers.
    ygrad : (Ny, Nx) array; derivatives w.r.t. y at cell centers.
    """
    nx: conint(gt=0)
    ny: conint(gt=0)
    dtype: DummyDtype
    vert: nplike.ndarray
    cntr: nplike.ndarray
    xface: nplike.ndarray
    yface: nplike.ndarray
    xgrad: nplike.ndarray
    ygrad: nplike.ndarray

    # validators
    _val_dtypes = validator(
        "vert", "cntr", "xface", "yface", "xgrad", "ygrad",
        allow_reuse=True)(_pydantic_val_dtype)
    _val_ny_1_nx_1 = validator("vert", allow_reuse=True)(_shape_val_factory([1, 1]))
    _val_ny_nx_1 = validator("xface", allow_reuse=True)(_shape_val_factory([0, 1]))
    _val_ny_1_nx = validator("yface", allow_reuse=True)(_shape_val_factory([1, 0]))
    _val_ny_nx = validator("cntr", "xgrad", "ygrad", allow_reuse=True)(_shape_val_factory([0, 0]))


class DummyDataModel:
    """A dummy class as a base for those needs the property `shape`."""
    @property
    def shape(self):
        "Shape of the arrays in this object."
        return (self.ny, self.nx)  # pylint: disable=no-member


class WHUHVModel(BaseConfig, DummyDataModel):
    """Data model with keys w, hu, and v."""
    nx: conint(gt=0)
    ny: conint(gt=0)
    dtype: DummyDtype
    w: nplike.ndarray
    hu: nplike.ndarray
    hv: nplike.ndarray

    # validators
    _val_arrays = validator("w", "hu", "hv", allow_reuse=True)(_pydantic_val_arrays)
    _val_valid_numbers = validator("w", "hu", "hv", allow_reuse=True)(_pydantic_val_nan_inf)


class HUVModel(BaseConfig, DummyDataModel):
    """Data model with keys h, u, and v."""
    nx: conint(gt=0)
    ny: conint(gt=0)
    dtype: DummyDtype
    h: nplike.ndarray
    u: nplike.ndarray
    v: nplike.ndarray

    # validators
    _val_arrays = validator("h", "u", "v", allow_reuse=True)(_pydantic_val_arrays)
    _val_valid_numbers = validator("h", "u", "v", allow_reuse=True)(_pydantic_val_nan_inf)


class FaceOneSideModel(BaseConfig, DummyDataModel):
    """Data model holding quantities on one side of cell faces normal to one direction."""
    nx: conint(gt=0)
    ny: conint(gt=0)
    dtype: DummyDtype
    w: nplike.ndarray
    hu: nplike.ndarray
    hv: nplike.ndarray
    h: nplike.ndarray
    u: nplike.ndarray
    v: nplike.ndarray
    a: nplike.ndarray
    flux: WHUHVModel

    # validator
    _val_arrays = validator(
        "w", "hu", "hv", "h", "u", "v", "a", "flux", allow_reuse=True)(_pydantic_val_arrays)
    _val_valid_numbers = validator(
        "w", "hu", "hv", "h", "u", "v", "a", allow_reuse=True)(_pydantic_val_nan_inf)


class FaceTwoSideModel(BaseConfig, DummyDataModel):
    """Date model holding quantities on both sides of cell faces normal to one direction."""
    nx: conint(gt=0)
    ny: conint(gt=0)
    dtype: DummyDtype
    plus: FaceOneSideModel
    minus: FaceOneSideModel
    num_flux: WHUHVModel

    # validator
    _val_arrays = validator("plus", "minus", "num_flux", allow_reuse=True)(_pydantic_val_arrays)


class FaceQuantityModel(BaseConfig):
    """Data model holding quantities on both sides of cell faces in both x and y directions."""
    nx: conint(gt=0)
    ny: conint(gt=0)
    dtype: DummyDtype
    x: FaceTwoSideModel
    y: FaceTwoSideModel

    @validator("x", "y")
    def _val_arrays(cls, v, values, field):
        try:
            if field.name == "x":
                shape = (values["ny"], values["nx"]+1)
            else:
                shape = (values["ny"]+1, values["nx"])
            dtype = values["dtype"]
        except KeyError as err:
            raise AssertionError("Validation failed due to other validation failures.") from err

        assert v.dtype == dtype, "Dtype mismatch. Should be {}, got {}".format(dtype, v.dtype)
        assert v.shape == shape, "Shape mismatch. Should be {}, got {}".format(shape, v.shape)
        return v


class Slopes(BaseConfig):
    """Data model for slopes at cell centers."""
    nx: conint(gt=0)
    ny: conint(gt=0)
    dtype: DummyDtype
    x: WHUHVModel
    y: WHUHVModel

    @validator("x", "y")
    def _val_arrays(cls, v, values, field):
        try:
            if field.name == "x":
                shape = (values["ny"], values["nx"]+2)
            else:
                shape = (values["ny"]+2, values["nx"])
            dtype = values["dtype"]
        except KeyError as err:
            raise AssertionError("Validation failed due to other validation failures.") from err

        assert v.dtype == dtype, "Dtype mismatch. Should be {}, got {}".format(dtype, v.dtype)
        assert v.shape == shape, "Shape mismatch. Should be {}, got {}".format(shape, v.shape)
        return v


class States(BaseConfig, DummyDataModel):
    """A jumbo data model of all arrays on a mesh patch.

    A brief overview of the structure in this jumbo model (ignoring scalars):
    State: {
        q: {w: ndarray hu: ndarray hv: ndarray},            # shape: (ny+2*ngh, nx+2*ngh)
        src: {w: ndarray hu: ndarray hv: ndarray},          # shape: (ny, nx)
        slp: {
            x: {w: ndarray hu: ndarray hv: ndarray},        # shape: (ny, nx+2)
            y: {w: ndarray hu: ndarray hv: ndarray},        # shape: (ny+2, nx)
        },
        rhs: {w: ndarray hu: ndarray hv: ndarray},          # shape: (ny, nx)
        face: {
            x: {                                            # shape: (ny, nx+1)
                plus: {
                    w: ndarray hu: ndarray hv: ndarray, h: ndarray u: ndarray v: ndarray,
                    a: ndarray,
                    flux: {w: ndarray, hu: ndarray, hv: ndarray}
                },
                minus: {
                    w: ndarray hu: ndarray hv: ndarray, h: ndarray u: ndarray v: ndarray,
                    a: ndarray,
                    flux: {w: ndarray, hu: ndarray, hv: ndarray}
                },
                num_flux: {w: ndarray, hu: ndarray, hv: ndarray},
            },
            y: {                                            # shape: (ny+1, nx)
                plus: {
                    w: ndarray hu: ndarray hv: ndarray, h: ndarray u: ndarray v: ndarray,
                    a: ndarray,
                    flux: {w: ndarray, hu: ndarray, hv: ndarray}
                },
                minus: {
                    w: ndarray hu: ndarray hv: ndarray, h: ndarray u: ndarray v: ndarray,
                    a: ndarray,
                    flux: {w: ndarray, hu: ndarray, hv: ndarray}
                },
                num_flux: {w: ndarray, hu: ndarray, hv: ndarray},
            }
        }
    }
    """

    # parameters
    nx: conint(gt=0)
    ny: conint(gt=0)
    ngh: conint(ge=2)
    dtype: DummyDtype

    # quantities defined at cell centers
    q: WHUHVModel
    src: WHUHVModel
    slp: Slopes
    rhs: WHUHVModel
    face: FaceQuantityModel

    @validator("q", "src", "rhs")
    def _val_1(cls, v, values, field):
        try:
            if field.name == "q":
                shape = (values["ny"]+2*values["ngh"], values["nx"]+2*values["ngh"])
            else:
                shape = (values["ny"], values["nx"])
            dtype = values["dtype"]
        except KeyError as err:
            raise AssertionError("Validation failed due to other validation failures.") from err

        assert v.dtype == dtype, "Dtype mismatch. Should be {}, got {}".format(dtype, v.dtype)
        assert v.shape == shape, "Shape mismatch. Should be {}, got {}".format(shape, v.shape)
        return v

    @validator("slp", "face")
    def _val_2(cls, v, values):
        try:
            nx = values["nx"]
            ny = values["ny"]
            dtype = values["dtype"]
        except KeyError as err:
            raise AssertionError("Validation failed due to other validation failures.") from err

        assert v.dtype == dtype, "Dtype mismatch. Should be {}, got {}".format(dtype, v.dtype)
        assert v.nx == nx, "Nx mismatch. Should be {}, got {}".format(nx, v.nx)
        assert v.ny == ny, "Ny mismatch. Should be {}, got {}".format(ny, v.ny)
        return v


def get_gridline(direction: str, n: int, start: float, end: float, dtype: str):
    """Get a Gridline object.

    Arguments
    ---------
    direction : str
        Either "x" or "y".
    n : int
        Number of cells.
    start, end : float
        Lower and upper bound of this axis.
    dtype : str, nplike.float32, or nplike.float64

    Returns
    -------
    gridline : Gridline
    """

    dtype = DummyDtype.validator(dtype)
    delta = (end - start) / n
    vert = nplike.linspace(start, end, n+1, dtype=dtype)
    cntr = nplike.linspace(start+delta/2., end-delta/2., n, dtype=dtype)

    if direction == "x":
        xface = copy.deepcopy(vert)
        yface = copy.deepcopy(cntr)
    else:  # if this is not "y", pydantic will let me know
        xface = copy.deepcopy(cntr)
        yface = copy.deepcopy(vert)

    # pydantic will validate the data here
    return Gridline(
        direction=direction, n=n, start=start, end=end, delta=delta, dtype=dtype,
        vert=vert, cntr=cntr, xface=xface, yface=yface)


def get_gridlines(spatial: SpatialConfig, temporal: TemporalConfig, dtype: str):
    """Get a Gridlines object using config object.

    Arguments
    ---------
    spatial : SpatialConfig
    temporal : TemporalConfig
    dtype : str, nplike.float32, nplike.float64

    Returns
    -------
    gridlines : Gridlines
    """

    # manually launch validation
    spatial.check()
    temporal.check()

    # write solutions to a file at give times
    if temporal.output[0] == "at":
        t = list(temporal.output[1])

    # output every `every_seconds` seconds `multiple` times from `t_start`
    elif temporal.output[0] == "t_start every_seconds multiple":
        bg, dt, n = temporal.output[1:]
        t = (nplike.arange(0, n+1) * dt + bg).tolist()  # including saving t_start

    # output every `every_steps` constant-size steps for `multiple` times from t=`t_start`
    elif temporal.output[0] == "t_start every_steps multiple":
        bg, steps, n = temporal.output[1:]
        dt = temporal.dt
        t = (nplike.arange(0, n+1) * dt * steps + bg).tolist()  # including saving t_start

    # from `t_start` to `t_end` evenly outputs `n_saves` times (including both ends)
    elif temporal.output[0] == "t_start t_end n_saves":
        bg, ed, n = temporal.output[1:]
        t = nplike.linspace(bg, ed, n+1).tolist()  # including saving t_start

    # run simulation from `t_start` to `t_end` but not saving solutions at all
    elif temporal.output[0] == "t_start t_end no save":
        t = temporal.output[1:]

    # should never reach this branch because pydantic has detected any invalid arguments
    else:
        raise ValueError("{} is not an allowed output method.".format(temporal.output[0]))

    return Gridlines(
        x=get_gridline("x", spatial.discretization[0], spatial.domain[0], spatial.domain[1], dtype),
        y=get_gridline("y", spatial.discretization[1], spatial.domain[2], spatial.domain[3], dtype),
        t=t)


def get_topography(topoconfig: TopoConfig, grid: Gridlines, dtype: str):
    """Get a Topography object from a config object.

    Arguments
    ---------
    topoconfig : TopoConfig
    grid : Gridlines
    dtype : str, nplike.float32, nplike.float64

    Returns
    -------
    topo : Topography
    """
    dtype = DummyDtype.validator(dtype)
    assert dtype == grid.x.dtype
    assert dtype == grid.y.dtype

    dem, _ = read_cf(topoconfig.file, [topoconfig.key])

    # copy to a nplike.ndarray
    vert = nplike.array(dem[topoconfig.key][:])

    # see if we need to do interpolation
    try:
        interp = not (
            nplike.allclose(grid.x.vert, nplike.array(dem["x"])) and
            nplike.allclose(grid.y.vert, nplike.array(dem["y"])))
    except ValueError:  # assume thie excpetion means a shape mismatch
        interp = True

    # unfortunately, we need to do interpolation in such a situation
    if interp:
        interpolator = RectBivariateSpline(dem["x"], dem["y"], vert.T)
        vert = nplike.array(interpolator(grid.x.vert, grid.y.vert).T)  # it uses vanilla numpy

    # cast to desired float type
    vert = vert.astype(dtype)

    # topography elevation at cell centers through linear interpolation
    cntr = vert[:-1, :-1] + vert[:-1, 1:] + vert[1:, :-1] + vert[1:, 1:]
    cntr /= 4

    # topography elevation at cell faces' midpoints through linear interpolation
    xface = (vert[:-1, :] + vert[1:, :]) / 2.
    yface = (vert[:, :-1] + vert[:, 1:]) / 2.

    # gradient at cell centers through central difference; here allows nonuniform grids
    # this function does not assume constant cell sizes, so we re-calculate dx, dy
    # the `delta`s in grid.x and y are constants (current solver only supports uniform grid)
    xgrad = (xface[:, 1:] - xface[:, :-1]) / (grid.x.vert[1:] - grid.x.vert[:-1])[None, :]
    ygrad = (yface[1:, :] - yface[:-1, :]) / (grid.y.vert[1:] - grid.y.vert[:-1])[:, None]

    # initialize DataModel and let pydantic validates data
    return Topography(
        nx=grid.x.n, ny=grid.y.n, dtype=dtype, vert=vert, cntr=cntr, xface=xface,
        yface=yface, xgrad=xgrad, ygrad=ygrad)


def get_empty_whuhvmodel(nx: int, ny: int, dtype: str):
    """Get an empty (i.e., zero arrays) WHUHVModel.

    Arguments
    ---------
    nx, ny : int
    dtype : str, nplike.float32, nplike.float64

    Returns
    -------
    A WHUHVModel with zero arrays.
    """
    dtype = DummyDtype.validator(dtype)
    w = nplike.zeros((ny, nx), dtype=dtype)
    hu = nplike.zeros((ny, nx), dtype=dtype)
    hv = nplike.zeros((ny, nx), dtype=dtype)
    return WHUHVModel(nx=nx, ny=ny, dtype=dtype, w=w, hu=hu, hv=hv)


def get_empty_huvmodel(nx: int, ny: int, dtype: str):
    """Get an empty (i.e., zero arrays) HUVModel.

    Arguments
    ---------
    nx, ny : int
    dtype : str, nplike.float32, nplike.float64

    Returns
    -------
    A HUVModel with zero arrays.
    """
    dtype = DummyDtype.validator(dtype)
    h = nplike.zeros((ny, nx), dtype=dtype)
    u = nplike.zeros((ny, nx), dtype=dtype)
    v = nplike.zeros((ny, nx), dtype=dtype)
    return WHUHVModel(nx=nx, ny=ny, dtype=dtype, h=h, u=u, v=v)


def get_empty_faceonesidemodel(nx: int, ny: int, dtype: str):
    """Get an empty (i.e., zero arrays) FaceOneSideModel.

    Arguments
    ---------
    nx, ny : int
    dtype : str, nplike.float32, nplike.float64

    Returns
    -------
    A FaceOneSideModel with zero arrays.
    """
    dtype = DummyDtype.validator(dtype)
    return FaceOneSideModel(
        nx=nx, ny=ny, dtype=dtype, w=nplike.zeros((ny, nx), dtype=dtype),
        hu=nplike.zeros((ny, nx), dtype=dtype), hv=nplike.zeros((ny, nx), dtype=dtype),
        h=nplike.zeros((ny, nx), dtype=dtype), u=nplike.zeros((ny, nx), dtype=dtype),
        v=nplike.zeros((ny, nx), dtype=dtype), a=nplike.zeros((ny, nx), dtype=dtype),
        flux=get_empty_whuhvmodel(nx, ny, dtype)
    )


def get_empty_facetwosidemodel(nx: int, ny: int, dtype: str):
    """Get an empty (i.e., zero arrays) FaceTwoSideModel.

    Arguments
    ---------
    nx, ny : int
    dtype : str, nplike.float32, nplike.float64

    Returns
    -------
    A FaceTwoSideModel with zero arrays.
    """
    dtype = DummyDtype.validator(dtype)
    return FaceTwoSideModel(
        nx=nx, ny=ny, dtype=dtype,
        plus=get_empty_faceonesidemodel(nx, ny, dtype),
        minus=get_empty_faceonesidemodel(nx, ny, dtype),
        num_flux=get_empty_whuhvmodel(nx, ny, dtype)
    )


def get_empty_facequantitymodel(nx: int, ny: int, dtype: str):
    """Get an empty (i.e., zero arrays) FaceQuantityModel.

    Arguments
    ---------
    nx, ny : int
    dtype : str, nplike.float32, nplike.float64

    Returns
    -------
    A FaceQuantityModel with zero arrays.
    """
    dtype = DummyDtype.validator(dtype)
    return FaceQuantityModel(
        nx=nx, ny=ny, dtype=dtype,
        x=get_empty_facetwosidemodel(nx+1, ny, dtype),
        y=get_empty_facetwosidemodel(nx, ny+1, dtype),
    )


def get_empty_slopes(nx: int, ny: int, dtype: str):
    """Get an empty (i.e., zero arrays) Slopes.

    Arguments
    ---------
    nx, ny : int
    dtype : str, nplike.float32, nplike.float64

    Returns
    -------
    A Slopes with zero arrays.
    """
    dtype = DummyDtype.validator(dtype)
    return Slopes(
        nx=nx, ny=ny, dtype=dtype,
        x=get_empty_whuhvmodel(nx+2, ny, dtype), y=get_empty_whuhvmodel(nx, ny+2, dtype),
    )


def get_empty_states(nx: int, ny: int, ngh: int, dtype: str):
    """Get an empty (i.e., zero arrays) States.

    Arguments
    ---------
    nx, ny : int
    ngh : int
    dtype : str, nplike.float32, nplike.float64

    Returns
    -------
    A States with zero arrays.
    """
    dtype = DummyDtype.validator(dtype)
    return States(
        nx=nx, ny=ny, ngh=ngh, dtype=dtype,
        q=get_empty_whuhvmodel(nx+2*ngh, ny+2*ngh, dtype),
        src=get_empty_whuhvmodel(nx, ny, dtype),
        slp=get_empty_slopes(nx, ny, dtype),
        rhs=get_empty_whuhvmodel(nx, ny, dtype),
        face=get_empty_facequantitymodel(nx, ny, dtype)
    )
