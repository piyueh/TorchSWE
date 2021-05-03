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
from typing import Literal, Tuple, List, Union

import numpy
from scipy.interpolate import RectBivariateSpline
from pydantic import validator, conint, confloat
from .config import BaseConfig, TemporalConfig, SpatialConfig, TopoConfig
from .netcdf import read_cf


def _pydantic_val_dtype(val: numpy.ndarray, values: dict) -> numpy.ndarray:
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
    dtype: either "float32" or "float64"
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
    dtype: Literal["float32", "float64"]
    vert: numpy.ndarray
    cntr: numpy.ndarray
    xface: numpy.ndarray
    yface: numpy.ndarray

    # common validators
    _val_vert = validator('vert', allow_reuse=True)(_shape_val_factory(1))
    _val_cntr = validator('cntr', allow_reuse=True)(_shape_val_factory(0))
    _val_dtypes = validator("vert", "cntr", "xface", "yface", allow_reuse=True)(_pydantic_val_dtype)

    def __init__(self, direction, n, start, end, dtype):

        delta = (end - start) / n
        vert = numpy.linspace(start, end, n+1, dtype=dtype)
        cntr = numpy.linspace(start+delta/2., end-delta/2., n, dtype=dtype)

        if direction == "x":
            xface = vert.copy()
            yface = cntr.copy()
        else:  # if this is not "y", pydantic will let me know
            xface = cntr.copy()
            yface = vert.copy()

        # pydantic will validate the data here
        super().__init__(
            direction=direction, n=n, start=start, end=end, delta=delta, dtype=dtype,
            vert=vert, cntr=cntr, xface=xface, yface=yface)

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

    def __init__(self, spatial: SpatialConfig, temporal: TemporalConfig, dtype: str):

        # manually launch validation
        spatial.check()
        temporal.check()

        # temporal gridline: not used in computation, so use native list here
        if temporal.output is None:  # no output
            t = []
        elif temporal.output[0] == "every":  # output every dt
            dt = temporal.output[1]  # alias # pylint: disable=invalid-name
            t = numpy.arange(temporal.start, temporal.end+dt/2., dt).tolist()
        elif temporal.output[0] == "at":  # output at the given times
            t = temporal.output[1]
            if temporal.start not in t:
                t.append(temporal.start)
            if temporal.end not in t:
                t.append(temporal.end)
            t.sort()
        # pydantic should detect invalid type of output; no need for else

        super().__init__(
            x=Gridline("x", spatial.discretization[0], spatial.domain[0], spatial.domain[1], dtype),
            y=Gridline("y", spatial.discretization[1], spatial.domain[2], spatial.domain[3], dtype),
            t=t)


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
    dtype: Literal["float32", "float64"]
    vert: numpy.ndarray
    cntr: numpy.ndarray
    xface: numpy.ndarray
    yface: numpy.ndarray
    xgrad: numpy.ndarray
    ygrad: numpy.ndarray

    # validators
    _val_dtypes = validator(
        "vert", "cntr", "xface", "yface", "xgrad", "ygrad",
        allow_reuse=True)(_pydantic_val_dtype)
    _val_ny_1_nx_1 = validator("vert", allow_reuse=True)(_shape_val_factory([1, 1]))
    _val_ny_nx_1 = validator("xface", allow_reuse=True)(_shape_val_factory([0, 1]))
    _val_ny_1_nx = validator("yface", allow_reuse=True)(_shape_val_factory([1, 0]))
    _val_ny_nx = validator("cntr", "xgrad", "ygrad", allow_reuse=True)(_shape_val_factory([0, 0]))

    def __init__(self, topoconfig: TopoConfig, grid: Gridlines, dtype: str):
        dem, _ = read_cf(topoconfig.file, [topoconfig.key])

        # copy to a numpy.ndarray
        vert = dem[topoconfig.key][:].copy()

        # see if we need to do interpolation
        try:
            interp = not (
                numpy.allclose(grid.x.vert, dem["x"]) and
                numpy.allclose(grid.y.vert, dem["y"]))
        except ValueError:  # assume thie excpetion means a shape mismatch
            interp = True

        # unfortunately, we need to do interpolation in such a situation
        if interp:
            interpolator = RectBivariateSpline(dem["x"], dem["y"], vert.T)
            vert = interpolator(grid.x.vert, grid.y.vert).T

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
        xgrad = (xface[:, 1:] - xface[:, :-1]) / (grid.x.vert[1:] - grid.x.vert[:-1])
        ygrad = (yface[1:, :] - yface[:-1, :]) / (grid.y.vert[1:] - grid.y.vert[:-1])[:, None]

        # initialize DataModel and let pydantic validates data
        super().__init__(
            nx=grid.x.n, ny=grid.y.n, dtype=dtype, vert=vert, cntr=cntr, xface=xface,
            yface=yface, xgrad=xgrad, ygrad=ygrad)


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
    dtype: Literal["float32", "float64"]
    w: numpy.ndarray
    hu: numpy.ndarray
    hv: numpy.ndarray

    # validators
    _val_arrays = validator("w", "hu", "hv", allow_reuse=True)(_pydantic_val_arrays)

    def __init__(self, nx, ny, dtype, w=None, hu=None, hv=None):

        kwargs = {"w": w, "hu": hu, "hv": hv}
        for key, val in kwargs.items():
            if val is None:
                kwargs[key] = numpy.zeros((ny, nx), dtype=dtype)

        # trigger pydantic validation
        super().__init__(nx=nx, ny=ny, dtype=dtype, **kwargs)


class HUVModel(BaseConfig, DummyDataModel):
    """Data model with keys h, u, and v."""
    nx: conint(gt=0)
    ny: conint(gt=0)
    dtype: Literal["float32", "float64"]
    h: numpy.ndarray
    u: numpy.ndarray
    v: numpy.ndarray

    # validators
    _val_arrays = validator("h", "u", "v", allow_reuse=True)(_pydantic_val_arrays)

    def __init__(self, nx, ny, dtype):
        super().__init__(  # trigger pydantic validation
            nx=nx, ny=ny, dtype=dtype, h=numpy.zeros((ny, nx), dtype=dtype),
            u=numpy.zeros((ny, nx), dtype=dtype), v=numpy.zeros((ny, nx), dtype=dtype))


class FaceOneSideModel(BaseConfig, DummyDataModel):
    """Data model holding quantities on one side of cell faces normal to one direction."""
    nx: conint(gt=0)
    ny: conint(gt=0)
    dtype: Literal["float32", "float64"]
    w: numpy.ndarray
    hu: numpy.ndarray
    hv: numpy.ndarray
    h: numpy.ndarray
    u: numpy.ndarray
    v: numpy.ndarray
    a: numpy.ndarray
    flux: WHUHVModel

    # validator
    _val_arrays = validator(
        "w", "hu", "hv", "h", "u", "v", "a", "flux", allow_reuse=True)(_pydantic_val_arrays)

    def __init__(self, nx, ny, dtype):
        super().__init__(  # trigger pydantic validation
            nx=nx, ny=ny, dtype=dtype, w=numpy.zeros((ny, nx), dtype=dtype),
            hu=numpy.zeros((ny, nx), dtype=dtype), hv=numpy.zeros((ny, nx), dtype=dtype),
            h=numpy.zeros((ny, nx), dtype=dtype), u=numpy.zeros((ny, nx), dtype=dtype),
            v=numpy.zeros((ny, nx), dtype=dtype), a=numpy.zeros((ny, nx), dtype=dtype),
            flux=WHUHVModel(nx, ny, dtype)
        )


class FaceTwoSideModel(BaseConfig, DummyDataModel):
    """Date model holding quantities on both sides of cell faces normal to one direction."""
    nx: conint(gt=0)
    ny: conint(gt=0)
    dtype: Literal["float32", "float64"]
    plus: FaceOneSideModel
    minus: FaceOneSideModel
    num_flux: WHUHVModel

    # validator
    _val_arrays = validator("plus", "minus", "num_flux", allow_reuse=True)(_pydantic_val_arrays)

    def __init__(self, nx, ny, dtype):
        super().__init__(  # trigger pydantic validation
            nx=nx, ny=ny, dtype=dtype, plus=FaceOneSideModel(nx, ny, dtype),
            minus=FaceOneSideModel(nx, ny, dtype), num_flux=WHUHVModel(nx, ny, dtype)
        )


class FaceQuantityModel(BaseConfig):
    """Data model holding quantities on both sides of cell faces in both x and y directions."""
    nx: conint(gt=0)
    ny: conint(gt=0)
    dtype: Literal["float32", "float64"]
    x: FaceTwoSideModel
    y: FaceTwoSideModel

    def __init__(self, nx, ny, dtype):
        super().__init__(  # trigger pydantic validation
            nx=nx, ny=ny, dtype=dtype,
            x=FaceTwoSideModel(nx+1, ny, dtype), y=FaceTwoSideModel(nx, ny+1, dtype),
        )

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
    dtype: Literal["float32", "float64"]
    x: WHUHVModel
    y: WHUHVModel

    def __init__(self, nx, ny, dtype):
        super().__init__(  # trigger pydantic validation
            nx=nx, ny=ny, dtype=dtype,
            x=WHUHVModel(nx+2, ny, dtype), y=WHUHVModel(nx, ny+2, dtype),
        )

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
    dtype: Literal["float32", "float64"]

    # quantities defined at cell centers
    q: WHUHVModel
    src: WHUHVModel
    slp: Slopes
    rhs: WHUHVModel
    face: FaceQuantityModel

    def __init__(self, nx, ny, ngh, dtype):
        super().__init__(
            nx=nx, ny=ny, ngh=ngh, dtype=dtype, q=WHUHVModel(nx+2*ngh, ny+2*ngh, dtype),
            src=WHUHVModel(nx, ny, dtype), slp=Slopes(nx, ny, dtype), rhs=WHUHVModel(nx, ny, dtype),
            face=FaceQuantityModel(nx, ny, dtype)
        )

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
