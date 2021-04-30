#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Pi-Yueh Chuang <pychuang@pm.me>
#
# Distributed under terms of the BSD 3-Clause license.

"""Objects holding simulation configuraions.
"""
import enum
import pathlib
from typing import Tuple, Union, Optional

import yaml
from pydantic import BaseModel, Field, validator, root_validator, conint, confloat


class TemporalScheme(enum.Enum):
    """Supported temporal schemes.
    """
    EULER = "Euler"
    RK2 = "RK2"
    RK4 = "RK4"


class OutoutType(enum.Enum):
    """Supported output mechanism.
    """
    EVERY = "every"
    AT = "at"


class BCType(enum.Enum):
    """Supported boundary condition types.
    """
    PERIODIC = "periodic"
    EXTRAP = "extrap"
    CONST = "const"
    INFLOW = "inflow"
    OUTFLOW = "outflow"


class SpatialConfig(BaseModel):
    """An object holding spatial configuration.

    Attributes
    ----------
    domain : a list/tuple of 4 floats
        The elements correspond the the bounds in west, east, south, and north.
    discretization : a list/tuple of 2 int
        The elements correspond the number of cells in west-east and south-north directions.
    """
    # pylint: disable=too-few-public-methods, no-self-argument, invalid-name, no-self-use

    domain: Tuple[float, float, float, float]
    discretization: Tuple[conint(strict=True, gt=0), conint(strict=True, gt=0)]

    @validator("domain")
    def domain_direction(cls, v):
        """Validate the East >= West and North >= South.
        """
        assert v[1] > v[0], "domain[1] must greater than domain[0]"
        assert v[3] > v[2], "domain[3] must greater than domain[2]"
        return v


class TemporalConfig(BaseModel):
    """An object holding temporal configuration.

    Attributes
    ----------
    start : float
        The start time of the simulation, i.e., the time that the initial conditions are applied.
    end : float
        The end time of the simulation, i.e., the simulation stops when reaching this time.
    output : list/tuple or None
        Three available formats:
            1. ["every", delta_t]: outputs every `delta_t` time.
            2. ["at", [t1, t2, t3, ...]]: outputs at t1, t2, t3, ...
            3. None: don't output any solutions.
        Default: None
    scheme : str or TemporalScheme
        Currently, either "Euler", "RK2", or "RK4". Default: "RK2"
    """
    # pylint: disable=too-few-public-methods, no-self-argument, invalid-name, no-self-use

    start: confloat(ge=0.)
    end: confloat(ge=0.)
    output: Optional[Tuple[OutoutType, Union[confloat(ge=0.), Tuple[confloat(ge=0), ...]]]] = None
    scheme: TemporalScheme = TemporalScheme.RK2

    @validator("end")
    def end_greater_than_start(cls, v, values):
        """Validate that end time > start time.
        """
        assert v > values["start"], "The end time should greater than the start time."
        return v

    @validator("output")
    def output_method_pair_check(cls, v):
        """Validate the content in output makes sense.
        """
        if v is None:
            return v

        if v[0] == OutoutType.AT:
            assert isinstance(v[1], (tuple, list)), \
                "When using \"at\", the second element should be a tuple/list."
        elif v[0] == OutoutType.EVERY:
            assert isinstance(v[1], float), \
                "When using \"every\", the second element should be a float."

        return v


class SingleBCConfig(BaseModel):
    """An object holding configuration of the boundary conditions on a single boundary.

    Attributes
    ----------
    types : a length-3 tuple/list of str/BCType
        Boundary conditions correspond to the three conservative quantities. If the type is
        "inflow", they correspond to non-conservative quantities, i.e., u and v. Applying "inflow"
        to depth h or elevation w seems not be make any sense.
    values : a length-3 tuple of floats or None
        Some BC types require user-provided values (e.g., "const"). Use this to give values.
        Usually, they are the conservative quantities, i.e., w, hu, and hv. For "inflow", however,
        they are non-conservative quantities, i.e., u and v. Defautl: [None, None, None]
    """
    # pylint: disable=too-few-public-methods, no-self-argument, invalid-name, no-self-use

    types: Tuple[BCType, BCType, BCType]
    values: Tuple[Optional[float], Optional[float], Optional[float]] = [None, None, None]

    @validator("values")
    def check_values(cls, v, values):
        """Check if values are set accordingly for some BC types.
        """
        for bctype, bcval in zip(values["types"], v):
            if bctype in (BCType.CONST, BCType.INFLOW):
                assert isinstance(bcval, float), \
                    "Using BC type \"{}\" requires setting a value.".format(bctype.value)
        return v


class BCConfig(BaseModel):
    """An object holding configuration of the boundary conditions of all boundaries.

    Attributes
    ----------
    west, east, north, south : SingleBCConfig
        Boundary conditions on west, east, north, and south boundaries.
    """
    # pylint: disable=too-few-public-methods, no-self-argument, invalid-name, no-self-use

    west: SingleBCConfig
    east: SingleBCConfig
    north: SingleBCConfig
    south: SingleBCConfig


class ICConfig(BaseModel):
    """An object holding configuration of the initial conditions.

    Attributes
    ----------
    file : None or str or path-like object
        The path to a NetCDF file containing IC data.
    keys : None or a tuple/list of str
        The variable names in the `file` that correspond to w, hu, and hv. If `file` is None, this
        can be None.
    values : None or a tuple/list of floats
        If `file` is None, use this attribute to specify constant IC values.
    """
    # pylint: disable=too-few-public-methods, no-self-argument, invalid-name, no-self-use

    file: Optional[pathlib.Path]
    keys: Optional[Tuple[str, str, str]]
    values: Optional[Tuple[float, float, float]]

    @root_validator(pre=True)
    def check_mutually_exclusive_attrs(cls, values):
        """\"file\" and \"values" should be mutually exclusive.
        """
        if "file" in values:
            if "values" in values and values["values"] is not None:
                raise AssertionError("Only one of \"file\" or \"values\" can be set for I.C.")

            if "keys" not in values:
                raise AssertionError("\"keys\" has to be set when \"file\" is not None for I.C.")

            if values["keys"] is None:
                raise AssertionError("\"keys\" has to be a length-3 tuple/list of floats.")
        else:
            if "values" not in values or values["values"] is None:
                raise AssertionError("Either \"file\" or \"values\" has to be set for I.C.")

        return values


class TopoConfig(BaseModel):
    """An object holding configuration of the topography file.

    Attributes
    ----------
    file : str or path-like object
        The path to a NetCDF file containing topography data.
    key : str
        The variable name in the `file` that corresponds to elevation data.
    """
    # pylint: disable=too-few-public-methods, no-self-argument, invalid-name, no-self-use

    file: pathlib.Path
    key: str


class ParamConfig(BaseModel):
    """An object holding configuration of miscellaneous parameters.

    Attributes
    ----------
    gravity : float
        Gravity in m^2/sec. Default: 9.81
    theta : float
        Parameter controlling numerical dissipation. 1.0 < theta < 2.0. Default: 1.3
    drytol : float
        Dry tolerance in meters. Default: 1.0e-4.
    """
    # pylint: disable=too-few-public-methods, no-self-argument, invalid-name, no-self-use

    gravity: confloat(ge=0.) = 9.81
    theta: confloat(ge=1., le=2.) = 1.3
    drytol: confloat(ge=0.) = 1.0e-4


class Config(BaseModel):
    """An object holding all configurations of a simulation case.

    Attributes
    ----------
    spatial : SpatialConfig
        Spatial information.
    temporal : TemporalConfig
        Temporal control.
    bc : BCConfig
        Boundary conditions.
    ic : ICConfig
        Initial conditions.
    topo : TopoConfig
        Topography information.
    params : ParamConfig
        Miscellaneous parameters.
    prehook : None or path-like
        The path to a Python script that will be executed before running a simulation.
    case : path-like
        The path to the case folder.
    ftype : type
        The floating number type. Default: numpy.float64
    """
    # pylint: disable=too-few-public-methods, no-self-argument, invalid-name, no-self-use

    spatial: SpatialConfig
    temporal: TemporalConfig
    bc: BCConfig = Field(..., alias="boundary")
    ic: ICConfig = Field(..., alias="initial")
    topo: TopoConfig = Field(..., alias="topography")
    params: ParamConfig = Field(..., alias="parameters")
    prehook: Optional[pathlib.Path]
    case: Optional[pathlib.Path]
    ftype: str = "float64"

    @validator("ftype")
    def check_ftype(cls, v):
        """Check ftype.
        """
        assert v in ("float32", "float64"), "\"ftype\" should be either \"float32\" or \"float64\"."
        return v


# register the Config class in yaml with tag !Config
yaml.add_constructor(
    u'!Config',
    lambda loader, node: Config(**loader.construct_mapping(node, deep=True))
)

yaml.add_representer(
    Config,
    lambda dumper, data: dumper.represent_mapping(
        tag=u"!Config", mapping=yaml.load(
            data.json(by_alias=True), Loader=yaml.Loader),
        flow_style=True
    )
)