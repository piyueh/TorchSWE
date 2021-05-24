#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Objects holding simulation configuraions.
"""
import pathlib
from typing import Literal, Tuple, Union, Optional

import yaml
from pydantic import BaseModel, Field, validator, root_validator, conint, confloat, validate_model


# alias to type hints
BCTypeHint = Literal["periodic", "extrap", "const", "inflow", "outflow"]

OutputTypeHint = Union[
    Tuple[Literal["at"], Tuple[confloat(ge=0), ...]],
    Tuple[Literal["t_start every_seconds multiple"], confloat(ge=0), confloat(gt=0), conint(ge=1)],
    Tuple[Literal["t_start every_steps multiple"], confloat(ge=0), conint(ge=1), conint(ge=1)],
    Tuple[Literal["t_start t_end n_saves"], confloat(ge=0), confloat(gt=0), conint(ge=1)],
    Tuple[Literal["t_start t_end no save"], confloat(ge=0), confloat(gt=0)],
    Tuple[Literal["t_start n_steps no save"], confloat(ge=0), conint(ge=1)],
]

TemporalTypeHint = Literal["Euler", "SSP-RK2", "SSP-RK3"]


class BaseConfig(BaseModel):
    """Extending pydantic.BaseModel with __getitem__ method."""

    class Config:  # pylint: disable=too-few-public-methods
        """pydantic configuration of this model."""
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        extra = "forbid"

    def __getitem__(self, key):
        return super().__getattribute__(key)

    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    def check(self):
        """Manually trigger the validation of the data in this instance."""
        _, _, validation_error = validate_model(self.__class__, self.__dict__)

        if validation_error:
            raise validation_error

        for field in self.__dict__.values():
            if isinstance(field, BaseConfig):
                field.check()


class SpatialConfig(BaseConfig):
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


class TemporalConfig(BaseConfig):
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
    scheme : str
        Currently, either "Euler", "RK2", or "RK4". Default: "RK2"
    """
    # pylint: disable=too-few-public-methods, no-self-argument, invalid-name, no-self-use

    dt: confloat(gt=0.) = 1e-3
    adaptive: bool = True
    output: OutputTypeHint
    max_iters: conint(gt=0) = 1000000
    scheme: TemporalTypeHint = "SSP-RK2"

    @validator("output")
    def _val_output_method(cls, v, values):
        """Validate that end time > start time."""

        if v[0] == "at":
            msg = "Times are not monotonically increasing"
            assert all(v[1][i]>v[1][i-1] for i in range(1, len(v[1]))), msg
        elif v[0] in ["t_start every_steps multiple", "t_start n_steps no save"]:
            assert not values["adaptive"], "Needs \"adaptive=False\"."
        elif v[0] in ["t_start t_end n_saves", "t_start t_end no save"]:
            assert v[2] > v[1], "End time is not greater than start time."

        return v

    @validator("max_iters")
    def _val_max_iters(cls, v, values):
        """Validate and modify max_iters."""
        if values["output"][0] in ["t_start every_steps multiple", "t_start n_steps no save"]:
            v = values["output"][2]  # use per_step as max_iters
        return v


class SingleBCConfig(BaseConfig):
    """An object holding configuration of the boundary conditions on a single boundary.

    Attributes
    ----------
    types : a length-3 tuple/list of str
        Boundary conditions correspond to the three conservative quantities. If the type is
        "inflow", they correspond to non-conservative quantities, i.e., u and v. Applying "inflow"
        to depth h or elevation w seems not be make any sense.
    values : a length-3 tuple of floats or None
        Some BC types require user-provided values (e.g., "const"). Use this to give values.
        Usually, they are the conservative quantities, i.e., w, hu, and hv. For "inflow", however,
        they are non-conservative quantities, i.e., u and v. Defautl: [None, None, None]
    """
    # pylint: disable=too-few-public-methods, no-self-argument, invalid-name, no-self-use

    types: Tuple[BCTypeHint, BCTypeHint, BCTypeHint]
    values: Tuple[Optional[float], Optional[float], Optional[float]] = [None, None, None]

    @validator("types")
    def check_periodicity(cls, v):
        """If one component is periodic, all components should be periodic."""
        if any(t == "periodic" for t in v):
            assert all(t == "periodic" for t in v), "All components should be periodic."
        return v

    @validator("values")
    def check_values(cls, v, values):
        """Check if values are set accordingly for some BC types.
        """
        if "types" not in values:
            return v

        for bctype, bcval in zip(values["types"], v):
            if bctype in ("const", "inflow"):
                assert isinstance(bcval, float), \
                    "Using BC type \"{}\" requires setting a value.".format(bctype.value)
        return v


class BCConfig(BaseConfig):
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

    @root_validator(pre=False)
    def check_periodicity(cls, values):
        """Check whether periodic BCs match at corresponding boundary pairs."""
        if any((t not in values) for t in ["west", "east", "south", "north"]):
            return values

        result = True
        for types in zip(values["west"]["types"], values["east"]["types"]):
            if any(t == "periodic" for t in types):
                result = all(t == "periodic" for t in types)
        for types in zip(values["north"]["types"], values["south"]["types"]):
            if any(t == "periodic" for t in types):
                result = all(t == "periodic" for t in types)
        if not result:
            raise ValueError("Periodic BCs do not match at boundaries and components.")
        return values


class ICConfig(BaseConfig):
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
        if "file" in values and values["file"] is not None:
            if "values" in values and values["values"] is not None:
                raise AssertionError("Only one of \"file\" or \"values\" can be set for I.C.")

            if "keys" not in values or values["keys"] is None:
                raise AssertionError("\"keys\" has to be set when \"file\" is not None for I.C.")
        else:  # "file" is not specified or is None
            if "values" not in values or values["values"] is None:
                raise AssertionError("Either \"file\" or \"values\" has to be set for I.C.")

        return values


class TopoConfig(BaseConfig):
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


class ParamConfig(BaseConfig):
    """An object holding configuration of miscellaneous parameters.

    Attributes
    ----------
    gravity : float
        Gravity in m^2/sec. Default: 9.81
    theta : float
        Parameter controlling numerical dissipation. 1.0 < theta < 2.0. Default: 1.3
    drytol : float
        Dry tolerance in meters. Default: 1.0e-4.
    ngh : int
        Number of ghost cell layers per boundary. At least 2 required.
    """
    # pylint: disable=too-few-public-methods, no-self-argument, invalid-name, no-self-use

    gravity: confloat(ge=0.) = 9.81
    theta: confloat(ge=1., le=2.) = 1.3
    drytol: confloat(ge=0.) = 1.0e-4
    ngh: conint(ge=2) = 2
    log_steps: conint(ge=1) = 100


class Config(BaseConfig):
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
    dtype : str
        The floating number type. Either "float32" or "float64". Default: "float64"
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
    dtype: Literal["float32", "float64"] = "float64"


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
