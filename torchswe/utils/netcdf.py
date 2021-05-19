#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Functions related to NetCDF I/O with the CF convention.
"""
import pathlib
from datetime import datetime, timezone

import netCDF4
import numpy  # real numpy because NetCDF library only works with real numpy ndarray
from torchswe.utils.dummy import DummyDict


def default_attrs(corner, delta):
    """Get basic attributes for a NetCDF4 file.

    Arguments
    ---------
    corner : tuple
        A tuple of two floats indicating the x and y coordinates at the west-north corner.
    delta : tuple
        A tuple of two floats indicatinf the dx and dy, i.e., cell sizes.

    Returns
    -------
    A dictionary with keys: title, institution, source, history, refernce, comment, and Conventions.
    """
    cur_t = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    wkt = \
        "PROJCS[\"WGS_1984_Web_Mercator_Auxiliary_Sphere\"," + \
            "GEOGCS[\"GCS_WGS_1984\"," + \
                "DATUM[\"D_WGS_1984\"," + \
                    "SPHEROID[\"WGS_1984\",6378137.0,298.257223563]]," + \
                "PRIMEM[\"Greenwich\",0.0]," + \
                "UNIT[\"Degree\",0.017453292519943295]]," + \
            "PROJECTION[\"Mercator_Auxiliary_Sphere\"]," + \
            "PARAMETER[\"False_Easting\",0.0]," + \
            "PARAMETER[\"False_Northing\",0.0]," + \
            "PARAMETER[\"Central_Meridian\",0.0]," + \
            "PARAMETER[\"Standard_Parallel_1\",0.0]," + \
            "PARAMETER[\"Auxiliary_Sphere_Type\",0.0]," + \
            "UNIT[\"Meter\",1.0]]"  # noqa:E127

    attrs = {
        "root": {
            "title": "Data created by TorchSWE",
            "institution": "The George Washington University",
            "source": "TorchSWE",
            "history": "Created " + cur_t,
            "reference": "https://github.com/piyueh/TorchSWE",
            "comment": "",
            "Conventions": "CF-1.7",
        },
        "x": {
            "units": "m",
            "long_name": "x-coordinate in EPSG:3857 WGS 84",
            "standard_name": "projection_x_coordinate",
        },
        "y": {
            "units": "m",
            "long_name": "y-coordinate in EPSG:3857 WGS 84",
            "standard_name": "projection_y_coordinate",
        },
        "time": {
            "axis": "T",
            "long_name": "Simulation time",
            "units": "seconds since " + cur_t,
            "calendar": "standard",
        },
        "mercator": {
            "GeoTransform": "{} {} 0 {} 0 {}".format(corner[0], delta[0], corner[1], -delta[1]),
            "grid_mapping_name": "mercator",
            "long_name": "CRS definition",
            "longitude_of_projection_origin": 0.0,
            "standard_parallel": 0.0,
            "false_easting": 0.0,
            "false_northing": 0.0,
            "spatial_ref": wkt,
        },
    }

    return attrs


def read(fpath, data_keys, **kwargs):
    """Read data from a NetCDF file in CF convention.

    The pure spatial array, data is in traditional numerical simulation style.
    For example, data[0, 0] is the most bottom-left data point in a structured
    grid. And data[1, :] represents all points in the second row from the bottom
    of the structured grid.

    For temporal data, the dimension is (time, y, x).

    Arguments
    ---------
    fpath : str or path-like
        Path to the input file.
    data_keys : a tuple/list of str
        Variable names in the file that will be read.
    **kwargs :
        Arbitrary keyword arguments passed into netCDF4.Dataset.__init__.

    Returns
    -------
    data : a dict
        This dict has key-value pairs of:
        - x: a 1D nplike.ndarray; gridline in x direction.
        - y: a 1D nplike.ndarray; gridline in y direction.
        - time: (optional) a 1D nplike.ndarray if gridline in time exists.
        - And all keys specified in data_key argument.
    attrs : a dict of dicts
        Attributes for each key in data (exclude root group's).
    """

    # use absolute path
    fpath = pathlib.Path(fpath).expanduser().resolve()

    # empty dictionary
    data = DummyDict()
    attrs = DummyDict()

    # create a NetCDF4 file/dataset
    with netCDF4.Dataset(fpath, "r", **kwargs) as rootgrp:  # pylint: disable=no-member

        # x and y
        data["x"] = rootgrp["x"][:].data
        data["y"] = rootgrp["y"][:].data

        # try to read t if it exists
        try:
            data["time"] = rootgrp["time"][:].data
        except IndexError:
            pass

        # read data of each specified key
        for key in data_keys:
            data[key] = rootgrp[key][:].data

        # other options
        for key in data.keys():
            attrs[key] = rootgrp[key].__dict__

    return data, attrs


def write(fpath, grid_x, grid_y, grid_t=None, data=None, options=None, **kwargs):
    """A wrapper to safely write to a NetCDF file.

    In case an I/O error happen, the NetCDF file will still be safely closed.

    Arguments
    ---------
    fpath : str or path-like
        Path to the target file.
    grid_x, grid_y : nplike.ndarray
        The gridlines in x and y directions
    grid_t : None, list, tuple, or nplike.ndarray
        The temporal gridlines. Can be length-zero list/array, meaning creating an unlimited axis.
    data : dict
        A dictionary of (variable name, value). If value is None, create all-NaN array for the
        variable. If not None, the shape of value should be either (ny, nx) or (ntime, ny, nx).
    options: a dict of dict
        The outer dictionary has pairs (variable name, dictionary). The inner dictionaries
        are the attributes of each vriable. A special key is "root", which holds attributes
        to the dataset itself. For options of variables, usually users may want to at least
        specify the attribute "units".
    **kwargs :
        Arbitrary keyword arguments that will be provide to netCDF4.Dataset.__init__.
    """

    fpath = pathlib.Path(fpath).expanduser().resolve()

    with netCDF4.Dataset(fpath, "w", **kwargs) as rootgrp:  # pylint: disable=no-member
        rootgrp = write_to_dataset(rootgrp, grid_x, grid_y, grid_t, data, options)


def write_to_dataset(dset, grid_x, grid_y, grid_t=None, data=None, user_options=None):
    """Create a NetCDF file of CF convention.

    Arguments
    ---------
    dset : netCDF4.Dataset
        The destination of data output.
    grid_x, grid_y : nplike.ndarray
        The gridlines in x and y directions
    grid_t : None, list, tuple, or nplike.ndarray
        The temporal gridlines. Can be length-zero list/array, meaning creating an unlimited axis.
    data : dict
        A dictionary of (variable name, value). If value is None, create all-NaN array for the
        variable. If not None, the shape of value should be either (ny, nx) or (ntime, ny, nx).
    user_options: a dict of dict
        The outer dictionary has pairs (variable name, dictionary). The inner dictionaries
        are the attributes of each vriable. A special key is "root", which holds attributes
        to the dataset itself. For options of variables, usually users may want to at least
        specify the attribute "units".

    Returns
    -------
    dset : netCDF4.Dataset
        The same input dataset but with axes, data, and a mercator.

    Notes
    -----
    - Note the input argument, data, is in regular numeric simulation style, i.e., data[0, 0] is the
      most bottom-left data point in structured grid. And data[1, :] is all values in the second row
      from the bottom in the structured grid.
    - The spatial gridline x an y are always assumed to be EPSG:3857 coordinates.
    - No sanity check will be done. Users have to be careful of dimensions mismatch.
    """

    # get default options
    options = default_attrs((grid_x[0], grid_y[-1]), (grid_x[1]-grid_x[0], grid_y[1]-grid_y[0]))
    options = {**user_options, **options}

    for key in ["root", "x", "y", "time", "mercator"]:
        options[key].update(user_options[key] if key in user_options else {})

    # global attributes
    dset.setncatts(options["root"])

    # create axes
    dset = add_axis_to_dataset(dset, "x", len(grid_x), grid_x, options["x"])
    dset = add_axis_to_dataset(dset, "y", len(grid_y), grid_y, options["y"])

    # temporal axis is optional
    if grid_t is not None:
        dset = add_axis_to_dataset(dset, "time", len(grid_t), grid_t, options["time"])

    # create mercator
    mercator = dset.createVariable("mercator", "S1")
    mercator.setncatts(options["mercator"])

    # create variables
    dset = add_variables_to_dataset(dset, data, options)

    return dset


def add_variables_to_dataset(dset, data, options=None):
    """Add variables to an existing dataset.

    Arguments
    ---------
    dset : netCDF4.Dataset
        The target dataset. Must be opened in "a" mode, i.e., appending mode.
    data : dict
        A dictionary of (variable name, value). If value is None, create all-NaN array for the
        variable. If not None, the shape of value should be either (ny, nx) or (ntime, ny, nx).
    options: a dict of dict
        The outer dictionary has pairs (variable name, dictionary). The inner dictionaries
        are the attributes of each vriable. Usually users may want to at least specify the
        attribute "units".

    Returns
    -------
    The same dataset instance (w/ new variable added).
    """

    nan = float("NaN")
    options = {} if options is None else options

    # create variables
    for key, val in data.items():

        # create the variable
        shape = ("ntime", "ny", "nx") if "ntime" in dset.dimensions else ("ny", "nx")
        dset.createVariable(key, "f8", shape, True, 9, fill_value=nan)  # all NaN in it right now

        # variable attributes
        dset[key].long_name = key
        dset[key].grid_mapping = "mercator"
        dset[key].setncatts(options[key] if key in options else {})

        if val is None:  # no need to copy data
            continue

        # otherwise, check th shape first
        if len(val.shape) == 2:
            assert val.shape[0] == dset.dimensions["ny"].size
            assert val.shape[1] == dset.dimensions["nx"].size
        elif len(val.shape) == 3:
            assert val.shape[0] == dset.dimensions["ntime"].size
            assert val.shape[1] == dset.dimensions["ny"].size
            assert val.shape[2] == dset.dimensions["nx"].size
        else:
            raise ValueError("{}.shape = {} is not supported.".format(key, val.shape))

        # otherwise, copy the data
        if len(val.shape) == 2 and "ntime" in dset.dimensions:  # assume this is the first snapshot
            _copy_data_to_slice(dset[key], val, 0)
        else:  # either no time axis or val has all snapshots (i.e., len(val.shape) == 3)
            _copy_data_to(dset[key], val)

    return dset


def add_time_data_to_dataset(dset, data, time, tidx=None):
    """Write/append new data at a specific time snapshot index to a NetCDF4 variable.

    Arguments
    ---------
    dset : netCDF4.Dataset
        The target dataset. Must be opened in "a" mode, i.e., appending mode.
    data : dict
        A dictionary of (variable name, value). The shape of value should be (ny, nx).
    time : float
        The value of time.
    tidx : int or None
        The index of the target time to write into. If None, append the data instead.

    Returns
    -------
    The same dataset instance (w/ updated values).

    Notes
    -----
    - the corresponding variables must already exist in the dataset and have temporal dimension.
    - fill_value is assumed to be float("NaN")
    - if the size of the temporal axis is NOT set to unlimited, appending doesn't work. Use specific
      time index to overwrite the slice instead.
    """

    if tidx is None:  # append to the variables
        assert dset.dimensions["ntime"].isunlimited()
        tidx = dset.dimensions["ntime"].size
        dset["time"][tidx] = time
    else:
        assert abs(dset["time"][tidx]-time) < 1e-10, "Time does not match the dataset's record."

    for key, val in data.items():
        assert len(val.shape) == 2  # should be a single snapshot
        assert val.shape == dset[key].shape[1:]
        _copy_data_to_slice(dset[key], val, tidx)

    return dset


def add_axis_to_dataset(dset, name, n, values, options=None):
    """Add an axes to a netCDF4.Dataset and handle different underlying np-like libraries.

    Arguments
    ---------
    dset : netCDF4.Dataset
        The target dataset.
    name : str
        The axis name.
    n : int
        The number of values in this axis.
    values: a dict of 1D nplike.ndarray
        Coordinates in "x" and "y".
    options: None or a dict
        A dictionary for setting attributes.

    Returns
    -------
    dset : netCDF4.Dataset
        The same input dataset but with a new spatial axis.
    """

    _ = dset.createDimension("n{}".format(name), n)
    var = dset.createVariable(name, "f8", ("n{}".format(name),))
    var.setncatts(options)
    _copy_data_to(var, values)

    return dset


def _copy_data_to(var, array):
    """Copy a non-completely np-compatible ndarray to a NetCDF4 variable."""

    try:
        var[...] = numpy.array(array)
    except TypeError as err:
        if str(err).startswith("Implicit conversion to a NumPy array is not allowe"):
            var[...] = array.get()  # cupy
        elif str(err).startswith("can't convert cuda:"):
            var[...] = array.cpu().numpy()
        else:
            raise


def _copy_data_to_slice(var, array, tidx):
    """Copy a non-completely np-compatible ndarray to a slice in a NetCDF4 variable."""

    try:
        var[tidx, ...] = numpy.array(array)
    except TypeError as err:
        if str(err).startswith("Implicit conversion to a NumPy array is not allowe"):
            var[tidx, ...] = array.get()  # cupy
        elif str(err).startswith("can't convert cuda:"):
            var[tidx, ...] = array.cpu().numpy()
        else:
            raise


def create_soln_snapshot_file(fpath, grid, soln, **kwargs):
    """Create a NetCDF file with a single snapshot of solutions.

    Arguments
    ---------
    fpath : str or PathLike
        The path to the file.
    grid : torchswe.utils.data.Gridlines
        The Gridlines instance corresponds to the solutions.
    soln : torchswe.utils.data.WHUHVModel or torchswe.utils.data.HUVModel
        The snapshot of the solution.
    **kwargs
        Keyword arguments sent to netCDF4.Dataset.
    """
    fpath = pathlib.Path(fpath).expanduser().resolve()

    try:
        data = {k: soln[k] for k in ["w", "hu", "hv"]}
        options = {"w": {"units": "m"}, "hu": {"units": "m2 s-1"}, "hv": {"units": "m2 s-1"}}
    except AttributeError as err:
        if "has no attribute \'w\'" in str(err):  # a HUVModel
            data = {k: soln[k] for k in ["h", "u", "v"]}
            options = {"h": {"units": "m"}, "u": {"units": "m s-1"}, "v": {"units": "m s-1"}}
        else:
            raise

    with netCDF4.Dataset(fpath, "w", **kwargs) as dset:  # pylint: disable=no-member
        write_to_dataset(dset, grid.x.cntr, grid.y.cntr, None, data, options)


def create_empty_soln_file(fpath, grid, model="whuhv", **kwargs):
    """Create an new NetCDF file for solutions using the corresponding grid object.

    Create an empty NetCDF4 file with axes `x`, `y`, and `time`. `x` and `y` are defined at cell
    centers. The spatial coordinates use EPSG 3856. The temporal axis is limited with dimension
    `ntime`. Also, it creates empty solution variables called `w`, `hu`, and `hv` to the dataset
    with `NaN` for all values. The shapes of these variables are `(ntime, ny, nx)`. The units of
    them are "m", "m2 s-1", and "m2 s-1", respectively.

    Arguments
    ---------
    fpath : str or PathLike
        The path to the file.
    grid : torchswe.utils.data.Gridlines
        The Gridlines instance corresponds to the solutions.
    model : str, either "whuhv" or "huv"
        The type of solution model: the conservative form (w, hu, hv) or non-conservative form (
        h, u, v).
    **kwargs
        Keyword arguments sent to netCDF4.Dataset.
    """
    fpath = pathlib.Path(fpath).expanduser().resolve()

    if model == "whuhv":
        data = {k: None for k in ["w", "hu", "hv"]}
        options = {"w": {"units": "m"}, "hu": {"units": "m2 s-1"}, "hv": {"units": "m2 s-1"}}
    elif model == "huv":
        data = {k: None for k in ["h", "u", "v"]}
        options = {"h": {"units": "m"}, "u": {"units": "m s-1"}, "v": {"units": "m s-1"}}

    with netCDF4.Dataset(fpath, "w", **kwargs) as dset:  # pylint: disable=no-member
        write_to_dataset(dset, grid.x.cntr, grid.y.cntr, grid.t, data, options)


def write_soln_to_file(fpath, soln, time, tidx, ngh=0, **kwargs):
    """Write a solution snapshot to an existing NetCDF file.

    Arguments
    ---------
    fpath : str or PathLike
        The path to the file.
    soln : torchswe.utils.data.WHUHVModel or torchswe.utils.data.HUVModel
        The States instance containing solutions.
    time : float
        The simulation time of this snapshot.
    tidx : int
        The index of the snapshot time in the temporal axis.
    ngh : int
        The number of ghost-cell layers out side each boundary.
    **kwargs
        Keyword arguments sent to netCDF4.Dataset.
    """
    fpath = pathlib.Path(fpath).expanduser().resolve()

    # determine if it's a WHUHVModel or HUVModel
    if hasattr(soln, "w"):
        keys = ["w", "hu", "hv"]
    else:
        keys = ["h", "u", "v"]

    if ngh == 0:
        data = {k: soln[k] for k in keys}
    else:
        slc = slice(ngh, -ngh)  # alias for convenience; non-ghost domain
        data = {k: soln[k][slc, slc] for k in keys}

    with netCDF4.Dataset(fpath, "a", **kwargs) as dset:  # pylint: disable=no-member
        add_time_data_to_dataset(dset, data, time, tidx)


def create_topography_file(fpath, x, y, data, options=None, **kwargs):
    """A helper to create a topography DEM file with NetCDF CF convention.

    The key of the elevation is fixed to `elevation` for convenience. By default, the spatial axes
    `x` and `y` use EPSG 3857 system. All length units are in meters (i.e., `m`).

    Arguments
    ---------
    fpath : str or PathLike
        The path to the file.
    x, y : nplike.ndarray
        The coordinates of the gridlines in x (west-east)and y (south-north) direction.
    data : nplike.ndarray
        The elevation data with shape (len(y), len(x))
    options : dict or None
        To overwrite the default attribute values of `x`, `y`, `elevation`, and `root`. See the
        docstring of `wrtie_to_dataset`.
    **kwargs
        Keyword arguments sent to netCDF4.Dataset.
    """
    fpath = pathlib.Path(fpath).expanduser().resolve()
    _options = {"elevation": {"units": "m"}}
    _options.update({} if options is None else options)

    with netCDF4.Dataset(fpath, "w", **kwargs) as dset:  # pylint: disable=no-member
        write_to_dataset(dset, x, y, None, {"elevation": data}, _options)
