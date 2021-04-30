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
from .dummydict import DummyDict  # pylint: disable=import-error


def read_cf(fpath, data_keys, **kwargs):
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
        - x: a 1D numpy.ndarray; gridline in x direction.
        - y: a 1D numpy.ndarray; gridline in y direction.
        - time: (optional) a 1D numpy.ndarray if gridline in time exists.
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


def write_cf(fpath, gridline, data, options=None, **kwargs):
    """A wrapper to safely write to a NetCDF file.

    In case an I/O error happen, the NetCDF file will still be safely closed.

    Arguments
    ---------
    fpath : str or path-like
        Path to the target file.
    gridline, data, options :
        See `write_cf_to_dataset` for their documentation.
    **kwargs :
        Arbitrary keyword arguments that will be provide to netCDF4.Dataset.__init__.
    """

    fpath = pathlib.Path(fpath).expanduser().resolve()

    with netCDF4.Dataset(fpath, "w", **kwargs) as rootgrp:  # pylint: disable=no-member
        rootgrp = write_cf_to_dataset(rootgrp, gridline, data, options)


def write_cf_to_dataset(dset, gridline, data, options=None):
    """Create a NetCDF file of CF convention.

    Note the input argument, data, is in regular numeric simulation style, i.e.,
    data[0, 0] is the most bottom-left data point in structured grid. And
    data[1, :] is all values in the second row from the bottom in the structured
    grid.

    The spatial gridline x an y are always assumed to be EPSG:3857 coordinates.

    No sanity check will be done. Users have to be careful of dimensions mismatch.

    Argument
    --------
    dset : netCDF4.Dataset
        The destination of data output.
    gridline : a dict
        This dict contains the following key-value pair
        - x, y : 1D numpy.ndarray of length Nx and Ny repectively
            The x and y coordinates
        - time : (optional) 1D numpy.ndarray of length Nt
            The coordinates in temporal axis.
    data: a dictionary of (variable name, array) pairs; the arrays are
            spatial data only, i.e., with shape (Ny, Nx)
    options: a dict of dict
        The outer dictionary has pairs (variable name, dictionary). The inner dictionaries
        are the attributes of each vriable. A special key is "root", which holds attributes
        to the dataset itself. For options of variables, usually users may want to at least
        specify the attribute "units".

    Returns
    -------
    dset : netCDF4.Dataset
        The same input dataset but with a mercator.
    """

    # sanity check
    assert "x" in gridline, "The key \"x\" is missing."
    assert "y" in gridline, "The key \"y\" is missing."
    for val in data.values():
        if "time" in gridline:
            assert len(val.shape) in (2, 3), "The arrays should be either 2D or 3D."
        else:
            assert len(val.shape) == 2, "The arrays should be 2D."

    # get current time
    cur_t = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    # complete the option dictonary to save time dealing with KeyError exception
    options = {} if options is None else options
    for key in ["root", "x", "y", "time", "mercator"]:
        options[key] = options[key] if key in options else {}

    # global attributes
    dset.title = "Data created by TorchSWE"
    dset.institution = "The George Washington University"
    dset.source = "TorchSWE"
    dset.history = "Created " + cur_t
    dset.reference = "https://github.com/piyueh/TorchSWE"
    dset.comment = ""
    dset.Conventions = "CF-1.7"

    # overwrite the root groups's attributes if users provide any
    dset.setncatts(options["root"])

    # create axes
    dset = add_spatial_axis(dset, gridline, options)

    if "time" in gridline:
        dset = add_time_axis(dset, gridline["time"], None, options["time"])

    # create spatial variables
    for key, val in data.items():
        if len(val.shape) == 2:
            dset.createVariable(key, "f8", ("y", "x"), True, 9, fill_value=float("NaN"))
        elif len(val.shape) == 3:
            dset.createVariable(key, "f8", ("time", "y", "x"), True, 9, fill_value=float("NaN"))

        # variable attributes
        dset[key][...] = val
        dset[key].long_name = key
        dset[key].grid_mapping = "mercator"

        # overwrite the data's attributes if users provide any
        try:
            dset[key].setncatts(options[key])
        except KeyError:
            pass

    # add mercator
    dset = add_mercator(
        dset, (gridline["x"][0], gridline["y"][-1]),
        (gridline["x"][1]-gridline["x"][1], gridline["y"][1]-gridline["y"][0]),
        options["mercator"])

    return dset


def append_time_data(fpath, time, data, options=None, **kwargs):
    """Append data to temporal dataset in a NetCDF file.

    Note:
        - the corresponding variables must either not exist in the dataset or already have temporal
          dimension.
        - fill_value is assumed to be float("NaN")

    Arguments
    ---------
    fpath : str or path-like
        Path to the target file.
    time : float
        The value of time.
    data : a dict
        Has key-value pairs of (variable name, array). The arrays have the shape (Ny, Nx).
    options : None or a dict
        A dicitonary of dictionary to specify the attributes of each variables if the variables will
        be created in this function.
    **kwargs :
        Arbitrary keyword arguments that will be provide to netCDF4.Dataset.__init__.
    """

    options = {} if options is None else options
    options["time"] = options["time"] if "time" in options else {}

    # use absolute path
    fpath = pathlib.Path(fpath).expanduser().resolve()

    # create a NetCDF4 file/dataset
    with netCDF4.Dataset(fpath, "a", **kwargs) as dset:  # pylint: disable=no-member

        # see if temporal dimension already exists
        if "time" not in dset.dimensions.keys():
            dset = add_time_axis(dset, options=options["time"])

            for key, array in data.items():
                dset.createVariable(key, "f8", ("time", "y", "x"), True, 9, fill_value=float("NaN"))
                dset[key].long_name = key
                dset[key].grid_mapping = "mercator"

                # overwrite or add attributes if users provide any
                try:
                    dset[key].setncatts(options[key])
                except KeyError:
                    pass

        tidx = len(dset["time"])
        dset["time"][tidx] = time

        # add data
        for key, array in data.items():
            dset[key][tidx, :, :] = array


def add_time_axis(dset, values=None, timestamp=None, options=None):
    """Add the time axis to a netCDF4.Dataset.

    Default unit is "seconds since {timestamp}".

    Arguments
    ---------
    dset : netCDF4.Dataset
        The target dataset.
    values : None, a list, or numpy.ndarray
        Given time values. None means users will assign values later.
    timestamp: None or str
        A custom timestamp used in "since ..." in CF convention. Use ISO time format. If None, the
        default value is the current UTC time.
    options: None or a dict
        An optional dictionary to overwrite default attributes or add new attributes.

    Returns
    -------
    dset : netCDF4.Dataset
        The same input dataset but with a new spatial axis.
    """

    if timestamp is None:
        timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    _ = dset.createDimension("time", None)
    axis = dset.createVariable("time", "f8", ("time",))
    axis[:] = values if values is not None else []
    axis.axis = "T"
    axis.long_name = "Simulation time"
    axis.units = "seconds since " + timestamp

    if options is None or "calendar" not in options:
        axis.calendar = "standard"

    if options is not None:
        axis.setncatts(options)

    return dset


def add_spatial_axis(dset, coords, options=None):
    """Add the spatial axes to a netCDF4.Dataset.

    The spatial coordinates are always assumed to be in EPSG:3857 standard. And the default unit is
    meter. The name of the gridlines are hard-coded to "x" and "y".

    Arguments
    ---------
    dset : netCDF4.Dataset
        The target dataset.
    coords: a dict of 1D numpy.ndarray
        Coordinates in "x" and "y".
    options: None or a dict
        An optional dictionary to overwrite default attributes or add new attributes. If not None,
        options should look like: options = {"x": {additional attrs of x ...}, "y": {additional
        attrs of y ...}}.

    Returns
    -------
    dset : netCDF4.Dataset
        The same input dataset but with a new spatial axis.
    """

    axes = {}
    for key in ("x", "y"):
        _ = dset.createDimension(key, len(coords[key]))
        axes[key] = dset.createVariable(key, "f8", (key,))
        axes[key][:] = coords["x"]
        axes[key].units = "m"
        axes[key].long_name = "{}-coordinate in EPSG:3857 WGS 84".format(key)
        axes[key].standard_name = "projection_{}_coordinate".format(key)

        # overwrite/add the attributes if users provide any
        if options is not None and options[key] is not None:
            axes[key].setncatts(options[key])

    return dset


def add_mercator(dset, wn_corner, delta, options=None):
    """Add mercator to a given netCDF4.Dataset.

    The projection is assumed to be EPSG:3857.

    Argument
    --------
    dset : netCDF4.Dataset
        The target dataset.
    wn_corner : a tuple/list of length 2
        The coordinate of the west-north (i.e., top-left) corner.
    delta : a tuple/list of length 2
        The gridspacing in x and y directions.
    options: None or a dict
        An optional dictionary to overwrite default attributes or add new attributes.

    Returns
    -------
    dset : netCDF4.Dataset
        The same input dataset but with a mercator.
    """

    # create the variable
    mrctr = dset.createVariable("mercator", "S1")

    # variable attributes: mercator
    mrctr.GeoTransform = "{} {} 0 {} 0 {}".format(wn_corner[0], delta[0], wn_corner[1], -delta[1])
    mrctr.grid_mapping_name = "mercator"
    mrctr.long_name = "CRS definition"
    mrctr.longitude_of_projection_origin = 0.0
    mrctr.standard_parallel = 0.0
    mrctr.false_easting = 0.0
    mrctr.false_northing = 0.0
    mrctr.spatial_ref = \
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

    # overwrite mercator's attributes if users provide any
    if options is not None:
        mrctr.setncatts(options)

    return dset
