#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the MIT license.

"""
Functions related to I/O with NetCDF files in CF convention.
"""
import os
import datetime
import netCDF4


def read_cf(filepath, data_keys, **kwargs):
    """Read data from a NetCDF file in CF convention.

    The pure spatial array, data is in traditional numerical simulation style.
    For example, data[0, 0] is the most bottom-left data point in a structured
    grid. And data[1, :] represents all points in the second row from the bottom
    of the structured grid.

    For temporal data, the dimension is (time, y, x).

    Args:
    -----
        filepath: path to the input file.
        data_keys: a list of variable names in the file that will be read.

        **kwargs are extra argument passed into netCDF4.Dataset.__init__.

    Returns:
    --------
        data: a dictionary that has key-value pairs of
            x: a 1D numpy.ndarray; gridline in x direction.
            y: a 1D numpy.ndarray; gridline in y direction.
            time: (optional) a 1D numpy.ndarray if gridline in time exists.
            And all keys specified in data_key argument.

        attrs: a dict of dicts; attributes for each key in data (exclude root
            group's).
    """

    # use absolute path
    filepath = os.path.abspath(filepath)

    # empty dictionary
    data = {}
    attrs = {}

    # create a NetCDF4 file/dataset
    with netCDF4.Dataset(filepath, "r", **kwargs) as rootgrp:

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

def write_cf(filepath, x, y, data, t=None, tdata=None, options={}, **kwargs):
    """A wrapper to safely write to a NetCDF file.

    In case an I/O error happen, the NetCDF file will still be safely closed.

    Args:
    -----
        filepath: path to the target file.

        See write_cf_to_dataset for other argument documentation. **kwargs are
        the arguments of netCDF4.Dataset.__init__.

    Returns:
    --------
        N/A
    """

    filepath = os.path.abspath(filepath)

    with netCDF4.Dataset(filepath, "w", **kwargs) as rootgrp:
        rootgrp = write_cf_to_dataset(rootgrp, x, y, data, t, tdata, options)

def write_cf_to_dataset(dataset, x, y, data, t=None, tdata=None, options={}):
    """Create a NetCDF file of CF convention.

    Note the input argument, data, is in regular numeric simulation style, i.e.,
    data[0, 0] is the most bottom-left data point in structured grid. And
    data[1, :] is all values in the second row from the bottom in the structured
    grid.

    The spatial gridline x an y are always assumed to be EPSG:3857 coordinates.

    No sanity check will be done. Users have to be careful of dimensions mismatch.

    Args:
    -----
        dataset: a netCDF4.Dataset; the destination of data output.
        x: a 1D numpy.ndarray of length Nx; gridlines in x.
        y: a 1D numpy.ndarray of length Nx; gridlines in y.
        data: a dictionary of (variable name, array) pairs; the arrays are
            spatial data only, i.e., with shape (Ny, Nx)
        t: an optional numpy.ndarray of length Nt for temporal gridline.
        tdata: a dictionary of (variable name, array) pairs; the arrays are
            temporal-spatial data, i.e, with shape (Nt, Ny, Nx).
        options: a dictionary of dictionary. The outer dictionary are of pairs
            (variable name, dictionary). The inner dictionaries are the
            attributes of each vriable. A special key is "root", which holds
            attributes to the dataset itself. For options of variables, usually
            users may want to at least specify the attribute "units".

    Return:
    -------
        The same input dataset but with data.
    """

    # get current time
    from datetime import datetime, timezone
    T = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    # complete the option dictonary to save time dealing with KeyError exception
    for key in ["root", "x", "y", "time", "mercator"]:
        options[key] = options[key] if key in options else {}

    # global attributes
    dataset.title = "Data created by TorchSWE"
    dataset.institution = "The George Washington University"
    dataset.source = "TorchSWE"
    dataset.history = "Created " + T
    dataset.reference = "https://github.com/piyueh/TorchSWE"
    dataset.comment = ""
    dataset.Conventions = "CF-1.7"

    # overwrite the root groups's attributes if users provide any
    dataset.setncatts(options["root"])

    # create spatial axes
    dataset = add_spatial_axis(dataset, "x", x, options["x"])
    dataset = add_spatial_axis(dataset, "y", y, options["y"])

    # create spatial variables
    for key, array in data.items():
        dataset.createVariable(
            key, "f8", ("y", "x"), True, 9, fill_value=float("NaN"))

        # variable attributes
        dataset[key][:, :] = array
        dataset[key].long_name = key
        dataset[key].grid_mapping = "mercator"

        # overwrite the data's attributes if users provide any
        try:
            dataset[key].setncatts(options[key])
        except KeyError:
            pass

    # if there're also temporal-spatial data
    if t is not None:
        dataset = add_time_axis(dataset, t, options=options["time"])

        # variable attributes
        for key, array in tdata.items():
            dataset.createVariable(
                key, "f8", ("time", "y", "x"), True, 9, fill_value=float("NaN"))

            # variable attributes
            dataset[key][:, :, :] = array
            dataset[key].long_name = key
            dataset[key].grid_mapping = "mercator"

            # overwrite or add attributes if users provide any
            try:
                dataset[key].setncatts(options[key])
            except KeyError:
                pass

    # add mercator
    dataset = add_mercator(
        dataset, x[0], x[1]-x[1], y[-1], y[1]-y[0], options["mercator"])

    return dataset

def append_time_data(filepath, time, data, options={}, **kwargs):
    """Append data to temporal data.

    Note:
        1. the corresponding variables must either not exist in the dataset or
            already have temporal dimension.
        2. fill_value is assumed to be float("NaN")

    Args:
    -----
        filepath: file path.
        time: a scalar; the value of time.
        data: a dictionary of (variable name, array); the arrays are the spatial
            data being written in NetCDF and with shape (Ny, Nx).
        options: a dicitonary of dictionary to specify the attributes of each
            variables if the variables will be created in this function.

        **kwargs are the arguments passed to netCDF4.Dataset.__init__.

    Returns:
    --------
        N/A
    """

    options["time"] = options["time"] if "time" in options else {}

    # use absolute path
    filepath = os.path.abspath(filepath)

    # create a NetCDF4 file/dataset
    with netCDF4.Dataset(filepath, "a", **kwargs) as rootgrp:

        # see if temporal dimension already exists
        if "time" not in rootgrp.dimensions.keys():
            rootgrp = add_time_axis(rootgrp, options=options["time"])

            for key, array in data.items():
                rootgrp.createVariable(
                    key, "f8", ("time", "y", "x"), True, 9, fill_value=float("NaN"))
                rootgrp[key].long_name = key
                rootgrp[key].grid_mapping = "mercator"

                # overwrite or add attributes if users provide any
                try:
                    rootgrp[key].setncatts(options[key])
                except KeyError:
                    pass

        ti = len(rootgrp["time"])
        rootgrp["time"][ti] = time

        # add data
        for key, array in data.items():
            rootgrp[key][ti, :, :] = array

def add_time_axis(dataset, values=[], timestamp=None, calendar=None, options={}):
    """Add the time axis to a netCDF4.Dataset.

    Default unit is "seconds since {timestamp}".

    Args:
    -----
        dataset: a netCDF4.Dataset.
        values: a list or numpy.ndarray of given time values; empty list means
            users will assign values later.
        timestamp: a custom timestam used for the time unit in CF convention.
            Default value is the current UTC time.
        calendar: a string for the calendar attribute of a time axis in CF
            convention. Default value is "standard".
        options: an optional dictionary to overwrite default attributes or add
            new attributes.

    Return:
    -------
        The same input dataset but with a time axis.
    """

    if timestamp is None:
        from datetime import datetime, timezone
        timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    if calendar is None:
        calendar = "standard"

    nt = dataset.createDimension("time", None)
    t = dataset.createVariable("time", "f8", ("time",))
    t.axis = "T"
    t.long_name = "Simulation time"
    t.calendar = calendar
    t.units = "seconds since " + timestamp

    # copy values in to the netCDF4.Variable
    t[:] = values

    # overwrite/add the attributes if users provide any
    t.setncatts(options)

    return dataset

def add_spatial_axis(dataset, name, gridline, options={}):
    """Add the time axis to a netCDF4.Dataset.

    The spatial gridline is always assumed to be EPSG:3857 coordinates. And the
    default is therefore meter.

    Args:
    -----
        dataset: a netCDF4.Dataset.
        name: the name of the gridline. The name will be the same for the
            corresponding netCDF$.Dimension and netCDF4.Variable.
        gridline: a 1D numpy.ndarray of the gridline data.
        options: an optional dictionary to overwrite default attributes or add
            new attributes.

    Return:
    --------
        The same input dataset but with a new spatial axis.
    """

    # set up dimension
    n = dataset.createDimension(name, gridline.size)

    # create the variable
    x = dataset.createVariable(name, "f8", (name,))

    # variable: x
    x[:] = gridline
    x.units = "m"
    x.long_name = "X-coordinate in EPSG:3857 WGS 84"
    x.standard_name = "projection_x_coordinate"

    # overwrite/add the attributes if users provide any
    x.setncatts(options)

    return dataset

def add_mercator(dataset, xbg, dx, yed, dy, options={}):
    """Add mercator to a given netCDF4.Dataset.

    The projection is assumed to be EPSG:3857.

    Args:
    -----
        dataset: a netCDF4.Dataset.
        xbg: a scalar denoting the domain's most left (west) x-coordinate.
        dx: gird (pixel) size in x direction.
        yed: a scalar denoting the domain's most top (north) x-coordinate.
        dy: gird (pixel) size in y direction.
        options: an optional dictionary to overwrite default attributes or add
            new attributes.

    Returns:
    --------
        The same input dataset but with a mercator.
    """

    # create the variable
    mercator = dataset.createVariable("mercator", "S1")

    # variable attributes: mercator
    mercator.GeoTransform = "{} {} 0 {} 0 {}".format(xbg, dx, yed, -dy)
    mercator.grid_mapping_name = "mercator"
    mercator.long_name = "CRS definition"
    mercator.longitude_of_projection_origin =0.0
    mercator.standard_parallel = 0.0
    mercator.false_easting = 0.0
    mercator.false_northing = 0.0
    mercator.spatial_ref = \
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
            "UNIT[\"Meter\",1.0]]"

    # overwrite mercator's attributes if users provide any
    mercator.setncatts(options)

    return dataset
