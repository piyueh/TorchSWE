#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Lower leve functions related to NetCDF I/O with the CF convention.
"""
from pathlib import Path as _Path
from datetime import datetime as _datetime, timezone as _timezone

from netCDF4 import Dataset as _Dataset  # pylint: disable=no-name-in-module
from torchswe import nplike as _nplike
from torchswe.utils.misc import DummyDict as _DummyDict


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
    cur_t = _datetime.now(_timezone.utc).replace(microsecond=0).isoformat()

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


def read(fpath, data_keys, domain=None, **kwargs):
    """Read data from a NetCDF file in CF convention (parallel version).

    The spatial data will have shape (ny, nx) or (ntime, ny, nx). For example, data[0, 0] is the
    most bottom-left data point in a structured grid. And data[1, :] represents all points in the
    second row from the bottom of the structured grid.

    Arguments
    ---------
    fpath : str or path-like
        Path to the input file.
    data_keys : a list/tuple of str
        The keys of data to read from the dataset
    domain : a list/tuple of 4 floats, or None
        Read data within the bounds (west, east, south, north) of the domain. If None, read all.
    **kwargs :
        Arbitrary keyword arguments passed into netCDF4.Dataset.__init__.

    Returns
    -------
    data : dict
        The (key, array) pairs with the following extra data:
            - x, y : 1D ndarray of the local gridlines correspond to the returnd data.
            - time : a list of the snapshot times
    attrs : dict
        The attributes to the data in the dictionary `data`.

    Notes
    -----
    If domain bounds do not exactly fall on a gridline, the returned data blocks/slices will be
    a little bit larger than the domain to cover the whole domain. Then, users can do interpolation
    later by themselves.
    """

    # use absolute path
    fpath = _Path(fpath).expanduser().resolve()

    with _Dataset(fpath, **kwargs) as dset:
        data, attrs = read_from_dataset(dset, data_keys, domain)
    return data, attrs


def read_from_dataset(dset, data_keys, domain=None):
    """Read a block of data from an opened dataset.

    Arguments
    ---------
    dset : netCDF4.Dataset
        The target dataset.
    data_keys : a list/tuple of str
        The keys of data to read from the dataset
    domain : a list/tuple of 4 floats, or None
        Read data within the bounds (west, east, south, north) of the domain. If None, read all.

    Returns
    -------
    data : dict
        The (key, array) pairs with the following extra data:
            - x, y : 1D ndarray of the local gridlines correspond to the returnd data.
            - time : a list of the snapshot times
    attrs : dict
        The attributes to the data in the dictionary `data`.

    Notes
    -----
    If domain bounds do not exactly fall on a gridline, the returned data blocks/slices will be
    a little bit larger than the domain to cover the whole domain. Then, users can do interpolation
    later by themselves.
    """

    # empty dictionary
    data = _DummyDict()
    attrs = _DummyDict()

    # make a local copy of the global fridline
    data["x"] = _nplike.array(dset["x"][:])
    data["y"] = _nplike.array(dset["y"][:])

    try:  # the raster data could be defined at cell centers, i.e., extent is different
        transform = dset["mercator"].GeoTransform.split()
        extent = [
            float(transform[0]), float(transform[0]) + float(transform[1]) * len(data["x"]),
            float(transform[3]) + float(transform[5]) * len(data["y"]), float(transform[3])
        ]
    except IndexError as err:
        if "mercator not found" in str(err):  # otherwise, use x, y data for extent
            extent = [data["x"][0], data["x"][-1], data["y"][0], data["y"][-1]]
        raise

    # determine the target domain in the index space
    if domain is None:
        ibg, ied, jbg, jed = None, None, None, None  # standard slicing: None:None means all
    else:
        # make sure the whole raster covers the required domain
        assert extent[0] <= domain[0], "{}, {}".format(extent[0], domain[0])
        assert extent[1] >= domain[1], "{}, {}".format(extent[1], domain[1])
        assert extent[2] <= domain[2], "{}, {}".format(extent[2], domain[2])
        assert extent[3] >= domain[3], "{}, {}".format(extent[3], domain[3])

        # find the start and end indices containing the provided domain
        ibg = int(_nplike.searchsorted(data["x"], domain[0]))
        ied = int(_nplike.searchsorted(data["x"], domain[1]))
        jbg = int(_nplike.searchsorted(data["y"], domain[2]))
        jed = int(_nplike.searchsorted(data["y"], domain[3]))

        # torch's searchsorted signature differs, so no right search; manual adjustment instead
        ied = len(data["x"]) - 1 if ied >= len(data["x"]) else ied
        jed = len(data["y"]) - 1 if jed >= len(data["y"]) else jed

        # make sure the target domain is big enough for interpolation, except for edge cases
        ibg = ibg - 1 if data["x"][ibg] > domain[0] and ibg != 0 else ibg
        ied = ied + 1 if data["x"][ied] < domain[1] and ied < len(data["x"])-2 else ied
        jbg = jbg - 1 if data["y"][jbg] > domain[2] and jbg != 0 else jbg
        jed = jed + 1 if data["y"][jed] < domain[3] and jed < len(data["y"])-2 else jed

        # the end has to shift one for slicing
        ied += 1
        jed += 1

        # save only the local gridlines to the output dictionary
        data["x"] = data["x"][ibg:ied]
        data["y"] = data["y"][jbg:jed]

    # try to read temporal axis if it exists
    try:
        data["time"] = list(dset["time"][:].data)
    except IndexError as err:
        if "time not found" not in str(err):  # only raise if the error is not about `time`
            raise

    # determine if the data is 2D or 3D
    if "time" in data:
        slc = (slice(None), slice(jbg, jed), slice(ibg, ied))
    else:
        slc = (slice(jbg, jed), slice(ibg, ied))

    # read data and attributes of each specified key
    for key in data_keys:
        data[key] = _nplike.array(dset[key][slc])
        attrs[key] = dset[key].__dict__

    return data, attrs


def write(
    fpath, axs, data=None, global_n=None, idx_bounds=None, corner=None, deltas=None, options=None,
    **kwargs
):
    """Write to a new NetCDF file with CF convention.

    Arguments
    ---------
    fpath : str or path-like
        Path to the target file.
    xyt : a list/tuple
        Can be a length-2 or length-3 list/tuple. If length is 2, it's type is [ndarray, ndarray],
        corresponding to x and y gridlines. If length is 3, the additional one is the temporal axis.
    data : dict
        A dictionary of (variable name, value). If value is None, create all-NaN array for the
        variable. If not None, the shape of value should be either (ny, nx) or (ntime, ny, nx).
    global_n : a list/tuple of 2 int
        Only useful for parallel write. The global size of the x, y gridlines. In this case,
        provided `axs` represent the local gridlines, so their shapes are different from `global_n`.
    idx_bounds : list/tuple of 4 int (west, east, south, north)
        Only useful for parallel write. Write data to the slice/block bounded in these indices.
    options: a dict of dict
        The outer dictionary has pairs (variable name, dictionary). The inner dictionaries
        are the attributes of each vriable. A special key is "root", which holds attributes
        to the dataset itself. For options of variables, usually users may want to at least
        specify the attribute "units".
    **kwargs :
        Arbitrary keyword arguments that will be provide to netCDF4.Dataset.__init__.
    """
    # pylint: disable=too-many-arguments

    fpath = _Path(fpath).expanduser().resolve()

    with _Dataset(fpath, "w", **kwargs) as rootgrp:  # pylint: disable=no-member
        rootgrp = write_to_dataset(
            rootgrp, axs, data, global_n, idx_bounds, corner, deltas, options)


def write_to_dataset(
    dset, axs, data=None, global_n=None, idx_bounds=None, corner=None, deltas=None, options=None
):
    """Write gridlines and data to an opened but empty NetCDF dataset using CF convention.

    Arguments
    ---------
    dset : netCDF4.Dataset
        The destination dataset.
    xyt : a list/tuple
        Can be a length-2 or length-3 list/tuple. If length is 2, it's type is [ndarray, ndarray],
        corresponding to x and y gridlines. If length is 3, the additional one is the temporal axis.
    data : dict
        A dictionary of (variable name, value). If value is None, create all-NaN array for the
        variable. If not None, the shape of value should be either (ny, nx) or (ntime, ny, nx).
    global_n : a list/tuple of 2 int
        Only useful for parallel write. The global size of the x, y gridlines. In this case,
        provided `axs` represent the local gridlines, so their shapes are different from `global_n`.
    idx_bounds : list/tuple of 4 int (west, east, south, north)
        Only useful for parallel write. Write data to the slice/block bounded in these indices.
    options: a dict of dict
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
    # pylint: disable=too-many-arguments

    # default values
    global_n = [len(axs[0]), len(axs[1])] if global_n is None else global_n
    idx_bounds = [None, None, None, None] if idx_bounds is None else idx_bounds
    corner = (axs[0][0], axs[1][-1]) if corner is None else corner
    deltas = (axs[0][1]-axs[0][0], axs[1][1]-axs[1][0]) if deltas is None else deltas

    # get default options
    options = {} if options is None else options
    _options = default_attrs(corner, deltas)
    _options = {**options, **_options}

    for key in ["root", "x", "y", "time", "mercator"]:
        _options[key].update(options[key] if key in options else {})

    # global attributes
    dset.setncatts(_options["root"])

    # create axs
    add_axis_to_dataset(dset, "x", axs[0], global_n[0], idx_bounds[:2], _options["x"])
    add_axis_to_dataset(dset, "y", axs[1], global_n[1], idx_bounds[2:], _options["y"])

    # temporal axis is optional; it is not a decomposed axis
    if len(axs) == 3:
        add_axis_to_dataset(dset, "time", axs[2], options=_options["time"])

    # create mercator
    dset.createVariable("mercator", "S1")
    dset["mercator"].setncatts(_options["mercator"])

    # create variables
    add_variables_to_dataset(dset, data, idx_bounds, _options)

    return dset


def add_variables_to_dataset(dset, data, idx_bounds=None, options=None):
    """Add variables to an existing dataset.

    Arguments
    ---------
    dset : netCDF4.Dataset
        The target dataset. Must be opened in "a" mode, i.e., appending mode.
    data : dict
        A dictionary of (variable name, value). If value is None, create all-NaN array for the
        variable. If not None, the shape of value should be either (ny, nx) or (ntime, ny, nx).
    idx_bounds : list/tuple of 4 int (west, east, south, north)
        Only write data to the slice/block bounded in these indices.
    options: a dict of dict
        The outer dictionary has pairs (variable name, dictionary). The inner dictionaries
        are the attributes of each vriable. Usually users may want to at least specify the
        attribute "units".

    Returns
    -------
    The same dataset instance (w/ new variable added).
    """

    # update the default values
    nan = float("NaN")
    idx_bounds = [None, None, None, None] if idx_bounds is None else idx_bounds
    options = {} if options is None else options

    # create variables
    for key, val in data.items():

        # create the variable
        shape = ("ntime", "ny", "nx") if "ntime" in dset.dimensions else ("ny", "nx")
        dset.createVariable(key, "f8", shape, fill_value=nan)  # all NaN in it right now

        # variable attributes
        dset[key].long_name = key
        dset[key].grid_mapping = "mercator"
        dset[key].setncatts(options[key] if key in options else {})

        if val is None:  # no need to copy data
            continue

        # spatial slice object
        slc = (slice(idx_bounds[3], idx_bounds[2]), slice(idx_bounds[1], idx_bounds[0]))

        # modify slices based on temporal axis
        if len(val.shape) == 2:
            if "ntime" in dset.dimensions:  # assume this is the first snapshot
                slc = (0,) + slc
        elif len(val.shape) == 3:
            slc = (slice(None),) + slc
        else:
            raise ValueError("\"{}\" should be either 2D or 3D.".format(key))

        _copy_data(dset[key], val, slc)

    return dset


def add_time_data_to_dataset(dset, data, time, tidx=None, idx_bounds=None):
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
    idx_bounds : list/tuple of 4 int (west, east, south, north)
        Only write data to the slice/block bounded in these indices.

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

    # update the default values
    idx_bounds = [None, None, None, None] if idx_bounds is None else idx_bounds

    if tidx is None:  # append to the variables
        assert dset.dimensions["ntime"].isunlimited()
        tidx = dset.dimensions["ntime"].size
        dset["time"][tidx] = time
    else:
        assert abs(dset["time"][tidx]-time) < 1e-10, "Time does not match the dataset's record."

    for key, val in data.items():
        assert len(val.shape) == 2  # should be a single snapshot
        slc = (tidx, slice(idx_bounds[2], idx_bounds[3]), slice(idx_bounds[0], idx_bounds[1]))
        _copy_data(dset[key], val, slc)

    return dset


def add_axis_to_dataset(dset, name, values, global_n=None, idx_bounds=None, options=None):
    """Add an axes to a netCDF4.Dataset.

    Arguments
    ---------
    dset : netCDF4.Dataset
        The target dataset.
    name : str
        The axis name.
    values: nplike.ndarray or list
        Coordinate values in this axis.
    global_n : int
        The global number of values in this axis. Only used for parallel write when `len(values)`
        does not equal to `global_n`.
    idx_bounds : a list/tuple of 2 int
        During parallel write, write `values` into the global axis using the slice from
        idx_bounds[0] to idx_bounds[1].
    options: None or a dict
        A dictionary for setting attributes.

    Returns
    -------
    dset : netCDF4.Dataset
        The same input dataset but with a new axis.
    """
    # pylint: disable=too-many-arguments

    # update default values
    global_n = len(values) if global_n is None else global_n
    idx_bounds = [None, None] if idx_bounds is None else idx_bounds
    options = {} if options is None else options

    dset.createDimension("n{}".format(name), global_n)
    dset.createVariable(name, "f8", ("n{}".format(name),))
    dset[name].setncatts(options)

    try:
        dset[name].set_collective(True)
    except RuntimeError as err:
        if "Parallel operation" not in str(err):  # only raise error if not complaining serial data
            raise

    _copy_data(dset[name], values, slice(idx_bounds[0], idx_bounds[1]))

    return dset


def _copy_data(var, array, slc):
    """Copy a non-completely np-compatible ndarray to a NetCDF4 variable."""

    try:
        var[slc] = array
    except TypeError as err:
        if str(err).startswith("Implicit conversion to a NumPy array is not allowe"):
            var[slc] = array.get()  # cupy
        elif str(err).startswith("can't convert cuda:"):
            var[slc] = array.cpu().numpy()  # pytorch
        else:
            raise
