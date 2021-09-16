#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Higher level api for writing data to files.
"""
from pathlib import Path as _Path
from netCDF4 import Dataset as _Dataset  # pylint: disable=no-name-in-module
from torchswe.utils.netcdf import write_to_dataset as _write_to_dataset
from torchswe.utils.netcdf import add_time_data_to_dataset as _add_time_data_to_dataset
from torchswe.utils.netcdf import read as _ncread


def create_soln_snapshot_file(fpath, domain, soln, **kwargs):
    """Create a NetCDF file with a single snapshot of solutions.

    Arguments
    ---------
    fpath : str or PathLike
        The path to the file.
    domain : torchswe.utils.data.Domain
        The Domain instance corresponds associated to the solutions.
    soln : torchswe.utils.data.WHUHVModel or torchswe.utils.data.HUVModel
        The snapshot of the solutions.
    **kwargs
        Keyword arguments sent to netCDF4.Dataset (excluding `parallel` and `comm`).
    """

    assert "parallel" not in kwargs, "`parallel` should not be included in `kwargs`"
    assert "comm" not in kwargs, "`parallel` should not be included in `kwargs`"

    fpath = _Path(fpath).expanduser().resolve()

    try:
        data = {k: soln[k] for k in ["w", "hu", "hv"]}
        options = {"w": {"units": "m"}, "hu": {"units": "m2 s-1"}, "hv": {"units": "m2 s-1"}}
    except AttributeError as err:
        if "has no attribute \'w\'" in str(err):  # a HUVModel
            data = {k: soln[k] for k in ["h", "u", "v"]}
            options = {"h": {"units": "m"}, "u": {"units": "m s-1"}, "v": {"units": "m s-1"}}
        else:
            raise

    with _Dataset(fpath, "w", parallel=True, comm=domain.process.comm, **kwargs) as dset:

        _write_to_dataset(
            dset=dset,
            axs=(domain.x.centers, domain.y.centers),
            data=data,
            global_n=(domain.x.gn, domain.y.gn),
            idx_bounds=(domain.x.ibegin, domain.x.iend, domain.y.ibegin, domain.y.iend),
            corner=(domain.x.glower, domain.y.gupper),
            deltas=(domain.x.delta, domain.y.delta),
            options=options
        )

        dset.sync()


def create_empty_soln_file(fpath, domain, t, model="whuhv", **kwargs):
    """Create an new NetCDF file for solutions using the corresponding grid object.

    Create an empty NetCDF4 file with axes `x`, `y`, and `time`. `x` and `y` are defined at cell
    centers. The spatial coordinates use EPSG 3856. The temporal axis is limited with dimension
    `ntime` (i.e., not using the unlimited axis feature from CF convention).

    Also, this function creates empty solution variables called `w`, `hu`, and `hv` in the dataset
    with `NaN` for all values. The shapes of these variables are `(ntime, ny, nx)`. The units of
    them are "m", "m2 s-1", and "m2 s-1", respectively.

    Arguments
    ---------
    fpath : str or PathLike
        The path to the file.
    domain : torchswe.utils.data.Domain
        The Domain instance corresponds to the solutions.
    t : torchswe.utils.data.Timeline
        The temporal axis object.
    model : str, either "whuhv" or "huv"
        The type of solution: the conservative form (w, hu, hv) or non-conservative form (h, u, v).
    **kwargs
        Keyword arguments sent to netCDF4.Dataset.
    """

    assert "parallel" not in kwargs, "`parallel` should not be included in `kwargs`"
    assert "comm" not in kwargs, "`parallel` should not be included in `kwargs`"

    fpath = _Path(fpath).expanduser().resolve()

    if model == "whuhv":
        data = {k: None for k in ["w", "hu", "hv"]}
        options = {"w": {"units": "m"}, "hu": {"units": "m2 s-1"}, "hv": {"units": "m2 s-1"}}
    elif model == "huv":
        data = {k: None for k in ["h", "u", "v"]}
        options = {"h": {"units": "m"}, "u": {"units": "m s-1"}, "v": {"units": "m s-1"}}

    with _Dataset(fpath, "w", parallel=True, comm=domain.process.comm, **kwargs) as dset:

        _write_to_dataset(
            dset=dset,
            axs=(domain.x.centers, domain.y.centers, t.values),
            data=data,
            global_n=(domain.x.gn, domain.y.gn),
            idx_bounds=(domain.x.ibegin, domain.x.iend, domain.y.ibegin, domain.y.iend),
            corner=(domain.x.glower, domain.y.gupper),
            deltas=(domain.x.delta, domain.y.delta),
            options=options
        )

        dset.sync()


def write_soln_to_file(fpath, domain, soln, time, tidx, ngh=0, **kwargs):
    """Write a solution snapshot to an existing NetCDF file.

    Arguments
    ---------
    fpath : str or PathLike
        The path to the file.
    domain : torchswe.utils.data.Domain
        A Domain instance associated to the solutions.
    soln : torchswe.utils.data.WHUHVModel or torchswe.utils.data.HUVModel
        The conservative solutions or non-conservative solutions.
    time : float
        The simulation time of this snapshot.
    tidx : int
        The index of the snapshot time in the temporal axis.
    ngh : int
        The number of ghost-cell layers outside each boundary. These layers will be excluded from
        the dataset.
    **kwargs
        Keyword arguments sent to netCDF4.Dataset.
    """
    fpath = _Path(fpath).expanduser().resolve()

    assert "parallel" not in kwargs, "`parallel` should not be included in `kwargs`"
    assert "comm" not in kwargs, "`parallel` should not be included in `kwargs`"

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

    with _Dataset(fpath, "a", parallel=True, comm=domain.process.comm, **kwargs) as dset:

        _add_time_data_to_dataset(
            dset=dset, data=data, time=time, tidx=tidx,
            idx_bounds=(domain.x.ibegin, domain.x.iend, domain.y.ibegin, domain.y.iend)
        )
        dset.sync()


def create_topography_file(fpath, axs, elevation, key="elevation", options=None, **kwargs):
    """A helper to create a topography DEM file with NetCDF CF convention.

    The key of the elevation is fixed to `elevation` for convenience. By default, the spatial axes
    `x` and `y` use EPSG 3857 system. All length units are in meters (i.e., `m`).

    Arguments
    ---------
    fpath : str or PathLike
        The path to the file.
    axs : a list/tuple of nplike.ndarray
        The coordinates of the gridlines in x (west-east) and y (south-north) direction.
    elevation : nplike.ndarray
        The elevation data with shape (ny, nx)
    key : str
        The key used to identify the elevation data in the dataset. (default: "elevation")
    options : dict or None
        To overwrite the default attribute values of `x`, `y`, `elevation`, and `root`.
    **kwargs
        Keyword arguments sent to netCDF4.Dataset.
    """
    fpath = _Path(fpath).expanduser().resolve()
    _options = {key: {"units": "m"}}
    _options.update({} if options is None else options)

    with _Dataset(fpath, "w", **kwargs) as dset:
        _write_to_dataset(dset, axs, {key: elevation}, options=_options)


def read_topography_file(fpath, key, extent, **kwargs):
    """Read the topograhy elevation data from a NetCDF DEM file with CF convention.

    Arguments
    ---------
    fpath : str or PathLike
        The path to the file.
    key : str
        The name/key of the elevation data in the NetCDF file.
    extent : a tuple/list of 4 floats
        The format is [west, east, south, north].
    **kwargs
        Other keyword arguments that will be supplied to open netCDF4.Dataset.

    Returns
    -------
    x, y : 1D nplike.ndarray
        The coordinates of where the returned elevations are defined at.
    topo : 2D nplike.ndarray
        The elevations. Its shape should be (y.size, x.size).

    Notes
    -----
    Common keyword arguments used for parallel read are `parallel=True` and `comm=...`.
    """

    dem, _ = _ncread(fpath, [key], extent, **kwargs)
    assert dem[key].shape == (dem["x"].size, dem["y"].size)
    return dem["x"], dem["y"], dem[key]
