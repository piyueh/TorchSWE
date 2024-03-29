#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Functions related Esri ASCII format.
"""
import os
from torchswe import nplike

def read(filepath):
    """Read an Esri ASCII raster file.

    Note, the output array, data, is in traditional numerical simulation style.
    That is to say, data[0, 0] is the most bottom-left data point in a
    structured grid. And data[-1, -1] represents the data point at upper-right
    corner of a structured grid.

    Args:
    -----
        filepath: path to the input file.

    Returns:
    --------
        data: a dictionary that has key-value pairs of
            x: a 1D nplike.ndarray; gridline in x direction.
            y: a 1D nplike.ndarray; gridline in y direction.
            data: a 2D nplike.ndarray; the data

        attrs: a mimic to the output of the read function from netcdf module. The only output is a
            dictionary: {"data": {"_fill_value": nodata_value}}.
    """

    filepath = os.path.abspath(filepath)

    with open(filepath, "r") as fobj:
        raw = fobj.read()

    raw = raw.splitlines()

    header = {
        "ncols": None, "nrows": None, "xllcenter": None, "xllcorner": None,
        "yllcenter": None, "yllcorner": None, "cellsize": None, "nodata_value": None
    }

    # header information
    for line in raw[:6]:
        line = line.split()
        assert len(line)==2
        if line[0].lower() not in header.keys():
            raise KeyError("{} is an illegal header key.".format(line[0]))
        header[line[0].lower()] = line[1]

    assert header["ncols"] is not None, "NCOLS or ncols does not exist in the header"
    assert header["nrows"] is not None, "NROWS or nrows does not exist in the header"
    assert header["cellsize"] is not None, "CELLSIZE or cellsize does not exist in the header"

    header["ncols"] = int(header["ncols"])
    header["nrows"] = int(header["ncols"])
    header["cellsize"] = float(header["cellsize"])

    try:
        header["nodata_value"] = float(header["nodata_value"])
    except TypeError:
        header["nodata_value"] = -9999.

    if (header["xllcenter"] is not None) and (header["yllcenter"] is not None):
        header["xll"] = float(header["xllcenter"])
        header["yll"] = float(header["yllcenter"])
    elif (header["xllcorner"] is not None) and (header["xllcorner"] is not None):
        header["xll"] = float(header["xllcorner"])
        header["yll"] = float(header["yllcorner"])
    else:
        raise KeyError("Missing xllcenter/xllcorner/yllcenter/yllcorner.")

    del header["xllcenter"], header["yllcenter"], header["xllcorner"], header["yllcorner"]

    x = nplike.linspace(
        header["xll"], header["xll"]+header["cellsize"]*(header["ncols"]-1), header["ncols"],
        dtype=nplike.float64)
    y = nplike.linspace(
        header["yll"], header["yll"]+header["cellsize"]*(header["nrows"]-1), header["nrows"],
        dtype=nplike.float64)

    assert nplike.all((x[1:]-x[:-1]) > 0.)
    assert nplike.all((y[1:]-y[:-1]) > 0.)

    data = nplike.zeros((header["nrows"], header["ncols"]), dtype=nplike.float64)

    for i, line in zip(range(header["nrows"]-1, -1, -1), raw[6:]):
        data[i, :] = nplike.fromstring(line, nplike.float64, -1, " ")

    return {"x": x, "y": y, "data": data}, {"data": {"_fill_value": header["nodata_value"]}}

def write(filepath, x, y, data, loc):
    """Write data to a file with Esri ASCII format.

    Note, the input data (with shape (Ny, Nx)) is in the traditional numerical
    convention. That is, data[0, :] is the data in south, and data[-1, :] in
    north. While in the Esri ASCII format, the order of rows folllows the
    raster/image convention, i.e., the first row represents the data in north,
    and the last row represents the data in south.

    Args:
    -----
        stream: a stream; usually an opened file's handle.
        x: a 1D nplike.ndarray; gridline in x direction.
        y: a 1D nplike.ndarray; gridline in y direction.
        data: a 2D nplike.ndarray with shape (Ny, Nx); data.
        loc: indicates whether the gridlines are defined at cell corners
            (vertices) or cell centers; allowed values: "center" or "corner".
        nodata_value: the nodata_value in the Esri ASCII file.

    Returns:
    --------
        N/A.
    """

    filepath = os.path.abspath(filepath)

    with open(filepath, "w") as fobj:
        write_to_stream(fobj, x, y, data, loc)

def write_to_stream(stream, x, y, data, loc, nodata_value=-9999):
    """Write data to a stream with Esri ASCII format.

    Note, the input data (with shape (Ny, Nx)) is in the traditional numerical
    convention. That is, data[0, :] is the data in south, and data[-1, :] in
    north. While in the Esri ASCII format, the order of rows folllows the
    raster/image convention, i.e., the first row represents the data in north,
    and the last row represents the data in south.

    Args:
    -----
        stream: a stream; usually an opened file's handle.
        x: a 1D nplike.ndarray of length Nx; gridline in x direction.
        y: a 1D nplike.ndarray of length Ny; gridline in y direction.
        data: a 2D nplike.ndarray with shape (Ny, Nx); data.
        loc: indicates whether the gridlines are defined at cell corners
            (vertices) or cell centers; allowed values: "center" or "corner".
        nodata_value: the nodata_value in the Esri ASCII file.

    Returns:
    --------
        N/A.
    """

    loc = loc.upper()
    stream.write("{:15s} {}\n".format("NCOLS", x.shape[0]))
    stream.write("{:15s} {}\n".format("NROWS", y.shape[0]))
    stream.write("{:15s} {}\n".format("XLL"+loc, x[0]))
    stream.write("{:15s} {}\n".format("YLL"+loc, y[0]))
    stream.write("{:15s} {} {}\n".format("CELLSIZE", x[1]-x[0], y[1]-y[0]))
    stream.write("{:15s} {}\n".format("NODATA_VALUE", nodata_value))

    for row in data[::-1, :]:
        string = nplike.array2string(row, precision=16, separator=' ', threshold=x.shape[0]+1)
        string = string.lstrip("[ \t").rstrip("] \t").replace("\n", "")
        stream.write(string+"\n")
