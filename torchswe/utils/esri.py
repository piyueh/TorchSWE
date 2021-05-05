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

def read_esri_ascii(filepath):
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

        attrs: a mimic to the output of read_cf. The only output is a dictionary:
            {"data": {"_fill_value": nodata_value}}.
    """

    filepath = os.path.abspath(filepath)

    with open(filepath, "r") as f:
        raw = f.read()

    raw = raw.splitlines()

    H = {
        "ncols": None, "nrows": None, "xllcenter": None, "xllcorner": None,
        "yllcenter": None, "yllcorner": None, "cellsize": None, "nodata_value": None
    }

    # header information
    for line in raw[:6]:
        line = line.split()
        assert len(line)==2
        if line[0].lower() not in H.keys():
            raise KeyError("{} is an illegal header key.".format(line[0]))
        H[line[0].lower()] = line[1]

    assert H["ncols"] is not None, "NCOLS or ncols does not exist in the header"
    assert H["nrows"] is not None, "NROWS or nrows does not exist in the header"
    assert H["cellsize"] is not None, "CELLSIZE or cellsize does not exist in the header"

    H["ncols"] = int(H["ncols"])
    H["nrows"] = int(H["ncols"])
    H["cellsize"] = float(H["cellsize"])

    try:
        H["nodata_value"] = float(H["nodata_value"])
    except TypeError:
        H["nodata_value"] = -9999.

    if (H["xllcenter"] is not None) and (H["yllcenter"] is not None):
        H["xll"] = float(H["xllcenter"])
        H["yll"] = float(H["yllcenter"])
    elif (H["xllcorner"] is not None) and (H["xllcorner"] is not None):
        H["xll"] = float(H["xllcorner"])
        H["yll"] = float(H["yllcorner"])
    else:
        raise KeyError("Missing xllcenter/xllcorner/yllcenter/yllcorner.")

    del H["xllcenter"], H["yllcenter"], H["xllcorner"], H["yllcorner"]

    x = nplike.linspace(
        H["xll"], H["xll"]+H["cellsize"]*(H["ncols"]-1), H["ncols"], dtype=nplike.float64)
    y = nplike.linspace(
        H["yll"], H["yll"]+H["cellsize"]*(H["nrows"]-1), H["nrows"], dtype=nplike.float64)

    data = nplike.zeros((H["nrows"], H["ncols"]), dtype=nplike.float64)

    for i, line in zip(range(H["nrows"]-1, -1, -1), raw[6:]):
        data[i, :] = nplike.fromstring(line, nplike.float64, -1, " ")

    return {"x": x, "y": y, "data": data}, {"data": {"_fill_value": H["nodata_value"]}}

def write_esri_ascii(filepath, x, y, data, loc):
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

    with open(filepath, "w") as f:
        write_esri_ascii_stream(f, x, y, data, loc)

def write_esri_ascii_stream(stream, x, y, data, loc, nodata_value=-9999):
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

    H = {
        "ncols": None, "nrows": None, "xllcenter": None, "xllcorner": None,
        "yllcenter": None, "yllcorner": None, "cellsize": None, "nodata_value": None
    }

    loc = loc.upper()
    stream.write("{:15s} {}\n".format("NCOLS", x.shape[0]))
    stream.write("{:15s} {}\n".format("NROWS", y.shape[0]))
    stream.write("{:15s} {}\n".format("XLL"+loc, x[0]))
    stream.write("{:15s} {}\n".format("YLL"+loc, y[0]))
    stream.write("{:15s} {} {}\n".format("CELLSIZE", x[1]-x[0], y[1]-y[0]))
    stream.write("{:15s} {}\n".format("NODATA_VALUE", nodata_value))

    for row in data[::-1, :]:
        s = nplike.array2string(row, precision=16, separator=' ', threshold=x.shape[0]+1)
        s = s.lstrip("[ \t").rstrip("] \t").replace("\n", "")
        stream.write(s+"\n")
