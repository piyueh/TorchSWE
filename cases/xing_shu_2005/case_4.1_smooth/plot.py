#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Plot.
"""
import pathlib
import yaml
import numpy
from matplotlib import pyplot
from torchswe.utils.netcdf import read as ncread


# paths
case = pathlib.Path(__file__).expanduser().resolve().parent

# unified style configuration
pyplot.style.use(case.joinpath("paper.mplstyle"))

# read case configuration
with open(case.joinpath("config.yaml"), 'r', encoding="utf-8") as f:
    config = yaml.load(f, Loader=yaml.Loader)

# read digital elevation model
dem, _ = ncread(case.joinpath("topo.nc"), [config.topo.key])

# read in solutions
data, _ = ncread(case.joinpath("solutions.nc"), ["w", "hu", "hv"])

# check
assert numpy.allclose(dem["elevation"][1:], numpy.tile(dem["elevation"][0][None, :], (200, 1)))
assert numpy.allclose(data["w"][1:], numpy.tile(data["w"][0][None, :, :], (10, 1, 1)))
assert numpy.allclose(data["hu"][1:], numpy.tile(data["hu"][0][None, :, :], (10, 1, 1)))
assert numpy.allclose(data["hv"][1:], numpy.tile(data["hv"][0][None, :, :], (10, 1, 1)))
assert numpy.allclose(data["w"][0][1:, :], numpy.tile(data["w"][0][0, :][None, :], (199, 1)))
assert numpy.allclose(data["hu"][0][1:, :], numpy.tile(data["hu"][0][0, :][None, :], (199, 1)))
assert numpy.allclose(data["hv"][0][1:, :], numpy.tile(data["hv"][0][0, :][None, :], (199, 1)))

# plot
pyplot.figure()
pyplot.plot(dem["x"], dem["elevation"][0], label="Topography elevation (m)", ls="--", lw=3)
pyplot.plot(data["x"], data["w"][-1][0], label="Flow elevation (m)", ls="-", lw=1.5)
pyplot.title("Flow and topography elevation")
pyplot.xlabel("x (m)")
pyplot.ylabel("Elevation (m)")
pyplot.legend(loc=(0.3, 0.6))
pyplot.savefig(case.joinpath("flow-elevation.png"))
