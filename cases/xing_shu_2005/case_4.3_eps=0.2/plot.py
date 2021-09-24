#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Pi-Yueh Chuang <pychuang@pm.me>
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
assert numpy.allclose(dem["elevation"][1:], numpy.tile(dem["elevation"][0][None, :], (100, 1)))
for i in range(21):
    assert numpy.allclose(data["w"][i][1:, :], numpy.tile(data["w"][i][0, :][None, :], (99, 1)))
    assert numpy.allclose(data["hu"][i][1:, :], numpy.tile(data["hu"][i][0, :][None, :], (99, 1)))
    assert numpy.allclose(data["hv"][i][1:, :], numpy.tile(data["hv"][i][0, :][None, :], (99, 1)))

# plot, T=0, w
i=0
pyplot.figure()
pyplot.plot(dem["x"], dem["elevation"][0], label="Topography elevation (m)", ls="--", lw=3)
pyplot.plot(data["x"], data["w"][i][0], label="Flow elevation (m)", ls="-", lw=1.5)
pyplot.title(f"Flow and topography elevation, T={i*0.01}s")
pyplot.xlabel("x (m)")
pyplot.ylabel("Elevation (m)")
pyplot.legend(loc=(0.3, 0.6))
pyplot.savefig(case.joinpath("flow-elevation-t=0.png"))

# plot, t=0.2, w
i=20
pyplot.figure()
pyplot.plot(data["x"], data["w"][i][0], label="Flow elevation (m)", ls="-", lw=1.5)
pyplot.title(f"Flow elevation, T={i*0.01}s")
pyplot.xlabel("x (m)")
pyplot.ylabel("Elevation (m)")
pyplot.ylim(0.8, 1.2)
pyplot.savefig(case.joinpath("flow-elevation-t=0.2.png"))

# plot, t=0.2, hu
i=20
pyplot.figure()
pyplot.plot(data["x"], data["hu"][i][0], label="Discharge, {} (m)".format(r"$hu$"), ls="-", lw=1.5)
pyplot.title("Flow discharge, {}, T={}s".format(r"$hu$", i*0.01))
pyplot.xlabel("x (m)")
pyplot.ylabel("Discharge, {} (m)".format(r"$hu$"))
pyplot.ylim(-0.5, 0.5)
pyplot.savefig(case.joinpath("flow-discharge-t=0.2.png"))
