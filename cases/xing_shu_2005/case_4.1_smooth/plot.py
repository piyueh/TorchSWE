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
import numpy
import h5py
from matplotlib import pyplot
from torchswe.utils.misc import DummyDict
from torchswe.utils.config import get_config

# read simulation data
case = pathlib.Path(__file__).expanduser().resolve().parent
case.joinpath("figs").mkdir(exist_ok=True)
config = get_config(case)

# unified style configuration
pyplot.style.use(case.joinpath("paper.mplstyle"))

# read in solutions
data = DummyDict()
with h5py.File(case.joinpath("solutions.h5"), "r") as root:
    data.x = root["grid/x/c"][...]
    data.w = root["10/states/w"][...]
    data.h = root["10/states/h"][...]
    data.hu = root["10/states/hu"][...]
    data.hv = root["10/states/hv"][...]

# read digital elevation model
dem = DummyDict()
with h5py.File(case.joinpath(config.topo.file), "r") as root:
    dem.x = root[config.topo.xykeys[0]][...]
    dem.y = root[config.topo.xykeys[0]][...]
    dem.elevation = root[config.topo.key][...]

# make sure y direction does not have variance
assert numpy.allclose(data.w, data.w[0].reshape(1, -1))
assert numpy.allclose(data.h, data.h[0].reshape(1, -1))
assert numpy.allclose(data.hu, data.hu[0].reshape(1, -1))
assert numpy.allclose(data.hv, data.hv[0].reshape(1, -1))

# plot
pyplot.figure()
pyplot.plot(dem["x"], dem["elevation"][0], label="Topography elevation (m)", ls="--", lw=3)
pyplot.plot(data["x"], data["w"][0], label="Flow elevation (m)", ls="-", lw=1.5)
pyplot.title("Flow and topography elevation")
pyplot.xlabel("x (m)")
pyplot.ylabel("Elevation (m)")
pyplot.legend(loc=(0.3, 0.6))
pyplot.savefig(case.joinpath("figs", "flow-elevation.png"))
