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


# paths
case = pathlib.Path(__file__).expanduser().resolve().parent
case.joinpath("figs").mkdir(exist_ok=True)
config = get_config(case)

# unified style configuration
pyplot.style.use(case.joinpath("paper.mplstyle"))

# read digital elevation model
dem = DummyDict()
with h5py.File(case.joinpath(config.topo.file), "r") as root:
    dem.x = root[config.topo.xykeys[0]][...]
    dem.y = root[config.topo.xykeys[0]][...]
    dem.elevation = root[config.topo.key][...]

# read in solutions
data = DummyDict()
with h5py.File(case.joinpath("solutions.h5"), "r") as root:
    data.x = root["grid/x/c"][...]
    data.w0 = root["0/states/w"][...]
    data.w20 = root["20/states/w"][...]
    data.hu20 = root["20/states/hu"][...]

# check
assert numpy.allclose(data.w0, data.w0[0].reshape(1, -1), 1e-15, 1e-15)
assert numpy.allclose(data.w20, data.w20[0].reshape(1, -1), 1e-15, 1e-15)
assert numpy.allclose(data.hu20, data.hu20[0].reshape(1, -1), 1e-15, 1e-15)

# plot, T=0, w
i=0
pyplot.figure()
pyplot.plot(dem["x"], dem["elevation"][0], label="Topography elevation (m)", ls="--", lw=3)
pyplot.plot(data["x"], data["w0"][0], label="Flow elevation (m)", ls="-", lw=1.5)
pyplot.title(f"Flow and topography elevation, T={i*0.01}s")
pyplot.xlabel("x (m)")
pyplot.ylabel("Elevation (m)")
pyplot.legend(loc=(0.3, 0.6))
pyplot.savefig(case.joinpath("figs", "flow-elevation-t=0.png"))

# plot, t=0.2, w
i=20
pyplot.figure()
pyplot.plot(data["x"], data["w20"][0], label="Flow elevation (m)", ls="-", lw=1.5)
pyplot.title(f"Flow elevation, T={i*0.01}s")
pyplot.xlabel("x (m)")
pyplot.ylabel("Elevation (m)")
pyplot.ylim(0.8, 1.2)
pyplot.savefig(case.joinpath("figs", "flow-elevation-t=0.2.png"))

# plot, t=0.2, hu
i=20
pyplot.figure()
pyplot.plot(data["x"], data["hu20"][0], label=r"Discharge, $hu$ (m)", ls="-", lw=1.5)
pyplot.title(rf"Flow discharge, $hu$, T={i*0.01}s")
pyplot.xlabel("x (m)")
pyplot.ylabel(r"Discharge, $hu$ (m)")
pyplot.ylim(-0.5, 0.5)
pyplot.savefig(case.joinpath("figs", "flow-discharge-t=0.2.png"))
