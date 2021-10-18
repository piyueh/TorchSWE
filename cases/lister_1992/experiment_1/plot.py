#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Pi-Yueh Chuang <pychuang@pm.me>
#
# Distributed under terms of the BSD 3-Clause license.

"""Post-processing solution.
"""
import pathlib
import yaml
import numpy
from matplotlib import pyplot
from torchswe.utils.netcdf import read as ncread
from torchswe.utils.init import get_timeline


# case path
case = pathlib.Path(__file__).expanduser().resolve().parent

# unified style configuration
pyplot.style.use(case.joinpath("paper.mplstyle"))

# read config
with open(case.joinpath("config.yaml"), "r", encoding="utf-8") as fobj:
    config = yaml.load(fobj, Loader=yaml.Loader)

# read in solution and digital elevation model
sim_data, _ = ncread(case.joinpath("solutions.nc"), ["w"])
dem, _ = ncread(case.joinpath("topo.nc"), [config.topo.key])

# 2D coordinates
dx = (config.spatial.domain[1] - config.spatial.domain[0]) / config.spatial.discretization[0]
dy = (config.spatial.domain[3] - config.spatial.domain[2]) / config.spatial.discretization[1]
x = sim_data["x"]
y = sim_data["y"]
X, Y = numpy.meshgrid(x, y)

# times
times = get_timeline(config.temporal.output, config.temporal.dt).values

# elevation at cell centers
elev = (
    dem["elevation"][:-1, :-1] + dem["elevation"][1:, :-1] +
    dem["elevation"][1:, 1:] + dem["elevation"][:-1, 1:]) / 4.

# get solutions
W = sim_data["w"][...]
H = W - elev
H = numpy.ma.array(H, mask=(H < config.params.drytol))

for h, t in zip(H[1:], times[1:]):
    # read in experimental data
    exp = numpy.loadtxt(
        case.joinpath("experimental_data", f"t={int(t+0.5)}.csv"),
        dtype=float, delimiter=",", skiprows=1
    )

    # print total volume
    sim_vol = numpy.sum(h) * dx * dy
    theo_vol = config.ptsource.rates[0] * t
    err = 100 * abs(sim_vol - theo_vol) / theo_vol if theo_vol != 0. else float("inf")
    print(f"Simulated volume: {sim_vol}; theoretical volume: {theo_vol}; relative err: {err:4.1f}%")

    # plot
    pyplot.figure()
    pyplot.title(f"Flow depth @T={t} sec")
    pyplot.contourf(X, Y, h, 128)
    pyplot.scatter(
        exp[:, 0], exp[:, 1], s=30, c="w", marker="^",
        linewidth=2, edgecolors="k", label="Experiments (Lister, 1992)")
    pyplot.colorbar(label="Depth (m)", orientation="horizontal")
    pyplot.xlabel("x (m)")
    pyplot.ylabel("y (m)")
    pyplot.legend(loc=0)
    pyplot.savefig(case.joinpath(f"depth_t_{int(t+0.5):03d}.png"))
