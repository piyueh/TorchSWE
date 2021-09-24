#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Comparison to the analytical solutions of planar water surface in a paraboloid.
"""
# pylint: disable=invalid-name
import sys
import pathlib
import importlib
from typing import List

import numpy
import matplotlib
from matplotlib import pyplot
from torchswe.utils.netcdf import read as ncread


# paths
case_dir = pathlib.Path(__file__).expanduser().resolve().parent

# unified style configuration
pyplot.style.use(case_dir.joinpath("paper.mplstyle"))

# import the "exact_soln" functions from cases
sys.path.insert(0, str(case_dir))
topo_fun = importlib.import_module("create_data").topo
exact_soln = importlib.import_module("create_data").exact_soln

# read data in
# -------------
sim_data, _ = ncread(case_dir.joinpath("solutions.nc"), ["w", "hu", "hv"])

# some parameters
# ----------------
L = 4.
h0 = 0.1
a = 1.
g = 9.81
eta = 0.5
omega = numpy.sqrt(2.*g*h0)

# topo
# -----
topo = topo_fun(*numpy.meshgrid(sim_data["x"], sim_data["y"]), h0, L, a)

# exact solutions
# ----------------
ext_data = {}
ext_data["w"], ext_data["hu"], ext_data["hv"] = exact_soln(
    *numpy.meshgrid(sim_data["x"], sim_data["y"], sim_data["time"]),
    g, h0, L, a, eta
)
ext_data = {k: v.transpose((2, 0, 1)) for k, v in ext_data.items()}  # fix order

# calculate depth
# ---------------
sim_data["h"] = sim_data["w"] - topo
ext_data["h"] = ext_data["w"] - topo

# errors
# -------
err = {}
for key in ["h", "hu", "hv"]:
    with numpy.errstate(divide='ignore', invalid="ignore"):  # shut warnings
        err[key] = numpy.abs((sim_data[key]-ext_data[key])/ext_data[key])
    err[key] = numpy.where(err[key] == 0., 1e-16, err[key])

# mask out cells of h = 0
# ------------------------
mask = numpy.abs(ext_data["h"]) <= 1e-4
for key in ["h", "hu", "hv"]:
    sim_data[key] = numpy.ma.array(sim_data[key], mask=mask)
    ext_data[key] = numpy.ma.array(ext_data[key], mask=mask)

# plot simulation depth
# --------------------
fig: pyplot.Figure = pyplot.figure(figsize=(13, 8.1))
gs: pyplot.GridSpec = fig.add_gridspec(3, 4, height_ratios=[20, 20, 1])

axs: List[pyplot.Axes] = []
axs.append(fig.add_subplot(gs[0, 0]))  # t = 1/8 period
axs.append(fig.add_subplot(gs[0, 1], sharey=axs[0]))  # t = 2/8 period
axs.append(fig.add_subplot(gs[0, 2], sharey=axs[0]))  # t = 3/8 period
axs.append(fig.add_subplot(gs[0, 3], sharey=axs[0]))  # t = 4/8 period
axs.append(fig.add_subplot(gs[1, 0], sharex=axs[0]))  # t = 5/8 period
axs.append(fig.add_subplot(gs[1, 1], sharex=axs[1], sharey=axs[4]))  # t = 6/8 period
axs.append(fig.add_subplot(gs[1, 2], sharex=axs[2], sharey=axs[4]))  # t = 7/8 period
axs.append(fig.add_subplot(gs[1, 3], sharex=axs[3], sharey=axs[4]))  # t = 8/8 period

cbarax = fig.add_subplot(gs[2, :])

cs: List[matplotlib.contour.QuadContourSet] = [None for _ in range(8)]

lvs = numpy.linspace(0., 0.1, 21)

for i in range(8):
    axs[i].set_title(r"$t={}/{}~T$".format(i+1, 8))
    axs[i].set_aspect("equal", adjustable="box")
    axs[i].set_xlim(0.4, 3.6)
    axs[i].set_ylim(0.4, 3.6)
    cs[i] = axs[i].contourf(sim_data["x"], sim_data["y"], sim_data["h"][i+1], lvs)

for i in range(4):
    pyplot.setp(axs[i].get_xticklabels(), visible=False)
    axs[i+4].set_xlabel("x ($m$)")
    axs[i+4].set_xticks(numpy.linspace(0.5, 3.5, 7))

for i in range(1, 4):
    pyplot.setp(axs[i].get_yticklabels(), visible=False)
    pyplot.setp(axs[i+4].get_yticklabels(), visible=False)

axs[0].set_ylabel("y ($m$)")
axs[4].set_ylabel("y ($m$)")

cbar = fig.colorbar(cs[0], cax=cbarax, ax=axs, ticks=lvs, orientation="horizontal")
cbar.set_label("Water depth ($m$)")

fig.suptitle("Planar water surface in a paraboloid, water depth ($m$)")
fig.savefig(case_dir.joinpath(case_dir.name+"-depth.png"), format="png")

# plot exact depth
# --------------------
fig: pyplot.Figure = pyplot.figure(figsize=(13, 8.1))
gs: pyplot.GridSpec = fig.add_gridspec(3, 4, height_ratios=[20, 20, 1])

axs: List[pyplot.Axes] = []
axs.append(fig.add_subplot(gs[0, 0]))  # t = 1/8 period
axs.append(fig.add_subplot(gs[0, 1], sharey=axs[0]))  # t = 2/8 period
axs.append(fig.add_subplot(gs[0, 2], sharey=axs[0]))  # t = 3/8 period
axs.append(fig.add_subplot(gs[0, 3], sharey=axs[0]))  # t = 4/8 period
axs.append(fig.add_subplot(gs[1, 0], sharex=axs[0]))  # t = 5/8 period
axs.append(fig.add_subplot(gs[1, 1], sharex=axs[1], sharey=axs[4]))  # t = 6/8 period
axs.append(fig.add_subplot(gs[1, 2], sharex=axs[2], sharey=axs[4]))  # t = 7/8 period
axs.append(fig.add_subplot(gs[1, 3], sharex=axs[3], sharey=axs[4]))  # t = 8/8 period

cbarax = fig.add_subplot(gs[2, :])

cs: List[matplotlib.contour.QuadContourSet] = [None for _ in range(8)]

lvs = numpy.linspace(0., 0.1, 21)

for i in range(8):
    axs[i].set_title(r"$t={}/{}~T$".format(i+1, 8))
    axs[i].set_aspect("equal", adjustable="box")
    axs[i].set_xlim(0.4, 3.6)
    axs[i].set_ylim(0.4, 3.6)
    cs[i] = axs[i].contourf(sim_data["x"], sim_data["y"], ext_data["h"][i+1], lvs)

for i in range(4):
    pyplot.setp(axs[i].get_xticklabels(), visible=False)
    axs[i+4].set_xlabel("x ($m$)")
    axs[i+4].set_xticks(numpy.linspace(0.5, 3.5, 7))

for i in range(1, 4):
    pyplot.setp(axs[i].get_yticklabels(), visible=False)
    pyplot.setp(axs[i+4].get_yticklabels(), visible=False)

axs[0].set_ylabel("y ($m$)")
axs[4].set_ylabel("y ($m$)")

cbar = fig.colorbar(cs[0], cax=cbarax, ax=axs, ticks=lvs, orientation="horizontal")
cbar.set_label("Water depth ($m$)")

fig.suptitle("Planar water surface in a paraboloid, water depth ($m$)")
fig.savefig(case_dir.joinpath(case_dir.name+"-exact-depth.png"), format="png")

# plot error of depth
# --------------------
fig: pyplot.Figure = pyplot.figure(figsize=(13, 8.1))
gs: pyplot.GridSpec = fig.add_gridspec(3, 4, height_ratios=[20, 20, 1])

axs: List[pyplot.Axes] = []
axs.append(fig.add_subplot(gs[0, 0]))  # t = 1/8 period
axs.append(fig.add_subplot(gs[0, 1], sharey=axs[0]))  # t = 2/8 period
axs.append(fig.add_subplot(gs[0, 2], sharey=axs[0]))  # t = 3/8 period
axs.append(fig.add_subplot(gs[0, 3], sharey=axs[0]))  # t = 4/8 period
axs.append(fig.add_subplot(gs[1, 0], sharex=axs[0]))  # t = 5/8 period
axs.append(fig.add_subplot(gs[1, 1], sharex=axs[1], sharey=axs[4]))  # t = 6/8 period
axs.append(fig.add_subplot(gs[1, 2], sharex=axs[2], sharey=axs[4]))  # t = 7/8 period
axs.append(fig.add_subplot(gs[1, 3], sharex=axs[3], sharey=axs[4]))  # t = 8/8 period

cbarax = fig.add_subplot(gs[2, :])

cs: List[matplotlib.contour.QuadContourSet] = [None for _ in range(8)]

lvs = 10**numpy.linspace(-6, 0, 7)

for i in range(8):
    axs[i].set_title(r"$t={}/{}~T$".format(i+1, 8))
    axs[i].set_aspect("equal", adjustable="box")
    axs[i].set_xlim(0.4, 3.6)
    axs[i].set_ylim(0.4, 3.6)
    cs[i] = axs[i].contourf(
        sim_data["x"], sim_data["y"], err["h"][i+1], lvs,
        norm=matplotlib.colors.LogNorm(), extend="both"
    )

for i in range(4):
    pyplot.setp(axs[i].get_xticklabels(), visible=False)
    axs[i+4].set_xlabel("x ($m$)")
    axs[i+4].set_xticks(numpy.linspace(0.5, 3.5, 7))

for i in range(1, 4):
    pyplot.setp(axs[i].get_yticklabels(), visible=False)
    pyplot.setp(axs[i+4].get_yticklabels(), visible=False)

axs[0].set_ylabel("y ($m$)")
axs[4].set_ylabel("y ($m$)")

cbar = fig.colorbar(cs[0], cax=cbarax, ax=axs, orientation="horizontal")
cbar.set_label("Relative error")

fig.suptitle("Planar surface in a paraboloid, relative error of depth")
fig.savefig(case_dir.joinpath(case_dir.name+"-error.png"), format="png")
