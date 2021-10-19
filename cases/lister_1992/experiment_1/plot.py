#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Post-processing solution.
"""
import pathlib
import yaml
import numpy
import pyvista
from matplotlib import pyplot
from torchswe.utils.netcdf import read as ncread
from torchswe.utils.init import get_timeline


# case path
case = pathlib.Path(__file__).expanduser().resolve().parent

# unified style configuration
pyplot.style.use(case.joinpath("paper.mplstyle"))
pyvista.global_theme.load_theme(str(case.joinpath("pyvista_theme.json")))

# read config
with open(case.joinpath("config.yaml"), "r", encoding="utf-8") as fobj:
    config = yaml.load(fobj, Loader=yaml.Loader)

# read in solution and digital elevation model
sim_data, _ = ncread(case.joinpath("solutions.nc"), ["w"])
dem, _ = ncread(case.joinpath("topo.nc"), [config.topo.key])

# 2D coordinates
dx = (config.spatial.domain[1] - config.spatial.domain[0]) / config.spatial.discretization[0]
dy = (config.spatial.domain[3] - config.spatial.domain[2]) / config.spatial.discretization[1]
soln_X, soln_Y = numpy.meshgrid(sim_data["x"], sim_data["y"])
dem_X, dem_Y = numpy.meshgrid(dem["x"], dem["y"])

# times
times = get_timeline(config.temporal.output, config.temporal.dt).values

# elevation at cell centers
elev = (
    dem["elevation"][:-1, :-1] + dem["elevation"][1:, :-1] +
    dem["elevation"][1:, 1:] + dem["elevation"][:-1, 1:]) / 4.

# get solutions
W = sim_data["w"][...]
H = W - elev

# 2D plots
#--------------------------------------------------------------------------------------------------
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
    pyplot.contourf(soln_X, soln_Y, numpy.ma.array(h, mask=(h < config.params.drytol)), 128)
    pyplot.scatter(
        exp[:, 0], exp[:, 1], s=30, c="w", marker="^",
        linewidth=2, edgecolors="k", label="Experiments (Lister, 1992)")
    pyplot.colorbar(label="Depth (m)", orientation="horizontal")
    pyplot.xlabel("x (m)")
    pyplot.ylabel("y (m)")
    pyplot.legend(loc=0)
    pyplot.savefig(case.joinpath(f"depth_t_{int(t+0.5):03d}.png"))

# 3D plots
#--------------------------------------------------------------------------------------------------

# create a structure grid object for PyVista
mesh = pyvista.StructuredGrid(dem_X.T, dem_Y.T, dem["elevation"].T)

# bind elevation data to vertices
mesh.point_data.set_array(dem["elevation"].flatten(), "elevation")

# plot 3D topo
plotter = pyvista.Plotter(off_screen=True, lighting="none")
plotter.add_mesh(mesh, "silver")
plotter.add_mesh(mesh.contour(32, "elevation"), cmap="viridis")
plotter.show_bounds(show_zaxis=False, xlabel="x (m)", ylabel="y (m)", grid=False, location="outer")
plotter.camera.zoom(1.)
plotter.add_light(pyvista.Light([-6,  6,  5.], color='white', light_type='scenelight'))
plotter.add_light(pyvista.Light(color='white', light_type='headlight'))
plotter.enable_parallel_projection()
plotter.screenshot(case.joinpath("topo_3d.png"))

# plot 3D depth
for w, h, t in zip(W[1:], H[1:], times[1:]):
    soln_mesh = mesh.cast_to_unstructured_grid()
    soln_mesh.cell_data.set_array(w.flatten(), "w")
    soln_mesh.cell_data.set_array(h.flatten(), "h")
    soln_mesh = soln_mesh.remove_cells(numpy.argwhere(h.flatten() < config.params.drytol)).ctp()

    plotter = pyvista.Plotter(off_screen=True, lighting="none")
    plotter.add_mesh(mesh, "silver")
    plotter.add_mesh(soln_mesh.warp_by_scalar("h"), color="003166", specular=0.5, specular_power=15)
    plotter.add_mesh(
        soln_mesh.contour(16, "h").warp_by_scalar(),
        color="w", line_width=1,
        show_scalar_bar=True, scalar_bar_args={"title": "Flow\ndepth (m)"},
    )
    plotter.show_bounds(show_zaxis=False, xlabel="x (m)", ylabel="y (m)", location="outer")
    plotter.camera.zoom(1.5)
    plotter.add_light(pyvista.Light([-6,  6,  5.], color='white', light_type='scenelight'))
    plotter.add_light(pyvista.Light(color="white", light_type="headlight"))
    plotter.enable_parallel_projection()
    plotter.screenshot(case.joinpath(f"depth_t_{int(t+0.5):03d}_3d.png"))
