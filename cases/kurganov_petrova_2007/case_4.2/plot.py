#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Plot topography and solutions for case 4.2 in Kurganov & Petrova (2007).
"""
import pathlib
import numpy
import h5py
import pyvista
from torchswe.utils.config import get_config
from torchswe.utils.misc import DummyDict


# paths
case = pathlib.Path(__file__).expanduser().resolve().parent
case.joinpath("figs").mkdir(exist_ok=True)

# unified style configuration
pyvista.global_theme.load_theme(str(case.joinpath("pyvista_theme.json")))

# read case configuration
config = get_config(case)

# read digital elevation model
dem = DummyDict()
with h5py.File(case.joinpath(config.topo.file), "r") as root:
    dem.x = root[config.topo.xykeys[0]][...]
    dem.y = root[config.topo.xykeys[1]][...]
    dem.x, dem.y = numpy.meshgrid(dem.x, dem.y)
    dem.elevation = root[config.topo.key][...]

# create a structure grid object for PyVista
mesh = pyvista.StructuredGrid(dem["x"].T, dem["y"].T, numpy.zeros_like(dem["x"].T))

# bind elevation data to vertices
mesh.point_data.set_array(dem["elevation"].flatten(), "elevation")

# plot 3D topo
plotter = pyvista.Plotter(off_screen=True, lighting="none")
plotter.add_mesh(mesh.warp_by_scalar("elevation"), "silver")
plotter.add_mesh(
    mesh.contour(32, "elevation").project_points_to_plane([0, 0, 1.4]),
    cmap="viridis", line_width=2,
    show_scalar_bar=True, scalar_bar_args={"title": "Topography\nelevation (m)"},
)
plotter.show_bounds(
    xlabel="x (m)", ylabel="y (m)", zlabel="Elevation (m)", grid=False, location="outer"
)
plotter.camera.zoom(1.)
plotter.add_light(pyvista.Light([-6,  6,  5.], color='white', light_type='scenelight'))
plotter.add_light(pyvista.Light(color='white', light_type='headlight'))
plotter.enable_parallel_projection()
plotter.screenshot(case.joinpath("figs", "topo.png"))

# read in solutions
data = DummyDict()
with h5py.File(case.joinpath("solutions.h5"), "r") as root:
    for k in ["w", "h", "hu", "hv"]:
        data[k] = DummyDict()
        for i in range(5):
            data[k][i] = root[f"{i}/states/{k}"][...]

    data.time = numpy.array([root[f"{i}"].attrs["simulation time"] for i in range(5)], float)

# create masked mesh for flow
wet_mesh = []
for i in range(5):
    wet_mesh.append(mesh.cast_to_unstructured_grid())
    wet_mesh[-1].cell_data.set_array(data["w"][i].flatten(), f"w.t{i}")
    wet_mesh[-1].cell_data.set_array(data["h"][i].flatten(), f"h.t{i}")
    wet_mesh[-1] = wet_mesh[-1].remove_cells(numpy.argwhere(data["h"][i].flatten() <= 5e-5)).ctp()

# plot depth
for i in range(5):
    plotter = pyvista.Plotter(off_screen=True, lighting="none")
    plotter.add_mesh(mesh.warp_by_scalar("elevation"), "silver")
    plotter.add_mesh(
        wet_mesh[i].warp_by_scalar(f"w.t{i}"), color="cyan", specular=0.5, specular_power=15
    )
    plotter.add_mesh(
        wet_mesh[i].contour(32, f"h.t{i}").project_points_to_plane([0, 0, 1.4]),
        cmap="viridis", line_width=2,
        show_scalar_bar=True, scalar_bar_args={"title": "Flow\ndepth (m)"},
    )
    plotter.show_bounds(
        xlabel="x (m)", ylabel="y (m)", zlabel="Elevation (m)", grid=False, location="outer"
    )
    plotter.camera.zoom(1.)
    plotter.add_light(pyvista.Light(color='white', light_type='headlight'))
    plotter.enable_parallel_projection()
    plotter.screenshot(case.joinpath("figs", f"depth-t={i}.png"))
