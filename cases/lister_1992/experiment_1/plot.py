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
import numpy
import h5py
import pyvista
from matplotlib import pyplot
from torchswe.utils.config import get_config
from torchswe.utils.misc import DummyDict
# pylint: disable=invalid-name


# paths
case = pathlib.Path(__file__).expanduser().resolve().parent
case.joinpath("figs").mkdir(exist_ok=True)

# unified style configuration
pyplot.style.use(case.joinpath("paper.mplstyle"))
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

# read in solutions
soln = DummyDict()
with h5py.File(case.joinpath("solutions.h5"), "r") as root:
    soln.y = root["grid/y/c"][...]
    soln.elevc = root["topo/c"][...]
    for k in ["w", "h"]:
        soln[k] = DummyDict()
        for i in range(7):
            soln[k][i] = root[f"{i}/states/{k}"][...]

    soln.times = numpy.array([root[f"{i}"].attrs["simulation time"] for i in range(7)], float)

# coordinate in flow direction but ON THE INCLINDE PLANE
dxp = 1.2 / config.spatial.discretization[0]
soln.x = numpy.linspace(-0.2+dxp/2., 1.0-dxp/2., config.spatial.discretization[0])

# 2D coordinates ON THE HORIZONTAL PLANE
dx = (config.spatial.domain[1] - config.spatial.domain[0]) / config.spatial.discretization[0]
dy = (config.spatial.domain[3] - config.spatial.domain[2]) / config.spatial.discretization[1]
soln.x, soln.y = numpy.meshgrid(soln.x, soln.y)

# read experimental data
exp = {}
for t in soln.times[1:]:
    exp[t] = numpy.loadtxt(
        case.joinpath("experimental_data", f"t={int(t+0.5)}.csv"),
        dtype=float, delimiter=",", skiprows=1)

# read model prediction data
with numpy.load(case.joinpath("model_prediction", "solution.npz")) as fobj:
    model = fobj["h"]
    modelx = fobj["x"]
    modely = fobj["y"]

# 2d contour
#--------------------------------------------------------------------------------------------------
for i, t in zip(range(1, len(soln.times)), soln.times[1:]):

    # print total volume
    sim_vol = soln.h[i].sum() * dx * dy
    theo_vol = config.ptsource.rates[0] * t
    err = 100 * abs(sim_vol - theo_vol) / theo_vol if theo_vol != 0. else float("inf")
    print(f"Simulated volume: {sim_vol}; theoretical volume: {theo_vol}; relative err: {err:4.1f}%")

    # plot
    label = "Experiments (Lister, 1992)"
    pyplot.figure()
    pyplot.title(f"Flow depth @T={t} sec")
    pyplot.contourf(soln.x, soln.y, numpy.ma.array(soln.h[i], mask=(soln.h[i] <= 1e-4)), 128)
    pyplot.colorbar(label="Depth (m)", orientation="horizontal")
    pyplot.scatter(exp[t][:, 0], exp[t][:, 1], 30, "w", "^", lw=2, ec="k", label=label)
    pyplot.xlabel("x (m)")
    pyplot.xlim(-0.2, 1.0)
    pyplot.ylabel("y (m)")
    pyplot.ylim(-0.3, 0.3)
    pyplot.legend(loc=0)
    pyplot.savefig(case.joinpath("figs", f"depth_t_{int(t+0.5):03d}.png"))

# comparing flow fronts
#--------------------------------------------------------------------------------------------------
fig, ax = pyplot.subplots(1, 1)
fig.suptitle("Silicone on an inclined plane w/ a point source")
lines = []
model_lines = []
scatters = []
markers = iter(["o", "s", "^", "v", "p", "X"])
for i, t in zip(range(1, len(soln.times)), soln.times[1:]):

    # find and plot the front from simulation data
    front = numpy.sum(soln.h[i][soln.h[i].shape[0]//2:, :] > 1e-4, 0)  # upper half
    front = numpy.where(front, soln.y[soln.y.shape[0]//2:, 0][front-1], 0.)
    dry = numpy.cumsum(front > 1e-4)
    bg = numpy.searchsorted(dry, 1, "left") - 1
    ed = numpy.searchsorted(dry, dry.max(), "left") + 2
    dry = numpy.ones(front.shape, dtype=bool)
    dry[bg:ed] = False
    front = numpy.ma.array(front, mask=dry)
    lines.append(ax.plot(soln.x[0, :], front, ls="-", lw=2, c="k", zorder=0)[0])

    # find and plot the front from the model prediction
    front = numpy.sum(model[i-1][model[i-1].shape[0]//2:, :] > 1e-4, 0)  # upper half
    front[front != 0] -= 1
    front = modely[modely.size//2:][front]
    dry = numpy.cumsum(front > 1e-4)  # temporarily borrow the variable name
    bg = numpy.searchsorted(dry, 1, "left") - 1
    ed = numpy.searchsorted(dry, dry.max(), "left") + 2
    dry = numpy.ones(front.shape, dtype=bool)
    dry[bg:ed] = False
    front = numpy.ma.array(front, mask=dry)
    model_lines.append(ax.plot(modelx[:], front, ls="--", lw=1, c="k", zorder=0)[0])

    # plot experimental data
    scatters.append(ax.scatter(exp[t][:, 0], exp[t][:, 1], 30, "w", next(markers), edgecolor="k"))

ax.set_title(r"$Q=1.48e^{-6}m^3/s$, $\theta=2.5\degree$, $\nu=1.13e^{-3} m^2/s$", fontsize=10)
ax.set_xlabel("x (m)")
ax.set_xlim(-0.2, 1.0)
ax.set_ylabel("y (m)")
ax.set_ylim(-0.01, 0.33)

leg1 = ax.legend(
    scatters, [f"t={int(ti+0.5)}" for ti in soln.times[1:]],
    title="Experiments (Lister, 1992)", title_fontsize=10,
    ncol=3, fontsize=8, bbox_to_anchor=(0.01, 0.99), loc="upper left", borderaxespad=0.,
    frameon=True, labelspacing=0.3
)
ax.add_artist(leg1)

leg2 = ax.legend(
    [model_lines[-1], lines[-1]], ["Model prediction (Lister, 1992)", "TorchSWE simulation"],
    ncol=1, fontsize=10, bbox_to_anchor=(0.99, 0.99), loc="upper right", borderaxespad=0.,
    frameon=True
)
ax.add_artist(leg2)

pyplot.savefig(case.joinpath("figs", "front_comparison.png"), bbox_inches='tight', dpi=166)

# 3D plots
#--------------------------------------------------------------------------------------------------

# create a structure grid object for PyVista
mesh = pyvista.StructuredGrid(dem.x.T, dem.y.T, dem["elevation"].T)

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
plotter.screenshot(case.joinpath("figs", "topo_3d.png"))

# plot 3D depth
for i, t in zip(range(1, len(soln.times)), soln.times[1:]):
    soln_mesh = mesh.cast_to_unstructured_grid()
    soln_mesh.cell_data.set_array(soln.w[i].flatten(), "w")
    soln_mesh.cell_data.set_array(soln.h[i].flatten(), "h")
    soln_mesh = soln_mesh.remove_cells(numpy.argwhere(soln.h[i].flatten() <= 1e-4)).ctp()

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
    plotter.screenshot(case.joinpath("figs", f"depth_t_{int(t+0.5):03d}_3d.png"))
