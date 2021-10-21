#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Plot
"""
import pathlib
import numpy
from matplotlib import pyplot


# unified style configuration
case = pathlib.Path(__file__).expanduser().resolve().parent
pyplot.style.use(case.parent.joinpath("paper.mplstyle"))

# read solution data
with numpy.load(case.joinpath("solution.npz")) as data:
    hs = data["h"]

# misc
dx = 1.2 / hs.shape[2]
dy = 0.6 / hs.shape[1]
x = numpy.linspace(-0.2+dx/2., 1.0-dx/2., hs.shape[2])
y = numpy.linspace(-0.3+dy/2., 0.3-dy/2., hs.shape[1])
x, y = numpy.meshgrid(x, y)
times = [32., 59., 122., 271., 486., 727.]

# 2D contourf
for h, t in zip(hs, times):
    # read in experimental data
    exp = numpy.loadtxt(
        case.parent.joinpath("experimental_data", f"t={int(t+0.5)}.csv"),
        dtype=float, delimiter=",", skiprows=1
    )

    # print total volume
    sim_vol = numpy.sum(h) * dx * dy
    theo_vol = 1.48e-6 * t
    err = 100 * abs(sim_vol - theo_vol) / theo_vol if theo_vol != 0. else float("inf")
    print(f"Simulated volume: {sim_vol}; theoretical volume: {theo_vol}; relative err: {err:4.1f}%")

    # plot
    pyplot.figure()
    pyplot.title(f"Flow depth (Lister's model prediction)@T={t} sec")
    pyplot.contourf(x, y, numpy.ma.array(h, mask=(h <= 1e-14)), 128)
    pyplot.colorbar(label="Depth (m)", orientation="horizontal")
    pyplot.scatter(
        exp[:, 0], exp[:, 1], s=30, c="w", marker="^",
        linewidth=2, edgecolors="k", label="Experiments (Lister, 1992)")
    pyplot.xlabel("x (m)")
    pyplot.ylabel("y (m)")
    pyplot.legend(loc=0)
    pyplot.savefig(case.joinpath(f"depth_t_{int(t+0.5):03d}.png"))

# comparing fronts
fig, ax = pyplot.subplots(1, 1)
fig.suptitle("Silicone on an inclined plane w/ a point source")
lines = []
scatters = []
markers = iter(["o", "s", "^", "v", "p", "X"])
for h, t in zip(hs, times):

    # find the front from the model prediction
    front = numpy.sum(h[h.shape[0]//2:, :] > 1e-14, 0)  # upper half
    front[front != 0] -= 1
    front = y[y.shape[0]//2:, 0][front]
    dry = numpy.cumsum(front > 1e-14)  # temporarily borrow this variable for counting wet cells
    bg = numpy.searchsorted(dry, 1, "left") - 1
    ed = numpy.searchsorted(dry, dry.max(), "left") + 2
    dry = numpy.ones(front.shape, dtype=bool)
    dry[bg:ed] = False
    front = numpy.ma.array(front, mask=dry)

    # plot line
    lines.append(ax.plot(x[0, :], front, ls="-", lw=1, c="k", zorder=0)[0])

    # read in experimental data
    exp = numpy.loadtxt(
        case.parent.joinpath("experimental_data", f"t={int(t+0.5)}.csv"),
        dtype=float, delimiter=",", skiprows=1
    )

    # plot scatters
    scatters.append(ax.scatter(exp[:, 0], exp[:, 1], 30, "w", next(markers), edgecolor="k"))

ax.set_title(r"$Q=1.48e^{-6}m^3/s$, $\theta=2.5\degree$, $\nu=1.13e^{-3} m^2/s$", fontsize=10)
ax.set_xlabel("x (m)")
ax.set_xlim(-0.2, 1.0)
ax.set_ylabel("y (m)")
ax.set_ylim(-0.01, 0.33)

leg1 = ax.legend(
    scatters, [f"t={int(ti+0.5)}" for ti in times],
    title="Experiments (Lister, 1992)", title_fontsize=10,
    ncol=3, fontsize=8, bbox_to_anchor=(0.01, 0.99), loc="upper left", borderaxespad=0.,
    frameon=True, labelspacing=0.3
)
ax.add_artist(leg1)

leg2 = ax.legend(
    [lines[-1], lines[-1]], ["Model prediction (Lister, 1992)", "TorchSWE simulation"],
    ncol=1, fontsize=10, bbox_to_anchor=(0.99, 0.99), loc="upper right", borderaxespad=0.,
    frameon=True
)
ax.add_artist(leg2)

pyplot.savefig(case.joinpath("front_comparison.png"), bbox_inches='tight', dpi=166)
