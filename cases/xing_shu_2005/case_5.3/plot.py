#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Plot.
"""
import pathlib
import numpy
import h5py
from matplotlib import pyplot
from torchswe.utils.misc import DummyDict


def main():
    """Plot and compare to analytical solutions."""
    # pylint: disable=invalid-name

    # paths
    case = pathlib.Path(__file__).expanduser().resolve().parent
    case.joinpath("figs").mkdir(exist_ok=True)

    # unified style configuration
    pyplot.style.use(case.joinpath("paper.mplstyle"))

    # read in solutions
    data = DummyDict({i: DummyDict() for i in range(5)})
    with h5py.File(case.joinpath("solutions.h5"), "r") as root:
        data.x, data.y = numpy.meshgrid(root["grid/x/c"][...], root["grid/y/c"][...])
        for i in range(5):
            data[i].w = root[f"{i+1}/states/w"][...]  # skip the soln@T=0

    # time labels
    t = [0.12, 0.24, 0.36, 0.48, 0.6]

    # contour line range
    n = 32
    r = [
        numpy.linspace(0.999703, 1.00629, n),
        numpy.linspace(0.994836, 1.01604, n),
        numpy.linspace(0.988582, 1.0117, n),
        numpy.linspace(0.990344, 1.00497, n),
        numpy.linspace(0.995065, 1.0056, n)]

    # contour lines: to compare with Xing & Shu
    for i in range(5):
        pyplot.figure(figsize=(10, 4), dpi=166)
        pyplot.contour(data.x, data.y, data[i].w, r[i], linewidths=1)
        pyplot.title(f"Xing & Shu (2005) case 5.3: water level @ T={t[i]} sec")
        pyplot.xlabel("x (m)")
        pyplot.ylabel("y (m)")
        pyplot.xlim(0., 2.)
        pyplot.ylim(0., 1.)
        pyplot.colorbar()
        pyplot.savefig(
            case.joinpath("figs", f"water_level_contourline_t={t[i]}.png"),
            dpi=166, bbox_inches="tight")

    # contourf
    for i in range(5):
        pyplot.figure(figsize=(10, 4), dpi=166)
        pyplot.contourf(data.x, data.y, data[i].w, 128)
        pyplot.title(f"Xing & Shu (2005) case 5.3: water level @ T={t[i]} sec")
        pyplot.xlabel("x (m)")
        pyplot.ylabel("y (m)")
        pyplot.xlim(0., 2.)
        pyplot.ylim(0., 1.)
        pyplot.colorbar()
        pyplot.savefig(
            case.joinpath("figs", f"water_level_contour_t={t[i]}.png"),
            dpi=166, bbox_inches="tight")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
