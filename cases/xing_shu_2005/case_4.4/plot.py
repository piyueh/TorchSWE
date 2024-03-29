#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Plot results.
"""
import pathlib
import h5py
from matplotlib import pyplot
from torchswe.utils.misc import DummyDict
from torchswe.utils.config import get_config


def main():
    """Main function."""

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
        dem.elevation = root[config.topo.key][...].mean(axis=0)

    # read in solutions
    data = DummyDict({i: DummyDict() for i in range(5)})
    with h5py.File(case.joinpath("solutions.h5"), "r") as root:
        data.x = root["grid/x/c"][...]
        for i in range(5):
            data[i].w = root[f"{i}/states/w"][...].mean(axis=0)

    # figure 5 (left) from Xing and Shu, 2005
    pyplot.figure()
    pyplot.plot(dem.x, dem.elevation, 'k-', lw=3, label="bottom")
    pyplot.plot(data.x, data[0].w, 'k--', lw=1, label="initial h+b")
    pyplot.plot(data.x, data[1].w, 'k-', lw=1, label="h+b @T=15s")
    pyplot.xlim(0., 1500.)
    pyplot.xlabel("x")
    pyplot.ylim(0., 22.)
    pyplot.ylabel("surface level h+b, bottom b")
    pyplot.legend(loc=0)
    pyplot.savefig(case.joinpath("figs", "figure_5_left.png"))

    # figure 5 (right) from Xing and Shu, 2005
    pyplot.figure()
    pyplot.plot(data.x, data[1].w, 'ko-', lw=1, ms=2, mew=1, mfc=None, label="h+b @T=15s")
    pyplot.xlim(0., 1500.)
    pyplot.xlabel("x")
    pyplot.ylim(14., 21.)
    pyplot.ylabel("surface level h+b, bottom b")
    pyplot.legend(loc=0)
    pyplot.savefig(case.joinpath("figs", "figure_5_right.png"))

    # figure 6 (left) from Xing and Shu, 2005
    pyplot.figure()
    pyplot.plot(dem.x, dem.elevation, 'k-', lw=3, label="bottom")
    pyplot.plot(data.x, data[0].w, 'k--', lw=1, label="initial h+b")
    pyplot.plot(data.x, data[4].w, 'k-', lw=1, label="h+b @T=60s")
    pyplot.xlim(0., 1500.)
    pyplot.xlabel("x")
    pyplot.ylim(0., 22.)
    pyplot.ylabel("surface level h+b, bottom b")
    pyplot.legend(loc=0)
    pyplot.savefig(case.joinpath("figs", "figure_6_left.png"))

    # figure 6 (right) from Xing and Shu, 2005
    pyplot.figure()
    pyplot.plot(data.x, data[4].w, 'ko-', lw=1, ms=2, mew=1, mfc=None, label="h+b @T=60s")
    pyplot.xlim(0., 1500.)
    pyplot.xlabel("x")
    pyplot.ylim(14., 20.)
    pyplot.ylabel("surface level h+b, bottom b")
    pyplot.legend(loc=0)
    pyplot.savefig(case.joinpath("figs", "figure_6_right.png"))


if __name__ == "__main__":
    import sys
    sys.exit(main())
