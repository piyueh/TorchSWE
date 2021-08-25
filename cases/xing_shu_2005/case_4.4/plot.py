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
from matplotlib import pyplot
from torchswe.utils.netcdf import read as ncread


def main():
    """Main function."""

    dirpath = pathlib.Path(__file__).expanduser().resolve().parent
    data, _ = ncread(dirpath.joinpath("solutions.nc"), ["w", "hu", "hv"])
    elev, _ = ncread(dirpath.joinpath("topo.nc"), ["elevation"])

    # use the average
    data.w = data.w.mean(axis=1)
    elev.elevation = elev.elevation.mean(axis=0)

    # figure 5 (left) from Xing and Shu, 2005
    pyplot.figure()
    pyplot.plot(elev.x, elev.elevation, 'k-', lw=3, label="bottom")
    pyplot.plot(data.x, data.w[0], 'k--', lw=1, label="initial h+b")
    pyplot.plot(data.x, data.w[1], 'k-', lw=1, label="h+b @T=15s")
    pyplot.xlim(0., 1500.)
    pyplot.xlabel("x")
    pyplot.ylim(0., 22.)
    pyplot.ylabel("surface level h+b, bottom b")
    pyplot.legend(loc=0)
    pyplot.savefig(dirpath.joinpath("figure_5_left.png"))

    # figure 5 (right) from Xing and Shu, 2005
    pyplot.figure()
    pyplot.plot(data.x, data.w[1], 'ko-', lw=1, ms=2, mew=1, mfc=None, label="h+b @T=15s")
    pyplot.xlim(0., 1500.)
    pyplot.xlabel("x")
    pyplot.ylim(14., 21.)
    pyplot.ylabel("surface level h+b, bottom b")
    pyplot.legend(loc=0)
    pyplot.savefig(dirpath.joinpath("figure_5_right.png"))

    # figure 6 (left) from Xing and Shu, 2005
    pyplot.figure()
    pyplot.plot(elev.x, elev.elevation, 'k-', lw=3, label="bottom")
    pyplot.plot(data.x, data.w[0], 'k--', lw=1, label="initial h+b")
    pyplot.plot(data.x, data.w[-1], 'k-', lw=1, label="h+b @T=60s")
    pyplot.xlim(0., 1500.)
    pyplot.xlabel("x")
    pyplot.ylim(0., 22.)
    pyplot.ylabel("surface level h+b, bottom b")
    pyplot.legend(loc=0)
    pyplot.savefig(dirpath.joinpath("figure_6_left.png"))

    # figure 6 (right) from Xing and Shu, 2005
    pyplot.figure()
    pyplot.plot(data.x, data.w[-1], 'ko-', lw=1, ms=2, mew=1, mfc=None, label="h+b @T=60s")
    pyplot.xlim(0., 1500.)
    pyplot.xlabel("x")
    pyplot.ylim(14., 20.)
    pyplot.ylabel("surface level h+b, bottom b")
    pyplot.legend(loc=0)
    pyplot.savefig(dirpath.joinpath("figure_6_right.png"))


if __name__ == "__main__":
    import sys
    sys.exit(main())
