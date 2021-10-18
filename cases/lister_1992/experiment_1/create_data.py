#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Create topography for an inclined plate with an angle of 2.5 degree..
"""
import pathlib
import yaml
import numpy
from torchswe.utils.netcdf import write


def main():
    """Main function"""

    case = pathlib.Path(__file__).expanduser().resolve().parent

    with open(case.joinpath("config.yaml"), 'r', encoding="utf-8") as fobj:
        config = yaml.load(fobj, Loader=yaml.Loader)

    # gridlines
    spatial = config.spatial
    x = numpy.linspace(*spatial.domain[:2], spatial.discretization[0]+1, dtype=config.params.dtype)
    y = numpy.linspace(*spatial.domain[2:], spatial.discretization[1]+1, dtype=config.params.dtype)

    # elevation
    elev = numpy.tile(numpy.abs(x-spatial.domain[1])*numpy.tan(2.5*numpy.pi/180.), (y.size, 1))

    # write topography file
    write(case.joinpath(config.topo.file), (x, y), {"elevation": elev})

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
