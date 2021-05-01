#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""
Create topography and I.C. file for case 5.3 in Xing and Shu (2005).

Note, the elevation data in the resulting NetCDF file is defined at vertices,
instead of cell centers. But the I.C. is defined at cell centers.
"""
import os
import numpy

def main():
    """Main function"""

    # it's users' responsibility to make sure TorchSWE package can be found
    from TorchSWE.utils.esri import write_esri_ascii

    case = os.path.dirname(os.path.abspath(__file__))

    Ncols = 2000
    Nrows = 1000
    xbg, xed = 0., 2.
    ybg, yed = 0., 1.

    x = numpy.linspace(xbg, xed, Ncols+1, dtype=numpy.float64)
    y = numpy.linspace(ybg, yed, Nrows+1, dtype=numpy.float64)

    # 2D X, Y for temporarily use
    X, Y = numpy.meshgrid(x, y)

    # topogeaphy elevation
    B = 0.8 * numpy.exp(-5.*numpy.power(X-0.9, 2)-50.*numpy.power(Y-0.5, 2))

    # write topography file
    write_esri_ascii(os.path.join(case, "topo.asc"), x, y, B, "corner")

if __name__ == "__main__":
    import sys

    # when execute this script directly, make sure TorchSWE can be found
    pkg_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
    sys.path.append(pkg_path)

    # execute the main function
    main()
