#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Test functions in torchswe.utils.init with only one MPI process.
"""
import pytest
from mpi4py import MPI
from torchswe import nplike
from torchswe.utils.init import get_process
from torchswe.utils.init import get_gridline
from torchswe.utils.init import get_timeline
from torchswe.utils.init import get_domain
from torchswe.utils.init import get_topography
from torchswe.utils.init import get_empty_whuhvmodel
from torchswe.utils.init import get_empty_huvmodel
from torchswe.utils.init import get_empty_faceonesidemodel
from torchswe.utils.init import get_empty_facetwosidemodel
from torchswe.utils.init import get_empty_facequantitymodel
from torchswe.utils.init import get_empty_slopes
from torchswe.utils.init import get_empty_states


def test_get_process():
    """Test `get_process(...)`."""

    # serial case
    proc = get_process(MPI.COMM_WORLD, 100, 75)
    assert proc.comm == MPI.COMM_WORLD
    assert proc.pnx == 1
    assert proc.pny == 1
    assert proc.pi == 0
    assert proc.pj == 0
    assert proc.proc_shape == (1, 1)
    assert proc.proc_loc == (0, 0)
    assert proc.west is None
    assert proc.east is None
    assert proc.south is None
    assert proc.north is None


def test_get_gridline():
    """Test `get_gridline(...)`."""
    gridlines = [None, None, None, None]
    gridlines[0] = get_gridline("x", 4, 0, 100, -1.2, 11.3, "float64")
    gridlines[1] = get_gridline("x", 4, 1, 100, -1.2, 11.3, "float64")
    gridlines[2] = get_gridline("x", 4, 2, 100, -1.2, 11.3, "float64")
    gridlines[3] = get_gridline("x", 4, 3, 100, -1.2, 11.3, "float64")

    assert gridlines[0].vertices[0] == -1.2
    assert gridlines[0].vertices[-1] == gridlines[1].vertices[0]
    assert gridlines[1].vertices[-1] == gridlines[2].vertices[0]
    assert gridlines[2].vertices[-1] == gridlines[3].vertices[0]
    assert gridlines[3].vertices[-1] == 11.3


def test_get_timeline():
    """Test `get_timeline(...)`."""

    timeline = get_timeline("at", nplike.linspace(0., 2., 5).tolist())
    assert nplike.allclose(timeline[:], (0., 0.5, 1., 1.5, 2.))
    assert timeline.save

    timeline = get_timeline("t_start every_seconds multiple", (-1.2, 3.3, 4))
    assert nplike.allclose(timeline[:], (-1.2, 2.1, 5.4, 8.7, 12.0))
    assert timeline.save

    timeline = get_timeline("t_start every_steps multiple", (-1.2, 10, 4), 0.33)
    assert nplike.allclose(timeline[:], (-1.2, 2.1, 5.4, 8.7, 12.0))
    assert timeline.save

    timeline = get_timeline("t_start t_end n_saves", (-2., 11.2, 4))
    assert nplike.allclose(timeline[:], (-2., 1.3, 4.6, 7.9, 11.2))
    assert timeline.save

    timeline = get_timeline("t_start t_end no save", (3.6, 7.9))
    assert nplike.allclose(timeline[:], (3.6, 7.9))
    assert not timeline.save

    timeline = get_timeline("t_start n_steps no save", (2.1, 11), 2.7)
    assert nplike.allclose(timeline[:], (2.1, 31.8))
    assert not timeline.save

    with pytest.raises(AssertionError, match="dt must be provided"):
        timeline = get_timeline("t_start every_steps multiple", (-1.2, 10, 4))

    with pytest.raises(AssertionError, match="dt must be provided"):
        timeline = get_timeline("t_start n_steps no save", (-1.2, 10))


def test_get_topography():
    """Test `get_topography`."""

    process = get_process(MPI.COMM_WORLD, 50, 50)
    x = get_gridline("x", process.pnx, process.pi, 50, 0., 5., "float64")
    y = get_gridline("y", process.pny, process.pj, 50, 0., 10., "float64")
    domain = get_domain(process, x, y)

    demx = nplike.linspace(0., 5., 51)
    demy = nplike.linspace(0., 10., 51)
    elev = (25. - (demy - 5.)**2)[:, None] * (6.25 - (demx - 2.5)**2)[None, :]

    topo = get_topography(domain, elev, demx, demy)
    topo.check()


def test_get_empty_whuhvmodel():
    """Test `get_empty_whuhvmodel(...)`."""
    result = get_empty_whuhvmodel(32, 40, "float32")
    result.check()


def test_get_empty_huvmodel():
    """Test `get_empty_huvmodel(...)`."""
    result = get_empty_huvmodel(38, 40, "float32")
    result.check()


def test_get_empty_faceonesidemodel():
    """Test `get_empty_faceonesidemodel(...)`."""
    result = get_empty_faceonesidemodel(38, 40, "float32")
    result.check()


def test_get_empty_facetwosidemodel():
    """Test `get_empty_facetwosidemodel(...)`."""
    result = get_empty_facetwosidemodel(38, 40, "float32")
    result.check()


def test_get_empty_facequantitymodel():
    """Test `get_empty_facequantitymodel(...)`."""
    result = get_empty_facequantitymodel(38, 40, "float32")
    result.check()


def test_get_empty_slopes():
    """Test `get_empty_slopes(...)`."""
    result = get_empty_slopes(38, 40, "float32")
    result.check()


def test_get_empty_states():
    """Test `get_empty_states(...)`."""

    process = get_process(MPI.COMM_WORLD, 50, 50)
    x = get_gridline("x", process.pnx, process.pi, 50, 0., 5., "float64")
    y = get_gridline("y", process.pny, process.pj, 50, 0., 10., "float64")
    domain = get_domain(process, x, y)
    results = get_empty_states(domain, 2)
    results.check()
