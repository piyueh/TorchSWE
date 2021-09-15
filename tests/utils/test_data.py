#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Tests for data models.
"""
import pydantic
from torchswe import nplike
from torchswe.utils.data import get_gridline, Gridline


def test_get_gridline():
    """Test Gridline model."""
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


def test_gridline_validation():
    """Test the validation mechanism."""

    data = {
        "dtype": "float64",
        "axis": "x",
        "gn": 100,
        "glower": -1.2,
        "gupper": 11.3,
        "n": 10,
        "lower": 0.,
        "upper": 1.25,
        "ibegin": 11,
        "iend": 21,
        "delta": 0.125,
        "vertices": nplike.linspace(0., 1.25, 11),
        "centers": nplike.linspace(0.125/2., 1.25-0.125/2., 10),
        "xfcenters": nplike.linspace(0., 1.25, 11),
        "yfcenters": nplike.linspace(0.125/2., 1.25-0.125/2., 10),
    }

    def _check_n_errors(nerr, **kwargs):
        try:
            Gridline(**kwargs)
            raise AssertionError("Expected exception was not raised.")
        except pydantic.ValidationError as err:
            assert len(err.errors()) == nerr

    def _get_data(*args):
        return {k: v for k, v in data.items() if k not in args}

    _check_n_errors(15)
    _check_n_errors(2, **_get_data("axis", "n"))
    _check_n_errors(1, **_get_data("gupper"), gupper=-3.)
    _check_n_errors(1, **_get_data("upper"), upper=-1.)
    _check_n_errors(1, **_get_data("lower"), lower=-2.)
    _check_n_errors(1, **_get_data("upper"), upper=12.)
    _check_n_errors(1, **_get_data("n"), n=120)
    _check_n_errors(1, **_get_data("n"), n=12)
    _check_n_errors(1, **_get_data("iend"), iend=9)
    _check_n_errors(1, **_get_data("vertices"), vertices=data["vertices"].astype("float32"))
    _check_n_errors(1, **_get_data("vertices"), vertices=nplike.linspace(0., -1.25, 11))
    _check_n_errors(1, **_get_data("centers"), centers=nplike.linspace(0., 1., 9))
    _check_n_errors(1, **_get_data("centers"), centers=nplike.linspace(0., 1., 10))
    _check_n_errors(1, **_get_data("xfcenters"), xfcenters=data["yfcenters"])
