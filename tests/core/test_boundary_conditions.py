#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Tests for boundary conditions.
"""
import numpy
from torchswe.core.boundary_conditions import _extrap_seq
from torchswe.core.boundary_conditions import _extrap_anchor
from torchswe.core.boundary_conditions import _extrap_delta_slc
from torchswe.core.boundary_conditions import _extrap_slc
from torchswe.core.boundary_conditions import _inflow_topo_slc
from torchswe.core.boundary_conditions import outflow_factory


def test_extrap_seq():
    """Test _extrap_seq."""

    result = _extrap_seq["west"](2, "float32")
    ans = numpy.zeros((1, 2), dtype="float32")
    ans[:, 0] = 2
    ans[:, 1] = 1
    assert numpy.allclose(result, ans)

    result = _extrap_seq["east"](2, "float32")
    ans = numpy.zeros((1, 2), dtype="float32")
    ans[:, 0] = 1
    ans[:, 1] = 2
    assert numpy.allclose(result, ans)

    result = _extrap_seq["south"](2, "float32")
    ans = numpy.zeros((2, 1), dtype="float32")
    ans[0, :] = 2
    ans[1, :] = 1
    assert numpy.allclose(result, ans)

    result = _extrap_seq["north"](2, "float32")
    ans = numpy.zeros((2, 1), dtype="float32")
    ans[0, :] = 1
    ans[1, :] = 2
    assert numpy.allclose(result, ans)


def test_extrap_anchor():
    """Test _extrap_anchor."""

    target = numpy.arange(253).reshape((11, 23))

    result = target[_extrap_anchor["west"](2)]
    assert numpy.allclose(result, numpy.arange(48, 187, 23).reshape((7, 1)))

    result = target[_extrap_anchor["east"](2)]
    assert numpy.allclose(result, numpy.arange(66, 205, 23).reshape((7, 1)))

    result = target[_extrap_anchor["south"](2)]
    assert numpy.allclose(result, numpy.arange(48, 67).reshape((1, 19)))

    result = target[_extrap_anchor["north"](2)]
    assert numpy.allclose(result, numpy.arange(186, 205).reshape((1, 19)))


def test_extrap_delta_slc():
    """Test _extrap_delta_slc."""

    target = numpy.arange(253).reshape((11, 23))

    result = target[_extrap_delta_slc["west"](2)]
    assert numpy.allclose(result, numpy.arange(49, 188, 23).reshape((7, 1)))

    result = target[_extrap_delta_slc["east"](2)]
    assert numpy.allclose(result, numpy.arange(65, 204, 23).reshape((7, 1)))

    result = target[_extrap_delta_slc["south"](2)]
    assert numpy.allclose(result, numpy.arange(71, 90).reshape((1, 19)))

    result = target[_extrap_delta_slc["north"](2)]
    assert numpy.allclose(result, numpy.arange(163, 182).reshape((1, 19)))


def test_extrap_slc():
    """Test _extrap_slc."""

    target = numpy.arange(253).reshape((11, 23))

    result = target[_extrap_slc["west"](2)]
    assert numpy.allclose(
        result, numpy.concatenate([
            numpy.arange(46, 185, 23).reshape((7, 1)),
            numpy.arange(47, 186, 23).reshape((7, 1))], 1))

    result = target[_extrap_slc["east"](2)]
    assert numpy.allclose(
        result, numpy.concatenate([
            numpy.arange(67, 206, 23).reshape((7, 1)),
            numpy.arange(68, 207, 23).reshape((7, 1))], 1))

    result = target[_extrap_slc["south"](2)]
    assert numpy.allclose(
        result, numpy.concatenate([
            numpy.arange(2, 21).reshape((1, 19)),
            numpy.arange(25, 44).reshape((1, 19))], 0))

    result = target[_extrap_slc["north"](2)]
    assert numpy.allclose(
        result, numpy.concatenate([
            numpy.arange(209, 228).reshape((1, 19)),
            numpy.arange(232, 251).reshape((1, 19))], 0))


def test_inflow_topo_slc():
    """Test _inflow_topo_slc."""

    target = numpy.arange(253).reshape((11, 23))

    result = target[_inflow_topo_slc["west"]]
    assert numpy.allclose(result, numpy.arange(0, 253, 23).reshape((11, 1)))

    result = target[_inflow_topo_slc["east"]]
    assert numpy.allclose(result, numpy.arange(22, 253, 23).reshape((11, 1)))

    result = target[_inflow_topo_slc["south"]]
    assert numpy.allclose(result, numpy.arange(0, 23).reshape((1, 23)))

    result = target[_inflow_topo_slc["north"]]
    assert numpy.allclose(result, numpy.arange(230, 253).reshape((1, 23)))


def test_outflow():
    """Test outflow bc."""

    orgin = numpy.arange(253).reshape((11, 23))

    func = outflow_factory(2, "west")
    target = orgin.copy()
    target = func(target)
    assert numpy.allclose(target[:, 2:], orgin[:, 2:])
    assert numpy.allclose(target[2:-2, 0], numpy.arange(48, 187, 23))
    assert numpy.allclose(target[2:-2, 1], numpy.arange(48, 187, 23))

    func = outflow_factory(2, "east")
    target = orgin.copy()
    target = func(target)
    assert numpy.allclose(target[:, :-2], orgin[:, :-2])
    assert numpy.allclose(target[2:-2, -2], numpy.arange(66, 205, 23))
    assert numpy.allclose(target[2:-2, -1], numpy.arange(66, 205, 23))

    func = outflow_factory(2, "south")
    target = orgin.copy()
    target = func(target)
    assert numpy.allclose(target[2:, :], orgin[2:, :])
    assert numpy.allclose(target[0, 2:-2], numpy.arange(48, 67))
    assert numpy.allclose(target[1, 2:-2], numpy.arange(48, 67))

    func = outflow_factory(2, "north")
    target = orgin.copy()
    target = func(target)
    assert numpy.allclose(target[:-2, :], orgin[:-2, :])
    assert numpy.allclose(target[-2, 2:-2], numpy.arange(186, 205))
    assert numpy.allclose(target[-1, 2:-2], numpy.arange(186, 205))
