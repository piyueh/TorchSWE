#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Test functions in torchswe.utils.init using 4 MPI processes.
"""
import pathlib
import shutil
import subprocess

data_dir = pathlib.Path(__file__).expanduser().resolve().parents[1].joinpath("data")

def test_get_process(tmp_path):
    """Test `get_process(...)`."""

    shutil.copyfile(data_dir.joinpath("test_get_process.script"), tmp_path.joinpath("test.py"))

    try:
        subprocess.run(
            ["mpiexec", "-n", "4", "python", str(tmp_path.joinpath("test.py"))],
            check=True, capture_output=True, timeout=10)
    except subprocess.CalledProcessError as err:
        raise AssertionError(
            "The MPI test reports the following error:\n{}".format(err.stderr.decode("utf-8"))
        ) from err


def test_get_gridline(tmp_path):
    """Test `get_gridline(...)`."""

    shutil.copyfile(data_dir.joinpath("test_get_gridline.script"), tmp_path.joinpath("test.py"))

    try:
        subprocess.run(
            ["mpiexec", "-n", "4", "python", str(tmp_path.joinpath("test.py"))],
            check=True, capture_output=True, timeout=10)
    except subprocess.CalledProcessError as err:
        raise AssertionError(
            "The MPI test reports the following error:\n{}".format(err.stderr.decode("utf-8"))
        ) from err


def test_get_domain(tmp_path):
    """Test `get_domain(...)`."""

    shutil.copyfile(data_dir.joinpath("test_get_domain.script"), tmp_path.joinpath("test.py"))

    try:
        subprocess.run(
            ["mpiexec", "-n", "4", "python", str(tmp_path.joinpath("test.py"))],
            check=True, capture_output=True, timeout=10)
    except subprocess.CalledProcessError as err:
        raise AssertionError(
            "The MPI test reports the following error:\n{}".format(err.stderr.decode("utf-8"))
        ) from err


def test_get_topography(tmp_path):
    """Test `get_topography(...)`."""

    shutil.copyfile(data_dir.joinpath("test_get_topography.script"), tmp_path.joinpath("test.py"))

    try:
        subprocess.run(
            ["mpiexec", "-n", "4", "python", str(tmp_path.joinpath("test.py"))],
            check=True, capture_output=True, timeout=10)
    except subprocess.CalledProcessError as err:
        raise AssertionError(
            "The MPI test reports the following error:\n{}".format(err.stderr.decode("utf-8"))
        ) from err


def test_get_empty_states(tmp_path):
    """Test `get_empty_states(...)`."""

    shutil.copyfile(data_dir.joinpath("test_get_empty_states.script"), tmp_path.joinpath("test.py"))

    try:
        subprocess.run(
            ["mpiexec", "-n", "4", "python", str(tmp_path.joinpath("test.py"))],
            check=True, capture_output=True, timeout=30)
    except subprocess.CalledProcessError as err:
        raise AssertionError(
            "The MPI test reports the following error:\n{}".format(err.stderr.decode("utf-8"))
        ) from err


def _assert_wrapper(fobj, pre, msg):
    fobj.write(pre+"assert {}, \"from rank {{}}\".format(rank)\n".format(msg))
