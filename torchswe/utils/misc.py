#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Pi-Yueh Chuang <pychuang@gwu.me>
#
# Distributed under terms of the BSD 3-Clause license.

"""A collection of some misc stuff.
"""
import os
import logging
import collections
from scipy.interpolate import RectBivariateSpline as _RectBivariateSpline

# instead of importing from torchswe, we do it here again to avoid circular importing
if "LEGATE_MAX_DIM" in os.environ and "LEGATE_MAX_FIELDS" in os.environ:
    from legate.numpy import float32, float64
elif "USE_CUPY" in os.environ and os.environ["USE_CUPY"] == "1":
    from cupy import float32, float64
elif "USE_TORCH" in os.environ and os.environ["USE_TORCH"] == "1":
    from torch import float32, float64
else:
    from numpy import float32, float64

logger = logging.getLogger("torchswe.utils.misc")


def dummy_function(*args, **kwargs):  #pylint: disable=unused-argument, useless-return
    """A dummy function for CuPy.

    Many functions in NumPy are not implemented in CuPy. However, most of them are not important.
    In order not to write another codepath for CuPy, we assign this dummy function to CuPy's
    corresponding attributes. Currenty, known functions

    - the member of the context manager: errstate
    - set_printoptions
    """
    logger.debug("This dummy function is called by CuPy.")
    return None


class DummyDict(collections.UserDict):  # pylint: disable=too-many-ancestors
    """A dummy dict of which the data can be accessed as attributes.
    """

    def __init__(self, init_attrs=None, /, **kwargs):
        # pylint: disable=super-init-not-called
        object.__setattr__(self, "data", {})

        if init_attrs is not None:
            self.data.update(init_attrs)

        if kwargs:
            self.data.update(kwargs)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __getattr__(self, key):
        return self.__getitem__(key)

    def __delattr__(self, key):
        self.__delitem__(key)


class DummyErrState:  # pylint: disable=too-few-public-methods
    """A dummy errstate context manager."""
    __enter__ = dummy_function
    __exit__ = dummy_function
    def __init__(self, *args, **kwargs):
        pass


class DummyDtype:  # pylint: disable=too-few-public-methods
    """A dummy dtype to make all NumPy, Legate, CuPy and PyTorch happy.

    PyTorch is the least numpy-compatible. This class is actually prepared for PyTorch!
    """
    @classmethod
    def __get_validators__(cls):
        """Iteratorate throuh available validators for pydantic's data model"""
        yield cls.validator

    @classmethod
    def validator(cls, v):  # pylint: disable=invalid-name
        """validator."""

        msg = "Either nplike.float32/nplike.float64 or their str representations."

        if isinstance(v, str):
            try:
                return {"float32": float32, "float64": float64}[v]
            except KeyError as err:
                raise ValueError(msg) from err
        elif v not in (float32, float64):
            raise ValueError(msg)

        return v


def interpolate(x_in, y_in, data_in, x_out, y_out):
    """A wrapper to interpolation with scipy.interpolate.RectBivariateSpline.

    scipy.interpolate.RectBivariateSpline only accpets vanilla NumPy array. Different np-like
    backends use different method to convert to vanilla numpy.ndarray. This function unifies them
    and the interpolation.

    The return is always vanilla numpy.ndarray.

    Arguments
    ---------
    x_in, y_in, data_in : nplike.ndarray
        The first three inputs to scipy.interpolate.RectBivariateSpline.
    x_out, y_out : nplike.ndarray
        The first two inputs to scipy.interpolate.RectBivariateSpline.__call__.

    Returns
    -------
    data_out : numpy.ndarray
        The output of scipy.interpolate.RectBivariateSpline.__call__.
    """

    try:
        func = _RectBivariateSpline(x_in, y_in, data_in)
    except TypeError as err:
        if str(err).startswith("Implicit conversion to a NumPy array is not allowe"):
            func = _RectBivariateSpline(x_in.get(), y_in.get(), data_in.get())  # cupy
            x_out = x_out.get()
            y_out = y_out.get()
        elif str(err).startswith("can't convert cuda:"):
            func = _RectBivariateSpline(
                x_in.cpu().numpy(), y_in.cpu().numpy(), data_in.cpu().numpy())  # pytorch
            x_out = x_out.cpu().numpy()
            y_out = y_out.cpu().numpy()
        else:
            raise

    return func(x_out, y_out)
