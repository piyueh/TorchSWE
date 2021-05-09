#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Pi-Yueh Chuang <pychuang@gwu.me>
#
# Distributed under terms of the BSD 3-Clause license.

"""A collection of some dummy stuff.
"""
import os
import logging
import collections

# instead of importing from torchswe, we do it here again to avoid circular importing
if "LEGATE_MAX_DIM" in os.environ and "LEGATE_MAX_FIELDS" in os.environ:
    from legate.numpy import float32, float64
elif "USE_CUPY" in os.environ and os.environ["USE_CUPY"] == "1":
    from cupy import float32, float64
elif "USE_TORCH" in os.environ and os.environ["USE_TORCH"] == "1":
    from torch import float32, float64
else:
    from numpy import float32, float64

logger = logging.getLogger("torchswe.utils.dummy")


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
