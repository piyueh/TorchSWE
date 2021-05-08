#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Pi-Yueh Chuang <pychuang@gwu.me>
#
# Distributed under terms of the BSD 3-Clause license.

"""A collection of some dummy stuff.
"""
import logging
import collections
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
