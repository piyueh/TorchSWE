#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Pi-Yueh Chuang <pychuang@gwu.me>
#
# Distributed under terms of the BSD 3-Clause license.

"""A dummy dict of which the data can be accessed as attributes.
"""
import collections


class DummyDict(collections.UserDict):
    """A dummy dict of which the data can be accessed as attributes.

    A dummy dictionary

    1. allows accessing data through `.` as if they are the attributes of this instance, and
    2. prohibiting setting new data/attribution is the key is not used to initialize this instance.
    """
    # pylint: disable=too-many-ancestors

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
