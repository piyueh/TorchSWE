#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Utilities of TorchSWE.
"""
import yaml as _yaml
from .config import Config as _Config


# register the Config class in yaml with tag !Config
_yaml.add_constructor(
    u'!Config',
    lambda loader, node: _Config(**loader.construct_mapping(node, deep=True))
)

_yaml.add_representer(
    _Config,
    lambda dumper, data: dumper.represent_mapping(
        tag=u"!Config", mapping=_yaml.load(
            data.json(by_alias=True), Loader=_yaml.Loader),
        flow_style=True
    )
)
