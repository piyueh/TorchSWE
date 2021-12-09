#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Data structs of TorchSWE.
"""
from torchswe.utils.data.grid import Gridline
from torchswe.utils.data.grid import Timeline
from torchswe.utils.data.grid import Domain
from torchswe.utils.data.grid import get_gridline_x
from torchswe.utils.data.grid import get_gridline_y
from torchswe.utils.data.grid import get_timeline
from torchswe.utils.data.grid import get_domain

from torchswe.utils.data.topography import Topography
from torchswe.utils.data.topography import get_topography

from torchswe.utils.data.states import States
from torchswe.utils.data.states import get_empty_states
from torchswe.utils.data.states import get_initial_states

from torchswe.utils.data.source import PointSource
from torchswe.utils.data.source import get_pointsource
