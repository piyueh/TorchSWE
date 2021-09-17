#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Linear reconstruction.
"""
from torchswe import nplike as _nplike
from torchswe.utils.data import States as _States
from torchswe.utils.data import Topography as _Topography


def correct_negative_depth(states: _States, topo: _Topography) -> _States:
    """Fix negative depth on the both sides of cell faces.

    Arguments
    ---------
    states : torchswe.utils.data.States
    topo : torchswe.utils.data.Topography

    Returns:
    --------
    states : torchswe.utils.data.States
        The same object as the input. Changed inplace. Returning it just for coding style.
    """

    # aliases
    ngh = states.ngh
    nx, ny = states.domain.x.n, states.domain.y.n

    # fix the case when the left depth of an interface is negative
    j, i = _nplike.nonzero(states.face.x.minus.w < topo.xfcenters)
    states.face.x.minus.w[j, i] = topo.xfcenters[j, i]
    j, i = j[i != 0], i[i != 0]  # to avoid those i - 1 = -1
    states.face.x.plus.w[j, i-1] = 2 * states.q.w[j+ngh, i-1+ngh] - topo.xfcenters[j, i]

    # fix the case when the right depth of an interface is negative
    j, i = _nplike.nonzero(states.face.x.plus.w < topo.xfcenters)
    states.face.x.plus.w[j, i] = topo.xfcenters[j, i]
    j, i = j[i != nx], i[i != nx]  # to avoid i + 1 = nx + 1
    states.face.x.minus.w[j, i+1] = 2 * states.q.w[j+ngh, i+ngh] - topo.xfcenters[j, i]

    # fix rounding errors in x.minus.w caused by the last calculation above
    j, i = _nplike.nonzero(states.face.x.minus.w < topo.xfcenters)
    states.face.x.minus.w[j, i] = topo.xfcenters[j, i]

    # fix the case when the bottom depth of an interface is negative
    j, i = _nplike.nonzero(states.face.y.minus.w < topo.yfcenters)
    states.face.y.minus.w[j, i] = topo.yfcenters[j, i]
    j, i = j[j != 0], i[j != 0]  # to avoid j - 1 = -1
    states.face.y.plus.w[j-1, i] = 2 * states.q.w[j-1+ngh, i+ngh] - topo.yfcenters[j, i]

    # fix the case when the top depth of an interface is negative
    j, i = _nplike.nonzero(states.face.y.plus.w < topo.yfcenters)
    states.face.y.plus.w[j, i] = topo.yfcenters[j, i]
    j, i = j[j != ny], i[j != ny]  # to avoid j + 1 = Ny + 1
    states.face.y.minus.w[j+1, i] = 2 * states.q.w[j+ngh, i+ngh] - topo.yfcenters[j, i]

    # fix rounding errors in y.minus.w caused by the last calculation above
    j, i = _nplike.nonzero(states.face.y.minus.w < topo.yfcenters)
    states.face.y.minus.w[j, i] = topo.yfcenters[j, i]

    return states


def get_discontinuous_cnsrv_q(states: _States):
    """Distinuous conservative quantity on both sides of cell faces in both x- and y-direction.

    Arguments
    ---------
    states : torchswe.utils.data.States
    grid : torchswe.utils.data.Gridlines

    Returns
    -------
    states : torchswe.utils.data.States
        The same object as the input. Changed inplace. Returning it just for coding style.

    Notes
    -----
    Linear interpolation.
    """

    # aliases; n_ghost must >= 2, so it should be safe to use -ngh+1 and ngh-1
    i = slice(states.ngh, -states.ngh)  # length ny or nx
    im1 = slice(states.ngh-1, -states.ngh)  # length ny+1 or nx+1
    ip1 = slice(states.ngh, -states.ngh+1)  # length ny+1 or nx+1

    delta_x_half = states.domain.x.delta / 2.
    delta_y_half = states.domain.y.delta / 2.

    states.face.x.minus.w = states.q.w[i, im1] + states.slp.x.w[:, :-1] * delta_x_half
    states.face.x.minus.hu = states.q.hu[i, im1] + states.slp.x.hu[:, :-1] * delta_x_half
    states.face.x.minus.hv = states.q.hv[i, im1] + states.slp.x.hv[:, :-1] * delta_x_half

    states.face.x.plus.w = states.q.w[i, ip1] - states.slp.x.w[:, 1:] * delta_x_half
    states.face.x.plus.hu = states.q.hu[i, ip1] - states.slp.x.hu[:, 1:] * delta_x_half
    states.face.x.plus.hv = states.q.hv[i, ip1] - states.slp.x.hv[:, 1:] * delta_x_half

    states.face.y.minus.w = states.q.w[im1, i] + states.slp.y.w[:-1, :] * delta_y_half
    states.face.y.minus.hu = states.q.hu[im1, i] + states.slp.y.hu[:-1, :] * delta_y_half
    states.face.y.minus.hv = states.q.hv[im1, i] + states.slp.y.hv[:-1, :] * delta_y_half

    states.face.y.plus.w = states.q.w[ip1, i] - states.slp.y.w[1:, :] * delta_y_half
    states.face.y.plus.hu = states.q.hu[ip1, i] - states.slp.y.hu[1:, :] * delta_y_half
    states.face.y.plus.hv = states.q.hv[ip1, i] - states.slp.y.hv[1:, :] * delta_y_half

    return states
