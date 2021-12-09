#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Higher level api for reading/writing data from/to TorchSWE's native HDF5 format.
"""
from os import PathLike as _PathLike
from typing import Tuple as _Tuple
from typing import Union as _Union
from h5py import File as _File
from torchswe.utils.misc import DummyDict as _DummyDict
from torchswe.utils.misc import find_index_bound as _find_index_bound
from torchswe.utils.config import Config as _Config
from torchswe.utils.data.states import States as _States
from torchswe.utils.data.grid import Domain as _Domain


def read_block(
    filename: _PathLike, xykeys: _Tuple[str, str],
    dkeys: _Union[str, _Tuple[str, ...]], domain: _Domain
):
    """Read only a region of data using a provided extent.

    Arguments
    ---------
    filename : os.PathLike
        The path of the file to be read.
    dkeys : str or a tuple of str
        The key(s) (or the path(s) in HDF5's terminology) to desired datasets.
    xykeys : a tuple of 2 str
        The keys to the x and y gridlines defining the underlying grid of the datasets.
    domain : torchswe.utils.data.grid.Domain
        The Domain object describing the local grid of this MPI rank. The desired extent is
        obtained from domain.

    Returns
    -------
    data : torchswe.utils.misc.DummyDict
        This data contains the keys x, y, and those in `dkeys`. The x and y keys provides
        only the dataset gridlines covering the desired extent.
    """

    data = _DummyDict()
    with _File(filename, "r", "mpio", comm=domain.comm) as root:

        # get index bounds
        data.x = root[xykeys[0]][...]
        data.y = root[xykeys[1]][...]
        ibg, ied, jbg, jed = _find_index_bound(data.x, data.y, domain.lextent)

        # extract only the gridlines covering the desired extent
        data.x = data.x[ibg:ied]
        data.y = data.y[jbg:jed]

        # if only providing one single key, make it a tuple for convenience
        if isinstance(dkeys, str):
            dkeys = (dkeys,)

        # the data slices that cover desired extent
        for key in dkeys:
            slc = (slice(None),) * (data[key].ndim - 2) + (slice(jbg, jed), slice(ibg, ied))
            data[key] = root[key][slc]

    return data


def write_debug_states(states: _States, config: _Config):
    """Write the local state of each MPI process to a .hdf5 file for debugging."""

    comm = states.domain.comm
    dtype = states.Q.dtype

    with _File(config.case.joinpath(f"debug-states-{comm.size}.{comm.rank+1}.hdf5"), 'w') as fobj:

        # cell-centered quantities
        fobj.create_dataset("Q", states.Q.shape, dtype, states.Q)
        fobj.create_dataset("U", states.U.shape, dtype, states.U)
        fobj.create_dataset("S", states.S.shape, dtype, states.S)
        fobj.create_dataset("slpx", states.slpx.shape, dtype, states.slpx)
        fobj.create_dataset("slpy", states.slpy.shape, dtype, states.slpy)

        if states.SS is not None:
            fobj.create_dataset("SS", states.SS.shape, dtype, states.SS)

        # quantities at faces facing x-direction
        fobj.create_dataset("xH", states.face.x.H.shape, dtype, states.face.x.H)
        fobj.create_dataset("xmQ", states.face.x.minus.Q.shape, dtype, states.face.x.minus.Q)
        fobj.create_dataset("xmU", states.face.x.minus.U.shape, dtype, states.face.x.minus.U)
        fobj.create_dataset("xma", states.face.x.minus.a.shape, dtype, states.face.x.minus.a)
        fobj.create_dataset("xmF", states.face.x.minus.F.shape, dtype, states.face.x.minus.F)
        fobj.create_dataset("xpQ", states.face.x.plus.Q.shape, dtype, states.face.x.plus.Q)
        fobj.create_dataset("xpU", states.face.x.plus.U.shape, dtype, states.face.x.plus.U)
        fobj.create_dataset("xpa", states.face.x.plus.a.shape, dtype, states.face.x.plus.a)
        fobj.create_dataset("xpF", states.face.x.plus.F.shape, dtype, states.face.x.plus.F)

        # quantities at faces facing y-direction
        fobj.create_dataset("yH", states.face.y.H.shape, dtype, states.face.y.H)
        fobj.create_dataset("ymQ", states.face.y.minus.Q.shape, dtype, states.face.y.minus.Q)
        fobj.create_dataset("ymU", states.face.y.minus.U.shape, dtype, states.face.y.minus.U)
        fobj.create_dataset("yma", states.face.y.minus.a.shape, dtype, states.face.y.minus.a)
        fobj.create_dataset("ymF", states.face.y.minus.F.shape, dtype, states.face.y.minus.F)
        fobj.create_dataset("ypQ", states.face.y.plus.Q.shape, dtype, states.face.y.plus.Q)
        fobj.create_dataset("ypU", states.face.y.plus.U.shape, dtype, states.face.y.plus.U)
        fobj.create_dataset("ypa", states.face.y.plus.a.shape, dtype, states.face.y.plus.a)
        fobj.create_dataset("ypF", states.face.y.plus.F.shape, dtype, states.face.y.plus.F)

        # the global index range of local grid
        fobj.attrs["ibg"] = states.domain.x.ibegin
        fobj.attrs["ied"] = states.domain.x.iend
        fobj.attrs["jbg"] = states.domain.y.ibegin
        fobj.attrs["jed"] = states.domain.y.iend
