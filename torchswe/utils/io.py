#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Higher level api for writing data to files.
"""
from pathlib import Path as _Path
from netCDF4 import Dataset as _Dataset  # pylint: disable=no-name-in-module
from h5py import File as _File
from torchswe import nplike as _nplike
from torchswe.utils.data import States as _States
from torchswe.utils.netcdf import write_to_dataset as _write_to_dataset
from torchswe.utils.netcdf import add_time_data_to_dataset as _add_time_data_to_dataset


def create_empty_soln_file(fpath, domain, t, **kwargs):
    """Create an new NetCDF file for solutions using the corresponding grid object.

    Create an empty NetCDF4 file with axes `x`, `y`, and `time`. `x` and `y` are defined at cell
    centers. The spatial coordinates use EPSG 3856. The temporal axis is limited with dimension
    `ntime` (i.e., not using the unlimited axis feature from CF convention).

    Also, this function creates empty solution variables (`w`, `hu`, `hv`, `h`, `u`, `v`) in the
    dataset with `NaN` for all values. The shapes of these variables are `(ntime, ny, nx)`.

    Arguments
    ---------
    fpath : str or PathLike
        The path to the file.
    domain : torchswe.utils.data.Domain
        The Domain instance corresponds to the solutions.
    t : torchswe.utils.data.Timeline
        The temporal axis object.
    **kwargs
        Keyword arguments sent to netCDF4.Dataset.
    """

    assert "parallel" not in kwargs, "`parallel` should not be included in `kwargs`"
    assert "comm" not in kwargs, "`parallel` should not be included in `kwargs`"

    fpath = _Path(fpath).expanduser().resolve()

    data = {k: None for k in ["w", "hu", "hv", "h"]}

    with _Dataset(fpath, "w", parallel=True, comm=domain.comm, **kwargs) as dset:

        _write_to_dataset(
            dset=dset,
            axs=(domain.x.centers, domain.y.centers, t.values),
            data=data,
            global_n=(domain.x.gn, domain.y.gn),
            idx_bounds=(domain.x.ibegin, domain.x.iend, domain.y.ibegin, domain.y.iend),
            corner=(domain.x.glower, domain.y.gupper),
            deltas=(domain.x.delta, domain.y.delta),
        )

        dset.sync()


def write_soln_to_file(fpath, soln, time, tidx, **kwargs):
    """Write a solution snapshot to an existing NetCDF file.

    Arguments
    ---------
    fpath : str or PathLike
        The path to the file.
    soln : torchswe.utils.data.State
        The solution object.
    time : float
        The simulation time of this snapshot.
    tidx : int
        The index of the snapshot time in the temporal axis.
    **kwargs
        Keyword arguments sent to netCDF4.Dataset.
    """
    fpath = _Path(fpath).expanduser().resolve()

    assert "parallel" not in kwargs, "`parallel` should not be included in `kwargs`"
    assert "comm" not in kwargs, "`parallel` should not be included in `kwargs`"

    # determine if it's a WHUHVModel or HUVModel
    data = {
        "w": soln.Q[(0,)+soln.domain.internal],
        "hu": soln.Q[(1,)+soln.domain.internal],
        "hv": soln.Q[(2,)+soln.domain.internal],
        "h": soln.U[(0,)+soln.domain.internal],
    }

    # alias
    domain = soln.domain

    with _Dataset(fpath, "a", parallel=True, comm=domain.comm, **kwargs) as dset:

        _add_time_data_to_dataset(
            dset=dset, data=data, time=time, tidx=tidx,
            idx_bounds=(domain.x.ibegin, domain.x.iend, domain.y.ibegin, domain.y.iend)
        )
        dset.sync()


def write_states(states: _States):
    """Write flatten states to a .npz file for debug."""

    comm = states.domain.comm
    gnx = states.domain.x.gn
    gny = states.domain.y.gn
    lnx = states.domain.x.n
    lny = states.domain.y.n
    ngh = states.domain.nhalo
    dtype = states.Q.dtype
    ibg = states.domain.x.ibegin
    ied = states.domain.x.iend
    jbg = states.domain.y.ibegin
    jed = states.domain.y.iend

    with _File(f"states-{comm.size}.{comm.rank+1}.hdf5", 'w') as fobj:  #, driver='mpio', comm=comm) as fobj:
        fobj.create_dataset("Q", states.Q.shape, dtype, states.Q)
        fobj.create_dataset("U", states.U.shape, dtype, states.U)
        fobj.create_dataset("S", states.S.shape, dtype, states.S)
        fobj.create_dataset("slpx", states.slpx.shape, dtype, states.slpx)
        fobj.create_dataset("slpy", states.slpy.shape, dtype, states.slpy)

        fobj.create_dataset("xH", states.face.x.H.shape, dtype, states.face.x.H)
        fobj.create_dataset("xmQ", states.face.x.minus.Q.shape, dtype, states.face.x.minus.Q)
        fobj.create_dataset("xmU", states.face.x.minus.U.shape, dtype, states.face.x.minus.U)
        fobj.create_dataset("xma", states.face.x.minus.a.shape, dtype, states.face.x.minus.a)
        fobj.create_dataset("xmF", states.face.x.minus.F.shape, dtype, states.face.x.minus.F)
        fobj.create_dataset("xpQ", states.face.x.plus.Q.shape, dtype, states.face.x.plus.Q)
        fobj.create_dataset("xpU", states.face.x.plus.U.shape, dtype, states.face.x.plus.U)
        fobj.create_dataset("xpa", states.face.x.plus.a.shape, dtype, states.face.x.plus.a)
        fobj.create_dataset("xpF", states.face.x.plus.F.shape, dtype, states.face.x.plus.F)

        fobj.create_dataset("yH", states.face.y.H.shape, dtype, states.face.y.H)
        fobj.create_dataset("ymQ", states.face.y.minus.Q.shape, dtype, states.face.y.minus.Q)
        fobj.create_dataset("ymU", states.face.y.minus.U.shape, dtype, states.face.y.minus.U)
        fobj.create_dataset("yma", states.face.y.minus.a.shape, dtype, states.face.y.minus.a)
        fobj.create_dataset("ymF", states.face.y.minus.F.shape, dtype, states.face.y.minus.F)
        fobj.create_dataset("ypQ", states.face.y.plus.Q.shape, dtype, states.face.y.plus.Q)
        fobj.create_dataset("ypU", states.face.y.plus.U.shape, dtype, states.face.y.plus.U)
        fobj.create_dataset("ypa", states.face.y.plus.a.shape, dtype, states.face.y.plus.a)
        fobj.create_dataset("ypF", states.face.y.plus.F.shape, dtype, states.face.y.plus.F)

        if states.SS is not None:
            fobj.create_dataset("SS", states.SS.shape, dtype, states.SS)

        fobj.attrs["ibg"] = ibg
        fobj.attrs["ied"] = ied
        fobj.attrs["jbg"] = jbg
        fobj.attrs["jed"] = jed
