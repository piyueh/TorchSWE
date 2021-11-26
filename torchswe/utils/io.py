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


def write_states(states: _States, fname: str):
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

    with _File(f"{fname}.hdf5", 'w', driver='mpio', comm=comm) as fobj:
        fobj.create_dataset("Q", (3, gny, gnx), dtype)
        fobj["Q"][:, jbg:jed, ibg:ied] = states.Q[:, ngh:-ngh, ngh:-ngh]

        fobj.create_dataset("H", (gny, gnx), dtype)
        fobj["U"][:, jbg:jed, ibg:ied] = states.U[:, ngh:-ngh, ngh:-ngh]

        fobj.create_dataset("S", (3, gny, gnx), dtype)
        fobj["S"][:, jbg:jed, ibg:ied] = states.S

        fobj.create_dataset("SS", (3, gny, gnx), dtype)
        fobj["SS"][:, jbg:jed, ibg:ied] = states.SS

        # how does h5py deal with cometition of shared cell faces between ranks?
        fobj.create_dataset("xH", (3, gny, gnx+1), dtype)
        fobj["xH"][:, jbg:jed, ibg:ied+1] = states.face.x.H

        fobj.create_dataset("xmQ", (3, gny, gnx+1), dtype)
        fobj["xmQ"][:, jbg:jed, ibg:ied+1] = states.face.x.minus.Q

        fobj.create_dataset("xmU", (3, gny, gnx+1), dtype)
        fobj["xmU"][:, jbg:jed, ibg:ied+1] = states.face.x.minus.U

        fobj.create_dataset("xma", (gny, gnx+1), dtype)
        fobj["xma"][jbg:jed, ibg:ied+1] = states.face.x.minus.a

        fobj.create_dataset("xmF", (3, gny, gnx+1), dtype)
        fobj["xmF"][:, jbg:jed, ibg:ied+1] = states.face.x.minus.F

        fobj.create_dataset("xpQ", (3, gny, gnx+1), dtype)
        fobj["xpQ"][:, jbg:jed, ibg:ied+1] = states.face.x.plus.Q

        fobj.create_dataset("xpU", (3, gny, gnx+1), dtype)
        fobj["xpU"][:, jbg:jed, ibg:ied+1] = states.face.x.plus.U

        fobj.create_dataset("xpa", (gny, gnx+1), dtype)
        fobj["xpa"][jbg:jed, ibg:ied+1] = states.face.x.plus.a

        fobj.create_dataset("xpF", (3, gny, gnx+1), dtype)
        fobj["xpF"][:, jbg:jed, ibg:ied+1] = states.face.x.plus.F


        fobj.create_dataset("yH", (3, gny+1, gnx), dtype)
        fobj["yH"][:, jbg:jed+1, ibg:ied] = states.face.y.H

        fobj.create_dataset("ymQ", (3, gny+1, gnx), dtype)
        fobj["ymQ"][:, jbg:jed+1, ibg:ied] = states.face.y.minus.Q

        fobj.create_dataset("ymU", (3, gny+1, gnx), dtype)
        fobj["ymU"][:, jbg:jed+1, ibg:ied] = states.face.y.minus.U

        fobj.create_dataset("yma", (gny+1, gnx), dtype)
        fobj["yma"][jbg:jed+1, ibg:ied] = states.face.y.minus.a

        fobj.create_dataset("ymF", (3, gny+1, gnx), dtype)
        fobj["ymF"][:, jbg:jed+1, ibg:ied] = states.face.y.minus.F

        fobj.create_dataset("ypQ", (3, gny+1, gnx), dtype)
        fobj["ypQ"][:, jbg:jed+1, ibg:ied] = states.face.y.plus.Q

        fobj.create_dataset("ypU", (3, gny+1, gnx), dtype)
        fobj["ypU"][:, jbg:jed+1, ibg:ied] = states.face.y.plus.U

        fobj.create_dataset("ypa", (gny+1, gnx), dtype)
        fobj["ypa"][jbg:jed+1, ibg:ied] = states.face.y.plus.a

        fobj.create_dataset("ypF", (3, gny+1, gnx), dtype)
        fobj["ypF"][:, jbg:jed+1, ibg:ied] = states.face.y.plus.F

        fobj.create_dataset("ibg", (comm.size,), int)
        fobj["ibg"][comm.rank] = ibg

        fobj.create_dataset("ied", (comm.size,), int)
        fobj["ied"][comm.rank] = ied

        fobj.create_dataset("jbg", (comm.size,), int)
        fobj["jbg"][comm.rank] = jbg

        fobj.create_dataset("jed", (comm.size,), int)
        fobj["jed"][comm.rank] = jed
