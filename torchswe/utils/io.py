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

    data = {k: None for k in ["w", "hu", "hv", "h", "u", "v"]}

    with _Dataset(fpath, "w", parallel=True, comm=domain.process.comm, **kwargs) as dset:

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
        "w": soln.Q[0, slice(soln.ngh, -soln.ngh), slice(soln.ngh, -soln.ngh)],
        "hu": soln.Q[1, slice(soln.ngh, -soln.ngh), slice(soln.ngh, -soln.ngh)],
        "hv": soln.Q[2, slice(soln.ngh, -soln.ngh), slice(soln.ngh, -soln.ngh)],
        "h": soln.U[0, slice(soln.ngh, -soln.ngh), slice(soln.ngh, -soln.ngh)],
        "u": soln.U[1, slice(soln.ngh, -soln.ngh), slice(soln.ngh, -soln.ngh)],
        "v": soln.U[2, slice(soln.ngh, -soln.ngh), slice(soln.ngh, -soln.ngh)],
    }

    # alias
    domain = soln.domain

    with _Dataset(fpath, "a", parallel=True, comm=domain.process.comm, **kwargs) as dset:

        _add_time_data_to_dataset(
            dset=dset, data=data, time=time, tidx=tidx,
            idx_bounds=(domain.x.ibegin, domain.x.iend, domain.y.ibegin, domain.y.iend)
        )
        dset.sync()


def write_states(states: _States, fname: str):
    """Write flatten states to a .npz file for debug."""

    keys = ["w", "hu", "hv"]
    keys2 = ["h", "u", "v", "a"]

    data = {}
    data.update({f"q_{k}": states.q[k] for k in keys})
    data.update({f"src_{k}": states.src[k] for k in keys})
    data.update({f"slp_x_{k}": states.slp.x[k] for k in keys})
    data.update({f"slp_y_{k}": states.slp.y[k] for k in keys})
    data.update({f"face_x_minus_{k}": states.face.x.minus[k] for k in keys})
    data.update({f"face_x_plus_{k}": states.face.x.plus[k] for k in keys})
    data.update({f"face_y_minus_{k}": states.face.y.minus[k] for k in keys})
    data.update({f"face_y_plus_{k}": states.face.y.plus[k] for k in keys})
    data.update({f"face_x_minus_{k}": states.face.x.minus[k] for k in keys2})
    data.update({f"face_x_plus_{k}": states.face.x.plus[k] for k in keys2})
    data.update({f"face_y_minus_{k}": states.face.y.minus[k] for k in keys2})
    data.update({f"face_y_plus_{k}": states.face.y.plus[k] for k in keys2})
    data.update({f"face_x_minus_flux_{k}": states.face.x.minus.flux[k] for k in keys})
    data.update({f"face_x_plus_flux_{k}": states.face.x.plus.flux[k] for k in keys})
    data.update({f"face_y_minus_flux_{k}": states.face.y.minus.flux[k] for k in keys})
    data.update({f"face_y_plus_flux_{k}": states.face.y.plus.flux[k] for k in keys})
    data.update({f"face_x_num_flux_{k}": states.face.x.num_flux[k] for k in keys})
    data.update({f"face_y_num_flux_{k}": states.face.y.num_flux[k] for k in keys})
    data.update({f"rhs_{k}": states.rhs[k] for k in keys})

    _nplike.savez(fname, **data)
