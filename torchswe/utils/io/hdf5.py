#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Higher level api for reading/writing data from/to TorchSWE's native HDF5 format.

Notes
-----
As of openmpi 4.1.1, the code in this module requires `--mca fcoll "^vulcan"`.
"""
# imports related to type hinting
from __future__ import annotations as _annotations  # allows us not using quotation marks for hints
from typing import TYPE_CHECKING as _TYPE_CHECKING  # indicates if we have type checking right now
if _TYPE_CHECKING:  # if we are having type checking, then we import corresponding classes/types
    from os import PathLike
    from typing import Tuple
    from h5py import Group
    from torchswe.utils.config import Config
    from torchswe.utils.misc import DummyDict
    from torchswe.utils.data.grid import Domain
    from torchswe.utils.data.topography import Topography
    from torchswe.utils.data.states import States
    from torchswe.utils.data.source import PointSource
    from torchswe.utils.data.source import FrictionModel

# pylint: disable=wrong-import-position, ungrouped-imports
from datetime import datetime as _datetime
from datetime import timezone as _timezone
from h5py import File as _File
from torchswe import nplike as _nplike
from torchswe.utils.misc import DummyDict as _DummyDict
from torchswe.utils.misc import find_index_bound as _find_index_bound


def read_block(filename: PathLike, xykeys: Tuple[str, str], dkeys: Tuple[str, ...], domain: Domain):
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
        data.x = _nplike.asarray(root[xykeys[0]][...])
        data.y = _nplike.asarray(root[xykeys[1]][...])
        ibg, ied, jbg, jed = _find_index_bound(data.x, data.y, domain.lextent)

        # extract only the gridlines covering the desired extent
        data.x = data.x[ibg:ied]
        data.y = data.y[jbg:jed]

        # if only providing one single key, make it a tuple for convenience
        if isinstance(dkeys, str):
            dkeys = (dkeys,)

        # the data slices that cover desired extent
        for key in dkeys:
            slc = (slice(None),) * (root[key].ndim - 2) + (slice(jbg, jed), slice(ibg, ied))
            data[key] = _nplike.asarray(root[key][slc])

    return data


def write_grid_to_group(domain: Domain, group: Group):
    """Write gridlines to an opened HDF5 group.

    Arguments
    ---------
    domain : torchswe.utils.data.grid.Domain
    group : h5py.Group or h5py.File
    """

    # aliases
    dtype = domain.dtype
    gny, gnx = domain.gshape

    _nplike.sync()

    # we use `require_dataset` instead of create_dataset
    group.require_dataset("grid/x/v", gnx+1, dtype, exact=True)
    group.require_dataset("grid/x/c", gnx, dtype, exact=True)
    group.require_dataset("grid/x/xf", gnx+1, dtype, exact=True)
    group.require_dataset("grid/x/yf", gnx, dtype, exact=True)
    group.require_dataset("grid/y/v", gny+1, dtype, exact=True)
    group.require_dataset("grid/y/c", gny, dtype, exact=True)
    group.require_dataset("grid/y/xf", gny, dtype, exact=True)
    group.require_dataset("grid/y/yf", gny+1, dtype, exact=True)

    group["grid/x/v"][domain.x.ibegin:domain.x.iend+1] = _nplike.get(domain.x.v)
    group["grid/x/c"][domain.x.ibegin:domain.x.iend] = _nplike.get(domain.x.c)
    group["grid/x/xf"][domain.x.ibegin:domain.x.iend+1] = _nplike.get(domain.x.xf)
    group["grid/x/yf"][domain.x.ibegin:domain.x.iend] = _nplike.get(domain.x.yf)
    group["grid/y/v"][domain.y.ibegin:domain.y.iend+1] = _nplike.get(domain.y.v)
    group["grid/y/c"][domain.y.ibegin:domain.y.iend] = _nplike.get(domain.y.c)
    group["grid/y/xf"][domain.y.ibegin:domain.y.iend] = _nplike.get(domain.y.xf)
    group["grid/y/yf"][domain.y.ibegin:domain.y.iend+1] = _nplike.get(domain.y.yf)

    # record delta (all ranks write to the same location; assume all ranks have the same delta)
    group["grid/x"].attrs["delta"] = float(domain.x.delta)
    group["grid/y"].attrs["delta"] = float(domain.y.delta)


def write_topo_to_group(topo: Topography, group: Group):
    """Write gridlines to an opened HDF5 group.

    Arguments
    ---------
    topo : torchswe.utils.data.topography.Topography
    group : h5py.Group or h5py.File
    """

    domain = topo.domain
    dtype = domain.dtype
    gny, gnx = domain.gshape

    _nplike.sync()

    # misc keywords for HDF5 datasets
    kwargs = _DummyDict()
    kwargs.exact = True
    kwargs.compression = "gzip"
    kwargs.compression_opts = 9
    kwargs.shuffle = True
    kwargs.fletcher32 = True
    kwargs.chunks = True

    group.require_dataset("topo/v", (gny+1, gnx+1), dtype, **kwargs)
    group.require_dataset("topo/c", (gny, gnx), dtype, **kwargs)
    group.require_dataset("topo/xf", (gny, gnx+1), dtype, **kwargs)
    group.require_dataset("topo/yf", (gny+1, gnx), dtype, **kwargs)
    group.require_dataset("topo/grad", (2, gny, gnx), dtype, **kwargs)

    dset = group["topo/v"]  # require this step; see h5py/h5py/issues/2017
    with dset.collective:
        dset[domain.global_v] = _nplike.get(topo.v[domain.nonhalo_v])

    dset = group["topo/c"]  # require this step; see h5py/h5py/issues/2017
    with dset.collective:
        dset[domain.global_c] = _nplike.get(topo.c[domain.nonhalo_c])

    dset = group["topo/xf"]  # require this step; see h5py/h5py/issues/2017
    with dset.collective:
        dset[domain.global_xf] = _nplike.get(topo.xf)

    dset = group["topo/yf"]  # require this step; see h5py/h5py/issues/2017
    with dset.collective:
        dset[domain.global_yf] = _nplike.get(topo.yf)

    dset = group["topo/grad"]  # require this step; see h5py/h5py/issues/2017
    with dset.collective:
        dset[(slice(None),)+domain.global_c] = _nplike.get(topo.grad)


def write_ptsource_to_group(ptsource: PointSource, group: Group):
    """Write PointSource data to an opened HDF5 group.

    Notes
    -----
    We only write data that are not related to domain decomposition adn are not in config files.

    Arguments
    ---------
    ptsource : torchswe.utils.data.source.PointSource
    group : h5py.Group or h5py.File
    """

    _nplike.sync()

    # point source (only write data unrelated to domain decomposition and not in config)
    group.require_dataset("ptsource/irate", (), int, exact=True)
    group.require_dataset("ptsource/active", (), bool, exact=True)

    # normally, only one rank contains the point source
    if ptsource is not None:
        group["ptsource/irate"][()] = ptsource.irate
        group["ptsource/active"][()] = ptsource.active


def write_frictionmodel_to_group(friction: FrictionModel, group: Group):
    """Write FrictionModel data to an opened HDF5 group.

    Notes
    -----
    We only write data that are not related to domain decomposition adn are not in config files.

    Arguments
    ---------
    friction : torchswe.utils.data.source.FrictionModel
    group : h5py.Group or h5py.File
    """
    domain = friction.domain

    _nplike.sync()

    # misc keywords for HDF5 datasets
    kwargs = _DummyDict()
    kwargs.exact = True
    kwargs.compression = "gzip"
    kwargs.compression_opts = 9
    kwargs.shuffle = True
    kwargs.fletcher32 = True
    kwargs.chunks = True

    dset = group.require_dataset("friction/roughness", domain.gshape, domain.dtype, **kwargs)

    with dset.collective:
        dset[domain.global_c] = _nplike.get(friction.roughness)


def write_states_to_group(states: States, group: Group):
    """Write states to a HDF5 group.

    Notes
    -----
    Currently we only write cell-centered conservative and non-conservative quantities.
    """

    # aliases
    domain = states.domain

    _nplike.sync()

    # misc keywords for HDF5 datasets
    kwargs = _DummyDict()
    kwargs.exact = True
    kwargs.compression = "gzip"
    kwargs.compression_opts = 9
    kwargs.shuffle = True
    kwargs.fletcher32 = True
    kwargs.chunks = True

    group.require_dataset("states/w", domain.gshape, domain.dtype, **kwargs)
    group.require_dataset("states/hu", domain.gshape, domain.dtype, **kwargs)
    group.require_dataset("states/hv", domain.gshape, domain.dtype, **kwargs)
    group.require_dataset("states/h", domain.gshape, domain.dtype, **kwargs)
    group.require_dataset("states/u", domain.gshape, domain.dtype, **kwargs)
    group.require_dataset("states/v", domain.gshape, domain.dtype, **kwargs)

    dset = group["states/w"]
    with dset.collective:
        dset[domain.global_c] = _nplike.get(states.q[(0,)+domain.nonhalo_c])

    dset = group["states/hu"]
    with dset.collective:
        dset[domain.global_c] = _nplike.get(states.q[(1,)+domain.nonhalo_c])

    dset = group["states/hv"]
    with dset.collective:
        dset[domain.global_c] = _nplike.get(states.q[(2,)+domain.nonhalo_c])

    dset = group["states/h"]
    with dset.collective:
        dset[domain.global_c] = _nplike.get(states.p[(0,)+domain.nonhalo_c])

    dset = group["states/u"]
    with dset.collective:
        dset[domain.global_c] = _nplike.get(states.p[(1,)+domain.nonhalo_c])

    dset = group["states/v"]
    with dset.collective:
        dset[domain.global_c] = _nplike.get(states.p[(2,)+domain.nonhalo_c])


def create_soln_file(states: States, runtime: DummyDict, config: Config):
    """Create a HDF5 with grid and topo info but empty states.
    """

    # aliases
    domain = states.domain

    # open a new HDF5 file and write in info that are unrelated to time marching
    with _File(runtime.outfile, "w", "mpio", comm=domain.comm) as root:
        # attributes
        root.attrs["time created"] = _datetime.now(_timezone.utc).isoformat()
        # gridlines
        write_grid_to_group(domain, root)
        # topo
        write_topo_to_group(runtime.topo, root)
        # friction
        if config.friction is not None:
            write_frictionmodel_to_group(runtime.friction, root)


def write_snapshot(states: States, runtime: DummyDict, config: Config):
    """Write a time snapshot of states to the solution file specified in config.

    If this is the first snapshot, the function also creates a new solution file.

    Arguments
    ---------
    states : torchswe.utils.data.states.States
    runtime : torchswe.utils.misc.DummyDict
    config : torchswe.utils.config.Config

    Returns
    -------
    states : torchswe.utils.data.states.States
        The same states as in the inputs. Returning it just for coding style.
    """

    if runtime.tidx == 0:  # the first time writing the solution
        create_soln_file(states, runtime, config)

    with _File(runtime.outfile, "r+", "mpio", comm=states.domain.comm) as root:
        snapshot = root.require_group(f"{runtime.tidx}")
        snapshot.attrs["dt"] = float(runtime.dt)
        snapshot.attrs["iterations"] = int(runtime.counter)
        snapshot.attrs["simulation time"] = float(runtime.cur_t)
        snapshot.attrs["time written"] = _datetime.now(_timezone.utc).isoformat()
        write_states_to_group(states, snapshot)

        if config.ptsource is not None:
            write_ptsource_to_group(runtime.ptsource, snapshot)

    return states


def read_snapshot(states: States, runtime: DummyDict, config: Config):
    """Read a time snapshot of states from the solution file specified in config.

    Only the cell-centered conservative and non-conservative quantities are read.

    Arguments
    ---------
    states : torchswe.utils.data.states.States
    runtime : torchswe.utils.misc.DummyDict
    config : torchswe.utils.config.Config

    Returns
    -------
    states : torchswe.utils.data.states.States
        The same states as in the inputs with updated values.
    runtime : torchswe.utils.misc.DummyDict
        The runtime data holder with update information for the point source.
    """

    # aliases
    domain = states.domain

    with _File(runtime.outfile, "r", "mpio", comm=domain.comm) as root:
        snapshot = root[f"{runtime.tidx}"]

        assert float(snapshot.attrs["simulation time"]) == float(runtime.cur_t)

        states.q[(0,)+domain.nonhalo_c] = _nplike.asarray(snapshot["states/w"][domain.global_c])
        states.q[(1,)+domain.nonhalo_c] = _nplike.asarray(snapshot["states/hu"][domain.global_c])
        states.q[(2,)+domain.nonhalo_c] = _nplike.asarray(snapshot["states/hv"][domain.global_c])
        states.p[(0,)+domain.nonhalo_c] = _nplike.asarray(snapshot["states/h"][domain.global_c])
        states.p[(1,)+domain.nonhalo_c] = _nplike.asarray(snapshot["states/u"][domain.global_c])
        states.p[(2,)+domain.nonhalo_c] = _nplike.asarray(snapshot["states/v"][domain.global_c])
        states.check()

        if config.ptsource is not None:
            runtime.ptsource.irate = snapshot["ptsource/irate"][()]
            runtime.ptsource.active = snapshot["ptsource/active"][()]
            runtime.ptsource.check()

        runtime.counter = int(snapshot.attrs["iterations"])

    return states, runtime


def write_debug_states(states: States, config: Config):
    """Write the local state of each MPI process to a .hdf5 file for debugging."""

    comm = states.domain.comm
    dtype = states.q.dtype

    with _File(config.case.joinpath(f"debug-states-{comm.size}.{comm.rank+1}.hdf5"), 'w') as fobj:

        # cell-centered quantities
        fobj.create_dataset("Q", states.q.shape, dtype, states.Q)
        fobj.create_dataset("U", states.p.shape, dtype, states.U)
        fobj.create_dataset("S", states.s.shape, dtype, states.S)
        fobj.create_dataset("slpx", states.slpx.shape, dtype, states.slpx)
        fobj.create_dataset("slpy", states.slpy.shape, dtype, states.slpy)

        if states.ss is not None:
            fobj.create_dataset("SS", states.ss.shape, dtype, states.SS)

        # quantities at faces facing x-direction
        fobj.create_dataset("xH", states.face.x.cf.shape, dtype, states.face.x.H)
        fobj.create_dataset("xmQ", states.face.x.minus.q.shape, dtype, states.face.x.minus.Q)
        fobj.create_dataset("xmU", states.face.x.minus.p.shape, dtype, states.face.x.minus.U)
        fobj.create_dataset("xma", states.face.x.minus.a.shape, dtype, states.face.x.minus.a)
        fobj.create_dataset("xmF", states.face.x.minus.f.shape, dtype, states.face.x.minus.F)
        fobj.create_dataset("xpQ", states.face.x.plus.q.shape, dtype, states.face.x.plus.Q)
        fobj.create_dataset("xpU", states.face.x.plus.p.shape, dtype, states.face.x.plus.U)
        fobj.create_dataset("xpa", states.face.x.plus.a.shape, dtype, states.face.x.plus.a)
        fobj.create_dataset("xpF", states.face.x.plus.f.shape, dtype, states.face.x.plus.F)

        # quantities at faces facing y-direction
        fobj.create_dataset("yH", states.face.y.cf.shape, dtype, states.face.y.H)
        fobj.create_dataset("ymQ", states.face.y.minus.q.shape, dtype, states.face.y.minus.Q)
        fobj.create_dataset("ymU", states.face.y.minus.p.shape, dtype, states.face.y.minus.U)
        fobj.create_dataset("yma", states.face.y.minus.a.shape, dtype, states.face.y.minus.a)
        fobj.create_dataset("ymF", states.face.y.minus.f.shape, dtype, states.face.y.minus.F)
        fobj.create_dataset("ypQ", states.face.y.plus.q.shape, dtype, states.face.y.plus.Q)
        fobj.create_dataset("ypU", states.face.y.plus.p.shape, dtype, states.face.y.plus.U)
        fobj.create_dataset("ypa", states.face.y.plus.a.shape, dtype, states.face.y.plus.a)
        fobj.create_dataset("ypF", states.face.y.plus.f.shape, dtype, states.face.y.plus.F)

        # the global index range of local grid
        fobj.attrs["ibg"] = states.domain.x.ibegin
        fobj.attrs["ied"] = states.domain.x.iend
        fobj.attrs["jbg"] = states.domain.y.ibegin
        fobj.attrs["jed"] = states.domain.y.iend
