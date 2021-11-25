#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Main function.
"""
# pylint: disable=wrong-import-position

import time
import logging

# due to openmpi's problematic implementation of one-sided communication
import mpi4py
mpi4py.rc.threads = False

from mpi4py import MPI
from torchswe import nplike
from torchswe.utils.init import get_cmd_arguments
from torchswe.utils.init import get_config
from torchswe.utils.init import get_timeline
from torchswe.utils.init import get_initial_states_from_config
from torchswe.utils.init import get_initial_states_from_snapshot
from torchswe.utils.init import get_topography_from_file
from torchswe.utils.init import get_pointsource
from torchswe.utils.init import get_friction_roughness
from torchswe.utils.misc import DummyDict
from torchswe.utils.misc import set_device
from torchswe.utils.misc import exchange_states
from torchswe.utils.io import create_empty_soln_file, write_soln_to_file
from torchswe.utils.friction import bellos_et_al_2018
from torchswe.kernels import reconstruct_cell_centers
from torchswe.bcs import get_ghost_cell_updaters
from torchswe.temporal import euler, ssprk2, ssprk3
from torchswe.sources import topography_gradient, point_mass_source, friction, zero_stiff_terms

# enforce print precision
nplike.set_printoptions(precision=15, linewidth=200)

# available time marching options
MARCHING_OPTIONS = {"Euler": euler, "SSP-RK2": ssprk2, "SSP-RK3": ssprk3}  # available options

# available friction coefficient models
FRICTION_MODELS = {"bellos_et_al_2018": bellos_et_al_2018}


def init(comm, args=None):
    """Initialize a simulation and read configuration.

    Attributes
    ----------
    comm : mpi4py.MPI.Comm
        The communicator.
    args : None or argparse.Namespace
        By default, None means getting arguments from command-line.

    Returns
    -------
    config : a torchswe.utils.config.Config
        A Config instance holding a case's simulation configurations.
    topo : torch.utils.data.Topography
        Topography elevation data.
    states : torch.utils.data.States
        Solutions and associated domain information.
    times : torchswe.utils.data.Timeline
        Times for saving solution snapshot.
    logger : logging.Logger
        Python's logging utility object.
    """

    # MPI size & rank
    size = comm.Get_size()
    rank = comm.Get_rank()

    # get cmd arguments
    if args is None:
        args = get_cmd_arguments()

    # setup the top-level (i.e., package-level/torchswe) logger
    logger = logging.getLogger("torchswe")

    if args.log_file is not None:
        # different ranks write to different log files
        if size != 1:
            args.log_file = args.log_file.with_name(args.log_file.name+f".proc.{rank:02d}")

        fmt = "%(asctime)s %(name)s %(funcName)s [%(levelname)s] %(message)s"  # format
        logger.setLevel(args.log_level)
        logger.addHandler(logging.FileHandler(args.log_file, "w"))
        logger.handlers[-1].setFormatter(logging.Formatter(fmt, "%m-%d %H:%M:%S"))
    else:
        if args.log_level == logging.INFO:
            if rank == 0:
                logger.setLevel(logging.INFO)
            else:
                logger.setLevel(logging.WARNING)
        else:
            logger.setLevel(args.log_level)

        fmt = f"[Rank {rank}] %(asctime)s %(message)s"
        logger.addHandler(logging.StreamHandler())
        logger.handlers[-1].setFormatter(logging.Formatter(fmt, "%H:%M:%S"))

    # make the final & returned logger refer to this specific file (main function)
    logger = logging.getLogger("torchswe.main")

    # log the backend
    logger.info("The np-like backend is: %s", nplike.__name__)

    # set GPU id
    if nplike.__name__ == "cupy":
        set_device(comm)

    # get configuration
    config = get_config(args)

    return config, logger


def config_runtime(comm, config, logger):
    """Configure a runtime object.
    """

    # get initial solution object
    states = get_initial_states_from_config(comm, config)
    logger.info("Obtained an initial solution object")

    # `runtime` holding things not available in config.yaml or may change during runtime
    runtime = DummyDict()  # it's just a dict and not a data model. so, no data validation

    # get temporal axis
    runtime.times = get_timeline(config.temporal.output, config.temporal.dt)
    logger.info("Obtained a Timeline object")

    # get dem (digital elevation model); assume dem values defined at cell centers
    runtime.topo = get_topography_from_file(config.topo.file, config.topo.key, states.domain)
    logger.info("Obtained a Topography object")

    # make sure initial depths are non-negative
    states.Q[(0,)+states.domain.internal] = nplike.maximum(
        runtime.topo.centers[states.domain.internal], states.Q[(0,)+states.domain.internal])
    states.check()

    runtime.dt = config.temporal.dt  # time step size; may be changed during runtime
    logger.info("Initial dt: %e", runtime.dt)

    runtime.cfl = 0.9
    logger.info("dt adaptive ratio: %e", runtime.cfl)

    runtime.dt_constraint = float("inf")
    logger.info("Initial dt constraint: %e", runtime.dt_constraint)

    runtime.tidx = 0
    logger.info("Initial output time index: %d", runtime.tidx)

    runtime.cur_t = runtime.times[0]  # the current simulation time
    logger.info("Initial t: %e", runtime.cur_t)

    runtime.next_t = runtime.times[1]
    logger.info("The next t: %s", runtime.next_t)

    runtime.counter = 0  # to count the current number of iterations
    logger.info("The current iteration counter: %d", runtime.counter)

    runtime.tol = 1e-12  # up to how big can be treated as zero
    logger.info("Tolerance: %e", runtime.tol)

    runtime.outfile = config.case.joinpath("solutions.nc")  # solution file
    logger.info("Output solution file: %s", str(runtime.outfile))

    runtime.marching = MARCHING_OPTIONS[config.temporal.scheme]  # time marching scheme
    logger.info("Time marching scheme: %s", config.temporal.scheme)

    runtime.gh_updater = get_ghost_cell_updaters(
        states, runtime.topo, config.bc, config.params.theta, runtime.tol, config.params.drytol)
    logger.info("Done setting ghost cell updaters")

    runtime.sources = [topography_gradient]
    logger.info("Explicit source term: topography gradients")

    if config.ptsource is not None:
        # add the function of calculating point source
        runtime.sources.append(point_mass_source)
        logger.info("Explicit source term: point source")

        # get a PointSource instance
        runtime.ptsource = get_pointsource(config.ptsource, states.domain, 0)
        logger.info("Obtained a point source object: %s", runtime.ptsource)

    runtime.stiff_sources = []
    if config.friction is not None:
        runtime.roughness = get_friction_roughness(states.domain, config.friction)
        logger.info("Friction roughness used")

        runtime.fc_model = FRICTION_MODELS[config.friction.model]
        logger.info("Friction coefficient model: %s", config.friction.model)

        runtime.stiff_sources.append(zero_stiff_terms)
        logger.info("Re-initialization fucntion of SS added to stiff source terms")

        runtime.stiff_sources.append(friction)
        logger.info("Friction fucntion added to stiff source terms")

    return states, runtime


def restart(states, runtime, cont, logger):
    """Update data if we're continue from a previous solution."""

    if cont is None:  # not restarting
        return states, runtime

    try:
        runtime.tidx = runtime.times.values.index(cont)
        logger.info("Initial output time index: %d", runtime.tidx)
    except ValueError as err:
        if "not in tuple" not in str(err):  # other kinds of ValueError
            raise
        raise ValueError(
            f"Target restarting time {cont} was not found in {runtime.times.values}"
        ) from err

    # update current time
    runtime.cur_t = cont
    logger.info("Initial t: %s", runtime.cur_t)

    # update next time
    runtime.next_t = runtime.times.values[runtime.tidx+1]
    logger.info("The next t: %s", runtime.next_t)

    # make the counter non-zero to avoid some functions using counter == 0 as condition
    runtime.counter = 1
    logger.info("The counter: %d", runtime.counter)

    # update initial solution
    states = get_initial_states_from_snapshot(runtime.outfile, runtime.tidx, states)

    # update point source timer
    if runtime.ptsource is not None:
        runtime.ptsource.irate = int(nplike.searchsorted(
            nplike.array(runtime.ptsource.times), nplike.array(runtime.cur_t), "left"))
        runtime.ptsource.active = (not runtime.ptsource.irate == len(runtime.ptsource.times))
        logger.info("Point source reset: %s", runtime.ptsource)

    return states, runtime


def main():
    """Main function."""

    # mpi communicator
    comm = MPI.COMM_WORLD

    # get CMD arguments
    args = get_cmd_arguments()

    # initialize
    config, logger = init(comm, args)
    logger.info("Done initialization.")

    # states and runtime data
    soln, runtime = config_runtime(comm, config, logger)
    logger.info("Done configuring runtime.")

    # update data if this is a continued run
    soln, runtime = restart(soln, runtime, args.cont, logger)

    # create an NetCDF file and append I.C.
    if runtime.times.save and runtime.tidx == 0:
        # exchange halo information and calculate cell-centered depths
        soln = exchange_states(soln)
        soln = reconstruct_cell_centers(soln, runtime, config)

        # write
        create_empty_soln_file(runtime.outfile, soln.domain, runtime.times)
        write_soln_to_file(runtime.outfile, soln, runtime.times[0], 0)
        logger.info("Done writing the initial solution to the NetCDF file.")
    else:
        logger.info("No need to save data for \"no save\" method or for a continued run.")

    # initialize counter and performance profiling variable
    perf_t0 = time.time()  # suppose to be wall time
    logger.info("Time marching starts at %s", time.ctime(perf_t0))

    # start running time marching until each output time
    for runtime.next_t in runtime.times[runtime.tidx+1:]:
        logger.info("Marching from T=%s to T=%s", runtime.cur_t, runtime.next_t)
        soln = runtime.marching(soln, runtime, config)

        # sanity check for the current time
        assert abs(runtime.next_t-runtime.cur_t) < 1e-10

        # update tidx
        runtime.tidx += 1

        # append to the NetCDF file
        if runtime.times.save:
            write_soln_to_file(runtime.outfile, soln, runtime.next_t, runtime.tidx)
            logger.info("Done writing the solution at T=%s to the NetCDF file.", runtime.next_t)

    logger.info("Done time marching.")
    logger.info("Run time (wall time): %s seconds", time.time()-perf_t0)
    logger.info("Program ends now.")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
