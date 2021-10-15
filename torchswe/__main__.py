#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Main function.
"""

import time
import logging

from mpi4py import MPI
from torchswe import nplike
from torchswe.utils.init import get_cmd_arguments
from torchswe.utils.init import get_config
from torchswe.utils.init import get_timeline
from torchswe.utils.init import get_initial_states_from_config
from torchswe.utils.init import get_topography_from_file
from torchswe.utils.init import get_pointsource
from torchswe.utils.misc import DummyDict
from torchswe.utils.misc import set_device
from torchswe.utils.io import create_empty_soln_file, write_soln_to_file
from torchswe.core.boundary_conditions import get_ghost_cell_updaters
from torchswe.core.temporal import euler, ssprk2, ssprk3
from torchswe.core.sources import topography_gradient, point_mass_source

# enforce print precision
nplike.set_printoptions(precision=15, linewidth=200)

# available time marching options
MARCHING_OPTIONS = {"Euler": euler, "SSP-RK2": ssprk2, "SSP-RK3": ssprk3}  # available options


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
    logger.debug("Obtained an initial solution object")

    # `runtime` holding things not available in config.yaml or may change during runtime
    runtime = DummyDict()  # it's just a dict and not a data model. so, no data validation

    # get temporal axis
    runtime.times = get_timeline(config.temporal.output, config.temporal.dt)
    logger.debug("Obtained a Timeline object")

    # get dem (digital elevation model); assume dem values defined at cell centers
    runtime.topo = get_topography_from_file(config.topo.file, config.topo.key, states.domain)
    logger.debug("Obtained a Topography object")

    # make sure initial depths are non-negative
    states.q.w[states.ngh:-states.ngh, states.ngh:-states.ngh] = nplike.maximum(
        runtime.topo.centers, states.q.w[states.ngh:-states.ngh, states.ngh:-states.ngh])
    states.check()

    runtime.dt = config.temporal.dt  # time step size; may be changed during runtime
    logger.debug("Initial dt: %e", runtime.dt)

    runtime.cfl = 0.95
    logger.debug("dt adaptive ratio: %e", runtime.cfl)

    runtime.dt_constraint = float("inf")
    logger.debug("Initial dt constraint: %e", runtime.dt_constraint)

    runtime.cur_t = runtime.times[0]  # the current simulation time
    logger.debug("Initial t: %e", runtime.cur_t)

    runtime.next_t = None  # next output time; will be set later
    logger.debug("The next t: %s", runtime.next_t)

    runtime.counter = 0  # to count the current number of iterations
    logger.debug("The current iteration counter: %d", runtime.counter)

    runtime.epsilon = config.params.drytol**4  # tolerance when dealing almost-dry cells
    logger.debug("Epsilon: %e", runtime.epsilon)

    runtime.tol = 1e-12  # up to how big can be treated as zero
    logger.debug("Tolerance: %e", runtime.tol)

    runtime.outfile = config.case.joinpath("solutions.nc")  # solution file
    logger.debug("Output solution file: %s", str(runtime.outfile))

    runtime.marching = MARCHING_OPTIONS[config.temporal.scheme]  # time marching scheme
    logger.debug("Time marching scheme: %s", config.temporal.scheme)

    runtime.gh_updater = get_ghost_cell_updaters(config.bc, states, runtime.topo)
    logger.debug("Done setting ghost cell updaters")

    runtime.sources = [topography_gradient]
    logger.debug("Explicit source term: topography gradients")

    if config.ptsource is not None:
        # get a PointSource instance
        runtime.ptsource = get_pointsource(
            config.ptsource.loc[0], config.ptsource.loc[1], config.ptsource.times,
            config.ptsource.rates, states.domain, 0)
        logger.debug("Setting a point source: %s", runtime.ptsource)

        # add the function of calculating point source
        runtime.sources.append(point_mass_source)
        logger.debug("Explicit source term: point source")


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

    # update ghost cells
    soln = runtime.gh_updater(soln)
    logger.info("Done updating ghost cells.")

    # create an NetCDF file and append I.C.
    if runtime.times.save:
        create_empty_soln_file(runtime.outfile, soln.domain, runtime.times)
        write_soln_to_file(runtime.outfile, soln.domain, soln.q, runtime.times[0], 0, soln.ngh)
        logger.info("Done writing the initial solution to the NetCDF file.")
    else:
        logger.info("No need to save data for \"no save\" method.")

    # initialize counter and performance profiling variable
    perf_t0 = time.time()  # suppose to be wall time
    logger.info("Time marching starts at %s", time.ctime(perf_t0))

    # start running time marching until each output time
    for tidx, runtime.next_t in zip(range(1, len(runtime.times)), runtime.times[1:]):
        logger.info("Marching from T=%s to T=%s", runtime.cur_t, runtime.next_t)
        soln = runtime.marching(soln, runtime, config)

        # sanity check for the current time
        assert abs(runtime.next_t-runtime.cur_t) < 1e-10

        # append to the NetCDF file
        if runtime.times.save:
            write_soln_to_file(runtime.outfile, soln.domain, soln.q, runtime.next_t, tidx, soln.ngh)
            logger.info("Done writing the solution at T=%s to the NetCDF file.", runtime.next_t)

    logger.info("Done time marching.")
    logger.info("Run time (wall time): %s seconds", time.time()-perf_t0)
    logger.info("Program ends now.")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
