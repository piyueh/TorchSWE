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
from torchswe.utils.init import get_cmd_arguments, get_config, get_initial_objects
from torchswe.utils.misc import DummyDict
from torchswe.utils.misc import set_device
from torchswe.utils.io import create_empty_soln_file, write_soln_to_file
from torchswe.core.boundary_conditions import get_ghost_cell_updaters
from torchswe.core.temporal import euler, ssprk2, ssprk3
from torchswe.core.fvm import fvm

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
            args.log_file = args.log_file.with_name(args.log_file.name+".proc.{:02d}".format(rank))

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

        fmt = "[Rank {}] %(asctime)s %(message)s".format(rank)
        logger.addHandler(logging.StreamHandler())
        logger.handlers[-1].setFormatter(logging.Formatter(fmt, "%H:%M:%S"))

    # make the final & returned logger refer to this specific file (main function)
    logger = logging.getLogger("torchswe.main")

    # set GPU id
    if nplike.__name__ == "cupy":
        set_device(comm)

    # get configuration
    config = get_config(args)

    # spatial discretization + output time values
    topo, states, times = get_initial_objects(comm, config)

    # make sure initial depths are non-negative
    states.q.w = nplike.maximum(topo.centers, states.q.w)

    return config, topo, states, times, logger


def main():
    """Main function."""

    # mpi communicator
    comm = MPI.COMM_WORLD

    # get CMD arguments
    args = get_cmd_arguments()

    # initialize
    config, topo, soln, times, logger = init(comm, args)
    logger.info("Done initialization.")
    logger.info("The np-like backend is: %s", nplike.__name__)

    # `runtime` holding things not available in config.yaml or may change during runtime
    runtime = DummyDict()  # it's just a dict and not a data model. so, no data validation
    runtime.dt = config.temporal.dt  # time step size; may be changed during runtime
    runtime.cur_t = times[0]  # the current simulation time
    runtime.next_t = None  # next output time; will be set later
    runtime.counter = 0  # to count the current number of iterations
    runtime.epsilon = config.params.drytol**4  # tolerance when dealing almost-dry cells
    runtime.tol = 1e-12  # up to how big can be treated as zero
    runtime.rhs_updater = fvm  # function to calculate right-hand-side
    runtime.gh_updater = get_ghost_cell_updaters(config.bc, soln, topo)  # ghost cell updater
    runtime.marching = MARCHING_OPTIONS[config.temporal.scheme]  # time marching scheme
    runtime.outfile = config.case.joinpath("solutions.nc")  # solution file
    logger.info("Done setting runtime data.")

    # update ghost cells
    soln = runtime.ghost_updater(soln)
    logger.info("Done updating ghost cells.")

    # create an NetCDF file and append I.C.
    if times.save:
        create_empty_soln_file(runtime.outfile, soln.domain, times)
        write_soln_to_file(runtime.outfile, soln.domain, soln.q, times[0], 0, soln.ngh)
        logger.info("Done writing the initial solution to the NetCDF file.")
    else:
        logger.info("No need to save data for \"no save\" method.")

    # initialize counter and performance profiling variable
    perf_t0 = time.time()  # suppose to be wall time
    logger.info("Time marching starts at %s", time.ctime(perf_t0))

    # start running time marching until each output time
    for tidx, runtime.next_t in zip(range(1, len(times)), times[1:]):
        logger.info("Marching from T=%s to T=%s", runtime.cur_t, runtime.next_t)
        soln = runtime.marching(soln, soln.domain, topo, config, runtime)

        # sanity check for the current time
        assert abs(runtime.next_t-runtime.cur_t) < 1e-10

        # append to the NetCDF file
        if times.save:
            write_soln_to_file(runtime.outfile, soln.domain, soln.q, runtime.next_t, tidx, soln.ngh)
            logger.info("Done writing the solution at T=%s to the NetCDF file.", runtime.next_t)

    logger.info("Done time marching.")
    logger.info("Run time (wall time): %s seconds", time.time()-perf_t0)
    logger.info("Program ends now.")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
