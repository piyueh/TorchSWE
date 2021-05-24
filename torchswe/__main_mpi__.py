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
from torchswe.utils.misc import DummyDict
from torchswe.mpi.data import get_empty_states
from torchswe.mpi.io import create_empty_soln_file, write_soln_to_file
from torchswe.mpi.initializer import init
from torchswe.mpi.boundary_conditions import get_ghost_cell_updaters
from torchswe.mpi.temporal import euler, ssprk2, ssprk3
from torchswe.core.initializer import get_cmd_arguments
from torchswe.core.fvm import fvm

# enforce print precision
nplike.set_printoptions(precision=15, linewidth=200)


def setup_logger(rank, log_level, log_file):
    """Setup logger."""

    # just for our convenience
    log_opts = {"quiet": logging.ERROR, "normal": logging.INFO, "debug": logging.DEBUG}

    # setup the top-level logger
    logger = logging.getLogger("torchswe")

    if log_file is not None:
        logger.setLevel(log_opts[log_level])
        log_file = log_file.expanduser().resolve()
        log_file = log_file.with_name(log_file.name+".proc.{:02d}".format(rank))
        logger.addHandler(logging.FileHandler(log_file, "w"))
        logger.handlers[-1].setFormatter(logging.Formatter(
            "%(asctime)s %(name)s %(funcName)s [%(levelname)s] %(message)s", "%m-%d %H:%M:%S"))
    else:
        if rank == 0 or log_level == "debug":
            logger.setLevel(log_opts[log_level])
        else:
            logger.setLevel(logging.WARNING)

        logger.addHandler(logging.StreamHandler())
        logger.handlers[-1].setFormatter(logging.Formatter(
            "[Rank {}] %(asctime)s %(message)s".format(rank), "%H:%M:%S"))

    # return the logger for this file
    logger = logging.getLogger("torchswe.main")
    return logger


def main():
    """Main function."""

    # mpi communicator and rank
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # get CMD arguments
    args = get_cmd_arguments()

    # setup loggier
    logger = setup_logger(rank, args.log_level, args.log_file)
    logger.info("Done parsing CMD arguments and setting up the logging system.")
    logger.info("The np-like backend is: %s", nplike.__name__)

    if nplike.__name__ == "legate.numpy":
        logger.info("Using nplike.where code path")
    else:
        logger.info("Using advanced-indexing code path")

    # configuration and required data
    config, grid, topo, ic_data = init(comm, args)
    logger.info("Done initializing.")

    # initialize an empty solution/states object
    soln = get_empty_states(comm, *config.spatial.discretization, config.params.ngh, config.dtype)
    logger.info("Done creating an empty state holder.")

    # copy initial data to the solution holder
    slc = slice(config.params.ngh, -config.params.ngh)  # slice indicating the non-ghost cells
    soln.q.w[slc, slc], soln.q.hu[slc, slc], soln.q.hv[slc, slc] = ic_data.w, ic_data.hu, ic_data.hv
    logger.info("Done applying initial conditions.")

    # runtime holding things not available in config.yaml or may change during runtime
    runtime = DummyDict()  # it's just a dict and not a data model. so, no data validation
    runtime.dt = config.temporal.dt  # time step size; may be changed during runtime
    runtime.cur_t = grid.t[0]  # the current simulation time
    runtime.next_t = None  # next output time; will be set later
    runtime.counter = 0  # to count the current number of iterations
    runtime.epsilon = config.params.drytol**4  # tolerance when dealing almost-dry cells
    runtime.tol = 1e-12  # up to how big can be treated as zero

    # function to calculate right-hand-side
    runtime.rhs_updater = fvm

    # get the callable to update ghost cells
    runtime.ghost_updater = get_ghost_cell_updaters(config.bc, soln, topo)
    logger.info("Done setting runtime data.")

    # update ghost cells
    soln = runtime.ghost_updater(soln)
    logger.info("Done updating ghost cells.")

    # select time marching function
    marching = {"Euler": euler, "SSP-RK2": ssprk2, "SSP-RK3": ssprk3}  # available options
    marching = marching[config.temporal.scheme]  # don't need the origianl dict anymore

    # create an NetCDF file and append I.C.
    if not "no save" in config.temporal.output[0]:
        outfile = config.case.joinpath("solutions.nc")  # initialize an empty solution file
        create_empty_soln_file(outfile, grid)
        write_soln_to_file(outfile, grid, soln.q, grid.t[0], 0, soln.ngh)
        logger.info("Done writing the initial solution to the NetCDF file.")
    else:
        logger.info("No need to save data for \"no save\" method.")

    # initialize counter and timing variable
    perf_t0 = time.time()  # suppose to be wall time
    logger.info("Time marching starts at %s", time.ctime(perf_t0))

    # start running time-march until each output time
    for tidx, runtime.next_t in enumerate(grid.t[1:]):
        logger.info("Marching from T=%s to T=%s", runtime.cur_t, runtime.next_t)
        soln = marching(soln, grid, topo, config, runtime)

        # sanity check for the current time
        assert abs(runtime.next_t-runtime.cur_t) < 1e-10

        # append to the NetCDF file
        if not "no save" in config.temporal.output[0]:
            write_soln_to_file(outfile, grid, soln.q, runtime.next_t, tidx+1, soln.ngh)
            logger.info("Done writing the solution at T=%s to the NetCDF file.", runtime.next_t)

    logger.info("Done time marching.")
    logger.info("Run time (wall time): %s seconds", time.time()-perf_t0)
    logger.info("Program ends now.")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
