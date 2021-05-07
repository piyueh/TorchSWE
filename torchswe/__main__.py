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
from torchswe import nplike
from torchswe.utils.dummydict import DummyDict
from torchswe.utils.data import States
from torchswe.utils.netcdf import write_cf, append_time_data
from torchswe.core.initializer import init, get_cmd_arguments
from torchswe.core.fvm import fvm
from torchswe.core.boundary_conditions import BoundaryGhostUpdater
from torchswe.core.temporal import euler, RK2, RK4

# enforce print precision
nplike.set_printoptions(precision=15, linewidth=200)


def setup_logger(log_level, log_file):
    """Setup logger."""

    # just for our convenience
    log_opts = {"quiet": logging.ERROR, "normal": logging.INFO, "debug": logging.DEBUG}

    # setup the top-level logger
    logger = logging.getLogger("torchswe")
    logger.setLevel(log_opts[log_level])

    if log_file is not None:
        logger.addHandler(logging.FileHandler(log_file.expanduser().resolve(), "w"))
        logger.handlers[-1].setFormatter(logging.Formatter(
            "%(asctime)s %(name)s %(funcName)s [%(levelname)s] %(message)s", "%m-%d %H:%M:%S"))
    else:
        logger.addHandler(logging.StreamHandler())
        logger.handlers[-1].setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))

    # return the logger for this file
    logger = logging.getLogger("torchswe.main")
    return logger


def main():
    """Main function."""

    # get CMD arguments
    args = get_cmd_arguments()

    # setup loggier
    logger = setup_logger(args.log_level, args.log_file)
    logger.info("Done parsing CMD arguments and setting up the logging system.")
    logger.info("The np-like backend is: %s", nplike.__name__)

    # configuration and required data
    config, grid, topo, ic_data = init(args)
    logger.info("Done initializing.")

    # runtime holding things not determined from config.yaml and my change during runtime
    # it's just a dict and not a data model. so, no data validation
    runtime = DummyDict()
    runtime.dt = 1e-3  # initial time step size
    runtime.epsilon = config.params.drytol**4  # tolerance when dealing almost-dry cells
    runtime.cur_t = config.temporal.start  # the current simulation time
    runtime.counter = 0  # to count the current number of iterations
    runtime.tol = 1e-12  # can be treated as zero
    runtime.rhs_updater = fvm

    # object to update ghost cells
    runtime.ghost_updater = BoundaryGhostUpdater(
        config.bc, config.spatial.discretization[0], config.spatial.discretization[1],
        config.params.ngh, topo)
    logger.info("Done getting a ghost cell updater.")

    # slice indicating the non-ghost cells
    slc = slice(config.params.ngh, -config.params.ngh)

    # solution object
    soln = States(
        config.spatial.discretization[0], config.spatial.discretization[1],
        config.params.ngh, config.dtype)

    # copy I.C.
    soln.q.w[slc, slc], soln.q.hu[slc, slc], soln.q.hv[slc, slc] = ic_data.w, ic_data.hu, ic_data.hv
    soln = runtime.ghost_updater.update_all(soln)
    logger.info("Done creating and initializing the state holder.")

    # select time marching function
    marching = {"Euler": euler, "RK2": RK2, "RK4": RK4}[config.temporal.scheme]

    # in case users don't want to write any output
    # =============================================================================================
    if len(grid.t) == 0:  # just run, and no output
        perf_t0 = time.time()  # suppose to be wall time
        logger.info("Time marching starts at %s", time.ctime(perf_t0))
        soln = marching(soln, grid, topo, config, runtime, config.temporal.end)
        logger.info("Done time marching.")
        logger.info("Run time (wall time): %s seconds", time.time()-perf_t0)
        logger.info("Program ends now.")
        return 0

    # otherside, we need to output I.C. append initial conditions to solution file
    # =============================================================================================
    outfile = config.case.joinpath("solutions.nc")  # initialize an empty solution file
    write_cf(outfile, {"x": grid.x.cntr, "y": grid.y.cntr}, {})  # empty file
    logger.info("Done creating an empty NerCDF file for solutions.")

    append_time_data(  # the first t index is supposed to be config.temporal.start
        outfile, grid.t[0], {"w": ic_data.w, "hu": ic_data.hu, "hv": ic_data.hv},
        {"w": {"units": "m"}, "hu": {"units": "m2 s-1"}, "hv": {"units": "m2 s-1"}})
    logger.info("Done writing the initial solution to the NetCDF file.")

    # initialize counter and timing variable
    perf_t0 = time.time()  # suppose to be wall time
    logger.info("Time marching starts at %s", time.ctime(perf_t0))

    # start running time-march until each output time
    for tend in grid.t[1:]:
        logger.info("Marching from T=%s to T=%s", runtime.cur_t, tend)
        soln = marching(soln, grid, topo, config, runtime, tend)

        # sanity check of the current time
        assert abs(tend-runtime.cur_t) < 1e-10

        # append to a NetCDF file
        append_time_data(
            outfile, tend,
            {"w": soln.q.w[slc, slc], "hu": soln.q.hu[slc, slc], "hv": soln.q.hv[slc, slc]})
        logger.info("Done writing the solution at T=%s to the NetCDF file.", tend)

    logger.info("Run time (wall time): %s seconds", time.time()-perf_t0)
    logger.info("Program ends now.")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
