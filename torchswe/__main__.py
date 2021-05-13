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
from torchswe.utils.dummy import DummyDict
from torchswe.utils.data import States
from torchswe.utils.netcdf import write_cf, append_time_data
from torchswe.core.initializer import init, get_cmd_arguments
from torchswe.core.fvm import fvm
from torchswe.core.boundary_conditions import BoundaryGhostUpdater
from torchswe.core.temporal import euler, ssprk2, ssprk3

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

    # object to update ghost cells
    runtime.ghost_updater = BoundaryGhostUpdater(
        config.bc, config.spatial.discretization[0], config.spatial.discretization[1],
        config.params.ngh, topo)
    logger.info("Done setting runtime data.")

    # initialize an empty solution/states object
    slc = slice(config.params.ngh, -config.params.ngh)  # slice indicating the non-ghost cells
    soln = States(
        config.spatial.discretization[0], config.spatial.discretization[1],
        config.params.ngh, config.dtype)
    soln.q.w[slc, slc], soln.q.hu[slc, slc], soln.q.hv[slc, slc] = ic_data.w, ic_data.hu, ic_data.hv
    soln = runtime.ghost_updater.update_all(soln)
    logger.info("Done creating initial state holder.")

    # select time marching function
    marching = {"Euler": euler, "SSP-RK2": ssprk2, "SSP-RK3": ssprk3}  # available options
    marching = marching[config.temporal.scheme]  # don't need the origianl dict anymore

    # create an NetCDF file and append I.C.
    if config.temporal.output[0] != "t_start t_end no save":
        outfile = config.case.joinpath("solutions.nc")  # initialize an empty solution file
        write_cf(outfile, {"x": grid.x.cntr, "y": grid.y.cntr}, {})  # empty file
        logger.info("Done creating an empty NetCDF file for solutions.")

        append_time_data(  # the first t index is supposed to be grid.t[0]
            outfile, grid.t[0], {"w": ic_data.w, "hu": ic_data.hu, "hv": ic_data.hv},
            {"w": {"units": "m"}, "hu": {"units": "m2 s-1"}, "hv": {"units": "m2 s-1"}})
        logger.info("Done writing the initial solution to the NetCDF file.")
    else:
        logger.info("No need to save data for \"no save\" method.")

    # initialize counter and timing variable
    perf_t0 = time.time()  # suppose to be wall time
    logger.info("Time marching starts at %s", time.ctime(perf_t0))

    # start running time-march until each output time
    for runtime.next_t in grid.t[1:]:
        logger.info("Marching from T=%s to T=%s", runtime.cur_t, runtime.next_t)
        soln = marching(soln, grid, topo, config, runtime)

        # sanity check for the current time
        assert abs(runtime.next_t-runtime.cur_t) < 1e-10

        # append to the NetCDF file
        if config.temporal.output[0] != "t_start t_end no save":
            append_time_data(
                outfile, runtime.next_t,
                {"w": soln.q.w[slc, slc], "hu": soln.q.hu[slc, slc], "hv": soln.q.hv[slc, slc]})
            logger.info("Done writing the solution at T=%s to the NetCDF file.", runtime.next_t)

    logger.info("Done time marching.")
    logger.info("Run time (wall time): %s seconds", time.time()-perf_t0)
    logger.info("Program ends now.")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
