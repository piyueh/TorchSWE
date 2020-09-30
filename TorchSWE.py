#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the MIT license.

"""
Main function.
"""

import os
import time
import torch
from utils.initializer import init
from utils.netcdf import write_cf, append_time_data
from src.fvm import fvm
from src.temporal import euler, RK2, RK4
from src.boundary_conditions import update_all_factory

# enforce print precision
torch.set_printoptions(precision=15, linewidth=200)

def main():
    """Main function."""

    # configuration and required data
    config, data = init()

    # the current scheme requires 2 layers of ghost cells
    Ngh = 2

    # aliases; for convenience ...
    Nx = config["discretization"]["Nx"]
    Ny = config["discretization"]["Ny"]

    # initial time step size
    dt = 1e-3

    # function to update all ghost cells in one call
    update_bc = update_all_factory(config["boundary conditions"], Ngh, data["Bf"], config["device"])

    # expand initial solution to (2, Ny+2*Ngh, Nx+2*Ngh)
    U = torch.zeros((3, Ny+2*Ngh, Nx+2*Ngh), dtype=config["dtype"], device=config["device"])
    U[:, Ngh:-Ngh, Ngh:-Ngh] = data["U0"]
    U = update_bc(U)

    # other parameters
    epsilon = config["drytol"]**4

    # initialize counter
    it = 0

    # initialize a solution file so we can append more to it
    outfile = os.path.join(config["path"], "solutions.nc")
    write_cf(outfile, data["xc"].cpu().numpy(), data["yc"].cpu().numpy(), {})

    # append initial conditions
    append_time_data(
        outfile, data["t"][0],
        {
            "w": data["U0"][0].cpu().numpy(),
            "hu": data["U0"][1].cpu().numpy(),
            "hv": data["U0"][2].cpu().numpy()
        },
        {
            "w": {"units": "m"},
            "hu": {"units": "m2 s-1"},
            "hv": {"units": "m2 s-1"}
        }
    )

    t0 = time.time()

    # temporal scheme
    if config["temporal"] == "RK4":
        TM = RK4
    elif config["temporal"] == "RK2":
        TM = RK2
    elif config["temporal"] == "euler":
        TM = euler
    else:
        raise RuntimeError

    # start running time-march until each outpu time
    for Ti in range(len(data["t"])-1):
        U, it, tc, dt = TM(
            U, update_bc, fvm, data["Bf"], data["Bc"], data["dBc"],
            data["dx"], Ngh, config["gravity"], epsilon, config["theta"],
            data["t"][Ti], data["t"][Ti+1], dt, it, 1)

        # sanity check
        assert abs(tc-data["t"][Ti+1]) < 1e-10

        # append to a NetCDF file
        append_time_data(
            outfile, data["t"][Ti+1],
            {
                "w": U[0, Ngh:-Ngh, Ngh:-Ngh].cpu().numpy(),
                "hu": U[1, Ngh:-Ngh, Ngh:-Ngh].cpu().numpy(),
                "hv": U[2, Ngh:-Ngh, Ngh:-Ngh].cpu().numpy()
            }
        )

    print("Run time (wall time): {} seconds".format(time.time()-t0))


if __name__ == "__main__":
    main()
