#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the MIT license.

"""
Time-marching.
"""
import warnings
import torch
from .misc import CFLWarning


def euler(U, update_ghost, rhs_fun, Bf, Bc, dBc, dx, Ngh, g, epsilon, theta,
          t_current, t_end, dt, it_current=0, print_steps=1, tol=1e-10):
    """A simple 1st-order forward Euler time-marching.
    """

    # total soil volume
    soil_vol = Bc.sum().item() * dx * dx

    # used to store maximum allowed time step size
    max_dt = None

    # information string formatter
    info_str = "Step {}: step size = {} sec, time = {} sec, total volume = {}"
    cfl_str1 = "Current dt (= {} sec)s is not safe, "
    cfl_str2 = "lower down to {} sec"

    # an initial updating, just in case
    U = update_ghost(U)

    # loop till t_current reaches t_end
    while True: # TODO: use iteration counter to avoid infinity loop

        # Euler step
        k, max_dt = rhs_fun(U, Bf, Bc, dBc, dx, Ngh, g, epsilon, theta)

        # adjust time step size; modify dt if next step will exceed target end time
        if t_current + dt > t_end:
            dt = t_end - t_current
        else:
            dt = max_dt * 0.9

        # update
        U[:, Ngh:-Ngh, Ngh:-Ngh] += dt * k
        U = update_ghost(U)

        # update iteration index and time
        it_current += 1
        t_current += dt

        # print out information
        if it_current % print_steps == 0:
            fluid_vol = U[0, Ngh:-Ngh, Ngh:-Ngh].sum().item() * dx * dx - soil_vol
            print(info_str.format(it_current, dt, t_current, fluid_vol))

        # break loop
        if abs(t_current-t_end) < tol:
            break

    return U, it_current, t_current, dt

def RK4(U, update_ghost, rhs_fun, Bf, Bc, dBc, dx, Ngh, g, epsilon, theta,
        t_current, t_end, dt, it_current=0, print_steps=1, tol=1e-10):
    """Commonly seen explicit 4th-order Runge-Kutta scheme.
    """

    # total soil volume
    soil_vol = Bc.sum().item() * dx * dx

    # used to store maximum allowed time step size
    max_dt = [None, None, None, None]

    # information string formatter
    info_str = "Step {}: step size = {} sec, time = {} sec, total volume = {}"
    cfl_str1 = "Current dt (= {} sec)s is not safe, "
    cfl_str2 = "lower down to {} sec"
    cfl_str3 = "lower down to {} sec and restart the iteration {}"

    # previous solution; should not be changed until the end of each time-step
    U = update_ghost(U)

    # to hold temporary solution at intermediate steps
    Utemp = torch.zeros_like(U)

    # loop till t_current reaches t_end
    while True: # TODO: use iteration counter to avoid infinity loop

        # the slope from the first step, using U
        k1, max_dt[0] = rhs_fun(U, Bf, Bc, dBc, dx, Ngh, g, epsilon, theta)

        # check if it is safe to update conservative variables
        if dt / 2. >= max_dt[0]:
            msg = cfl_str1.format(dt)
            dt = max_dt[0] * 0.9
            msg += cfl_str2.format(dt)
            warnings.warn(msg, CFLWarning)

        # update for the first step
        Utemp[:, Ngh:-Ngh, Ngh:-Ngh] = U[:, Ngh:-Ngh, Ngh:-Ngh] + dt * k1 / 2.
        Utemp = update_ghost(Utemp)

        # the slope from the second step, using Utemp
        k2, max_dt[1] = rhs_fun(Utemp, Bf, Bc, dBc, dx, Ngh, g, epsilon, theta)

        # check if it is safe to update conservative variables
        if dt / 2. >= max_dt[1]:
            msg = cfl_str1.format(dt)
            dt = max_dt[1] * 0.9
            msg += cfl_str3.format(dt, it_current)
            warnings.warn(msg, CFLWarning)
            continue # restart the iteration to get correct soln from previos steps

        # update for the second step
        Utemp[:, Ngh:-Ngh, Ngh:-Ngh] = U[:, Ngh:-Ngh, Ngh:-Ngh] + dt * k2 / 2.
        Utemp = update_ghost(Utemp)

        # the slope from the second step, using Utemp
        k3, max_dt[2] = rhs_fun(Utemp, Bf, Bc, dBc, dx, Ngh, g, epsilon, theta)

        # check if it is safe to update conservative variables
        if dt >= max_dt[2]:
            msg = cfl_str1.format(dt)
            dt = max_dt[2] * 0.9
            msg += cfl_str3.format(dt, it_current)
            warnings.warn(msg, CFLWarning)
            continue # restart the iteration to get correct U from previos steps

        # update for the third step
        Utemp[:, Ngh:-Ngh, Ngh:-Ngh] = U[:, Ngh:-Ngh, Ngh:-Ngh] + dt * k3
        Utemp = update_ghost(Utemp)

        # the final step, using Utemp
        k4, max_dt[3] = rhs_fun(Utemp, Bf, Bc, dBc, dx, Ngh, g, epsilon, theta)

        # update U directly because we reach the end of this time step
        U[:, Ngh:-Ngh, Ngh:-Ngh] += dt * (k1 + 2. * k2 + 2. * k3 + k4) / 6.
        U = update_ghost(U)

        # update step counter and current time and print information
        it_current += 1
        t_current += dt

        if it_current % print_steps == 0:
            fluid_vol = U[0, Ngh:-Ngh, Ngh:-Ngh].sum().item() * dx * dx - soil_vol
            print(info_str.format(it_current, dt, t_current, fluid_vol))

        # break loop
        if abs(t_current-t_end) < tol:
            break

        # update dt; modify dt if the next step will exceed target end time
        if t_current + dt > t_end:
            dt = t_end - t_current
        else:
            dt = min(max_dt) * 0.9

    return U, it_current, t_current, dt
