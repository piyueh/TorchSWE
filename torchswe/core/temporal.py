#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Time-marching.
"""
import copy
import warnings
from torchswe.core.misc import CFLWarning
from torchswe.utils.data import States, Gridlines, Topography, WHUHVModel
from torchswe.utils.config import Config
from torchswe.utils.dummydict import DummyDict


def euler(
    states: States, grid: Gridlines, topo: Topography, config: Config,
        runtime: DummyDict, t_end: float):
    """A simple 1st-order forward Euler time-marching.
    """
    # non-ghost domain slice
    internal = slice(states.ngh, -states.ngh)

    # cell area and total soil volume
    cell_area = grid.x.delta * grid.y.delta
    soil_vol = topo.cntr.sum() * cell_area

    # information string formatter
    info_str = "Step {}: step size = {} sec, time = {} sec, total volume = {}"

    # an initial updating, just in case
    states = runtime.ghost_updater.update_all(states)

    # initial time
    runtime.dt = min(t_end-runtime.cur_t, runtime.dt)  # make sure 1st step won't exceed end t

    # loop till cur_t reaches t_range[0]
    while True:  # TODO: use iteration counter to avoid infinity loop

        # Euler step
        states, max_dt = runtime.rhs_updater(states, grid, topo, config, runtime)

        runtime.dt = max_dt * 0.9  # adjust time step size
        runtime.dt = min(t_end-runtime.cur_t, runtime.dt)  # make sure we won't exceed end t

        # update
        states.q.w[internal, internal] += runtime.dt * states.rhs.w
        states.q.hu[internal, internal] += runtime.dt * states.rhs.hu
        states.q.hv[internal, internal] += runtime.dt * states.rhs.hv
        states = runtime.ghost_updater.update_all(states)

        # update iteration index and time
        runtime.counter += 1
        runtime.cur_t += runtime.dt

        # print out information
        if runtime.counter % config.params.log_steps == 0:
            fluid_vol = states.q.w[internal, internal].sum() * cell_area - soil_vol
            print(info_str.format(runtime.counter, runtime.dt, runtime.cur_t, fluid_vol))

        # break loop
        if abs(runtime.cur_t-t_end) < runtime.tol:
            break

    return states


def RK2(  # pylint: disable=invalid-name, too-many-locals
    states: States, grid: Gridlines, topo: Topography, config: Config,
        runtime: DummyDict, t_end: float):
    """Commonly seen explicit 2th-order Runge-Kutta scheme.
    """
    # non-ghost domain slice
    internal = slice(states.ngh, -states.ngh)

    # cell area and total soil volume
    cell_area = grid.x.delta * grid.y.delta
    soil_vol = topo.cntr.sum() * cell_area

    # used to store maximum allowed time step size
    max_dt = [None, None]

    # information string formatter
    info_str = "Step {}: step size = {} sec, time = {} sec, total volume = {}"
    cfl_str1 = "Current dt (= {} sec)s is not safe, "
    cfl_str2 = "lower down to {} sec"

    # previous solution; should not be changed until the end of each time-step
    states = runtime.ghost_updater.update_all(states)

    # to hold previous solution
    prev_q = copy.deepcopy(states.q)

    # initial time
    runtime.dt = min(t_end-runtime.cur_t, runtime.dt)  # make sure 1st step won't exceed end t

    # loop till t_current reaches t_end
    while True:  # TODO: use iteration counter to avoid infinity loop

        # the slope from the first step
        states, max_dt[0] = runtime.rhs_updater(states, grid, topo, config, runtime)

        # check if it is safe to update conservative variables
        if runtime.dt / 2. >= max_dt[0]:
            msg = cfl_str1.format(runtime.dt)
            runtime.dt = max_dt[0] * 0.9
            msg += cfl_str2.format(runtime.dt)
            warnings.warn(msg, CFLWarning)

        # update for the first step
        states.q.w[internal, internal] += (runtime.dt * states.rhs.w / 2.)
        states.q.hu[internal, internal] += (runtime.dt * states.rhs.hu / 2.)
        states.q.hv[internal, internal] += (runtime.dt * states.rhs.hv / 2.)
        states = runtime.ghost_updater.update_all(states)

        # the final step; k2 is an alias to mid
        states, max_dt[1] = runtime.rhs_updater(states, grid, topo, config, runtime)

        # swap object so the solution holder in `states` is from the previous time step
        states.q, prev_q = prev_q, states.q

        # update solution from previous steps and slopes from the 2nd stage in RK2
        states.q.w[internal, internal] += (runtime.dt * states.rhs.w)
        states.q.hu[internal, internal] += (runtime.dt * states.rhs.hu)
        states.q.hv[internal, internal] += (runtime.dt * states.rhs.hv)
        states = runtime.ghost_updater.update_all(states)

        # update iteration index and time
        runtime.counter += 1
        runtime.cur_t += runtime.dt

        # print out information
        if runtime.counter % config.params.log_steps == 0:
            fluid_vol = states.q.w[internal, internal].sum() * cell_area - soil_vol
            print(info_str.format(runtime.counter, runtime.dt, runtime.cur_t, fluid_vol))

        # break loop
        if abs(runtime.cur_t-t_end) < runtime.tol:
            break

        # for the next time step; copying values should be faster than allocating new arrays
        prev_q.w[...], prev_q.hu[...], prev_q.hv[...] = states.q.w, states.q.hu, states.q.hv

        # update dt; modify dt if the next step will exceed target end time
        runtime.dt = min(max_dt) * 0.9  # adjust time step size
        runtime.dt = min(t_end-runtime.cur_t, runtime.dt)  # make sure we won't exceed end t

    return states


def RK4(  # pylint: disable=invalid-name, too-many-locals, too-many-statements
    states: States, grid: Gridlines, topo: Topography, config: Config,
        runtime: DummyDict, t_end: float):
    """Commonly seen explicit 4th-order Runge-Kutta scheme.
    """
    # non-ghost domain slice
    internal = slice(states.ngh, -states.ngh)

    # cell area and total soil volume
    cell_area = grid.x.delta * grid.y.delta
    soil_vol = topo.cntr.sum() * cell_area

    # used to store maximum allowed time step size
    max_dt = [None, None, None, None]

    # information string formatter
    info_str = "Step {}: step size = {} sec, time = {} sec, total volume = {}"
    cfl_str1 = "Current dt (= {} sec)s is not safe, "
    cfl_str2 = "lower down to {} sec"
    cfl_str3 = "lower down to {} sec and restart the iteration {}"

    # previous solution; should not be changed until the end of each time-step
    states = runtime.ghost_updater.update_all(states)

    # to hold previous solution
    prev_q = copy.deepcopy(states.q)

    # to hold slopes from intermediate RK4 stages
    k1 = WHUHVModel(states.rhs.nx, states.rhs.ny, states.rhs.dtype)
    k2 = WHUHVModel(states.rhs.nx, states.rhs.ny, states.rhs.dtype)
    k3 = WHUHVModel(states.rhs.nx, states.rhs.ny, states.rhs.dtype)

    # initial time
    runtime.dt = min(t_end-runtime.cur_t, runtime.dt)  # make sure 1st step won't exceed end t

    # loop till t_current reaches t_end
    while True:  # TODO: use iteration counter to avoid infinity loop

        # stage 1
        # =========================================================================================
        states, max_dt[0] = runtime.rhs_updater(states, grid, topo, config, runtime)

        # check if it is safe to update conservative variables
        if runtime.dt / 2. >= max_dt[0]:
            msg = cfl_str1.format(runtime.dt)
            runtime.dt = max_dt[0] * 0.9
            msg += cfl_str2.format(runtime.dt)
            warnings.warn(msg, CFLWarning)

        # swap underlying objects
        k1, states.rhs = states.rhs, k1

        # update for the first step
        states.q.w[internal, internal] = prev_q.w[internal, internal] + runtime.dt * k1.w / 2.
        states.q.hu[internal, internal] = prev_q.hu[internal, internal] + runtime.dt * k1.hu / 2.
        states.q.hv[internal, internal] = prev_q.hv[internal, internal] + runtime.dt * k1.hv / 2.
        states = runtime.ghost_updater.update_all(states)

        # stage 2
        # =========================================================================================
        states, max_dt[1] = runtime.rhs_updater(states, grid, topo, config, runtime)

        # check if it is safe to update conservative variables
        if runtime.dt / 2. >= max_dt[1]:
            msg = cfl_str1.format(runtime.dt)
            runtime.dt = max_dt[1] * 0.9
            msg += cfl_str3.format(runtime.dt, runtime.counter)
            warnings.warn(msg, CFLWarning)
            continue  # restart the iteration to get correct soln from previos steps

        # swap underlying objects
        k2, states.rhs = states.rhs, k2

        # update for the second step using k2
        states.q.w[internal, internal] = prev_q.w[internal, internal] + runtime.dt * k2.w / 2.
        states.q.hu[internal, internal] = prev_q.hu[internal, internal] + runtime.dt * k2.hu / 2.
        states.q.hv[internal, internal] = prev_q.hv[internal, internal] + runtime.dt * k2.hv / 2.
        states = runtime.ghost_updater.update_all(states)

        # stage 3
        # =========================================================================================
        states, max_dt[2] = runtime.rhs_updater(states, grid, topo, config, runtime)

        # check if it is safe to update conservative variables
        if runtime.dt >= max_dt[2]:
            msg = cfl_str1.format(runtime.dt)
            runtime.dt = max_dt[2] * 0.9
            msg += cfl_str3.format(runtime.dt, runtime.counter)
            warnings.warn(msg, CFLWarning)
            continue  # restart the iteration to get correct U from previos steps

        # swap underlying objects
        k3, states.rhs = states.rhs, k3

        # update for the third step using k3
        states.q.w[internal, internal] = prev_q.w[internal, internal] + runtime.dt * k3.w
        states.q.hu[internal, internal] = prev_q.hu[internal, internal] + runtime.dt * k3.hu
        states.q.hv[internal, internal] = prev_q.hv[internal, internal] + runtime.dt * k3.hv
        states = runtime.ghost_updater.update_all(states)

        # stage 3
        # =========================================================================================
        states, max_dt[3] = runtime.rhs_updater(states, grid, topo, config, runtime)

        # update U directly because we reach the end of this time step
        states.q.w[internal, internal] = prev_q.w[internal, internal] + \
            runtime.dt * (k1.w + 2. * k2.w + 2. * k3.w + states.rhs.w) / 6.
        states.q.hu[internal, internal] = prev_q.w[internal, internal] + \
            runtime.dt * (k1.hu + 2. * k2.hu + 2. * k3.hu + states.rhs.hu) / 6.
        states.q.hv[internal, internal] = prev_q.w[internal, internal] + \
            runtime.dt * (k1.hv + 2. * k2.hv + 2. * k3.hv + states.rhs.hv) / 6.
        states = runtime.ghost_updater.update_all(states)

        # update iteration index and time
        runtime.counter += 1
        runtime.cur_t += runtime.dt

        # print out information
        if runtime.counter % config.params.log_steps == 0:
            fluid_vol = states.q.w[internal, internal].sum() * cell_area - soil_vol
            print(info_str.format(runtime.counter, runtime.dt, runtime.cur_t, fluid_vol))

        # break loop
        if abs(runtime.cur_t-t_end) < runtime.tol:
            break

        # for the next time step; copying values should be faster than allocating new arrays
        prev_q.w[...], prev_q.hu[...], prev_q.hv[...] = states.q.w, states.q.hu, states.q.hv

        # update dt; modify dt if the next step will exceed target end time
        runtime.dt = min(max_dt) * 0.9  # adjust time step size
        runtime.dt = min(t_end-runtime.cur_t, runtime.dt)  # make sure we won't exceed end t

    return states


def euler_debug(
    states: States, grid: Gridlines, topo: Topography, config: Config,
        runtime: DummyDict, t_end: float):  # pylint: disable=unused-argument
    """A simple 1st-order forward Euler for debug.
    """
    # non-ghost domain slice
    internal = slice(states.ngh, -states.ngh)

    # an initial updating, just in case
    states = runtime.ghost_updater.update_all(states)

    # loop till cur_t reaches t_range[0]
    for _ in range(runtime.max_it):

        # Euler step
        states, _ = runtime.rhs_updater(states, grid, topo, config, runtime)

        # update
        states.q.w[internal, internal] += runtime.dt * states.rhs.w
        states.q.hu[internal, internal] += runtime.dt * states.rhs.hu
        states.q.hv[internal, internal] += runtime.dt * states.rhs.hv
        states = runtime.ghost_updater.update_all(states)

        print(runtime.counter)

        # update iteration index and time
        runtime.counter += 1
        runtime.cur_t += runtime.dt

    return states
