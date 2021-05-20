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
import logging
from torchswe.utils.data import States, Gridlines, Topography
from torchswe.utils.config import Config
from torchswe.utils.dummy import DummyDict

logger = logging.getLogger("torchswe.core.temporal")


def _dt_adapter(delta_t: float, max_dt: float, coeff: float = 0.95):
    """Adapt dt according to max_dt and log it at debug level.

    Notes
    -----
    If max_dt is returned by CuPy calculation, it may be a CuPy's data type (e.g., 1-element
    ndarray). Assigning the return of this function to a varaible will make that variable also a
    data type of CuPy.
    """
    max_dt *= coeff  # it's safe because Python's not pass-by-reference
    logger.debug("Adjust dt from %e to %e to meet CFL condition.", delta_t, max_dt)
    return max_dt


def _dt_adapter_log_only(delta_t: float, max_dt: float, coeff=None):
    """Log a warning if dt is greater than max_dt but don't do anything else."""
    # pylint: disable=unused-argument
    if delta_t > max_dt:
        logger.warning("dt=%e is fixed but exceeds max safe value (%e).", delta_t, max_dt)
    return delta_t


def _dt_fixer(cur_t: float, next_t: float, delta_t: float):
    """Check if current time plus dt exceeds next_t and fix it if true."""
    gap_dt = next_t - cur_t
    if delta_t > gap_dt:
        logger.warning("Adjust dt from %e to %e to meet the target time.", delta_t, gap_dt)
        return gap_dt
    return delta_t


def euler(states: States, grid: Gridlines, topo: Topography, config: Config, runtime: DummyDict):
    """A simple 1st-order forward Euler time-marching."""

    if config.temporal.adaptive:
        adapter = _dt_adapter
    else:
        adapter = _dt_adapter_log_only

    # non-ghost domain slice
    internal = slice(states.ngh, -states.ngh)

    # cell area and total soil volume
    cell_area = grid.x.delta * grid.y.delta
    soil_vol = topo.cntr.sum() * cell_area

    # information string formatter
    info_str = "Step %d: step size = %e sec, time = %e sec, total volume = %e"

    # an initial updating, just in case
    states = runtime.ghost_updater(states)

    # loop till cur_t reaches the target t or hitting max iterations
    for _ in range(config.temporal.max_iters):

        # Euler step
        states, max_dt = runtime.rhs_updater(states, grid, topo, config, runtime)

        # adaptive dt
        runtime.dt = adapter(runtime.dt, max_dt, 0.95)  # may exceed next_t

        # make sure cur_t + dt won't exceed next_t
        runtime.dt = _dt_fixer(runtime.cur_t, runtime.next_t, runtime.dt)

        # update
        states.q.w[internal, internal] += (states.rhs.w * runtime.dt)
        states.q.hu[internal, internal] += (states.rhs.hu * runtime.dt)
        states.q.hv[internal, internal] += (states.rhs.hv * runtime.dt)
        states = runtime.ghost_updater(states)

        # update iteration index and time
        runtime.counter += 1
        runtime.cur_t += runtime.dt

        # print out information
        if runtime.counter % config.params.log_steps == 0:
            fluid_vol = states.q.w[internal, internal].sum() * cell_area - soil_vol
            logger.info(info_str, runtime.counter, runtime.dt, runtime.cur_t, fluid_vol)

        # break loop
        if abs(runtime.cur_t-runtime.next_t) < runtime.tol:
            break

    return states


def ssprk2(states: States, grid: Gridlines, topo: Topography, config: Config, runtime: DummyDict):
    """An optimal 2-stage 2nd-order SSP-RK, a.k.a.m Heun's method.

    Notes
    -----
    If presenting the scheme using the Butcher table, it is

        0 |
        1 |  1
        ------------
          | 1/2  1/2

    A better expression that is easier for programming is

        u^1 = u_{n} + dt * RHS(u_{n})
        u_{n+1} = 0.5 * u_{n} + 0.5 * u^1 + 0.5 * dt * RHS(u^1)

    References
    ----------
    Gottlieb, S., Shu, C.-W., & Tadmor, E. (2001). Strong Stability-Preserving High-Order Time
    Discretization Methods. SIAM Review, 43(1), 89-112.
    """

    if config.temporal.adaptive:
        adapter = _dt_adapter
    else:
        adapter = _dt_adapter_log_only

    # non-ghost domain slice
    nongh = slice(states.ngh, -states.ngh)

    # cell area and total soil volume
    cell_area = grid.x.delta * grid.y.delta
    soil_vol = topo.cntr.sum() * cell_area

    # information string formatter
    info_str = "Step %d: step size = %e sec, time = %e sec, total volume = %e"

    # previous solution; should not be changed until the end of each time-step
    states = runtime.ghost_updater(states)

    # to hold previous solution
    prev_q = copy.deepcopy(states.q)

    # loop till cur_t reaches the target t or hitting max iterations
    for _ in range(config.temporal.max_iters):

        # stage 1: now states.rhs is RHS(u_{n})
        states, max_dt = runtime.rhs_updater(states, grid, topo, config, runtime)

        # adaptive dt based on the CFL of 1st order Euler
        runtime.dt = adapter(runtime.dt, max_dt, 0.95)  # may exceed next_t

        # make sure cur_t + dt won't exceed next_t
        runtime.dt = _dt_fixer(runtime.cur_t, runtime.next_t, runtime.dt)

        # update for the first step; now states.q is u1 = u_{n} + dt * RHS(u_{n})
        states.q.w[nongh, nongh] += (states.rhs.w * runtime.dt)
        states.q.hu[nongh, nongh] += (states.rhs.hu * runtime.dt)
        states.q.hv[nongh, nongh] += (states.rhs.hv * runtime.dt)
        states = runtime.ghost_updater(states)

        # stage 2: now states.rhs is RHS(u^1)
        states, _ = runtime.rhs_updater(states, grid, topo, config, runtime)

        # calculate u_{n+1} = (u_{n} + u^1 + dt * RHS(u^1)) / 2.
        states.q.w[nongh, nongh] += (prev_q.w[nongh, nongh] + states.rhs.w * runtime.dt)
        states.q.w[nongh, nongh] /= 2
        states.q.hu[nongh, nongh] += (prev_q.hu[nongh, nongh] + states.rhs.hu * runtime.dt)
        states.q.hu[nongh, nongh] /= 2
        states.q.hv[nongh, nongh] += (prev_q.hv[nongh, nongh] + states.rhs.hv * runtime.dt)
        states.q.hv[nongh, nongh] /= 2
        states = runtime.ghost_updater(states)

        # update iteration index and time
        runtime.counter += 1
        runtime.cur_t += runtime.dt

        # print out information
        if runtime.counter % config.params.log_steps == 0:
            fluid_vol = states.q.w[nongh, nongh].sum() * cell_area - soil_vol
            logger.info(info_str, runtime.counter, runtime.dt, runtime.cur_t, fluid_vol)

        # break loop
        if abs(runtime.cur_t-runtime.next_t) < runtime.tol:
            break

        # for the next time step; copying values should be faster than allocating new arrays
        prev_q.w[...], prev_q.hu[...], prev_q.hv[...] = states.q.w, states.q.hu, states.q.hv

    return states


def ssprk3(states: States, grid: Gridlines, topo: Topography, config: Config, runtime: DummyDict):
    """An optimal 3-stage 3rd-order SSP-RK.

    Notes
    -----
    If presenting the scheme using the Butcher table, it is

    0    |
    1    |   1
    1/2  |  1/4  1/4
    ---------------------
         |  1/6  1/6  2/3

    A better expression that is easier for programming is

    u^1 = u_{n} + dt * RHS(u_{n})
    u^2 = 3/4 u_{n} + 1/4 u^1 + 1/4 dt * RHS(u^1)
    u_{n+1} = 1/3 u_{n} + 2/3 u^2 + 2/3 dt * RHS(u^2)

    References
    ----------
    Gottlieb, S., Shu, C.-W., & Tadmor, E. (2001). Strong Stability-Preserving High-Order Time
    Discretization Methods. SIAM Review, 43(1), 89-112.
    """

    if config.temporal.adaptive:
        adapter = _dt_adapter
    else:
        adapter = _dt_adapter_log_only

    # non-ghost domain slice
    nongh = slice(states.ngh, -states.ngh)

    # cell area and total soil volume
    cell_area = grid.x.delta * grid.y.delta
    soil_vol = topo.cntr.sum() * cell_area

    # information string formatter
    info_str = "Step %d: step size = %e sec, time = %e sec, total volume = %e"

    # previous solution; should not be changed until the end of each time-step
    states = runtime.ghost_updater(states)

    # to hold previous solution
    prev_q = copy.deepcopy(states.q)

    # loop till cur_t reaches the target t or hitting max iterations
    for _ in range(config.temporal.max_iters):

        # stage 1: now states.rhs is RHS(u_{n})
        states, max_dt = runtime.rhs_updater(states, grid, topo, config, runtime)

        # adaptive dt based on the CFL of 1st order Euler
        runtime.dt = adapter(runtime.dt, max_dt, 0.95)  # may exceed next_t

        # make sure cur_t + dt won't exceed next_t
        runtime.dt = _dt_fixer(runtime.cur_t, runtime.next_t, runtime.dt)

        # update for the first step; now states.q is u1 = u_{n} + dt * RHS(u_{n})
        states.q.w[nongh, nongh] += (states.rhs.w * runtime.dt)
        states.q.hu[nongh, nongh] += (states.rhs.hu * runtime.dt)
        states.q.hv[nongh, nongh] += (states.rhs.hv * runtime.dt)
        states = runtime.ghost_updater(states)

        # stage 2: now states.rhs is RHS(u^1)
        states, _ = runtime.rhs_updater(states, grid, topo, config, runtime)

        # now states.q = u^2 = (3 * u_{n} + u^1 + dt * RHS(u^1)) / 4
        states.q.w[nongh, nongh] += (prev_q.w[nongh, nongh] * 3. + states.rhs.w * runtime.dt)
        states.q.w[nongh, nongh] /= 4
        states.q.hu[nongh, nongh] += (3 * prev_q.hu[nongh, nongh] * 3. + states.rhs.hu * runtime.dt)
        states.q.hu[nongh, nongh] /= 4
        states.q.hv[nongh, nongh] += (prev_q.hv[nongh, nongh] * 3. + states.rhs.hv * runtime.dt)
        states.q.hv[nongh, nongh] /= 4
        states = runtime.ghost_updater(states)

        # stage 3: now states.rhs is RHS(u^2)
        states, _ = runtime.rhs_updater(states, grid, topo, config, runtime)

        # now states.q = u_{n+1} = (u_{n} + 2 * u^2 + 2 * dt * RHS(u^1)) / 3
        states.q.w[nongh, nongh] *= 2  # 2 * u^2
        states.q.w[nongh, nongh] += (prev_q.w[nongh, nongh] + states.rhs.w * runtime.dt * 2)
        states.q.w[nongh, nongh] /= 3
        states.q.hu[nongh, nongh] *= 2  # 2 * u^2
        states.q.hu[nongh, nongh] += (prev_q.hu[nongh, nongh] + states.rhs.hu * runtime.dt * 2)
        states.q.hu[nongh, nongh] /= 3
        states.q.hv[nongh, nongh] *= 2  # 2 * u^2
        states.q.hv[nongh, nongh] += (prev_q.hv[nongh, nongh] + states.rhs.hv * runtime.dt * 2)
        states.q.hv[nongh, nongh] /= 3
        states = runtime.ghost_updater(states)

        # update iteration index and time
        runtime.counter += 1
        runtime.cur_t += runtime.dt

        # print out information
        if runtime.counter % config.params.log_steps == 0:
            fluid_vol = states.q.w[nongh, nongh].sum() * cell_area - soil_vol
            logger.info(info_str, runtime.counter, runtime.dt, runtime.cur_t, fluid_vol)

        # break loop
        if abs(runtime.cur_t-runtime.next_t) < runtime.tol:
            break

        # for the next time step; copying values should be faster than allocating new arrays
        prev_q.w[...], prev_q.hu[...], prev_q.hv[...] = states.q.w, states.q.hu, states.q.hv

    return states
