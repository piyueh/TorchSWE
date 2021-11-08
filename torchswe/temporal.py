#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Time-marching schemes.
"""
import copy as _copy
import logging as _logging
from mpi4py import MPI as _MPI
from torchswe.fvm import prepare_rhs as _prepare_rhs
from torchswe.utils.data import States as _States
from torchswe.utils.config import Config as _Config
from torchswe.utils.misc import DummyDict as _DummyDict
from torchswe.utils.misc import exchange_states as _exchange_states

_logger = _logging.getLogger("torchswe.temporal")


def _cfl_dt_adapter(delta_t: float, max_dt: float, coeff: float):
    """Adapt dt according to max_dt and log it at debug level.

    Notes
    -----
    If max_dt is returned by CuPy calculation, it may be a CuPy's data type (e.g., 1-element
    ndarray). Assigning the return of this function to a varaible will make that variable also a
    data type of CuPy.
    """
    max_dt *= coeff  # it's safe because Python's not pass-by-reference
    _logger.debug("Adjust dt from %e to %e to meet CFL condition.", delta_t, max_dt)
    return max_dt


def _cfl_dt_adapter_log_only(delta_t: float, max_dt: float, *args, **kwargs):
    """Log a warning if dt is greater than max_dt but don't do anything else."""
    # pylint: disable=unused-argument
    if delta_t > max_dt:
        _logger.warning("dt=%e is fixed but exceeds max safe value (%e).", delta_t, max_dt)
    return delta_t


def _stiff_terms(states, slc, dt):
    """update with stiff terms."""
    states.Q[:, slc, slc] /= (1. - dt * states.SS)
    return states


def _stiff_terms_null(states, *args, **kwargs):
    """Dummy function"""
    return states


def euler(states: _States, runtime: _DummyDict, config: _Config):
    """A simple 1st-order forward Euler time-marching."""

    # adaptive time stepping
    adapter = _cfl_dt_adapter if config.temporal.adaptive else _cfl_dt_adapter_log_only

    # stiff term handling
    semi_implicit_step = _stiff_terms if states.SS is not None else _stiff_terms_null

    # non-ghost domain slice
    internal = slice(states.ngh, -states.ngh)

    # cell area and total soil volume
    cell_area = states.domain.x.delta * states.domain.y.delta
    soil_vol = runtime.topo.centers.sum() * cell_area

    # information string formatter
    info_str = "Step %d: step size = %e sec, time = %e sec, total volume = %e"

    # loop till cur_t reaches the target t or hitting max iterations
    for _ in range(config.temporal.max_iters):

        # re-initialize time-step size constraint by not exceeding the next output time
        runtime.dt_constraint = runtime.next_t - runtime.cur_t

        # update values of the halo-ring cells
        states = _exchange_states(states)

        # Euler step
        states, max_dt = _prepare_rhs(states, runtime, config)

        # adaptive dt based on CFL condition
        runtime.dt = adapter(runtime.dt, max_dt, runtime.cfl)  # may exceed next_t

        # re-evaluate dt with other constraints; dt_constraint might be modified during _prepare_rhs
        runtime.dt = min(runtime.dt, runtime.dt_constraint)

        # synchronize dt across all processes
        runtime.dt = states.domain.process.comm.allreduce(runtime.dt, _MPI.MIN)

        # update
        states.Q[:, internal, internal] += (states.S * runtime.dt)
        states = semi_implicit_step(states, internal, runtime.dt)

        # update iteration index and time
        runtime.counter += 1
        runtime.cur_t += runtime.dt

        # print out information
        if runtime.counter % config.params.log_steps == 0:
            fluid_vol = states.Q[0, internal, internal].sum() * cell_area - soil_vol
            fluid_vol = states.domain.process.comm.allreduce(fluid_vol, _MPI.SUM)
            _logger.info(info_str, runtime.counter, runtime.dt, runtime.cur_t, fluid_vol)

        # break loop
        if abs(runtime.cur_t-runtime.next_t) < runtime.tol:
            break

    return states


def ssprk2(states: _States, runtime: _DummyDict, config: _Config):
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

    # adaptive time stepping
    adapter = _cfl_dt_adapter if config.temporal.adaptive else _cfl_dt_adapter_log_only

    # non-ghost domain slice
    nongh = slice(states.ngh, -states.ngh)

    # cell area and total soil volume
    cell_area = states.domain.x.delta * states.domain.y.delta
    soil_vol = runtime.topo.centers.sum() * cell_area

    # information string formatter
    info_str = "Step %d: step size = %e sec, time = %e sec, total volume = %e"

    # to hold previous solution
    prev_q = _copy.deepcopy(states.Q[:, nongh, nongh])

    # loop till cur_t reaches the target t or hitting max iterations
    for _ in range(config.temporal.max_iters):

        # re-initialize time-step size constraint by not exceeding the next output time
        runtime.dt_constraint = runtime.next_t - runtime.cur_t

        # update values of the halo-ring cells
        states = _exchange_states(states)

        # stage 1: now states.rhs is RHS(u_{n})
        states, max_dt = _prepare_rhs(states, runtime, config)

        # adaptive dt based on the CFL of 1st order Euler
        runtime.dt = adapter(runtime.dt, max_dt, runtime.cfl)  # may exceed next_t

        # re-evaluate dt with other constraints; dt_constraint might be modified during _prepare_rhs
        runtime.dt = min(runtime.dt, runtime.dt_constraint)

        # synchronize dt across all processes
        runtime.dt = states.domain.process.comm.allreduce(runtime.dt, _MPI.MIN)

        # update for the first step; now states.q is u1 = u_{n} + dt * RHS(u_{n})
        states.Q[:, nongh, nongh] += (states.S * runtime.dt)

        # update values of the halo-ring cells
        states = _exchange_states(states)

        # stage 2: now states.rhs is RHS(u^1)
        states, _ = _prepare_rhs(states, runtime, config)

        # calculate u_{n+1} = (u_{n} + u^1 + dt * RHS(u^1)) / 2.
        states.Q[:, nongh, nongh] += (prev_q + states.S * runtime.dt)
        states.Q /= 2  # doesn't matter whether ghost cells are also divided by 2

        # update iteration index and time
        runtime.counter += 1
        runtime.cur_t += runtime.dt

        # print out information
        if runtime.counter % config.params.log_steps == 0:
            fluid_vol = states.Q[0, nongh, nongh].sum() * cell_area - soil_vol
            fluid_vol = states.domain.process.comm.allreduce(fluid_vol, _MPI.SUM)
            _logger.info(info_str, runtime.counter, runtime.dt, runtime.cur_t, fluid_vol)

        # break loop
        if abs(runtime.cur_t-runtime.next_t) < runtime.tol:
            break

        # for the next time step; copying values should be faster than allocating new arrays
        prev_q[...] = states.Q[:, nongh, nongh]

    return states


def ssprk3(states: _States, runtime: _DummyDict, config: _Config):
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

    # adaptive time stepping
    adapter = _cfl_dt_adapter if config.temporal.adaptive else _cfl_dt_adapter_log_only

    # non-ghost domain slice
    nongh = slice(states.ngh, -states.ngh)

    # cell area and total soil volume
    cell_area = states.domain.x.delta * states.domain.y.delta
    soil_vol = runtime.topo.centers.sum() * cell_area

    # information string formatter
    info_str = "Step %d: step size = %e sec, time = %e sec, total volume = %e"

    # to hold previous solution
    prev_q = _copy.deepcopy(states.Q[:, nongh, nongh])

    # loop till cur_t reaches the target t or hitting max iterations
    for _ in range(config.temporal.max_iters):

        # re-initialize time-step size constraint by not exceeding the next output time
        runtime.dt_constraint = runtime.next_t - runtime.cur_t

        # update values of the halo-ring cells
        states = _exchange_states(states)

        # stage 1: now states.rhs is RHS(u_{n})
        states, max_dt = _prepare_rhs(states, runtime, config)

        # adaptive dt based on the CFL of 1st order Euler
        runtime.dt = adapter(runtime.dt, max_dt, runtime.cfl)  # may exceed next_t

        # re-evaluate dt with other constraints; dt_constraint might be modified during _prepare_rhs
        runtime.dt = min(runtime.dt, runtime.dt_constraint)

        # synchronize dt across all processes
        runtime.dt = states.domain.process.comm.allreduce(runtime.dt, _MPI.MIN)

        # update for the first step; now states.q is u1 = u_{n} + dt * RHS(u_{n})
        states.Q[:, nongh, nongh] += (states.S * runtime.dt)

        # update values of the halo-ring cells
        states = _exchange_states(states)

        # stage 2: now states.rhs is RHS(u^1)
        states, _ = _prepare_rhs(states, runtime, config)

        # now states.q = u^2 = (3 * u_{n} + u^1 + dt * RHS(u^1)) / 4
        states.Q[:, nongh, nongh] += (prev_q * 3. + states.S * runtime.dt)
        states.Q /= 4.

        # update values of the halo-ring cells
        states = _exchange_states(states)

        # stage 3: now states.rhs is RHS(u^2)
        states, _ = _prepare_rhs(states, runtime, config)

        # now states.q = u_{n+1} = (u_{n} + 2 * u^2 + 2 * dt * RHS(u^1)) / 3
        states.Q[:, nongh, nongh] += (states.S * runtime.dt)
        states.Q *= 2.
        states.Q[:, nongh, nongh] += prev_q
        states.Q /= 3.

        # update iteration index and time
        runtime.counter += 1
        runtime.cur_t += runtime.dt

        # print out information
        if runtime.counter % config.params.log_steps == 0:
            fluid_vol = states.Q[0, nongh, nongh].sum() * cell_area - soil_vol
            fluid_vol = states.domain.process.comm.allreduce(fluid_vol, _MPI.SUM)
            _logger.info(info_str, runtime.counter, runtime.dt, runtime.cur_t, fluid_vol)

        # break loop
        if abs(runtime.cur_t-runtime.next_t) < runtime.tol:
            break

        # for the next time step; copying values should be faster than allocating new arrays
        prev_q[...] = states.Q[:, nongh, nongh]

    return states
