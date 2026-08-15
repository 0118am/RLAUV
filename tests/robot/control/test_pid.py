from __future__ import annotations

import torch
from tensordict import TensorDict

from robot.control import PIDGains, PIDTrajectoryController
from robot.propulsion.curves import (
    measured_thruster_body_forces,
    reduce_point_forces_to_wrench,
)
from robot.dynamics.parameters import AUV
from robot.control.trajectory.observation_contract import ACTION_DIM, BASE_OBSERVATION_DIM


def _controller(gains: PIDGains, *, num_envs: int = 1) -> PIDTrajectoryController:
    positions = torch.tensor(AUV.thruster_positions_body_m)
    return PIDTrajectoryController(
        num_envs=num_envs,
        dt=0.02,
        thruster_positions_b=positions,
        thruster_force_curve_coefficients=torch.tensor(AUV.thruster_force_curve_coefficients),
        mass_kg=10.0,
        gains=gains,
    )


def _realized_wrench(commands: torch.Tensor) -> torch.Tensor:
    positions = torch.tensor(AUV.thruster_positions_body_m).unsqueeze(0).expand(commands.shape[0], -1, -1)
    forces = measured_thruster_body_forces(
        commands,
        torch.tensor(AUV.thruster_force_curve_coefficients),
    )
    return reduce_point_forces_to_wrench(positions, forces)


def _identity_observation() -> torch.Tensor:
    observation = torch.zeros(1, BASE_OBSERVATION_DIM)
    observation[:, 9] = 1.0
    return observation


def test_pid_zero_error_outputs_neutral_commands() -> None:
    controller = _controller(PIDGains())
    assert torch.allclose(controller(_identity_observation()), torch.zeros(1, ACTION_DIM))


def test_pid_accepts_rsl_rl_tensordict_observations() -> None:
    controller = _controller(PIDGains())
    observations = TensorDict({"policy": _identity_observation()}, batch_size=[1])

    assert torch.allclose(controller(observations), torch.zeros(1, ACTION_DIM))


def test_nonlinear_allocator_recovers_a_feasible_measured_wrench() -> None:
    controller = _controller(PIDGains())
    source_commands = torch.tensor([[0.62, -0.48, 0.55, -0.41, 0.58, -0.52, 0.46, -0.57]])
    desired = _realized_wrench(source_commands)

    allocated = controller.allocate_wrench(desired)
    realized = _realized_wrench(allocated)

    assert torch.all(allocated.abs() <= 1.0)
    assert torch.allclose(realized, desired, atol=3e-3)


def test_nonlinear_allocator_bounds_infeasible_requests() -> None:
    controller = _controller(PIDGains())
    commands = controller.allocate_wrench(torch.full((1, 6), 1.0e6))

    assert torch.all(torch.isfinite(commands))
    assert torch.all(commands >= -1.0)
    assert torch.all(commands <= 1.0)
