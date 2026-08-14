#!/usr/bin/env python3
"""Validate an exported AUV policy in an independent MuJoCo rollout."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
from typing import Protocol

import numpy as np


MUJOCO_WORKFLOW_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MUJOCO_WORKFLOW_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.isaac.agents.ppo.architectures import available_mlp_architectures, get_mlp_architecture
from simulation.mujoco.bridge import (
    ACTION_DIM,
    coefficient_matrix,
    HydrodynamicsModel,
    HydrodynamicsParameters,
    PolicyObservationAdapter,
    ReferenceGenerator,
    ThrusterModel,
    ThrusterParameters,
    TrajectoryConfig,
    VehicleState,
    quaternion_rotate,
    quaternion_rotate_inverse,
    summarize_validation,
)
from environment.profiles.pool_profile import load_pool_dynamics_profile_json
from robot.dynamics.parameters import AUV


DEFAULT_MODEL = PROJECT_ROOT / "robot/assets/mujoco/auv.xml"
DEFAULT_PROFILE = PROJECT_ROOT / "environment/hydrodynamics/coefficients/auv_pool_openfoam_hydrodynamics_v1.json"


class Policy(Protocol):
    def __call__(self, observation: np.ndarray) -> np.ndarray: ...


class OnnxPolicy:
    def __init__(self, path: Path, expected_observation_dim: int):
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError(
                "ONNX validation requires onnxruntime. Install "
                "`simulation/mujoco/requirements.txt` in env_isaaclab."
            ) from error
        self._session = ort.InferenceSession(
            str(path),
            providers=["CPUExecutionProvider"],
        )
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if len(inputs) != 1 or len(outputs) < 1:
            raise ValueError("ONNX policy must expose exactly one input and at least one output.")
        self._input_name = inputs[0].name
        self._output_name = outputs[0].name
        shape = inputs[0].shape
        if len(shape) != 2 or shape[-1] not in (None, "None", expected_observation_dim):
            raise ValueError(
                f"ONNX input shape {shape} does not match observation dimension "
                f"{expected_observation_dim}."
            )

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        result = self._session.run(
            [self._output_name],
            {self._input_name: observation.reshape(1, -1).astype(np.float32)},
        )[0]
        return np.asarray(result, dtype=np.float64).reshape(-1)


class TorchCheckpointPolicy:
    def __init__(
        self,
        path: Path,
        *,
        observation_dim: int,
        hidden_dims: list[int],
        activation: str,
    ):
        import torch

        from simulation.isaac.agents.ppo.evaluation import load_evaluation_actor

        self._torch = torch
        self._actor = load_evaluation_actor(
            path,
            observation_dim=observation_dim,
            action_dim=ACTION_DIM,
            hidden_dims=hidden_dims,
            activation=activation,
            device="cpu",
        )

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        with self._torch.inference_mode():
            action = self._actor(
                self._torch.from_numpy(observation.reshape(1, -1).astype(np.float32))
            )
        return action.detach().cpu().numpy().reshape(-1).astype(np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an ONNX or RSL-RL checkpoint policy against the independent "
            "MuJoCo AUV model and enforce trajectory-tracking gates."
        )
    )
    parser.add_argument("--policy", type=Path, required=True, help="Exported .onnx or model_*.pt policy.")
    parser.add_argument(
        "--mlp-architecture",
        choices=available_mlp_architectures(),
        default="mlp_history_5",
        help="Actor observation/history and hidden-layer contract.",
    )
    parser.add_argument("--activation", default="elu", help="Activation used by a .pt checkpoint.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="MuJoCo XML model.")
    parser.add_argument(
        "--profile",
        type=Path,
        default=DEFAULT_PROFILE,
        help="PoolDynamicsProfile supplying nominal hydrodynamics and actuators.",
    )
    parser.add_argument(
        "--trajectory",
        choices=("hold", "circle", "lissajous", "helix"),
        default="lissajous",
    )
    parser.add_argument("--duration", type=float, default=32.0)
    parser.add_argument("--settling-time", type=float, default=2.0)
    parser.add_argument("--policy-dt", type=float, default=0.02, help="Actor update interval in seconds.")
    parser.add_argument("--amplitude-x", type=float, default=0.75)
    parser.add_argument("--amplitude-y", type=float, default=0.65)
    parser.add_argument("--amplitude-z", type=float, default=0.16)
    parser.add_argument("--period", type=float, default=12.0)
    parser.add_argument(
        "--center",
        type=float,
        nargs=3,
        default=(0.0, 0.0, -3.0),
        metavar=("X", "Y", "Z"),
        help="World-frame trajectory center in metres.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true", help="Open the MuJoCo passive viewer.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/mujoco"))
    parser.add_argument(
        "--max-position-rmse",
        type=float,
        default=0.50,
        help="Fail when post-settling position RMSE exceeds this value in metres.",
    )
    parser.add_argument(
        "--max-action-clip-fraction",
        type=float,
        default=0.10,
        help="Fail when this fraction of raw Actor outputs lies outside [-1, 1].",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if not args.policy.is_file():
        raise FileNotFoundError(f"Policy not found: {args.policy}")
    if args.policy.suffix.lower() not in {".onnx", ".pt"}:
        raise ValueError("--policy must use .onnx or .pt.")
    if not args.model.is_file():
        raise FileNotFoundError(f"MuJoCo model not found: {args.model}")
    if not args.profile.is_file():
        raise FileNotFoundError(f"Dynamics profile not found: {args.profile}")
    if args.duration <= 0.0 or args.policy_dt <= 0.0:
        raise ValueError("--duration and --policy-dt must be positive.")
    if not 0.0 <= args.settling_time < args.duration:
        raise ValueError("--settling-time must satisfy 0 <= value < duration.")
    if args.max_position_rmse <= 0.0:
        raise ValueError("--max-position-rmse must be positive.")
    if not 0.0 <= args.max_action_clip_fraction <= 1.0:
        raise ValueError("--max-action-clip-fraction must be in [0, 1].")


def _load_policy(args: argparse.Namespace, observation_dim: int, hidden_dims: list[int]) -> Policy:
    if args.policy.suffix.lower() == ".onnx":
        return OnnxPolicy(args.policy, observation_dim)
    return TorchCheckpointPolicy(
        args.policy,
        observation_dim=observation_dim,
        hidden_dims=hidden_dims,
        activation=args.activation,
    )


def _thruster_parameters(profile) -> ThrusterParameters:
    thrusters = profile.thrusters
    return ThrusterParameters(
        positions_b=np.asarray(AUV.thruster_positions_body_m, dtype=np.float64),
        force_curve_coefficients=np.asarray(
            AUV.thruster_force_curve_coefficients,
            dtype=np.float64,
        ),
        time_constant_s=float(thrusters.dyn_time_constant),
        max_command_rate_per_s=float(thrusters.max_command_rate),
        command_delay_steps=int(thrusters.command_delay_steps),
        command_resolution=float(thrusters.command_resolution),
        dropout_probability=float(thrusters.command_dropout_probability),
        pwm_center_us=float(AUV.thruster_pwm_center_us),
        pwm_half_range_us=float(AUV.thruster_pwm_half_range_us),
        pwm_deadband_us=float(AUV.thruster_pwm_deadband_us),
    )


def _hydrodynamics_parameters(profile) -> HydrodynamicsParameters:
    body = profile.rigid_body
    hydro = profile.hydrodynamics
    if hydro.speed_dependent_damping_enabled:
        raise ValueError("MuJoCo validation does not yet support speed-dependent damping curves.")
    if hydro.water_current_field_enabled:
        raise ValueError("MuJoCo validation does not yet support gridded water-current fields.")
    return HydrodynamicsParameters(
        fluid_density_kg_m3=float(body.water_rho),
        displaced_volume_m3=float(body.volume),
        center_of_buoyancy_from_com_b=np.asarray(body.com_to_cob_offset, dtype=np.float64),
        linear_damping=np.asarray(hydro.linear_damping, dtype=np.float64),
        quadratic_damping=np.asarray(hydro.quadratic_damping, dtype=np.float64),
        added_mass=np.asarray(hydro.added_mass, dtype=np.float64),
        added_mass_inertia_scale=float(hydro.added_mass_inertia_scale),
        added_mass_acceleration_filter_alpha=float(hydro.added_mass_accel_filter_alpha),
        water_current_w=np.asarray(hydro.water_current_w, dtype=np.float64),
        periodic_current_enabled=bool(hydro.water_current_periodic_enabled),
        periodic_current_amplitude_w=np.asarray(
            hydro.water_current_periodic_amplitude_w,
            dtype=np.float64,
        ),
        periodic_current_period_s=np.asarray(
            hydro.water_current_periodic_period_s,
            dtype=np.float64,
        ),
        periodic_current_phase_rad=np.asarray(
            hydro.water_current_periodic_phase_rad,
            dtype=np.float64,
        ),
    )


def _vehicle_state(mujoco, model, data, body_id: int) -> VehicleState:
    """Read MuJoCo state in the project's x-forward body frame.

    ``mj_objectVelocity(..., flg_local=1)`` uses MuJoCo's diagonalized inertial
    frame when ``fullinertia`` has products of inertia. That frame is not the
    AUV body frame used by the Actor, so read world velocity and rotate it
    explicitly with the body's world quaternion.
    """

    velocity = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(
        model,
        data,
        mujoco.mjtObj.mjOBJ_BODY,
        body_id,
        velocity,
        0,
    )
    quaternion_wxyz = np.asarray(data.xquat[body_id], dtype=np.float64).copy()
    return VehicleState(
        position_w=np.asarray(data.xpos[body_id], dtype=np.float64).copy(),
        quaternion_wxyz=quaternion_wxyz,
        linear_velocity_b=quaternion_rotate_inverse(quaternion_wxyz, velocity[3:]),
        angular_velocity_b=quaternion_rotate_inverse(quaternion_wxyz, velocity[:3]),
    )


def _set_initial_state(mujoco, model, data, body_id: int, joint_id: int, reference) -> None:
    qpos_address = model.jnt_qposadr[joint_id]
    dof_address = model.jnt_dofadr[joint_id]
    data.qpos[qpos_address : qpos_address + 3] = reference.position_w
    data.qpos[qpos_address + 3 : qpos_address + 7] = reference.quaternion_wxyz
    data.qvel[dof_address : dof_address + 6] = 0.0
    mujoco.mj_forward(model, data)
    if not np.allclose(data.xpos[body_id], reference.position_w, atol=1.0e-9):
        raise RuntimeError("MuJoCo initial-state application failed.")


def _write_results(
    output_dir: Path,
    rows: list[dict[str, float]],
    summary: dict,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "rollout.csv"
    summary_path = output_dir / "summary.json"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return csv_path, summary_path


def run(args: argparse.Namespace) -> dict:
    _validate_args(args)
    try:
        import mujoco
    except ImportError as error:
        raise RuntimeError(
            "MuJoCo validation requires the `mujoco` package. Install "
            "`simulation/mujoco/requirements.txt` in env_isaaclab."
        ) from error
    if not hasattr(mujoco, "MjModel"):
        raise RuntimeError(
            "The imported `mujoco` module is not the MuJoCo SDK. Install "
            "`simulation/mujoco/requirements.txt` in env_isaaclab."
        )

    architecture = get_mlp_architecture(args.mlp_architecture)
    observation_adapter = PolicyObservationAdapter(
        history_steps=architecture.history_steps,
        history_fields=architecture.history_fields,
    )
    if observation_adapter.observation_dim != architecture.observation_dim:
        raise RuntimeError(
            "MuJoCo observation bridge disagrees with the selected Actor architecture: "
            f"{observation_adapter.observation_dim} != {architecture.observation_dim}."
        )
    policy = _load_policy(
        args,
        architecture.observation_dim,
        list(architecture.actor_hidden_dims),
    )
    profile = load_pool_dynamics_profile_json(args.profile)
    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "auv")
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "auv_freejoint")
    if body_id < 0 or joint_id < 0:
        raise ValueError("MuJoCo model must define body 'auv' and joint 'auv_freejoint'.")

    profile_mass = float(profile.rigid_body.mass)
    if not np.isclose(model.body_mass[body_id], profile_mass, rtol=0.0, atol=1.0e-6):
        raise ValueError(
            f"MuJoCo body mass {model.body_mass[body_id]} does not match profile mass "
            f"{profile_mass}."
        )
    profile_inertia = np.asarray(profile.rigid_body.inertia_diag, dtype=np.float64)
    if profile_inertia.shape == (6,):
        profile_inertia = np.diag(profile_inertia)
    if profile_inertia.shape != (3, 3):
        raise ValueError(
            f"Profile inertia must be a 3-vector or 3x3 matrix, got {profile_inertia.shape}."
        )
    expected_principal_inertia = np.sort(np.linalg.eigvalsh(profile_inertia))
    actual_principal_inertia = np.sort(np.asarray(model.body_inertia[body_id]))
    if not np.allclose(
        actual_principal_inertia,
        expected_principal_inertia,
        rtol=0.0,
        atol=1.0e-8,
    ):
        raise ValueError(
            "MuJoCo principal inertia does not match the selected profile: "
            f"{actual_principal_inertia.tolist()} != {expected_principal_inertia.tolist()}."
        )
    for label, expected_position in zip(AUV.thruster_labels, AUV.thruster_positions_body_m):
        site_name = f"thruster_{label}"
        site_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            site_name,
        )
        if site_id < 0 or model.site_bodyid[site_id] != body_id:
            raise ValueError(f"MuJoCo model is missing AUV site {site_name}.")
        if not np.allclose(
            model.site_pos[site_id],
            expected_position,
            rtol=0.0,
            atol=1.0e-8,
        ):
            raise ValueError(f"MuJoCo site {site_name} does not match AUV geometry.")
    physics_dt = float(model.opt.timestep)
    decimation = int(round(args.policy_dt / physics_dt))
    if decimation < 1 or not np.isclose(decimation * physics_dt, args.policy_dt, atol=1.0e-10):
        raise ValueError(
            f"--policy-dt {args.policy_dt} must be an integer multiple of model timestep "
            f"{physics_dt}."
        )

    trajectory = ReferenceGenerator(
        TrajectoryConfig(
            kind=args.trajectory,
            center_w=tuple(args.center),
            amplitude_x_m=args.amplitude_x,
            amplitude_y_m=args.amplitude_y,
            amplitude_z_m=args.amplitude_z,
            period_s=args.period,
        ),
        args.policy_dt,
    )
    initial_reference = trajectory.sample(0.0)
    hydro_parameters = _hydrodynamics_parameters(profile)
    # Explicitly feeding the previous-step acceleration back as
    # ``-M_A dot(nu_r)`` is unstable when an added-mass axis exceeds the rigid
    # mass. MuJoCo's armature term adds positive generalized inertia directly
    # to the implicit mass matrix. The profile's small off-diagonal terms stay
    # in the full added-mass Coriolis calculation below.
    dof_address = model.jnt_dofadr[joint_id]
    added_mass_diagonal = (
        np.diag(coefficient_matrix(hydro_parameters.added_mass, "added_mass"))
        * hydro_parameters.added_mass_inertia_scale
    )
    if np.any(added_mass_diagonal < 0.0):
        raise ValueError("Added-mass diagonal must be non-negative.")
    model.dof_armature[dof_address : dof_address + 6] += added_mass_diagonal
    hydro_parameters = replace(hydro_parameters, added_mass_inertia_scale=0.0)
    _set_initial_state(mujoco, model, data, body_id, joint_id, initial_reference)
    trajectory.reset()
    thrusters = ThrusterModel(_thruster_parameters(profile), seed=args.seed)
    hydrodynamics = HydrodynamicsModel(hydro_parameters)

    requested_action = np.zeros(ACTION_DIM)
    rows: list[dict[str, float]] = []
    position_errors: list[float] = []
    raw_actions: list[np.ndarray] = []
    total_steps = int(np.ceil(args.duration / physics_dt))
    viewer = None
    if args.render:
        from mujoco import viewer as mujoco_viewer

        viewer = mujoco_viewer.launch_passive(model, data)

    try:
        for physics_step in range(total_steps):
            policy_step = physics_step % decimation == 0
            state = _vehicle_state(mujoco, model, data, body_id)
            if policy_step:
                reference = trajectory.sample(float(data.time))
                observation = observation_adapter.build(
                    state,
                    reference,
                    thrusters.applied_command,
                )
                raw_action = np.asarray(policy(observation), dtype=np.float64).reshape(-1)
                if raw_action.shape != (ACTION_DIM,):
                    raise ValueError(
                        f"Policy output must contain {ACTION_DIM} actions, got {raw_action.shape}."
                    )
                if not np.all(np.isfinite(raw_action)):
                    raise ValueError("Policy produced non-finite actions.")
                requested_action = np.clip(raw_action, -1.0, 1.0)
                if data.time >= args.settling_time:
                    position_error = float(np.linalg.norm(reference.position_w - state.position_w))
                    position_errors.append(position_error)
                    raw_actions.append(raw_action.copy())
                    row: dict[str, float] = {
                        "time_s": float(data.time),
                        "position_error_m": position_error,
                    }
                    for axis, label in enumerate(("x", "y", "z")):
                        row[f"actual_{label}_m"] = float(state.position_w[axis])
                        row[f"target_{label}_m"] = float(reference.position_w[axis])
                    for index in range(ACTION_DIM):
                        row[f"raw_action_{index}"] = float(raw_action[index])
                        row[f"applied_action_{index}"] = float(thrusters.applied_command[index])
                    rows.append(row)

            thruster_step = thrusters.step(requested_action, physics_dt)
            state = _vehicle_state(mujoco, model, data, body_id)
            fluid_wrench_b = hydrodynamics.step(state, float(data.time), physics_dt)
            total_wrench_b = thruster_step.wrench_b + fluid_wrench_b
            data.xfrc_applied.fill(0.0)
            data.xfrc_applied[body_id, :3] = quaternion_rotate(
                state.quaternion_wxyz,
                total_wrench_b[:3],
            )
            data.xfrc_applied[body_id, 3:] = quaternion_rotate(
                state.quaternion_wxyz,
                total_wrench_b[3:],
            )
            mujoco.mj_step(model, data)
            if (
                not np.all(np.isfinite(data.qpos))
                or not np.all(np.isfinite(data.qvel))
                or np.max(np.abs(data.qvel)) > 1.0e4
            ):
                raise RuntimeError(
                    f"MuJoCo state became unstable at simulation time {data.time:.6f} s; "
                    f"qvel={np.asarray(data.qvel).tolist()}."
                )
            if viewer is not None:
                viewer.sync()
                time.sleep(max(0.0, physics_dt))
    finally:
        if viewer is not None:
            viewer.close()

    summary = summarize_validation(
        position_errors,
        raw_actions,
        max_position_rmse_m=args.max_position_rmse,
        max_action_clip_fraction=args.max_action_clip_fraction,
    )
    summary.update(
        {
            "policy": str(args.policy.resolve()),
            "model": str(args.model.resolve()),
            "profile": str(args.profile.resolve()),
            "profile_name": profile.name,
            "mlp_architecture": architecture.name,
            "observation_dim": architecture.observation_dim,
            "action_dim": ACTION_DIM,
            "trajectory": args.trajectory,
            "duration_s": float(args.duration),
            "settling_time_s": float(args.settling_time),
            "physics_dt_s": physics_dt,
            "policy_dt_s": float(args.policy_dt),
            "added_mass_armature_diagonal": added_mass_diagonal.tolist(),
            "principal_inertia_kg_m2": actual_principal_inertia.tolist(),
        }
    )
    run_name = f"{args.policy.stem}_{architecture.name}_{args.trajectory}"
    result_dir = args.output_dir / run_name
    summary["rollout_csv"] = str((result_dir / "rollout.csv").resolve())
    summary["summary_json"] = str((result_dir / "summary.json").resolve())
    _write_results(result_dir, rows, summary)
    return summary


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    if not summary["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
