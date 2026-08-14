from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MUJOCO_WORKFLOW_ROOT = PROJECT_ROOT / "simulation/mujoco"

from simulation.mujoco.bridge import (
    ACTION_DIM,
    HydrodynamicsModel,
    HydrodynamicsParameters,
    PolicyObservationAdapter,
    ReferenceGenerator,
    ReferenceState,
    ThrusterModel,
    ThrusterParameters,
    TrajectoryConfig,
    VehicleState,
    summarize_validation,
)
from robot.dynamics.parameters import AUV


def _vehicle_state(
    *,
    position=(0.0, 0.0, -3.0),
    linear_velocity=(0.0, 0.0, 0.0),
) -> VehicleState:
    return VehicleState(
        position_w=np.asarray(position, dtype=np.float64),
        quaternion_wxyz=np.array((1.0, 0.0, 0.0, 0.0)),
        linear_velocity_b=np.asarray(linear_velocity, dtype=np.float64),
        angular_velocity_b=np.zeros(3),
    )


def _reference_state() -> ReferenceState:
    return ReferenceState(
        position_w=np.array((1.0, 0.0, -3.0)),
        linear_velocity_w=np.array((0.25, 0.0, 0.0)),
        linear_acceleration_w=np.array((0.5, 0.0, 0.0)),
        quaternion_wxyz=np.array((1.0, 0.0, 0.0, 0.0)),
        angular_velocity_w=np.zeros(3),
    )


def _thruster_parameters() -> ThrusterParameters:
    return ThrusterParameters(
        positions_b=np.asarray(AUV.thruster_positions_body_m),
        force_curve_coefficients=np.asarray(AUV.thruster_force_curve_coefficients),
        time_constant_s=0.0,
        max_command_rate_per_s=0.0,
        command_delay_steps=0,
        command_resolution=0.0,
        dropout_probability=0.0,
        pwm_center_us=AUV.thruster_pwm_center_us,
        pwm_half_range_us=AUV.thruster_pwm_half_range_us,
        pwm_deadband_us=AUV.thruster_pwm_deadband_us,
    )


def _hydrodynamics_parameters() -> HydrodynamicsParameters:
    return HydrodynamicsParameters(
        fluid_density_kg_m3=AUV.water_density_kg_m3,
        displaced_volume_m3=AUV.displaced_volume_m3,
        center_of_buoyancy_from_com_b=np.asarray(AUV.center_of_buoyancy_from_com_m),
        linear_damping=np.array((10.0, 12.0, 14.0, 1.0, 1.2, 1.4)),
        quadratic_damping=np.array((2.0, 2.0, 2.0, 0.2, 0.2, 0.2)),
        added_mass=np.zeros(6),
        added_mass_inertia_scale=0.0,
        added_mass_acceleration_filter_alpha=0.35,
        water_current_w=np.zeros(3),
        periodic_current_enabled=False,
        periodic_current_amplitude_w=np.zeros(3),
        periodic_current_period_s=np.ones(3),
        periodic_current_phase_rad=np.zeros(3),
    )


def test_observation_adapter_matches_30d_and_135d_actor_contracts() -> None:
    current_only = PolicyObservationAdapter(history_steps=0, history_fields=())
    observation_30d = current_only.build(_vehicle_state(), _reference_state(), np.zeros(ACTION_DIM))
    assert observation_30d.shape == (30,)
    assert observation_30d[0] == pytest.approx(0.5)
    assert observation_30d[3] == pytest.approx(0.25)
    assert observation_30d[9] == pytest.approx(1.0)
    assert observation_30d[19] == pytest.approx(1.0)

    history = PolicyObservationAdapter(
        history_steps=5,
        history_fields=(
            "position_error_b",
            "linear_velocity_error_b",
            "attitude_error_quat",
            "angular_velocity_b",
            "applied_action",
        ),
    )
    first = history.build(_vehicle_state(), _reference_state(), np.zeros(ACTION_DIM))
    second = history.build(_vehicle_state(), _reference_state(), np.ones(ACTION_DIM))
    assert first.shape == (135,)
    assert np.all(first[30:] == 0.0)
    assert second[30] == pytest.approx(first[0])
    assert np.all(second[-21:] == 0.0)


def test_reference_generator_is_finite_and_keeps_quaternion_sign_continuous() -> None:
    generator = ReferenceGenerator(TrajectoryConfig(kind="lissajous"), policy_dt_s=0.02)
    previous = generator.sample(0.0)
    for step in range(1, 100):
        current = generator.sample(step * 0.02)
        assert np.all(np.isfinite(current.position_w))
        assert np.linalg.norm(current.quaternion_wxyz) == pytest.approx(1.0)
        assert np.dot(previous.quaternion_wxyz, current.quaternion_wxyz) >= 0.0
        previous = current


def test_thruster_bridge_uses_measured_vector_curve_and_wrench_geometry() -> None:
    model = ThrusterModel(_thruster_parameters())
    result = model.step(np.ones(ACTION_DIM), 0.005)
    assert np.all(result.applied_command == 1.0)
    expected_forces = np.array(
        (
            (0.952406656250, 0.158601187500, 3.870241375000),
            (0.457870437500, 0.909703375000, 7.300617712500),
            (0.952406656250, -0.158601187500, 3.870241375000),
            (0.457870437500, -0.909703375000, 7.300617712500),
            (3.058405437500, -0.653428125000, -1.535113387500),
            (3.058405437500, 0.653428125000, -1.535113387500),
            (-1.386683375000, -0.835819250000, -2.739376937500),
            (-1.386683375000, 0.835819250000, -2.739376937500),
        )
    )
    assert np.allclose(result.forces_b_n, expected_forces, rtol=0.0, atol=1.0e-8)
    expected_wrench = np.concatenate(
        (
            np.sum(expected_forces, axis=0),
            np.sum(
                np.cross(np.asarray(AUV.thruster_positions_body_m), expected_forces),
                axis=0,
            ),
        )
    )
    assert np.allclose(result.wrench_b, expected_wrench)

    model.reset()
    reverse = model.step(-np.ones(ACTION_DIM), 0.005)
    expected_reverse_forces = np.array(
        (
            (0.371767550000, -0.670443252500, -7.831306000000),
            (-0.041407625000, -0.513367312500, -6.307440562500),
            (0.371767550000, 0.670443252500, -7.831306000000),
            (-0.041407625000, 0.513367312500, -6.307440562500),
            (-4.566026062500, -1.889365191875, -0.357713116250),
            (-4.566026062500, 1.889365191875, -0.357713116250),
            (5.888204875000, -2.108183577500, -1.785394187500),
            (5.888204875000, 2.108183577500, -1.785394187500),
        )
    )
    assert np.allclose(reverse.forces_b_n, expected_reverse_forces, rtol=0.0, atol=1.0e-8)

    model.reset()
    deadband_action = AUV.thruster_pwm_deadband_us / AUV.thruster_pwm_half_range_us
    deadband = model.step(np.full(ACTION_DIM, deadband_action), 0.005)
    assert np.array_equal(deadband.forces_b_n, np.zeros((ACTION_DIM, 3)))


def test_hydrodynamics_bridge_is_dissipative_and_adds_buoyancy() -> None:
    model = HydrodynamicsModel(_hydrodynamics_parameters())
    state = _vehicle_state(linear_velocity=(1.0, 0.0, 0.0))
    wrench = model.step(state, time_s=0.0, dt_s=0.005)
    assert wrench[0] < 0.0
    assert wrench[2] == pytest.approx(
        AUV.water_density_kg_m3 * AUV.displaced_volume_m3 * 9.81
    )
    assert np.dot(wrench[:3], state.linear_velocity_b) < 0.0


def test_validation_summary_enforces_tracking_and_action_gates() -> None:
    passing = summarize_validation(
        [0.1, 0.2, 0.3],
        np.zeros((3, ACTION_DIM)),
        max_position_rmse_m=0.5,
        max_action_clip_fraction=0.1,
    )
    assert passing["passed"] is True

    failing = summarize_validation(
        [0.8, 0.9],
        np.full((2, ACTION_DIM), 1.2),
        max_position_rmse_m=0.5,
        max_action_clip_fraction=0.1,
    )
    assert failing["passed"] is False
    assert len(failing["failures"]) == 2


def test_mujoco_model_and_entrypoint_exist_without_importing_optional_runtime() -> None:
    model_path = PROJECT_ROOT / "robot/assets/mujoco/t60_auv.xml"
    entrypoint = MUJOCO_WORKFLOW_ROOT / "validate_policy.py"
    document = ET.parse(model_path)
    body = document.find(".//body[@name='t60_auv']")
    joint = document.find(".//freejoint[@name='t60_auv_freejoint']")
    inertial = document.find(".//body[@name='t60_auv']/inertial")
    hull = document.find(".//body[@name='t60_auv']/geom[@name='hull']")
    sites = document.findall(".//site")

    assert body is not None
    assert joint is not None
    assert inertial is not None
    assert hull is not None
    assert float(inertial.attrib["mass"]) == pytest.approx(AUV.mass_kg)
    full_inertia = tuple(float(value) for value in inertial.attrib["fullinertia"].split())
    inertia = np.asarray(AUV.inertia_tensor_body_kg_m2)
    assert full_inertia == pytest.approx(
        (inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2])
    )
    hull_half_size = tuple(float(value) for value in hull.attrib["size"].split())
    assert hull_half_size == pytest.approx(np.asarray(AUV.visual_bounds_size_m) / 2.0)
    assert len(sites) == ACTION_DIM
    assert [site.attrib["name"] for site in sites] == [f"thruster_{label}" for label in AUV.thruster_labels]
    for site, expected_position in zip(sites, AUV.thruster_positions_body_m):
        actual_position = tuple(float(value) for value in site.attrib["pos"].split())
        assert actual_position == pytest.approx(expected_position)
    source = entrypoint.read_text(encoding="utf-8")
    assert "import mujoco" in source
    assert "summarize_validation" in source
    assert "onnxruntime" in source


def test_mujoco_full_inertia_velocity_is_converted_to_project_body_frame() -> None:
    mujoco = pytest.importorskip("mujoco")
    if not hasattr(mujoco, "MjModel"):
        pytest.skip("MuJoCo SDK is not installed")
    from simulation.mujoco.validate_policy import _vehicle_state as read_vehicle_state

    model = mujoco.MjModel.from_xml_path(str(PROJECT_ROOT / "robot/assets/mujoco/t60_auv.xml"))
    data = mujoco.MjData(model)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "auv")
    # Free-joint qvel is world translation followed by body angular velocity
    # at identity attitude. MuJoCo's local object velocity instead uses its
    # diagonalized inertia frame, so the bridge must request world velocity
    # and perform the body rotation itself.
    data.qvel[:] = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    mujoco.mj_forward(model, data)

    state = read_vehicle_state(mujoco, model, data, body_id)

    assert np.allclose(state.linear_velocity_b, (1.0, 2.0, 3.0))
    assert np.allclose(state.angular_velocity_b, (4.0, 5.0, 6.0))
