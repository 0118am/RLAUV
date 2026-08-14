"""Checks for selectable trajectory rewards and notebook command plumbing."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[5]


def _load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_package(name: str, relative_dir: str):
    package_dir = REPO_ROOT / relative_dir
    spec = importlib.util.spec_from_file_location(
        name,
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


reward_policies = _load_package("tracking_reward_policies", "simulation/isaac/agents/rewards")
experiment_tools = _load_module("trajectory_experiment_tools", "simulation/isaac/workflows/common/trajectory_experiment.py")


def _perfect_reward(policy_name: str) -> torch.Tensor:
    policy = reward_policies.TRACKING_REWARD_POLICIES[policy_name]
    zeros3 = torch.zeros((1, 3))
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    target_velocity = torch.tensor([[0.4, 0.0, 0.0]])
    zeros8 = torch.zeros((1, 8))
    reward_function = reward_policies.get_tracking_reward_function(policy_name)
    args = (
        policy.position_weight,
        policy.attitude_weight,
        policy.velocity_weight,
        policy.angular_velocity_weight,
        policy.forward_alignment_weight,
        policy.motion_alignment_weight,
        policy.action_weight,
        policy.action_rate_weight,
        policy.position_sigma,
        policy.attitude_sigma,
        policy.velocity_sigma,
        policy.angular_velocity_sigma,
        policy.forward_min_speed,
        zeros3,
        identity,
        target_velocity,
        zeros3,
        zeros3,
        identity,
        target_velocity,
        zeros8,
        zeros8,
    )
    if policy.requires_action_rate_limit:
        return reward_function(*args, torch.full((1, 1), 0.08))
    return reward_function(*args)


def test_named_policies_preserve_their_documented_positive_reward_maximum():
    assert reward_policies.available_tracking_reward_policies() == (
        "policy_0",
        "policy_1",
        "policy_2",
        "policy_3",
        "policy_4",
        "policy_5",
        "policy_6",
        "custom",
    )
    for policy_name in ("policy_0", "policy_1", "policy_2", "policy_3", "policy_4"):
        policy = reward_policies.TRACKING_REWARD_POLICIES[policy_name]
        assert torch.allclose(_perfect_reward(policy_name), torch.tensor([4.87]), atol=1.0e-6)
        assert math.isclose(policy.maximum_positive_reward, 4.87)

    policy_5 = reward_policies.TRACKING_REWARD_POLICIES["policy_5"]
    assert torch.allclose(_perfect_reward("policy_5"), torch.tensor([1.0]), atol=1.0e-6)
    assert math.isclose(policy_5.maximum_positive_reward, 1.0)
    assert policy_5.action_source == "applied"
    assert policy_5.termination_penalty == 1.0

    policy_6 = reward_policies.TRACKING_REWARD_POLICIES["policy_6"]
    assert torch.allclose(_perfect_reward("policy_6"), torch.tensor([1.0]), atol=1.0e-6)
    assert math.isclose(policy_6.maximum_positive_reward, 1.0)
    assert policy_6.action_source == "applied"
    assert policy_6.termination_penalty == 1.0


def test_policy_1_uses_control_effort_scales_that_suppress_command_chatter():
    policy = reward_policies.TRACKING_REWARD_POLICIES["policy_1"]

    assert policy.action_weight == 0.01
    assert policy.action_rate_weight == 0.05


def _isolated_reward(
    policy_name: str,
    *,
    scales=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    sigmas=(1.0, 1.0, 1.0, 1.0),
    root_pos=None,
    root_lin_vel_b=None,
    target_pos_w=None,
    target_lin_vel_b=None,
    actions=None,
    previous_actions=None,
) -> torch.Tensor:
    zeros3 = torch.zeros((1, 3))
    zeros8 = torch.zeros((1, 8))
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    reward_function = reward_policies.get_tracking_reward_function(policy_name)
    args = (
        *scales,
        *sigmas,
        1.0e-3,
        zeros3 if root_pos is None else root_pos,
        identity,
        zeros3 if root_lin_vel_b is None else root_lin_vel_b,
        zeros3,
        zeros3 if target_pos_w is None else target_pos_w,
        identity,
        zeros3 if target_lin_vel_b is None else target_lin_vel_b,
        zeros8 if actions is None else actions,
        zeros8 if previous_actions is None else previous_actions,
    )
    if policy_name in {"policy_5", "policy_6"}:
        return reward_function(*args, torch.full((1, 1), 0.08))
    return reward_function(*args)


def test_heading_reward_distinguishes_forward_and_reverse_velocity():
    zeros3 = torch.zeros((1, 3))
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    zeros8 = torch.zeros((1, 8))
    reward_function = reward_policies.get_tracking_reward_function("policy_1")

    def score(target_velocity):
        return reward_function(
            0.0,
            0.0,
            0.0,
            0.0,
            0.8,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0e-3,
            zeros3,
            identity,
            zeros3,
            zeros3,
            zeros3,
            identity,
            target_velocity,
            zeros8,
            zeros8,
        )

    assert torch.allclose(score(torch.tensor([[1.0, 0.0, 0.0]])), torch.tensor([0.8]))
    assert torch.allclose(score(torch.tensor([[-1.0, 0.0, 0.0]])), torch.tensor([0.0]))


def test_policy_2_gaussian_sigma_remains_a_half_reward_tolerance():
    reward = _isolated_reward(
        "policy_2",
        scales=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        sigmas=(0.7, 1.0, 1.0, 1.0),
        root_pos=torch.tensor([[0.7, 0.0, 0.0]]),
    )
    assert torch.allclose(reward, torch.tensor([0.5]), atol=1.0e-6)


def test_policy_3_requires_actual_motion_to_follow_the_nose():
    target_velocity = torch.tensor([[1.0, 0.0, 0.0]])

    def alignment_reward(actual_velocity):
        return _isolated_reward(
            "policy_3",
            scales=(0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0),
            root_lin_vel_b=actual_velocity,
            target_lin_vel_b=target_velocity,
        )

    assert torch.allclose(alignment_reward(target_velocity), torch.tensor([0.8]))
    assert torch.allclose(alignment_reward(torch.zeros((1, 3))), torch.tensor([0.0]))
    assert torch.allclose(alignment_reward(-target_velocity), torch.tensor([0.0]))


def test_policies_0_to_4_reward_nose_alignment_with_actual_motion():
    forward = torch.tensor([[1.0, 0.0, 0.0]])
    reverse = -forward
    for policy_name in ("policy_0", "policy_1", "policy_2", "policy_3", "policy_4"):
        forward_reward = _isolated_reward(
            policy_name,
            scales=(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
            root_lin_vel_b=forward,
            target_lin_vel_b=forward,
        )
        reverse_reward = _isolated_reward(
            policy_name,
            scales=(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
            root_lin_vel_b=reverse,
            target_lin_vel_b=forward,
        )
        assert torch.allclose(forward_reward, torch.tensor([1.0]), atol=1.0e-6)
        assert torch.allclose(reverse_reward, torch.tensor([0.0]), atol=1.0e-6)


def test_policy_5_penalizes_normalized_applied_action_rate_with_mean_over_thrusters():
    applied_now = torch.full((1, 8), 0.08)
    applied_previous = torch.zeros((1, 8))
    reward = _isolated_reward(
        "policy_5",
        scales=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.03),
        actions=applied_now,
        previous_actions=applied_previous,
    )
    # Effort is 0.02 * mean(0.08^2); rate is 0.03 * mean((0.08 / 0.08)^2).
    assert torch.allclose(reward, torch.tensor([-(0.02 * 0.08**2 + 0.03)]), atol=1.0e-6)


def test_policy_6_huber_residual_is_half_at_sigma_and_linear_afterward():
    half_reward = _isolated_reward(
        "policy_6",
        scales=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        sigmas=(0.7, 1.0, 1.0, 1.0),
        root_pos=torch.tensor([[0.7, 0.0, 0.0]]),
    )
    large_error_reward = _isolated_reward(
        "policy_6",
        scales=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        sigmas=(0.7, 1.0, 1.0, 1.0),
        root_pos=torch.tensor([[1.4, 0.0, 0.0]]),
    )
    assert torch.allclose(half_reward, torch.tensor([0.5]), atol=1.0e-6)
    assert torch.allclose(large_error_reward, torch.tensor([-0.5]), atol=1.0e-6)


def test_policy_4_matches_policy_1_penalty_at_action_bounds_but_suppresses_small_actions_more():
    policy_1 = reward_policies.TRACKING_REWARD_POLICIES["policy_1"]
    policy_4 = reward_policies.TRACKING_REWARD_POLICIES["policy_4"]
    upper = torch.ones((1, 8))
    lower = -torch.ones((1, 8))
    policy_1_bound = _isolated_reward(
        "policy_1",
        scales=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, policy_1.action_weight, policy_1.action_rate_weight),
        actions=upper,
        previous_actions=lower,
    )
    policy_4_bound = _isolated_reward(
        "policy_4",
        scales=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, policy_4.action_weight, policy_4.action_rate_weight),
        actions=upper,
        previous_actions=lower,
    )
    assert torch.allclose(policy_1_bound, policy_4_bound, atol=1.0e-6)

    small = torch.full((1, 8), 0.1)
    policy_1_small = _isolated_reward(
        "policy_1",
        scales=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, policy_1.action_weight, 0.0),
        actions=small,
        previous_actions=small,
    )
    policy_4_small = _isolated_reward(
        "policy_4",
        scales=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, policy_4.action_weight, 0.0),
        actions=small,
        previous_actions=small,
    )
    assert policy_4_small < policy_1_small


def test_policy_application_legacy_alias_and_custom_mode():
    cfg = SimpleNamespace(tracking_reward_profile="policy_1")
    reward_policies.apply_tracking_reward_policy(cfg)
    assert cfg.rew_scale_pos == 3.0
    assert cfg.rew_scale_forward == 0.4
    assert cfg.rew_scale_motion_alignment == 0.4

    custom = SimpleNamespace(tracking_reward_profile="custom", rew_scale_pos=1.23)
    assert reward_policies.apply_tracking_reward_policy(custom) is None
    assert custom.rew_scale_pos == 1.23

    legacy = SimpleNamespace(tracking_reward_profile="heading_v1")
    reward_policies.apply_tracking_reward_policy(legacy)
    assert legacy.tracking_reward_profile == "policy_1"

    anti_chatter = SimpleNamespace(tracking_reward_profile="policy_4")
    reward_policies.apply_tracking_reward_policy(anti_chatter)
    assert anti_chatter.rew_scale_actions == 0.08
    assert anti_chatter.rew_scale_action_rate == 1.6

    compact = SimpleNamespace(tracking_reward_profile="policy_5")
    reward_policies.apply_tracking_reward_policy(compact)
    assert compact.rew_scale_pos == 0.55
    assert compact.rew_scale_forward == 0.0
    assert compact.rew_scale_motion_alignment == 0.0
    assert compact.rew_action_source == "applied"
    assert compact.rew_scale_terminated == 1.0


def test_train_and_eval_commands_carry_the_reward_policy():
    spec = experiment_tools.ExperimentSpec(isaaclab_root=Path("/tmp/IsaacLab"))
    train = experiment_tools.TrainRequest(reward_profile="policy_1")
    train_command = experiment_tools.build_train_command(spec, train)
    assert "env.tracking_reward_profile=policy_1" in train_command

    evaluation = experiment_tools.EvalRequest(reward_profile="policy_1")
    eval_command = experiment_tools.build_eval_command(
        spec,
        evaluation,
        "2026-01-01_00-00-00_trajectory_policy_1",
        "model_300.pt",
        "racetrack",
    )
    profile_index = eval_command.index("--reward_profile")
    assert eval_command[profile_index + 1] == "policy_1"
