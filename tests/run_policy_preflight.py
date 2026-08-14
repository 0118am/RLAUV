"""Fast checks to run before launching a long policy-training job."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.simulation.isaac.agents.rewards import test_policies as reward_policy_tests
from tests.simulation.isaac.envs.auv.trajectory import test_guidance


PREFLIGHT_CHECKS = (
    reward_policy_tests.test_named_policies_preserve_their_documented_positive_reward_maximum,
    reward_policy_tests.test_heading_reward_distinguishes_forward_and_reverse_velocity,
    reward_policy_tests.test_policy_2_gaussian_sigma_remains_a_half_reward_tolerance,
    reward_policy_tests.test_policy_3_requires_actual_motion_to_follow_the_nose,
    reward_policy_tests.test_policies_0_to_4_reward_nose_alignment_with_actual_motion,
    reward_policy_tests.test_policy_5_penalizes_normalized_applied_action_rate_with_mean_over_thrusters,
    reward_policy_tests.test_policy_6_huber_residual_is_half_at_sigma_and_linear_afterward,
    reward_policy_tests.test_policy_4_matches_policy_1_penalty_at_action_bounds_but_suppresses_small_actions_more,
    reward_policy_tests.test_policy_application_legacy_alias_and_custom_mode,
    reward_policy_tests.test_train_and_eval_commands_carry_the_reward_policy,
    test_guidance.test_body_x_aligns_with_three_dimensional_velocity,
    test_guidance.test_near_zero_velocity_keeps_previous_attitude,
    test_guidance.test_quaternion_sign_stays_continuous_across_yaw_wrap,
    test_guidance.test_quaternion_step_returns_shortest_body_angular_velocity,
)


def main() -> None:
    for check in PREFLIGHT_CHECKS:
        check()
        print(f"[PASS] {check.__module__}.{check.__name__}")
    print(f"Policy-training preflight passed: {len(PREFLIGHT_CHECKS)} checks.")


if __name__ == "__main__":
    main()
