"""Launch IsaacLab's RSL-RL trainer with GPU-batched episode bookkeeping."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import runpy
import sys

repo_root = Path(__file__).resolve().parents[4]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

def _require_training_environment() -> None:
    """Fail early when training is launched outside ``env_isaaclab``."""

    expected = "env_isaaclab"
    active_conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    interpreter_env = Path(sys.prefix).name
    if expected not in {active_conda_env, interpreter_env}:
        raise RuntimeError(
            "Trajectory training requires the env_isaaclab environment. "
            "Run `conda activate env_isaaclab` before invoking isaaclab.sh."
        )


def main() -> None:
    _require_training_environment()

    import rsl_rl.runners
    import rsl_rl.runners.on_policy_runner as on_policy_runner_module

    from simulation.isaac.agents.ppo.algorithm import RolloutAdaptivePPO
    from simulation.isaac.agents.ppo.runner import GpuBatchedOnPolicyRunner

    isaaclab_root = Path.cwd()
    official_script = isaaclab_root / "scripts" / "reinforcement_learning" / "rsl_rl" / "train.py"
    if not official_script.is_file():
        raise FileNotFoundError(f"IsaacLab trainer not found: {official_script}")

    # The official script imports the public symbol after launching Isaac Sim.
    # Replacing that symbol keeps all official CLI/Hydra setup unchanged.
    # Its runner resolves algorithm class names in the on_policy_runner module,
    # so register the AUV rollout-level PPO there before construction.
    on_policy_runner_module.RolloutAdaptivePPO = RolloutAdaptivePPO
    rsl_rl.runners.OnPolicyRunner = GpuBatchedOnPolicyRunner
    sys.path.insert(0, str(official_script.parent))

    class _DirectIoDescriptorFilter(logging.Filter):
        """Drop one unconditional upstream warning irrelevant to DirectRLEnv."""

        def filter(self, record: logging.LogRecord) -> bool:
            return not record.getMessage().startswith(
                "IO descriptors are only supported for manager based RL environments."
            )

    direct_logger = logging.getLogger("__main__")
    log_filter = _DirectIoDescriptorFilter()
    direct_logger.addFilter(log_filter)
    try:
        runpy.run_path(str(official_script), run_name="__main__")
    finally:
        direct_logger.removeFilter(log_filter)


if __name__ == "__main__":
    main()
