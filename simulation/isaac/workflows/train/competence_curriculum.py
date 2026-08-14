"""Detached checkpoint-gated trajectory curriculum supervisor.

The notebook writes one JSON campaign specification, then starts this script
in its own session.  Training and both held-out evaluations are therefore
separate child processes and survive a VS Code/Jupyter disconnect.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.isaac.workflows.common.trajectory_experiment import (  # noqa: E402
    CompetenceGateCriteria,
    ExperimentSpec,
    TrainRequest,
    TrajectoryCurriculumRequest,
    run_competence_gate_cycle,
)


def _trajectory_curriculum(payload: dict[str, Any]) -> TrajectoryCurriculumRequest:
    return TrajectoryCurriculumRequest(
        enabled=bool(payload["enabled"]),
        amplitude_x_range=tuple(payload["amplitude_x_range"]),
        amplitude_y_range=tuple(payload["amplitude_y_range"]),
        amplitude_z_range=tuple(payload["amplitude_z_range"]),
        period_range=tuple(payload["period_range"]),
        stage_steps=tuple(payload["stage_steps"]),
        stage_0_types=tuple(payload["stage_0_types"]),
        stage_1_types=tuple(payload["stage_1_types"]),
        stage_2_types=tuple(payload["stage_2_types"]),
        stage_3_types=tuple(payload["stage_3_types"]),
        amplitude_scales=tuple(payload["amplitude_scales"]),
        vertical_amplitude_scales=tuple(payload["vertical_amplitude_scales"]),
        period_min_by_stage=tuple(payload["period_min_by_stage"]),
        period_max_by_stage=tuple(payload["period_max_by_stage"]),
        speed_levels_mps=tuple(payload.get("speed_levels_mps", (0.1, 0.2, 0.3, 0.4))),
    )


def _criteria(payload: dict[str, Any]) -> CompetenceGateCriteria:
    values = dict(payload)
    for name in (
        "nominal_position_error_p95_m",
        "nominal_velocity_rmse_mps",
        "robust_position_error_p95_m",
        "robust_velocity_rmse_mps",
    ):
        if name in values:
            values[name] = tuple(values[name])
    return CompetenceGateCriteria(**values)


def _load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        return dict(default)
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a checkpoint-gated AUV trajectory curriculum campaign.")
    parser.add_argument("--config", type=Path, required=True, help="JSON campaign specification written by trajectory_train.ipynb.")
    parser.add_argument("--restart", action="store_true", help="Discard only the supervisor state, not prior checkpoints.")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    spec_data = config["experiment"]
    spec = ExperimentSpec(
        isaaclab_root=Path(spec_data["isaaclab_root"]),
        mlp_architecture=spec_data["mlp_architecture"],
        task_name=spec_data.get("task_name", "Isaac-AUV-Traj-Direct-v1"),
    )
    train_data = dict(config["train"])
    train_data["trajectory_curriculum"] = _trajectory_curriculum(train_data["trajectory_curriculum"])
    request = TrainRequest(**train_data)
    criteria = _criteria(config.get("criteria", {}))
    segment_iterations = int(config["segment_iterations"])
    total_iterations = int(config["total_iterations"])
    if segment_iterations <= 0 or total_iterations <= 0:
        raise ValueError("segment_iterations and total_iterations must be positive.")

    state_path = Path(config["state_path"])
    initial_state = {
        "status": "running",
        "completed_iterations": 0,
        "stage": 0,
        "consecutive_passes": 0,
        "latest_run": "",
        "latest_checkpoint": "",
        "history": [],
    }
    state = dict(initial_state) if args.restart else _load_json(state_path, initial_state)
    if state.get("status") == "complete":
        print(f"[CURRICULUM] already complete: {state_path}")
        return
    _write_json_atomic(state_path, state)

    while int(state["completed_iterations"]) < total_iterations:
        completed_iterations = int(state["completed_iterations"])
        remaining = total_iterations - completed_iterations
        this_segment = min(segment_iterations, remaining)
        stage = int(state["stage"])
        disturbance_offset = completed_iterations * int(request.rollout_steps_per_env or 0)
        print(
            f"[CURRICULUM] segment={this_segment} completed={state['completed_iterations']}/{total_iterations} "
            f"stage={stage} dr_step_offset={disturbance_offset} streak={state['consecutive_passes']}"
        )
        run_name, checkpoint, decision = run_competence_gate_cycle(
            spec,
            request,
            stage=stage,
            segment_iterations=this_segment,
            completed_iterations=completed_iterations,
            previous_consecutive_passes=int(state["consecutive_passes"]),
            resume_load_run=str(state.get("latest_run", "")),
            resume_checkpoint=str(state.get("latest_checkpoint", "")),
            criteria=criteria,
            execute=True,
        )
        if run_name is None or checkpoint is None or decision is None:
            raise RuntimeError("Gate cycle did not produce a trained checkpoint and decision.")
        state["completed_iterations"] = completed_iterations + this_segment
        state["stage"] = decision.next_stage
        state["consecutive_passes"] = decision.consecutive_passes
        state["latest_run"] = run_name
        state["latest_checkpoint"] = checkpoint
        state["history"].append(
            {
                "run": run_name,
                "checkpoint": checkpoint,
                "segment_iterations": this_segment,
                "disturbance_curriculum_global_step_offset": disturbance_offset,
                "decision": {
                    "evaluated_stage": decision.evaluated_stage,
                    "next_stage": decision.next_stage,
                    "passed": decision.passed,
                    "promoted": decision.promoted,
                    "consecutive_passes": decision.consecutive_passes,
                    "metrics": decision.metrics,
                },
            }
        )
        _write_json_atomic(state_path, state)

    state["status"] = "complete"
    _write_json_atomic(state_path, state)
    print(f"[CURRICULUM] complete: {state_path}")


if __name__ == "__main__":
    main()
