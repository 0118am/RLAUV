"""Stable public API for configuring and managing AUV training campaigns."""

from .campaign import (
    TrainingCampaign,
    build_training_campaign,
)
from .recipe import (
    DEFAULT_TRAINING_RECIPE,
    PROJECT_ROOT,
    RunInputPaths,
    TrainingRecipe,
    apply_training_recipe,
    apply_trajectory_kinematic_limits,
    load_training_recipe,
    materialize_run_inputs,
    run_input_paths,
)

__all__ = [
    "DEFAULT_TRAINING_RECIPE",
    "PROJECT_ROOT",
    "TrainingCampaign",
    "RunInputPaths",
    "TrainingRecipe",
    "apply_training_recipe",
    "apply_trajectory_kinematic_limits",
    "build_training_campaign",
    "load_training_recipe",
    "materialize_run_inputs",
    "run_input_paths",
]
