"""Stable public API for configuring and managing AUV training campaigns."""

from .campaign import (
    CampaignProcess,
    TrainingCampaign,
    build_training_campaign,
    campaign_processes,
    follow_campaign_log,
    latest_launcher_record,
    launch_campaign,
    launch_or_attach_campaign,
    render_campaign_status,
    stop_campaign,
)
from .recipe import (
    DEFAULT_TRAINING_RECIPE,
    PROJECT_ROOT,
    RunInputPaths,
    TrainingRecipe,
    load_training_recipe,
    materialize_run_inputs,
)
from .manifest import RunManifest, load_run_manifest

__all__ = [
    "CampaignProcess",
    "DEFAULT_TRAINING_RECIPE",
    "PROJECT_ROOT",
    "TrainingCampaign",
    "RunInputPaths",
    "RunManifest",
    "TrainingRecipe",
    "build_training_campaign",
    "campaign_processes",
    "follow_campaign_log",
    "latest_launcher_record",
    "launch_campaign",
    "launch_or_attach_campaign",
    "load_run_manifest",
    "load_training_recipe",
    "materialize_run_inputs",
    "render_campaign_status",
    "stop_campaign",
]
