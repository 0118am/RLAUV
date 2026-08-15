"""Stable public API for configuring and managing AUV training campaigns."""

from simulation.isaac.training_campaign import (
    CampaignProcess,
    TrainingCampaign,
    build_default_campaign,
    campaign_processes,
    follow_campaign_log,
    latest_launcher_record,
    launch_campaign,
    launch_or_attach_campaign,
    render_campaign_status,
    stop_campaign,
)
from simulation.isaac.training_profiles import (
    DEFAULT_DOMAIN_RANDOMIZATION_SPEC,
    DEFAULT_ENVIRONMENT_PROFILE,
    PROJECT_ROOT,
    TrainingProfilePaths,
    default_trajectory_curriculum,
    materialize_training_profiles,
)

__all__ = [
    "CampaignProcess",
    "DEFAULT_DOMAIN_RANDOMIZATION_SPEC",
    "DEFAULT_ENVIRONMENT_PROFILE",
    "PROJECT_ROOT",
    "TrainingCampaign",
    "TrainingProfilePaths",
    "build_default_campaign",
    "campaign_processes",
    "default_trajectory_curriculum",
    "follow_campaign_log",
    "latest_launcher_record",
    "launch_campaign",
    "launch_or_attach_campaign",
    "materialize_training_profiles",
    "render_campaign_status",
    "stop_campaign",
]
