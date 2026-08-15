"""Shared names and matrix conventions for PMM identification."""

from __future__ import annotations

import numpy as np


DOFS = ("sway", "heave", "pitch", "yaw")


DOF_RESPONSE = {"sway": "Y", "heave": "Z", "pitch": "M", "yaw": "N"}


DOF_VARIABLE = {"sway": "v", "heave": "w", "pitch": "q", "yaw": "r"}


DOF_INDEX = {"sway": 1, "heave": 2, "pitch": 4, "yaw": 5}


DOF_FAMILY = {
    "sway": "pure_sway",
    "heave": "vertical_sway",
    "pitch": "vertical_yaw",
    "yaw": "pure_yaw",
}


DOF_RAW_KIND = {"sway": "sway", "heave": "sway", "pitch": "yaw", "yaw": "yaw"}


DOF_TIMING_GROUP = {"sway": "horizontal", "heave": "vertical", "pitch": "vertical", "yaw": "horizontal"}


SENSOR_COLUMNS = ("TX", "TY", "TZ", "FX", "FY", "FZ")


ROW_ORDER = ("X", "Y", "Z", "K", "M", "N")


COLUMN_ORDER = ("u", "v", "w", "p", "q", "r")


HORIZONTAL_SENSOR_TO_H_WRENCH = np.asarray(
    [
        [0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
    ],
    dtype=float,
)


VERTICAL_SENSOR_TO_H_WRENCH = HORIZONTAL_SENSOR_TO_H_WRENCH.copy()


COEFFICIENT_FIELDS = (
    "dof", "experiment", "response", "variable", "term", "coefficient", "unit",
    "nondimensional", "nominal_frequency_hz", "mean_estimated_frequency_hz",
    "min_estimated_frequency_hz", "max_estimated_frequency_hz", "included_repeats",
    "reference_speed_m_s", "full_r2", "fit_quality_flag",
    "standardized_condition_number", "assumption_status",
)


DIAGNOSTIC_FIELDS = (
    "dof", "repeat", "file_id", "nominal_frequency_hz", "estimated_frequency_hz",
    "timing_group", "gather_file", "sensor_file", "status", "exclusion_reason",
    "gather_rows", "sensor_rows", "required_sensor_rows", "paired_rows", "motion_fit_r2",
    "u_mean_m_s", "surge_force_mean_n", "surge_force_oscillatory_rms_n",
    "q_amplitude", "condition_raw", "sensor_time_shift_ms", "quality_flag",
)


TIMING_FIELDS = (
    "sensor_time_shift_ms", "timing_group", "dof", "nominal_frequency_hz",
    "mean_estimated_frequency_hz", "term", "coefficient", "full_r2",
)
