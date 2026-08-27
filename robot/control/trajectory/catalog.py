"""Stable trajectory identifiers shared by training, evaluation, and deployment."""

TRAJECTORY_GENERATOR_VERSION = "curve_v5"

CIRCLE = 0
LISSAJOUS = 1
AXIS_SINE = 2
WAVY_LOOP = 3
BREATHING_LOOP = 4
CHIRP = 5
RACETRACK = 6
RANDOM_SMOOTH = 7
LATERAL_WAVE = 8
VERTICAL_WAVE = 9
SPATIAL_HELIX = 10
REVERSE_SPATIAL_HELIX = 11

TRAJECTORY_TYPE_IDS = {
    "lissajous": LISSAJOUS,
    "surge_sine": AXIS_SINE,
    "sway_sine": AXIS_SINE,
    "heave_sine": AXIS_SINE,
    "lateral_wave": LATERAL_WAVE,
    "vertical_wave": VERTICAL_WAVE,
    "wavy_loop": WAVY_LOOP,
    "breathing_loop": BREATHING_LOOP,
    "chirp": CHIRP,
    "racetrack": RACETRACK,
    "random_smooth": RANDOM_SMOOTH,
    "spatial_helix": SPATIAL_HELIX,
    "reverse_spatial_helix": REVERSE_SPATIAL_HELIX,
}
TRAJECTORY_AXIS_BY_NAME = {
    "surge_sine": 0,
    "sway_sine": 1,
    "heave_sine": 2,
}
EVALUATION_TRAJECTORY_NAMES = tuple(TRAJECTORY_TYPE_IDS)
SPEED_CONTROLLED_TYPES = (
    LISSAJOUS,
    AXIS_SINE,
    LATERAL_WAVE,
    VERTICAL_WAVE,
    SPATIAL_HELIX,
    REVERSE_SPATIAL_HELIX,
)
