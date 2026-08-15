"""Stable trajectory identifiers shared by training, evaluation, and deployment."""

TRAJECTORY_GENERATOR_VERSION = "curve_v2"

CIRCLE = 0
LISSAJOUS = 1
AXIS_SINE = 2
WAVY_LOOP = 3
BREATHING_LOOP = 4
CHIRP = 5
RACETRACK = 6
RANDOM_SMOOTH = 7
LATERAL_SINE = 8
VERTICAL_SINE = 9
SPATIAL_HELIX = 10

# Public evaluation names preserve the existing helix/spiral aliases used by
# run directories and notebooks.
TRAJECTORY_TYPE_IDS = {
    "lissajous": LISSAJOUS,
    "helix": WAVY_LOOP,
    "spiral": BREATHING_LOOP,
    "chirp": CHIRP,
    "racetrack": RACETRACK,
    "random_smooth": RANDOM_SMOOTH,
    "lateral_sine": LATERAL_SINE,
    "vertical_sine": VERTICAL_SINE,
    "spatial_helix": SPATIAL_HELIX,
}
EVALUATION_TRAJECTORY_NAMES = tuple(TRAJECTORY_TYPE_IDS)
SPEED_CONTROLLED_TYPES = (LATERAL_SINE, VERTICAL_SINE, SPATIAL_HELIX)
