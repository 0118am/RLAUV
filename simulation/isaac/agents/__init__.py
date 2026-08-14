# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trajectory-policy packages.

Keep this namespace import-light: notebook command construction imports only
``ppo.architectures`` and must not start IsaacLab just to select a profile.
Task registration imports ``ppo.config`` explicitly when it needs IsaacLab
runner classes.
"""
