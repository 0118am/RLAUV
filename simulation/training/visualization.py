"""Isaac UI and geometry helpers for AUV environments."""

from __future__ import annotations

import torch
from isaaclab.envs.ui import BaseEnvWindow

from common.tensor_math import quat_apply_wxyz


_AXIS_COLORS = (
    (1.0, 0.05, 0.05),  # +X: red, body forward
    (0.05, 1.0, 0.05),  # +Y: green, body left
    (0.05, 0.25, 1.0),  # +Z: blue, body up
)


def frame_line_data(
    origins_w: torch.Tensor,
    quaternions_wxyz: torch.Tensor,
    *,
    axis_length: float,
    alpha: float,
    shaft_thickness: float,
) -> tuple[list, list, list, list]:
    """Build colored XYZ shafts and arrowheads for batched world poses."""
    if origins_w.ndim != 2 or origins_w.shape[-1] != 3:
        raise ValueError(f"Expected origins with shape [N, 3], got {tuple(origins_w.shape)}.")
    if quaternions_wxyz.shape != (origins_w.shape[0], 4):
        raise ValueError(
            f"Expected quaternions with shape [{origins_w.shape[0]}, 4], "
            f"got {tuple(quaternions_wxyz.shape)}."
        )

    count = origins_w.shape[0]
    local_axes = torch.eye(3, dtype=origins_w.dtype, device=origins_w.device)
    local_axes = local_axes.unsqueeze(0).expand(count, -1, -1)
    frame_quaternions = quaternions_wxyz.unsqueeze(1).expand(-1, 3, -1)
    directions_w = quat_apply_wxyz(
        frame_quaternions.reshape(-1, 4),
        local_axes.reshape(-1, 3),
    ).reshape(count, 3, 3)

    origins = origins_w.unsqueeze(1).expand(-1, 3, -1)
    tips = origins + float(axis_length) * directions_w
    # Use the following frame axis as an arrowhead width direction. Since the
    # frame is orthonormal, every arrowhead remains perpendicular to its shaft.
    perpendicular_w = torch.roll(directions_w, shifts=-1, dims=1)
    head_base = tips - (0.20 * float(axis_length)) * directions_w
    head_half_width = 0.075 * float(axis_length)
    head_side_positive = head_base + head_half_width * perpendicular_w
    head_side_negative = head_base - head_half_width * perpendicular_w

    starts = torch.cat((origins, tips, tips), dim=1).reshape(-1, 3)
    ends = torch.cat((tips, head_side_positive, head_side_negative), dim=1).reshape(-1, 3)
    axis_colors = [(*color, float(alpha)) for color in _AXIS_COLORS] * count
    colors = axis_colors * 3
    frame_thicknesses = [float(shaft_thickness)] * 3
    frame_thicknesses += [max(1.0, float(shaft_thickness) - 1.0)] * 6
    thicknesses = frame_thicknesses * count
    return (
        starts.detach().cpu().tolist(),
        ends.detach().cpu().tolist(),
        colors,
        thicknesses,
    )


class AUVTrajEnvWindow(BaseEnvWindow):
    """IsaacLab window extension that exposes AUV targets for debugging."""

    def _visualize_manager(self, title: str, class_name: str):
        """Skip manager widgets that do not exist in a DirectRLEnv."""
        return None

    def __init__(self, env, window_name: str = "IsaacLab"):
        super().__init__(env, window_name)
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    self._create_debug_vis_ui_element("targets", self.env)


class AUVVisualizationMixin:
    """Draw pose-aligned frames without USD PointInstancers."""

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis and not hasattr(self, "_auv_debug_draw"):
            # Import lazily: headless jobs with debug visualization disabled do
            # not need to load or acquire the viewport drawing interface.
            from isaacsim.util.debug_draw import _debug_draw

            self._auv_debug_draw = _debug_draw.acquire_debug_draw_interface()
        self._auv_debug_draw_visible = bool(debug_vis)
        if not debug_vis and hasattr(self, "_auv_debug_draw"):
            self._auv_debug_draw.clear_lines()

    def _debug_vis_callback(self, event):
        if not getattr(self, "_auv_debug_draw_visible", False):
            return

        # Rendering can run between policy callbacks. Synchronize the command
        # cache with the current episode clock before drawing.
        self._update_tracking_targets()
        body_data = frame_line_data(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            axis_length=0.35,
            alpha=1.0,
            shaft_thickness=4.0,
        )
        target_data = frame_line_data(
            self._target_pos_w,
            self._target_quat_w,
            axis_length=0.25,
            alpha=0.65,
            shaft_thickness=2.0,
        )
        starts = body_data[0] + target_data[0]
        ends = body_data[1] + target_data[1]
        colors = body_data[2] + target_data[2]
        thicknesses = body_data[3] + target_data[3]

        # Debug Draw owns one transient line list. Replacing it each rendered
        # frame avoids accumulation while completely bypassing Fabric's broken
        # PointInstancer prototype initialization path.
        self._auv_debug_draw.clear_lines()
        self._auv_debug_draw.draw_lines(starts, ends, colors, thicknesses)
