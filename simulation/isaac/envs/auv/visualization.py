"""Optional IsaacLab debug visualization for AUV environments."""

from __future__ import annotations

from isaaclab.envs.ui import BaseEnvWindow

from .visualization_geometry import frame_line_data


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
