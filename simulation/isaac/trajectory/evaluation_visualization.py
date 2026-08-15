"""Optional Isaac Sim visualization for trajectory evaluation."""

from collections import deque

import torch


class TrajectoryEvalVisualizer:
    """Draw one desired/actual trajectory pair without affecting evaluation data."""

    def __init__(
        self,
        enabled: bool,
        trajectory: str,
        checkpoint_name: str,
        max_points: int,
        stride: int,
    ) -> None:
        self.enabled = enabled
        self.trajectory = trajectory
        self.checkpoint_name = checkpoint_name
        self.stride = max(1, stride)
        self.desired_points = deque(maxlen=max(2, max_points))
        self.actual_points = deque(maxlen=max(2, max_points))
        self._draw = None
        self._labels = {}

        if self.enabled:
            self._init_debug_draw()
            self._init_status_window()

    def _init_debug_draw(self) -> None:
        try:
            try:
                from isaacsim.util.debug_draw import _debug_draw
            except Exception:
                from omni.isaac.debug_draw import _debug_draw

            self._draw = _debug_draw.acquire_debug_draw_interface()
            self._clear_draw()
        except Exception as exc:
            self.enabled = False
            print(f"[WARN]: Live trajectory drawing is unavailable: {exc}")

    def _init_status_window(self) -> None:
        try:
            import omni.ui as ui

            self._window = ui.Window("Trajectory Eval", width=360, height=150)
            with self._window.frame:
                with ui.VStack(spacing=4):
                    ui.Label(f"trajectory: {self.trajectory}")
                    ui.Label(f"checkpoint: {self.checkpoint_name}")
                    ui.Label("blue: desired target trail")
                    ui.Label("orange: actual AUV trail")
                    self._labels["time"] = ui.Label("time: 0.00 s")
                    self._labels["error"] = ui.Label("pos err: -- m | vel err: -- m/s")
        except Exception as exc:
            print(f"[WARN]: Trajectory status window is unavailable: {exc}")

    @staticmethod
    def _point(tensor: torch.Tensor) -> tuple[float, float, float]:
        values = tensor.detach().cpu().tolist()
        return float(values[0]), float(values[1]), float(values[2])

    def update(
        self,
        step: int,
        time_s: float,
        desired_pos_w: torch.Tensor,
        actual_pos_w: torch.Tensor,
        position_error: float,
        velocity_error: float,
    ) -> None:
        if not self.enabled:
            return

        self.desired_points.append(self._point(desired_pos_w))
        self.actual_points.append(self._point(actual_pos_w))
        if step % self.stride == 0:
            self._draw_trails()
            self._update_labels(time_s, position_error, velocity_error)

    def _draw_trails(self) -> None:
        if self._draw is None:
            return
        desired = list(self.desired_points)
        actual = list(self.actual_points)
        start_points = desired[:-1] + actual[:-1]
        end_points = desired[1:] + actual[1:]
        colors = [(0.1, 0.45, 1.0, 1.0)] * max(0, len(desired) - 1)
        colors += [(1.0, 0.45, 0.05, 1.0)] * max(0, len(actual) - 1)
        self._clear_draw()
        if start_points:
            self._draw.draw_lines(start_points, end_points, colors, [3.0] * len(start_points))
        if desired and actual:
            self._draw.draw_points(
                [desired[-1], actual[-1]],
                [(1.0, 0.95, 0.05, 1.0), (1.0, 0.95, 0.95, 1.0)],
                [18.0, 12.0],
            )

    def _clear_draw(self) -> None:
        if self._draw is None:
            return
        if hasattr(self._draw, "clear_lines"):
            self._draw.clear_lines()
        if hasattr(self._draw, "clear_points"):
            self._draw.clear_points()

    def _update_labels(self, time_s: float, position_error: float, velocity_error: float) -> None:
        time_label = self._labels.get("time")
        error_label = self._labels.get("error")
        if time_label is not None:
            time_label.text = f"time: {time_s:.2f} s"
        if error_label is not None:
            error_label.text = f"pos err: {position_error:.3f} m | vel err: {velocity_error:.3f} m/s"

    def set_status(self, message: str) -> None:
        label = self._labels.get("time")
        if label is not None:
            label.text = message
