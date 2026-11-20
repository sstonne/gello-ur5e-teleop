"""Lightweight training and evaluation visualizations for the rope task."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


COLORS = {
    "initial": (25, 25, 25),      # BGR: black
    "current": (220, 80, 30),     # BGR: blue
    "goal": (30, 30, 230),        # BGR: red
}


def _polyline_pixels(centerline, width, height, workspace_low, workspace_high):
    points = np.asarray(centerline, dtype=np.float32)
    low = np.asarray(workspace_low, dtype=np.float32)
    high = np.asarray(workspace_high, dtype=np.float32)
    margin = 32
    x = margin + (points[:, 0] - low[0]) / (high[0] - low[0]) * (width - 2 * margin)
    y = height - margin - (points[:, 1] - low[1]) / (high[1] - low[1]) * (height - 2 * margin)
    return np.rint(np.stack([x, y], axis=1)).astype(np.int32)


def draw_centerline_panel(
    initial,
    current,
    goal,
    workspace_low,
    workspace_high,
    width=512,
    height=512,
    title=None,
):
    """Draw initial/current/goal centerlines without invoking the GS renderer."""
    panel = np.full((height, width, 3), 255, dtype=np.uint8)
    for name, points in (("initial", initial), ("current", current), ("goal", goal)):
        pixels = _polyline_pixels(points, width, height, workspace_low, workspace_high)
        cv2.polylines(panel, [pixels], False, COLORS[name], 2, cv2.LINE_AA)
        for point in pixels:
            cv2.circle(panel, tuple(point), 3, COLORS[name], -1, cv2.LINE_AA)

    labels = (("initial", 20), ("current", 45), ("goal", 70))
    for name, y in labels:
        cv2.line(panel, (18, y), (45, y), COLORS[name], 3, cv2.LINE_AA)
        cv2.putText(panel, name, (54, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS[name], 1, cv2.LINE_AA)
    if title:
        cv2.putText(panel, title, (18, height - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
    return panel


class RopeMonitoringCallback(BaseCallback):
    """Log task metrics and save state-only centerline snapshots during learning."""

    def __init__(
        self,
        output_dir,
        snapshot_frequency,
        workspace_low,
        workspace_high,
        success_curriculum=None,
    ):
        super().__init__(verbose=0)
        self.output_dir = Path(output_dir)
        self.snapshot_dir = self.output_dir / "centerline_snapshots"
        self.snapshot_frequency = int(snapshot_frequency)
        self.workspace_low = np.asarray(workspace_low, dtype=np.float32)
        self.workspace_high = np.asarray(workspace_high, dtype=np.float32)
        self.initial = None
        self.goal = None
        self.best_distance = np.inf
        self.success_curriculum = success_curriculum
        self.curriculum_tightened = False

    @staticmethod
    def _first_observation(observation):
        return {key: np.asarray(value)[0] for key, value in observation.items()}

    def _on_training_start(self):
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        obs = self._first_observation(self.model._last_obs)
        self.initial = obs["rope_centerline"].copy()
        self.goal = obs["goal_centerline"].copy()
        if self.success_curriculum and bool(self.success_curriculum.enabled):
            start_threshold = float(self.success_curriculum.start_threshold)
            self.training_env.env_method("set_success_threshold", start_threshold)
            self.logger.record("task/success_threshold", start_threshold)
        self._save_snapshot(obs["rope_centerline"], 0, "training start")

    def _save_snapshot(self, current, step, title):
        panel = draw_centerline_panel(
            self.initial,
            current,
            self.goal,
            self.workspace_low,
            self.workspace_high,
            title=title,
        )
        cv2.imwrite(str(self.snapshot_dir / f"step_{step:08d}.png"), panel)

    def _on_step(self):
        if (
            self.success_curriculum
            and bool(self.success_curriculum.enabled)
            and not self.curriculum_tightened
            and self.num_timesteps >= int(self.success_curriculum.tighten_at_step)
        ):
            final_threshold = float(self.success_curriculum.final_threshold)
            self.training_env.env_method("set_success_threshold", final_threshold)
            self.curriculum_tightened = True
        if self.success_curriculum and bool(self.success_curriculum.enabled):
            threshold = (
                float(self.success_curriculum.final_threshold)
                if self.curriculum_tightened
                else float(self.success_curriculum.start_threshold)
            )
            self.logger.record("task/success_threshold", threshold)

        infos = self.locals.get("infos", [])
        distances = [float(info["shape_distance"]) for info in infos if "shape_distance" in info]
        successes = [float(info["is_success"]) for info in infos if "is_success" in info]
        early_stops = [
            float(info["early_stopped"]) for info in infos if "early_stopped" in info
        ]
        approach_distances = [
            float(info["approach_distance"]) for info in infos if "approach_distance" in info
        ]
        contacts = [
            float(info["contact_established"]) for info in infos if "contact_established" in info
        ]
        if distances:
            distance = float(np.mean(distances))
            self.best_distance = min(self.best_distance, distance)
            self.logger.record("task/shape_distance", distance)
            self.logger.record("task/best_shape_distance", self.best_distance)
        if successes:
            self.logger.record("task/is_success", float(np.mean(successes)))
        if early_stops:
            self.logger.record("task/early_stopped", float(np.mean(early_stops)))
        if approach_distances:
            self.logger.record("task/approach_distance", float(np.mean(approach_distances)))
        if contacts:
            self.logger.record("task/contact_established", float(np.mean(contacts)))

        if self.snapshot_frequency > 0 and self.num_timesteps % self.snapshot_frequency == 0:
            obs = self._first_observation(self.locals["new_obs"])
            distance_text = f"step {self.num_timesteps}"
            if distances:
                distance_text += f" | distance {distances[0]:.4f} m"
            self._save_snapshot(obs["rope_centerline"], self.num_timesteps, distance_text)
        return True
