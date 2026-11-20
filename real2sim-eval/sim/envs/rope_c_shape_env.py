"""State-only PhysTwin task for shaping a free-ended rope into a demonstrated C."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import kornia
import numpy as np
import torch

from .env import BaseEnv
from ..utils.env.registration import register_env
from ..utils.rope_centerline import RopeCenterlineExtractor


@register_env("RopeCShape-v0", max_episode_steps=300)
class RopeCShapeEnv(BaseEnv):
    """Control EE XY deltas while PhysTwin supplies rope contact dynamics.

    Both rope ends remain free: this environment introduces no pinned particles or
    endpoint constraints. Height, orientation and gripper command are fixed.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, cfg, exp_root, randomize=False, local_rank=0, **kwargs):
        super().__init__(cfg, exp_root, randomize, local_rank, **kwargs)
        self.rl_cfg = cfg.task
        self.goal_path = Path(self.rl_cfg.goal_path)
        if not self.goal_path.exists():
            raise FileNotFoundError(
                f"processed rope goal not found: {self.goal_path}; run prepare_rope_c_goal.py"
            )
        with np.load(self.goal_path) as goal:
            self.num_keypoints = int(goal["num_keypoints"])
            self.goal_centerline = goal["goal_centerline"].astype(np.float32)
            labels = goal["particle_bin_labels"]
        self.centerline_extractor = RopeCenterlineExtractor(self.num_keypoints).set_labels(labels)

        limit = np.finfo(np.float32).max
        self.observation_space = gym.spaces.Dict(
            {
                "rope_centerline": gym.spaces.Box(-limit, limit, (self.num_keypoints, 3), np.float32),
                "goal_centerline": gym.spaces.Box(-limit, limit, (self.num_keypoints, 3), np.float32),
                "eef_pose": gym.spaces.Box(-limit, limit, (7,), np.float32),
            }
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.fixed_quat = None
        self.fixed_gripper = float(self.rl_cfg.closed_gripper)
        self.previous_distance = None
        self.elapsed_steps = 0
        self.success_threshold = float(self.rl_cfg.success_threshold)
        self.contact_established = False
        self.previous_approach_distance = None
        self.best_distance = None
        self.consecutive_worsening_steps = 0

    def set_success_threshold(self, threshold):
        self.success_threshold = float(threshold)

    @staticmethod
    def _numpy(value):
        return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)

    def _state_observation(self):
        state = self.renderer.get_state()
        centerline = self.centerline_extractor.transform(state["x"])
        xyz = self._numpy(state["eef_xyz"])[0]
        quat = self._numpy(state["eef_quat"])[0]
        return {
            "rope_centerline": centerline,
            "goal_centerline": self.goal_centerline.copy(),
            "eef_pose": np.concatenate([xyz, quat]).astype(np.float32),
        }

    def _target_action(self, xyz, gripper):
        rot = kornia.geometry.conversions.quaternion_to_rotation_matrix(self.fixed_quat)
        return torch.cat([xyz, rot.reshape(1, 9), gripper], dim=1)

    def _apply_target(self, target_xyz, gripper_value):
        state = self.renderer.get_state()
        gripper = torch.full((1, 1), gripper_value, dtype=torch.float32, device=state["x"].device)
        action = self._target_action(target_xyz, gripper)
        next_state = self.physics.step(state, action)
        self.renderer.update_state(next_state)

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed, options=options)
        self._reset_simulation(seed)
        state = self.renderer.get_state()
        self.fixed_quat = state["eef_quat"].clone()

        # Stabilize the already-closed, initially non-contact boundary.
        initial_xyz = torch.as_tensor(
            self.rl_cfg.initial_eef_xyz, dtype=torch.float32, device=state["x"].device
        ).reshape(1, 3)
        start_open = float(self._numpy(state["eef_gripper"])[0, 0])
        for openness in np.linspace(start_open, self.fixed_gripper, int(self.rl_cfg.grasp_steps)):
            self._apply_target(initial_xyz, float(openness))

        obs = self._state_observation()
        self.previous_distance = self._shape_distance(obs["rope_centerline"])
        self.best_distance = self.previous_distance
        self.consecutive_worsening_steps = 0
        self.previous_approach_distance = self._approach_distance(obs)
        self.contact_established = (
            self.previous_approach_distance <= float(self.rl_cfg.approach_contact_threshold)
        )
        self.elapsed_steps = 0
        return obs, {
            "shape_distance": self.previous_distance,
            "approach_distance": self.previous_approach_distance,
            "contact_established": self.contact_established,
            "is_success": False,
            "early_stopped": False,
        }

    def _shape_distance(self, centerline):
        # XY defines the table-plane C shape; Z remains in the observation for diagnostics.
        return float(np.linalg.norm(centerline[:, :2] - self.goal_centerline[:, :2], axis=1).mean())

    def _approach_distance(self, observation):
        centerline = observation["rope_centerline"]
        midpoint = centerline[len(centerline) // 2, :2]
        return float(np.linalg.norm(observation["eef_pose"][:2] - midpoint))

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,):
            raise ValueError(f"expected XY action shaped (2,), got {action.shape}")
        action = np.clip(action, -1.0, 1.0)

        state = self.renderer.get_state()
        current_xyz = state["eef_xyz"].clone()
        target_xyz = current_xyz.clone()
        target_xyz[0, :2] += torch.as_tensor(
            action * float(self.rl_cfg.action_delta), device=target_xyz.device
        )
        workspace_low = torch.as_tensor(self.rl_cfg.workspace_low, device=target_xyz.device)
        workspace_high = torch.as_tensor(self.rl_cfg.workspace_high, device=target_xyz.device)
        target_xyz[0] = torch.maximum(torch.minimum(target_xyz[0], workspace_high), workspace_low)
        target_xyz[0, 2] = float(self.rl_cfg.initial_eef_xyz[2])
        self._apply_target(target_xyz, self.fixed_gripper)

        obs = self._state_observation()
        distance = self._shape_distance(obs["rope_centerline"])
        improvement = self.previous_distance - distance
        approach_distance = self._approach_distance(obs)
        was_in_contact = self.contact_established
        if approach_distance <= float(self.rl_cfg.approach_contact_threshold):
            self.contact_established = True
        reward = (
            -float(self.rl_cfg.distance_weight) * distance
            + float(self.rl_cfg.improvement_weight) * improvement
            - float(self.rl_cfg.action_penalty) * float(np.square(action).sum())
        )
        if not was_in_contact:
            reward += (
                float(self.rl_cfg.approach_improvement_weight)
                * (self.previous_approach_distance - approach_distance)
                - float(self.rl_cfg.approach_distance_weight) * approach_distance
            )
        success = distance <= self.success_threshold
        if success:
            reward += float(self.rl_cfg.success_bonus)

        early_stop_cfg = self.rl_cfg.early_stop
        if distance < self.best_distance:
            self.best_distance = distance
        if (
            bool(early_stop_cfg.enabled)
            and self.best_distance <= float(early_stop_cfg.activation_distance)
            and distance > self.previous_distance + float(early_stop_cfg.worsening_tolerance)
        ):
            self.consecutive_worsening_steps += 1
        else:
            self.consecutive_worsening_steps = 0
        early_stopped = (
            bool(early_stop_cfg.enabled)
            and not success
            and self.consecutive_worsening_steps >= int(early_stop_cfg.patience)
        )
        if early_stopped:
            reward -= float(early_stop_cfg.penalty)
        self.previous_distance = distance
        self.previous_approach_distance = approach_distance
        self.elapsed_steps += 1
        time_limit = self.elapsed_steps >= int(self.rl_cfg.episode_steps)
        info = {
            "shape_distance": distance,
            "approach_distance": approach_distance,
            "contact_established": self.contact_established,
            "is_success": success,
            "early_stopped": early_stopped,
            "best_shape_distance": self.best_distance,
        }
        terminated = success or early_stopped
        return obs, float(reward), bool(terminated), bool(time_limit and not terminated), info
