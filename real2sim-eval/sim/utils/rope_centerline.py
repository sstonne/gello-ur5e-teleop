"""Stable, particle-ID based centerline extraction for the PhysTwin rope."""

from __future__ import annotations

import numpy as np


class RopeCenterlineExtractor:
    """Group a straight rope's particles into ordered longitudinal bins.

    Bins are fitted once on the initial straight configuration.  The same particle
    IDs are averaged after deformation, so the centerline ordering remains stable
    even when the rope becomes a C or folds back over itself.
    """

    def __init__(self, num_keypoints: int = 24):
        if num_keypoints < 2:
            raise ValueError("num_keypoints must be at least 2")
        self.num_keypoints = int(num_keypoints)
        self.labels: np.ndarray | None = None

    @staticmethod
    def _points_array(points) -> np.ndarray:
        if hasattr(points, "detach"):
            points = points.detach().cpu().numpy()
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"expected particle positions shaped (N, 3), got {points.shape}")
        if not np.isfinite(points).all():
            raise ValueError("particle positions contain non-finite values")
        return points

    def fit(self, straight_points) -> "RopeCenterlineExtractor":
        points = self._points_array(straight_points)
        if len(points) < self.num_keypoints:
            raise ValueError("there are fewer particles than centerline keypoints")

        centered = points - points.mean(axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis = vh[0]
        # Resolve PCA's arbitrary sign to make saved results reproducible.
        dominant = int(np.argmax(np.abs(axis)))
        if axis[dominant] < 0:
            axis = -axis
        order = np.argsort(centered @ axis)

        labels = np.empty(len(points), dtype=np.int32)
        for label, particle_ids in enumerate(np.array_split(order, self.num_keypoints)):
            labels[particle_ids] = label
        self.labels = labels
        return self

    def set_labels(self, labels) -> "RopeCenterlineExtractor":
        labels = np.asarray(labels, dtype=np.int32)
        expected = np.arange(self.num_keypoints)
        if labels.ndim != 1 or not np.array_equal(np.unique(labels), expected):
            raise ValueError("labels must contain every keypoint ID exactly as a contiguous range")
        self.labels = labels.copy()
        return self

    def transform(self, points) -> np.ndarray:
        points = self._points_array(points)
        if self.labels is None:
            raise RuntimeError("fit or set_labels must be called before transform")
        if len(points) != len(self.labels):
            raise ValueError(
                f"particle count differs from fitted reference: {len(points)} != {len(self.labels)}"
            )
        return np.stack(
            [points[self.labels == i].mean(axis=0) for i in range(self.num_keypoints)]
        ).astype(np.float32)
