"""Trocar goal-overlay drawing utilities for SutureBot-style visual conditioning.

Draws a small colored disk at the trocar entry pixel on each tool_camera frame.
At train time UV comes from HDF5 `observations/keypoints_wrist[:, 2:4]` (GT
projection produced by Sim/Save_dataset_*.py:project_to_2d, normalized [0,1]).
At eval time UV comes from the KeypointInferencer head (sim_eval_align_only.py).

Single-frame public API:
    draw_overlay(frame, uv_norm, visible, color, radius, jitter_std_px, rng) -> frame

Batch helper:
    apply_overlay_batch(frames, uv_norm, visibility, color, radius,
                        dropout_prob, jitter_std_px, rng) -> frames (modified in place)
"""
from __future__ import annotations
import cv2
import numpy as np


def draw_overlay(frame: np.ndarray,
                 uv_norm: tuple[float, float],
                 *,
                 color: tuple[int, int, int] = (255, 0, 0),
                 radius_px: int = 3,
                 jitter_std_px: float = 0.0,
                 rng: np.random.Generator | None = None) -> np.ndarray:
    """Draw a filled disk on a single uint8 HxWx3 RGB frame.

    uv_norm in [0,1]^2 (sim convention: u=horizontal, v=vertical).
    Out-of-range UV is silently skipped (no draw).
    Returns the same array (mutated).
    """
    h, w = frame.shape[:2]
    u, v = float(uv_norm[0]), float(uv_norm[1])
    if not np.isfinite(u) or not np.isfinite(v):
        return frame
    if jitter_std_px > 0.0:
        if rng is None:
            rng = np.random.default_rng()
        u += float(rng.normal(0.0, jitter_std_px)) / max(w, 1)
        v += float(rng.normal(0.0, jitter_std_px)) / max(h, 1)
    # Reject overlays whose center is outside the image — partial-edge disks
    # bias the model to use the disk edge rather than the center.
    if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
        return frame
    px = int(round(u * (w - 1)))
    py = int(round(v * (h - 1)))
    cv2.circle(frame, (px, py), int(radius_px), tuple(int(c) for c in color), thickness=-1, lineType=cv2.LINE_AA)
    return frame


def apply_overlay_batch(frames: np.ndarray,
                        uv_norm: np.ndarray,
                        visibility: np.ndarray | None = None,
                        *,
                        color: tuple[int, int, int] = (255, 0, 0),
                        radius_px: int = 3,
                        dropout_prob: float = 0.0,
                        jitter_std_px: float = 0.0,
                        rng: np.random.Generator | None = None) -> np.ndarray:
    """Apply overlay to a stack of frames (N, H, W, 3) uint8.

    uv_norm: (N, 2) normalized coords.
    visibility: (N,) optional confidence (1=visible). Frames with vis<=0.5 get
                no overlay regardless of dropout — they're out-of-frame.
    dropout_prob: per-frame chance to skip overlay even when visible — exposes
                  the model to "no signal" cases (KP head failures at test time).
    """
    if rng is None:
        rng = np.random.default_rng()
    n = frames.shape[0]
    assert uv_norm.shape == (n, 2), f"uv_norm must be ({n}, 2), got {uv_norm.shape}"
    if visibility is None:
        visibility = np.ones(n, dtype=np.float32)
    for i in range(n):
        if visibility[i] <= 0.5:
            continue
        if dropout_prob > 0.0 and rng.random() < dropout_prob:
            continue
        draw_overlay(frames[i], (float(uv_norm[i, 0]), float(uv_norm[i, 1])),
                     color=color, radius_px=radius_px,
                     jitter_std_px=jitter_std_px, rng=rng)
    return frames
