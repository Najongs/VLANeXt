"""Shared post-processing for the needle-tip distance sensor.

Raw `sensor_dist` (mm) measured by the on-tool ToF/raycast sensor has a
known failure mode: when the needle stares directly into the trocar
opening, the ray passes through the hole and reports the *back wall*
distance — usually 20-30mm — at the very moment the alignment is best.
This inverts the signal ("close-to-aligned ↔ large reading") and
breaks any downstream model that treats the value as monotonic.

Empirically (see 5mm_fine_align_00_tip3 dataset): below 5mm the reading
tracks the actual tip-trocar surface distance; above 5mm it is dominated
by hole-through cases and is unreliable.

Convention used everywhere (training, sim eval, real eval):
    dist_clipped: float in [0, SENSOR_MAX_MM]
    valid:        1.0 if reading is informative, else 0.0
When invalid, dist_clipped is set to SENSOR_MAX_MM (saturated "far")
so the model can still see a consistent value with valid=0.
"""

import numpy as np


SENSOR_MAX_MM = 5.0  # raw values above this are unreliable (hole-through)
SENSOR_PROPRIO_CHANNELS = 2  # [dist, valid]


def process_sensor_dist(raw_mm, max_mm: float = SENSOR_MAX_MM):
    """Convert raw mm reading(s) to (dist_clipped, valid) — both float32.

    Works for scalars and numpy arrays. Negative values (no detection)
    and values > max_mm are flagged invalid; dist saturates to max_mm.
    """
    raw = np.asarray(raw_mm, dtype=np.float32)
    valid = ((raw >= 0.0) & (raw <= max_mm)).astype(np.float32)
    dist = np.clip(raw, 0.0, max_mm).astype(np.float32)
    dist = np.where(valid > 0, dist, np.float32(max_mm))
    return dist, valid


def process_sensor_dist_scalar(raw_mm: float, max_mm: float = SENSOR_MAX_MM):
    """Scalar variant — returns plain Python floats."""
    d, v = process_sensor_dist(np.float32(raw_mm), max_mm=max_mm)
    return float(d), float(v)


if __name__ == "__main__":
    cases = [-1.0, 0.0, 2.5, 5.0, 7.0, 25.0, 30.0]
    for r in cases:
        d, v = process_sensor_dist_scalar(r)
        print(f"raw={r:6.2f}  dist={d:5.2f}  valid={v:.0f}")
