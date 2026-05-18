# Champion Model — Performance Summary

**ckpt**: `b24_ft10mm_HARD_cotrain/checkpoint_1000.pt`
**Pipeline**: VLA + KP servo×3 + sensor sweep + polish

## SR across grids

| Grid | n | SR | lat_med(succ) | min(min_dist) |
|---|---|---|---|---|
| 12ep (max=250) | 12 | **66.7%** | 2.60mm | 1.10mm |
| 12ep (max=400) | 12 | **75.0%** | 2.60mm | 1.10mm |
| 24ep (max=250) | 24 | **75.0%** | 2.60mm | 1.60mm |
| 90ep (max=250) | 90 | **82.2%** | 2.70mm | 0.90mm |