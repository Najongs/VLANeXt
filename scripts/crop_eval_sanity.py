"""Sanity check: eval-time cropped frame vs train-time cropped HDF5 frame.

목적: crop_zoom_v2 eval fail이 train/eval mismatch인지 확인.
Train 시점에 모델이 본 cropped image와 eval 첫 step의 cropped image를 시각 비교.
"""
import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ.setdefault('__EGL_VENDOR_LIBRARY_FILENAMES', '/usr/share/glvnd/egl_vendor.d/50_mesa.json')

import mujoco
import numpy as np
import h5py
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("/home/najo/NAS/VLANeXt/vqa_samples/crop_eval_sanity.png")
MODEL_PATH = "/home/najo/NAS/VLANeXt/Sim/meca_add.xml"
HOME_POSE = np.array([0.75, -0.5, 0.5, 0, 0.6, 1.0])

CROP_WIN = 256
CROP_TGT = 512


def center_crop(img, win=CROP_WIN, tgt=CROP_TGT):
    H, W = img.shape[:2]
    x0 = (W - win) // 2
    y0 = (H - win) // 2
    patch = img[y0:y0 + win, x0:x0 + win, :]
    return cv2.resize(patch, (tgt, tgt), interpolation=cv2.INTER_LINEAR)


# === Train side: HDF5 한 episode 첫 frame ===
hdf5_path = "/home/najo/NAS/VLANeXt/dataset/fine_align/NEARGOAL_eval_match_v2/collected_data_merged/w0_episode_20260520_152751.h5"
with h5py.File(hdf5_path, "r") as f:
    img_grp = f["observations"]["images"]
    buf = np.array(img_grp["tool_camera"][0]).flatten().astype(np.uint8)
    train_raw = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    train_raw = cv2.cvtColor(train_raw, cv2.COLOR_BGR2RGB)
train_cropped = center_crop(train_raw)
print(f"Train raw: {train_raw.shape}, cropped: {train_cropped.shape}")


# === Eval side: live MuJoCo render at home_pose + phantom at (0,0,0) ===
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=480, width=640)

mujoco.mj_resetData(model, data)
data.qpos[:6] = HOME_POSE.copy()
data.ctrl[:6] = HOME_POSE.copy()
mujoco.mj_forward(model, data)
renderer.update_scene(data, camera='tool_camera')
eval_raw = renderer.render().copy()
eval_cropped = center_crop(eval_raw)
print(f"Eval raw: {eval_raw.shape}, cropped: {eval_cropped.shape}")


# === Side-by-side comparison ===
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes[0, 0].imshow(train_raw); axes[0, 0].set_title(f"Train raw (HDF5) {train_raw.shape}"); axes[0, 0].axis('off')
axes[0, 1].imshow(train_cropped); axes[0, 1].set_title(f"Train CROPPED (model input) {train_cropped.shape}"); axes[0, 1].axis('off')
axes[1, 0].imshow(eval_raw); axes[1, 0].set_title(f"Eval raw (MuJoCo render) {eval_raw.shape}"); axes[1, 0].axis('off')
axes[1, 1].imshow(eval_cropped); axes[1, 1].set_title(f"Eval CROPPED (model input) {eval_cropped.shape}"); axes[1, 1].axis('off')

fig.suptitle("Train vs Eval cropped images — model 입력 일관성 검증", fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(OUT, dpi=110, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {OUT}")
