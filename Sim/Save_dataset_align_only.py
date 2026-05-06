"""
Fine-alignment only dataset collection.

1. 트로카 위치 고정 (랜덤화 없음)
2. 기존 정렬 알고리즘으로 needle tip을 trocar entry까지 이동 (녹화 X)
3. 랜덤 perturbation 적용
4. 미세 정렬만 녹화 (삽입 X)
5. 정렬 완료 시 에피소드 종료

# 기존 방식 (uniform, 변경 없음)
python Sim/Save_dataset_align_only.py

# X 음수 방향 편향 수집 (2000개, 별도 폴더)
python Sim/Save_dataset_align_only.py \
    --save-dir dataset/fine_align/bias_x_neg \
    --num-episodes 2000 \
    --bias x_neg

# Y 음수 방향 편향 수집
python Sim/Save_dataset_align_only.py \
    --save-dir dataset/fine_align/bias_y_neg \
    --num-episodes 2000 \
    --bias y_neg

Usage:
    python Save_dataset_align_only.py
"""

import os
os.environ['MUJOCO_GL'] = 'egl'

import json
import mujoco
import numpy as np
import cv2
import time
import h5py
import datetime
import threading
import pathlib
import argparse
from collections import deque

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, total=None: x


def project_to_2d(point_3d, model, data, cam_name, img_w, img_h):
    """Project a 3D world point to normalized 2D image coordinates [0,1] using MuJoCo camera."""
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    cam_pos = data.cam_xpos[cam_id]
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3)
    fovy = model.cam_fovy[cam_id]

    p_cam = cam_mat.T @ (point_3d - cam_pos)
    f = img_h / (2.0 * np.tan(np.deg2rad(fovy) / 2.0))
    u = -f * (p_cam[0] / p_cam[2]) + (img_w - 1) / 2.0
    v =  f * (p_cam[1] / p_cam[2]) + (img_h - 1) / 2.0

    return np.array([u / img_w, v / img_h], dtype=np.float32)


# ============================================================
# === Configuration (수정하기 쉽게 상단에 모아놓음) ===
# ============================================================

MODEL_PATH = "meca_add.xml"
SAVE_DIR = "collected_data_fine_align"
MAX_EPISODES = 1
IMG_WIDTH = 640
IMG_HEIGHT = 480
CAMERA_LIST = ["side_camera", "tool_camera", "top_camera"]

# --- 정렬 속도 ---
ALIGN_SPEED = 0.15          # 초기 정렬 속도 (m/s) — 녹화 전 이동용
FINE_ALIGN_SPEED = 0.0025    # 미세 정렬 속도 (m/s) — 녹화 중
# Velocity profile shape during recording (fine_align loop).
# 0.5 = pure cubic smoothstep (slow tails dominate), 0.2 = 20% accel + 60% cruise + 20% decel,
# 0.1 = sharper trapezoid. Lower = more time at constant velocity, less "stop-near-goal" bias.
FINE_ALIGN_ACCEL_FRAC = 0.15

# --- Perturbation 설정 (미세 정렬 시작 전 흐트러뜨리는 범위) ---
PERTURB_POS_XY_MM = 5.0    # XY 평면 perturbation 범위 (±mm)
PERTURB_POS_Z_MIN_MM = -5.0 # Z축 하한 (mm) — 음수 시 occlusion check로 가려진 케이스 자동 폐기
PERTURB_POS_Z_MAX_MM = 5.0  # Z축 상한 (mm)
PERTURB_ANGLE_DEG = 5.0    # 각도 perturbation 범위 (±deg)
ALLOW_OCCLUDED = False      # True 시 tool_camera에서 needle tip이 가려져도 폐기하지 않음

# --- 성공 조건 ---
ALIGN_THRESHOLD_M = 0.002   # needle tip - trocar entry 거리 (m)
ALIGN_HOLD_STEPS = 10       # threshold 이내 연속 유지 횟수

# --- Task Instruction ---
TASK_INSTRUCTION = "Align the needle tip to the small grey circular trocar port on the eye model, next to the larger lens opening"

# --- Holding (정렬 완료 후 자세 유지 녹화) ---
HOLD_RECORD_STEPS = 10           # 정렬 완료 후 녹화 control steps

# --- 기타 ---
ACTION_CLIP_MM = 1.0        # IK spike 방지용 delta position 클리핑 (mm)
TIMEOUT_SEC = 30.0          # 에피소드 전체 타임아웃 (초)
MAX_CTRL_STEPS = 250        # 녹화 control step 상한 (초과 시 에피소드 폐기)

# --- Retreat (goal_tip을 trocar entry에서 뒤로 빼는 거리) ---
RETREAT_MM = 1.0           # insertion axis 반대 방향 retreat (mm)

# --- Bias collection (set via CLI --bias) ---
BIAS_DIRECTION = None       # e.g. "x_neg", "y_pos"
BIAS_RATIO = 0.8            # fraction of episodes with biased perturbation

# --- Grid collection (set via CLI --grid-cells-file) ---
GRID_CELLS = None           # list of [x_lo, x_hi, y_lo, y_hi, z_lo, z_hi] in mm

# ============================================================


class SimRecorder:
    def __init__(self, output_dir):
        self.out = pathlib.Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.buffer = []
        self.episode_metadata = {}
        self.recording = False
        self.is_saving = False
        self.save_threads = []

    def start(self, episode_metadata=None):
        self.buffer = []
        self.episode_metadata = dict(episode_metadata or {})
        self.recording = True

    def add(self, frames, qpos, ee_pose, action, timestamp, phase, sensor_dist,
            needle_tip_mm=None, trocar_entry_mm=None, keypoints_wrist=None,
            keypoints_visibility=None, instruction="", action_sim=None):
        if not self.recording: return
        self.buffer.append({
            "ts": timestamp,
            "imgs": frames,
            "q": qpos,
            "p": ee_pose,
            "act": action,
            "act_sim": action_sim,
            "phase": phase,
            "sd": sensor_dist,
            "needle_tip_mm": needle_tip_mm,
            "trocar_entry_mm": trocar_entry_mm,
            "keypoints_wrist": keypoints_wrist,
            "keypoints_visibility": keypoints_visibility,
            "instruction": instruction,
        })

    def save_async(self):
        if not self.buffer: return
        data_snapshot = self.buffer
        metadata_snapshot = dict(self.episode_metadata)
        self.buffer = []
        self.recording = False
        self.is_saving = True

        def worker(data, metadata, filename):
            try:
                with h5py.File(filename, 'w') as f:
                    obs = f.create_group("observations")
                    img_grp = obs.create_group("images")
                    meta_grp = f.create_group("metadata")

                    q_data = np.array([x['q'] for x in data], dtype=np.float32)
                    p_data = np.array([x['p'] for x in data], dtype=np.float32)
                    act_data = np.array([x['act'] for x in data], dtype=np.float32)
                    ts_data = np.array([x['ts'] for x in data], dtype=np.float32)
                    phase_data = np.array([x['phase'] for x in data], dtype=np.int32)
                    sensor_data = np.array([x['sd'] for x in data], dtype=np.float32)

                    # Fine alignment only → gripper always open
                    action_gripper = np.full((len(data), 1), -1.0, dtype=np.float32)
                    act_data = np.concatenate([act_data, action_gripper], axis=-1)  # (N, 7)

                    proprio_gripper = np.full((len(data), 1), 0.0, dtype=np.float32)
                    p_data = np.concatenate([p_data, proprio_gripper], axis=-1)  # (N, 7)

                    obs.create_dataset("qpos", data=q_data, compression="gzip")
                    obs.create_dataset("ee_pose", data=p_data, compression="gzip")
                    obs.create_dataset("sensor_dist", data=sensor_data, compression="gzip")
                    f.create_dataset("action", data=act_data, compression="gzip")

                    # Optional sim-side action label (if any frame supplied it).
                    if any(x.get("act_sim") is not None for x in data):
                        n = len(data)
                        sim_act_arr = np.full((n, 6), np.nan, dtype=np.float32)
                        for i, x in enumerate(data):
                            v = x.get("act_sim")
                            if v is not None:
                                sim_act_arr[i] = np.asarray(v, dtype=np.float32)[:6]
                        sim_act_arr = np.concatenate(
                            [sim_act_arr, np.full((n, 1), -1.0, dtype=np.float32)],
                            axis=-1,
                        )
                        f.create_dataset("action_sim", data=sim_act_arr, compression="gzip")

                    f.create_dataset("timestamp", data=ts_data, compression="gzip")
                    f.create_dataset("phase", data=phase_data, compression="gzip")

                    if data[0].get("needle_tip_mm") is not None:
                        needle_tip_data = np.array([x['needle_tip_mm'] for x in data], dtype=np.float32)
                        trocar_entry_data = np.array([x['trocar_entry_mm'] for x in data], dtype=np.float32)
                        kp_wrist_data = np.array([x['keypoints_wrist'] for x in data], dtype=np.float32)
                        kp_vis_data = np.array([x['keypoints_visibility'] for x in data], dtype=np.float32)
                        obs.create_dataset("needle_tip_pos", data=needle_tip_data, compression="gzip")
                        obs.create_dataset("trocar_entry_pos", data=trocar_entry_data, compression="gzip")
                        obs.create_dataset("keypoints_wrist", data=kp_wrist_data, compression="gzip")
                        obs.create_dataset("keypoints_visibility", data=kp_vis_data, compression="gzip")

                    instruction = data[0].get("instruction", "")
                    f.create_dataset("language_instruction", data=instruction)

                    for key, value in metadata.items():
                        value_arr = np.asarray(value)
                        if value_arr.shape == ():
                            meta_grp.create_dataset(key, data=value_arr)
                        else:
                            meta_grp.create_dataset(key, data=value_arr, compression="gzip")

                    first_imgs = data[0]["imgs"]
                    for cam_name in first_imgs.keys():
                        jpeg_list = []
                        for step in data:
                            img = step["imgs"][cam_name]
                            success, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                            if success: jpeg_list.append(buf.flatten())
                            else: jpeg_list.append(np.zeros(1, dtype=np.uint8))

                        dt = h5py.special_dtype(vlen=np.dtype('uint8'))
                        dset = img_grp.create_dataset(cam_name, (len(jpeg_list),), dtype=dt)
                        for i, code in enumerate(jpeg_list): dset[i] = code

            except Exception as e:
                print(f"Save Failed: {e}")
            finally:
                self.is_saving = False

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = self.out / f"episode_{timestamp}.h5"
        t = threading.Thread(target=worker, args=(data_snapshot, metadata_snapshot, fname))
        t.start()
        self.save_threads.append(t)

    def discard(self):
        self.buffer = []
        self.recording = False

    def wait_for_all(self):
        if self.save_threads:
            print(f"\nWaiting for {len(self.save_threads)} files to finish saving...")
            for t in self.save_threads:
                t.join()
            print("All files saved!")
            self.save_threads = []


def smooth_step(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def trapezoid_step(t, accel_frac=0.2):
    """Trapezoidal velocity profile with cubic-smoothed accel/decel.

    accel_frac in (0, 0.5): fraction of total duration spent on accel (= decel).
    Cruise (constant velocity) covers the remaining 1 - 2*accel_frac fraction.
    Total area normalized to 1, so position(0)=0, position(1)=1.

    accel_frac=0.5 falls back to cubic smoothstep (no cruise).
    accel_frac=0.2: peak velocity = 1.25 * avg, 60% of duration at constant speed.

    Compared to cubic smoothstep (peak velocity 1.5×, 0 at both ends, slow tails):
    less time spent at low frame-Δ near goal → less "stop-near-goal" bias in
    learned policy. Recommended for recording the fine-alignment phase.
    """
    t = np.clip(t, 0.0, 1.0)
    ta = float(accel_frac)
    if ta <= 0 or ta >= 0.5:
        return smooth_step(t)
    v_max = 1.0 / (1.0 - ta)
    if t < ta:
        u = t / ta
        # ∫₀ᵘ smoothstep(s) ds = u³ - u⁴/2, scaled by v_max·ta
        return v_max * ta * (u**3 - 0.5 * u**4)
    if t < 1 - ta:
        return v_max * ta * 0.5 + v_max * (t - ta)
    u = (1 - t) / ta
    return 1.0 - v_max * ta * (u**3 - 0.5 * u**4)


def randomize_phantom_pos(model, data, phantom_id, rot_id, combo_counts=None):
    """팬텀 위치/회전 랜덤화.
    combo_counts: {(angle, z_dir): count} — 4-way 균등 배분용 카운터.
                  덜 모인 각도 구간을 우선 선택. None이면 50/50 랜덤.
    """
    # X: [-0.03, 0.05] (phantom_grid_test_v3 결과: X=-0.05 좌측, X=0.053+ 우측 실패)
    offset_x = np.random.uniform(-0.03, 0.05)
    # Y=-0.24~-0.20 제외 (회전 전환 경계 + IK 실패 다발 구간, phantom_grid_test_v3 결과)
    # combo_counts에서 각도별 합산으로 덜 모인 쪽 우선 선택
    if combo_counts is not None:
        count_0 = combo_counts.get((0, "pos"), 0) + combo_counts.get((0, "neg"), 0)
        count_m90 = combo_counts.get((-90, "pos"), 0) + combo_counts.get((-90, "neg"), 0)
        pick_m90 = count_m90 <= count_0
    else:
        pick_m90 = np.random.random() < 0.5
    if pick_m90:
        offset_y = np.random.uniform(-0.4, -0.24)   # → angle -90°
    else:
        offset_y = np.random.uniform(-0.20, 0.0)    # → angle 0°
    offset_z = 0.0

    model.body_pos[phantom_id] = np.array([offset_x, offset_y, offset_z])

    if offset_y >= -0.25:
        random_angle_deg = 0 # np.random.uniform(-15, 15)
    else:
        random_angle_deg = -90 # np.random.uniform(-15 - 90, 15 - 90)

    new_quat = np.zeros(4)
    mujoco.mju_euler2Quat(new_quat, [0, 0, np.deg2rad(random_angle_deg)], "xyz")
    model.body_quat[rot_id] = new_quat
    print(f">>> Randomize: Pos=({offset_x:.2f}, {offset_y:.2f}), Angle={random_angle_deg:.1f} deg")
    mujoco.mj_forward(model, data)
    return np.array([offset_x, offset_y, offset_z], dtype=np.float32), new_quat.astype(np.float32), np.float32(random_angle_deg)


RANDOM_SEED = None
RANDOMIZE_PHANTOM = False
PHANTOM_POS = None  # (x, y) 고정 위치, e.g. (0.0, -0.2)

def main():
    if RANDOM_SEED is not None:
        np.random.seed(RANDOM_SEED)
        print(f"Random seed: {RANDOM_SEED}")
    print(f"Loading Model: {MODEL_PATH}")
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=IMG_HEIGHT, width=IMG_WIDTH)

    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "needle_tip")
    back_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "needle_back")
    target_entry_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "trocar_target")
    target_depth_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "trocar_depth")
    link6_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "6_Link")
    n_motors = model.nu
    dof = model.nv

    # Phantom body IDs (for randomization)
    phantom_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "phantom_assembly")
    rotating_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rotating_assembly")

    # tool_camera ID (occlusion check용)
    tool_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "tool_camera")

    def check_tip_occluded():
        """tool_camera → needle_tip ray cast로 팬텀에 가려지는지 확인."""
        cam_pos = data.cam_xpos[tool_cam_id].copy()
        tip_pos = data.site_xpos[tip_id].copy()
        direction = tip_pos - cam_pos
        dist_to_tip = np.linalg.norm(direction)
        direction_norm = direction / (dist_to_tip + 1e-10)
        geomid_out = np.zeros(1, dtype=np.int32)
        hit_dist = mujoco.mj_ray(model, data, cam_pos, direction_norm,
                                  None, 1, -1, geomid_out)
        if hit_dist > 0 and hit_dist < dist_to_tip - 0.001:
            return True  # 팁보다 앞에 뭔가 있음 → 가려짐
        return False

    recorder = SimRecorder(SAVE_DIR)

    # 초기 home pose (정렬 시작점)
    home_pose = np.array([0, 0, 0, 0, 0, 0])
    ik_speed = 0.5

    def get_ee_pose_6d_scaled():
        # TCP shifted to needle_tip site (177.5mm along flange +Z; rotation identical).
        # See src/utils/tip_frame.py and Sim/meca_add.xml:87.
        if tip_id >= 0:
            pos = data.site_xpos[tip_id].copy() * 1000
            mat = data.site_xmat[tip_id].reshape(3, 3)
            sy = np.sqrt(mat[0, 0] ** 2 + mat[1, 0] ** 2)
            if sy > 1e-6:
                r = np.arctan2(mat[2, 1], mat[2, 2])
                p = np.arctan2(-mat[2, 0], sy)
                y = np.arctan2(mat[1, 0], mat[0, 0])
            else:
                r = np.arctan2(-mat[1, 2], mat[1, 1])
                p = np.arctan2(-mat[2, 0], sy)
                y = 0.0
            return np.concatenate([pos, [r, p, y]])
        return np.zeros(6, dtype=np.float32)

    def run_ik_step(target_tip_pos, target_back_pos):
        """IK solver 1스텝 실행 (기존 로직 그대로)"""
        curr_tip = data.site_xpos[tip_id].copy()
        curr_back = data.site_xpos[back_id].copy()

        err_tip = target_tip_pos - curr_tip
        err_back = target_back_pos - curr_back

        tip_rot_mat = data.site_xmat[tip_id].reshape(3, 3)
        offset_angle = np.deg2rad(180 + 30)
        offset_local_vec = np.array([np.cos(offset_angle), np.sin(offset_angle), 0])
        current_side_vec = tip_rot_mat @ offset_local_vec

        needle_axis_curr = (curr_tip - curr_back) / (np.linalg.norm(curr_tip - curr_back) + 1e-10)
        target_side_vec = np.cross(needle_axis_curr, np.array([0, 0, 1]))
        target_side_vec = target_side_vec / np.linalg.norm(target_side_vec) if np.linalg.norm(target_side_vec) > 1e-3 else np.array([1, 0, 0])
        err_roll = np.cross(current_side_vec, target_side_vec)

        jac_tip_full = np.zeros((6, dof))
        jac_back = np.zeros((3, dof))
        mujoco.mj_jacSite(model, data, jac_tip_full[:3], jac_tip_full[3:], tip_id)
        mujoco.mj_jacSite(model, data, jac_back, None, back_id)

        J_p1 = jac_tip_full[:3, :n_motors]
        e_p1 = err_tip * 50.0
        if np.linalg.norm(e_p1) > 1.0:
            e_p1 = e_p1 / np.linalg.norm(e_p1) * 1.0
        J_p1_pinv = np.linalg.pinv(J_p1, rcond=1e-4)
        dq_p1 = J_p1_pinv @ e_p1

        P_null_1 = np.eye(n_motors) - (J_p1_pinv @ J_p1)
        J_p2_proj = jac_back[:, :n_motors] @ P_null_1
        dq_p2 = np.linalg.pinv(J_p2_proj, rcond=1e-4) @ ((err_back * 50.0) - jac_back[:, :n_motors] @ dq_p1)

        P_null_2 = P_null_1 - (np.linalg.pinv(J_p2_proj, rcond=1e-4) @ J_p2_proj)
        J_p3_proj = jac_tip_full[3:, :n_motors] @ P_null_2
        dq_p3 = np.linalg.pinv(J_p3_proj, rcond=1e-4) @ ((err_roll * 10.0) - jac_tip_full[3:, :n_motors] @ (dq_p1 + dq_p2))

        data.ctrl[:n_motors] = data.qpos[:n_motors] + (dq_p1 + dq_p2 + dq_p3) * ik_speed

    print(f"Starting Fine-Alignment Data Collection ({MAX_EPISODES} episodes)...")
    pbar = tqdm(total=MAX_EPISODES, desc="Collecting", unit="ep")

    # ============================================================
    # Phase 0: 최초 1회만 정렬 → aligned state 저장
    # ============================================================
    print("Running initial pre-alignment (one-time)...")
    mujoco.mj_resetData(model, data)
    home_pose = np.array([0, 0, 0, 0, 0, 0])
    data.qpos[:6] = home_pose
    mujoco.mj_forward(model, data)

    p_entry = data.site_xpos[target_entry_id].copy()
    p_depth = data.site_xpos[target_depth_id].copy()
    curr_tip = data.site_xpos[tip_id].copy()
    curr_back = data.site_xpos[back_id].copy()
    needle_len = np.linalg.norm(curr_tip - curr_back)

    axis_dir = (p_depth - p_entry) / (np.linalg.norm(p_depth - p_entry) + 1e-10)
    retreat_m = RETREAT_MM / 1000.0
    goal_tip = p_entry - (axis_dir * retreat_m)
    goal_back = p_entry - (axis_dir * (retreat_m + needle_len))

    start_tip_pos = curr_tip.copy()
    start_back_pos = curr_back.copy()
    align_distance = np.linalg.norm(goal_tip - start_tip_pos)
    dynamic_duration = align_distance / ALIGN_SPEED
    traj_start_time = data.time
    align_timer = 0

    while True:
        progress = smooth_step((data.time - traj_start_time) / dynamic_duration) if dynamic_duration > 0 else 1.0
        target_tip_pos = (1 - progress) * start_tip_pos + progress * goal_tip
        target_back_pos = (1 - progress) * start_back_pos + progress * goal_back

        run_ik_step(target_tip_pos, target_back_pos)
        mujoco.mj_step(model, data)

        if progress >= 1.0:
            curr_tip = data.site_xpos[tip_id].copy()
            if np.linalg.norm(curr_tip - goal_tip) < ALIGN_THRESHOLD_M:
                align_timer += 1
            else:
                align_timer = 0
            if align_timer > ALIGN_HOLD_STEPS:
                break

        if data.time - traj_start_time > 50.0:
            print("Pre-alignment failed! Exiting.")
            return

    # 정렬된 상태 저장 (이후 매 에피소드에서 재사용)
    aligned_qpos = data.qpos[:n_motors].copy()
    aligned_qvel = data.qvel[:n_motors].copy()
    print(f"Pre-alignment done. Reusing aligned state for all episodes.")

    # ============================================================
    # 에피소드 루프: reset → perturb → 녹화 (pre-alignment 스킵)
    # ============================================================
    episode_count = 0
    grid_cell_index = 0          # Grid mode: 항상 다음 셀로 진행 (실패해도 스킵)
    grid_fail_cells = []         # Grid mode: 실패한 셀 기록
    grid_max_retries = 3         # Grid mode: 셀당 최대 재시도 횟수
    grid_retry_count = 0
    # 4-way 교차 카운터: (angle, z_dir) → 성공 에피소드 수
    # 고정 위치 모드에서는 angle이 항상 같으므로 자동으로 Z+/Z- 2-way로 동작
    combo_counts = {
        (0, "pos"): 0, (0, "neg"): 0,
        (-90, "pos"): 0, (-90, "neg"): 0,
    }
    while episode_count < MAX_EPISODES:
        # 정렬된 상태로 즉시 리셋
        mujoco.mj_resetData(model, data)
        data.qpos[:n_motors] = aligned_qpos
        data.qvel[:n_motors] = aligned_qvel
        mujoco.mj_forward(model, data)

        # --- 팬텀 랜덤화: 매 에피소드마다 위치 변경 + 재정렬 ---
        phantom_offset = np.zeros(3, dtype=np.float32)
        phantom_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        phantom_angle_deg = np.float32(0.0)
        if PHANTOM_POS is not None and phantom_body_id >= 0:
            # 고정 위치에 팬텀 배치
            px, py = PHANTOM_POS
            model.body_pos[phantom_body_id] = np.array([px, py, 0.0])
            # 회전: Y 위치에 따라 고정 각도
            if py >= -0.25:
                random_angle_deg_val = 0
            else:
                random_angle_deg_val = -90
            new_quat = np.zeros(4)
            mujoco.mju_euler2Quat(new_quat, [0, 0, np.deg2rad(random_angle_deg_val)], "xyz")
            model.body_quat[rotating_id] = new_quat
            mujoco.mj_forward(model, data)
            phantom_offset = np.array([px, py, 0.0], dtype=np.float32)
            phantom_quat = new_quat.astype(np.float32)
            phantom_angle_deg = np.float32(random_angle_deg_val)
            print(f">>> Fixed phantom: Pos=({px:.2f}, {py:.2f}), Angle={random_angle_deg_val:.1f} deg")
        elif RANDOMIZE_PHANTOM and phantom_body_id >= 0:
            phantom_offset, phantom_quat, phantom_angle_deg = randomize_phantom_pos(
                model, data, phantom_body_id, rotating_id, combo_counts=combo_counts)

        # 팬텀이 이동된 경우 재정렬 필요
        need_realign = (PHANTOM_POS is not None or RANDOMIZE_PHANTOM) and phantom_body_id >= 0
        if need_realign:
            # 팬텀 이동 후 재정렬 (trocar 위치가 바뀌었으므로)
            p_entry = data.site_xpos[target_entry_id].copy()
            p_depth = data.site_xpos[target_depth_id].copy()
            curr_tip = data.site_xpos[tip_id].copy()
            curr_back = data.site_xpos[back_id].copy()
            needle_len_local = np.linalg.norm(curr_tip - curr_back)
            axis_dir_local = (p_depth - p_entry) / (np.linalg.norm(p_depth - p_entry) + 1e-10)
            retreat_m_local = RETREAT_MM / 1000.0
            re_goal_tip = p_entry - (axis_dir_local * retreat_m_local)
            re_goal_back = p_entry - (axis_dir_local * (retreat_m_local + needle_len_local))

            start_tip = data.site_xpos[tip_id].copy()
            start_back = data.site_xpos[back_id].copy()
            re_dist = np.linalg.norm(re_goal_tip - start_tip)
            re_duration = max(re_dist / ALIGN_SPEED, 0.1)
            re_start_time = data.time
            re_timer = 0

            realign_ok = False
            for _ in range(50000):
                t_re = (data.time - re_start_time) / re_duration
                alpha_re = smooth_step(min(t_re, 1.0))
                target_tip_re = (1 - alpha_re) * start_tip + alpha_re * re_goal_tip
                target_back_re = (1 - alpha_re) * start_back + alpha_re * re_goal_back
                run_ik_step(target_tip_re, target_back_re)
                mujoco.mj_step(model, data)

                if t_re >= 1.0:
                    if np.linalg.norm(data.site_xpos[tip_id] - re_goal_tip) < ALIGN_THRESHOLD_M:
                        re_timer += 1
                    else:
                        re_timer = 0
                    if re_timer > ALIGN_HOLD_STEPS:
                        realign_ok = True
                        break

                if data.time - re_start_time > 50.0:
                    print(f"Re-alignment failed for phantom offset={phantom_offset}, retrying with fresh state...")
                    break

            if not realign_ok:
                # 실패 시: aligned_qpos 업데이트하지 않고 에피소드 스킵
                # 다음 에피소드에서 이전 성공 상태로 복원됨
                continue

            # 재정렬 성공 시에만 상태 갱신
            aligned_qpos = data.qpos[:n_motors].copy()
            aligned_qvel = data.qvel[:n_motors].copy()

            # trocar 위치도 갱신
            p_entry = data.site_xpos[target_entry_id].copy()
            p_depth = data.site_xpos[target_depth_id].copy()
            axis_dir = (p_depth - p_entry) / (np.linalg.norm(p_depth - p_entry) + 1e-10)
            retreat_m = RETREAT_MM / 1000.0
            goal_tip = p_entry - (axis_dir * retreat_m)
            goal_back = p_entry - (axis_dir * (retreat_m + needle_len))

        # ============================================================
        # Phase 1: Perturbation 적용 (녹화 X)
        # ============================================================

        # --- Perturbation 생성 ---
        if GRID_CELLS is not None:
            # Grid mode: 셀 범위 내 랜덤 샘플링
            if grid_cell_index >= len(GRID_CELLS):
                break  # 모든 셀 소진
            cell = GRID_CELLS[grid_cell_index]
            perturb_xyz = np.array([
                np.random.uniform(cell[0], cell[1]) / 1000.0,  # x: [x_lo, x_hi] mm
                np.random.uniform(cell[2], cell[3]) / 1000.0,  # y: [y_lo, y_hi] mm
                np.random.uniform(cell[4], cell[5]) / 1000.0,  # z: [z_lo, z_hi] mm
            ])
        else:
            # Random mode — Z 방향 균등 배분 (4-way combo_counts 기반)
            # 현재 angle에서 Z+/Z- 중 덜 모인 쪽 선택
            cur_angle_key = int(round(float(phantom_angle_deg)))
            pick_z_neg = combo_counts.get((cur_angle_key, "neg"), 0) <= combo_counts.get((cur_angle_key, "pos"), 0)
            if pick_z_neg and PERTURB_POS_Z_MIN_MM < 0:
                z_val = np.random.uniform(PERTURB_POS_Z_MIN_MM, 0) / 1000.0
            else:
                z_val = np.random.uniform(0, PERTURB_POS_Z_MAX_MM) / 1000.0
            perturb_xyz = np.array([
                np.random.uniform(-PERTURB_POS_XY_MM, PERTURB_POS_XY_MM) / 1000.0,
                np.random.uniform(-PERTURB_POS_XY_MM, PERTURB_POS_XY_MM) / 1000.0,
                z_val,
            ])
            # Apply directional bias if configured
            if BIAS_DIRECTION is not None and np.random.random() < BIAS_RATIO:
                for bias_part in BIAS_DIRECTION.split(","):
                    axis, sign = bias_part.strip().split("_")
                    idx = {"x": 0, "y": 1, "z": 2}[axis]
                    limit = PERTURB_POS_Z_MAX_MM if axis == "z" else PERTURB_POS_XY_MM
                    if sign == "neg":
                        perturb_xyz[idx] = np.random.uniform(-limit, -limit * 0.15) / 1000.0
                    else:
                        perturb_xyz[idx] = np.random.uniform(limit * 0.15, limit) / 1000.0
        perturb_angle_rad = np.deg2rad(np.random.uniform(-PERTURB_ANGLE_DEG, PERTURB_ANGLE_DEG))
        random_axis = np.random.randn(3)
        random_axis = random_axis / (np.linalg.norm(random_axis) + 1e-10)

        perturbed_tip = goal_tip + perturb_xyz
        rot_mat_perturb = np.eye(3)
        if abs(perturb_angle_rad) > 1e-6:
            K = np.array([
                [0, -random_axis[2], random_axis[1]],
                [random_axis[2], 0, -random_axis[0]],
                [-random_axis[1], random_axis[0], 0],
            ])
            rot_mat_perturb = np.eye(3) + np.sin(perturb_angle_rad) * K + (1 - np.cos(perturb_angle_rad)) * (K @ K)
        perturbed_back_dir = rot_mat_perturb @ (goal_back - goal_tip)
        perturbed_back = perturbed_tip + perturbed_back_dir

        # IK로 perturbed 위치까지 이동 (aligned → perturbed smooth interpolation)
        perturb_reached = False
        move_speed = 0.05  # m/s — perturbation 이동 속도
        move_dist = np.linalg.norm(perturbed_tip - goal_tip)
        move_duration = max(move_dist / move_speed, 0.1)
        move_start_time = data.time

        for ps in range(5000):
            t = (data.time - move_start_time) / move_duration
            alpha = smooth_step(min(t, 1.0))
            interp_tip = (1 - alpha) * goal_tip + alpha * perturbed_tip
            interp_back = (1 - alpha) * goal_back + alpha * perturbed_back

            run_ik_step(interp_tip, interp_back)
            mujoco.mj_step(model, data)

            if t >= 1.0:
                if np.linalg.norm(data.site_xpos[tip_id] - perturbed_tip) < 0.001:
                    # 안정화
                    for _ in range(200):
                        run_ik_step(perturbed_tip, perturbed_back)
                        mujoco.mj_step(model, data)
                    perturb_reached = True
                    break
                # 보간 끝났는데 아직 도달 못함 → 조금 더 시도
                if ps > 4500:
                    break

        perturb_dist_mm = np.linalg.norm(perturb_xyz) * 1000
        ik_err_mm = np.linalg.norm(data.site_xpos[tip_id] - perturbed_tip) * 1000
        reach_tag = "OK" if perturb_reached else f"IK_FAIL(err={ik_err_mm:.1f}mm)"

        # Occlusion check: tool_camera에서 바늘 팁이 팬텀에 가려지는지 확인
        tip_occluded = False
        if perturb_reached:
            tip_occluded = check_tip_occluded()
            if tip_occluded:
                reach_tag = "OCCLUDED" if not ALLOW_OCCLUDED else "OCCLUDED(kept)"

        print(f"  Episode {episode_count}: perturbation applied "
              f"(pos={perturb_dist_mm:.1f}mm, angle={np.rad2deg(perturb_angle_rad):.1f}deg) [{reach_tag}]")

        if tip_occluded and not ALLOW_OCCLUDED:
            print(f"  Episode {episode_count} discarded. Reason: needle tip occluded by phantom")
            continue

        # ============================================================
        # Phase 2: 미세 정렬 녹화
        # ============================================================
        last_ee_pose = get_ee_pose_6d_scaled()
        episode_meta = {
            "aligned_qpos": np.rad2deg(aligned_qpos).astype(np.float32),
            "perturb_xyz_mm": (perturb_xyz * 1000).astype(np.float32),
            "perturb_angle_deg": np.array(np.rad2deg(perturb_angle_rad), dtype=np.float32),
            "target_entry_world": p_entry.astype(np.float32),
            "target_depth_world": p_depth.astype(np.float32),
            "retreat_mm": np.float32(RETREAT_MM),
        }
        if RANDOMIZE_PHANTOM:
            episode_meta["phantom_offset"] = phantom_offset
            episode_meta["phantom_quat"] = phantom_quat
            episode_meta["phantom_angle_deg"] = phantom_angle_deg
        recorder.start(episode_meta)

        record_start_time = data.time
        step_count = 0
        ctrl_step_count = 0
        align_timer = 0
        success = False

        start_tip_pos = data.site_xpos[tip_id].copy()
        start_back_pos = data.site_xpos[back_id].copy()
        fine_align_distance = np.linalg.norm(goal_tip - start_tip_pos)
        fine_duration = fine_align_distance / FINE_ALIGN_SPEED if FINE_ALIGN_SPEED > 0 else 1.0
        fine_traj_start = data.time

        while True:
            # Trapezoidal velocity profile (less slow tail near goal — see FINE_ALIGN_ACCEL_FRAC).
            progress = (
                trapezoid_step((data.time - fine_traj_start) / fine_duration, FINE_ALIGN_ACCEL_FRAC)
                if fine_duration > 0
                else 1.0
            )

            target_tip_pos = (1 - progress) * start_tip_pos + progress * goal_tip
            target_back_pos = (1 - progress) * start_back_pos + progress * goal_back

            run_ik_step(target_tip_pos, target_back_pos)

            # Sensor
            curr_tip = data.site_xpos[tip_id].copy()
            curr_back = data.site_xpos[back_id].copy()
            needle_dir = (curr_tip - curr_back) / (np.linalg.norm(curr_tip - curr_back) + 1e-10)
            dist_to_surface = mujoco.mj_ray(model, data, curr_tip, needle_dir, None, 1, link6_id, np.zeros(1, dtype=np.int32))
            current_sensor_dist = dist_to_surface * 1000.0 if dist_to_surface >= 0 else -1.0

            mujoco.mj_step(model, data)
            step_count += 1

            # 녹화 (67 스텝마다)
            if step_count % 67 == 0:
                current_qpos_deg = np.rad2deg(data.qpos[:n_motors].copy())
                current_ee_pose_mm = get_ee_pose_6d_scaled()
                delta_ee_action = current_ee_pose_mm - last_ee_pose
                delta_ee_action[3:6] = (delta_ee_action[3:6] + np.pi) % (2 * np.pi) - np.pi

                pos_mag = np.linalg.norm(delta_ee_action[:3])
                if pos_mag > ACTION_CLIP_MM:
                    delta_ee_action[:3] *= ACTION_CLIP_MM / pos_mag

                frames = {}
                for cam_name in CAMERA_LIST:
                    renderer.update_scene(data, camera=cam_name)
                    frames[cam_name] = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)

                needle_tip_mm = data.site_xpos[tip_id].copy() * 1000
                trocar_entry_mm = data.site_xpos[target_entry_id].copy() * 1000

                tip_uv_wrist = project_to_2d(data.site_xpos[tip_id], model, data, "tool_camera", IMG_WIDTH, IMG_HEIGHT)
                trocar_uv_wrist = project_to_2d(data.site_xpos[target_entry_id], model, data, "tool_camera", IMG_WIDTH, IMG_HEIGHT)
                keypoints_wrist = np.concatenate([tip_uv_wrist, trocar_uv_wrist]).astype(np.float32)

                tip_visible = float(0.0 <= tip_uv_wrist[0] <= 1.0 and 0.0 <= tip_uv_wrist[1] <= 1.0)
                trocar_visible = float(0.0 <= trocar_uv_wrist[0] <= 1.0 and 0.0 <= trocar_uv_wrist[1] <= 1.0)
                keypoints_visibility = np.array([tip_visible, trocar_visible], dtype=np.float32)

                recorder.add(
                    frames, current_qpos_deg, current_ee_pose_mm, delta_ee_action,
                    data.time, 1,  # phase=1 (정렬)
                    current_sensor_dist,
                    needle_tip_mm=needle_tip_mm, trocar_entry_mm=trocar_entry_mm,
                    keypoints_wrist=keypoints_wrist, keypoints_visibility=keypoints_visibility,
                    instruction=TASK_INSTRUCTION,
                )
                last_ee_pose = current_ee_pose_mm.copy()

            # 성공 조건 체크
            if progress >= 1.0:
                curr_tip = data.site_xpos[tip_id].copy()
                if np.linalg.norm(curr_tip - goal_tip) < ALIGN_THRESHOLD_M:
                    align_timer += 1
                else:
                    align_timer = 0
                if align_timer > ALIGN_HOLD_STEPS:
                    success = True
                    break

            # 타임아웃
            if data.time - record_start_time > TIMEOUT_SEC:
                break

            # control step 상한 초과
            if len(recorder.buffer) >= MAX_CTRL_STEPS:
                break

        # ============================================================
        # Phase 3: Holding — 정렬 완료 후 자세 유지 녹화 (깨끗한 expert data)
        # ============================================================
        if success:
            for hold_step in range(HOLD_RECORD_STEPS):
                for _ in range(67):
                    run_ik_step(goal_tip, goal_back)
                    mujoco.mj_step(model, data)

                curr_tip = data.site_xpos[tip_id].copy()
                curr_back = data.site_xpos[back_id].copy()
                needle_dir = (curr_tip - curr_back) / (np.linalg.norm(curr_tip - curr_back) + 1e-10)
                dist_to_surface = mujoco.mj_ray(model, data, curr_tip, needle_dir, None, 1, link6_id, np.zeros(1, dtype=np.int32))
                current_sensor_dist = dist_to_surface * 1000.0 if dist_to_surface >= 0 else -1.0

                current_qpos_deg = np.rad2deg(data.qpos[:n_motors].copy())
                current_ee_pose_mm = get_ee_pose_6d_scaled()
                delta_ee_action = current_ee_pose_mm - last_ee_pose
                delta_ee_action[3:6] = (delta_ee_action[3:6] + np.pi) % (2 * np.pi) - np.pi

                pos_mag = np.linalg.norm(delta_ee_action[:3])
                if pos_mag > ACTION_CLIP_MM:
                    delta_ee_action[:3] *= ACTION_CLIP_MM / pos_mag

                frames = {}
                for cam_name in CAMERA_LIST:
                    renderer.update_scene(data, camera=cam_name)
                    frames[cam_name] = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)

                needle_tip_mm = data.site_xpos[tip_id].copy() * 1000
                trocar_entry_mm = data.site_xpos[target_entry_id].copy() * 1000

                tip_uv_wrist = project_to_2d(data.site_xpos[tip_id], model, data, "tool_camera", IMG_WIDTH, IMG_HEIGHT)
                trocar_uv_wrist = project_to_2d(data.site_xpos[target_entry_id], model, data, "tool_camera", IMG_WIDTH, IMG_HEIGHT)
                keypoints_wrist = np.concatenate([tip_uv_wrist, trocar_uv_wrist]).astype(np.float32)

                tip_visible = float(0.0 <= tip_uv_wrist[0] <= 1.0 and 0.0 <= tip_uv_wrist[1] <= 1.0)
                trocar_visible = float(0.0 <= trocar_uv_wrist[0] <= 1.0 and 0.0 <= trocar_uv_wrist[1] <= 1.0)
                keypoints_visibility = np.array([tip_visible, trocar_visible], dtype=np.float32)

                recorder.add(
                    frames, current_qpos_deg, current_ee_pose_mm, delta_ee_action,
                    data.time, 1,  # phase=1 (유지)
                    current_sensor_dist,
                    needle_tip_mm=needle_tip_mm, trocar_entry_mm=trocar_entry_mm,
                    keypoints_wrist=keypoints_wrist, keypoints_visibility=keypoints_visibility,
                    instruction=TASK_INSTRUCTION,
                )
                last_ee_pose = current_ee_pose_mm.copy()

        if success and len(recorder.buffer) > 0:
            recorder.save_async()
            episode_count += 1
            # 4-way 교차 카운터 업데이트 (angle × Z방향 균등 배분용)
            angle_key = int(round(float(phantom_angle_deg)))
            z_dir_key = "neg" if perturb_xyz[2] < 0 else "pos"
            combo_key = (angle_key, z_dir_key)
            if combo_key in combo_counts:
                combo_counts[combo_key] += 1
            pbar.update(1)
            if GRID_CELLS is not None:
                grid_cell_index += 1
                grid_retry_count = 0
        else:
            if not perturb_reached:
                fail_reason = "IK_FAIL"
            elif len(recorder.buffer) >= MAX_CTRL_STEPS:
                fail_reason = f"MaxSteps({MAX_CTRL_STEPS})"
            else:
                fail_reason = "Timeout"
            recorder.discard()
            if GRID_CELLS is not None:
                grid_retry_count += 1
                if grid_retry_count >= grid_max_retries:
                    cell_center = [(cell[0]+cell[1])/2, (cell[2]+cell[3])/2, (cell[4]+cell[5])/2]
                    grid_fail_cells.append({
                        "center": cell_center,
                        "reason": fail_reason,
                    })
                    print(f"  Cell {grid_cell_index} SKIPPED [{fail_reason}] after {grid_max_retries} retries "
                          f"(center={cell_center[0]:.1f},{cell_center[1]:.1f},{cell_center[2]:.1f}mm)")
                    grid_cell_index += 1
                    grid_retry_count = 0
                else:
                    print(f"  Episode {episode_count} retry {grid_retry_count}/{grid_max_retries} "
                          f"(cell {grid_cell_index}) [{fail_reason}]")
            else:
                print(f"  Episode {episode_count} discarded. Reason: {fail_reason}")

    pbar.close()
    recorder.wait_for_all()

    if GRID_CELLS is not None and grid_fail_cells:
        fail_path = pathlib.Path(SAVE_DIR) / "grid_failed_cells.json"
        import json as _json
        with open(fail_path, 'w') as f:
            _json.dump(grid_fail_cells, f, indent=2)
        print(f"\n{len(grid_fail_cells)} cells failed (saved to {fail_path})")
        print(f"Succeeded: {episode_count}/{len(GRID_CELLS)} cells")
    else:
        print(f"\nAll collections finished! ({episode_count} episodes saved to {SAVE_DIR})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-alignment dataset collection")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help="Output directory for h5 files")
    parser.add_argument("--num-episodes", type=int, default=MAX_EPISODES,
                        help="Number of episodes to collect")
    parser.add_argument("--bias", type=str, default=None,
                        help="Bias perturbation direction(s). Single: 'x_neg', Combined: 'x_neg,y_neg'")
    parser.add_argument("--bias-ratio", type=float, default=0.8,
                        help="Fraction of perturbations in biased direction (default: 0.8)")
    parser.add_argument("--grid-cells-file", type=str, default=None,
                        help="JSON file with grid cells [[x_lo,x_hi,y_lo,y_hi,z_lo,z_hi], ...]")
    parser.add_argument("--perturb", type=float, nargs=3, default=None,
                        metavar=("X", "Y", "Z"),
                        help="Fixed perturbation in mm. e.g. --perturb 30 30 30")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible perturbations")
    parser.add_argument("--randomize-phantom-pos", dest="randomize_phantom_pos",
                        action="store_true", default=False,
                        help="Enable phantom position randomization per episode")
    parser.add_argument("--no-randomize-phantom-pos", dest="randomize_phantom_pos",
                        action="store_false",
                        help="Disable phantom position randomization (default)")
    parser.add_argument("--phantom-pos", type=float, nargs=2, default=None,
                        metavar=("X", "Y"),
                        help="Fixed phantom position (x, y). e.g. --phantom-pos 0.0 -0.2")
    parser.add_argument("--retreat-mm", type=float, default=RETREAT_MM,
                        help="Retreat goal_tip from trocar entry along -axis_dir (mm, default: 20)")
    parser.add_argument("--no-side-camera", action="store_true",
                        help="Skip side_camera rendering/saving (saves storage)")
    parser.add_argument("--cameras", type=str, nargs="+", default=None,
                        help="Explicit camera list (overrides --no-side-camera)")
    parser.add_argument("--allow-occluded", action="store_true",
                        help="Keep episodes even when tool_camera view of needle tip is occluded by phantom")
    args = parser.parse_args()

    # Override globals from CLI args
    SAVE_DIR = args.save_dir
    MAX_EPISODES = args.num_episodes
    RETREAT_MM = args.retreat_mm

    # Store bias config as global for use in main()
    BIAS_DIRECTION = args.bias
    BIAS_RATIO = args.bias_ratio

    # Seed
    RANDOM_SEED = args.seed

    # Allow occluded episodes
    ALLOW_OCCLUDED = args.allow_occluded
    if ALLOW_OCCLUDED:
        print("ALLOW_OCCLUDED=True: occluded episodes will be kept")

    # Phantom randomization / fixed position
    RANDOMIZE_PHANTOM = args.randomize_phantom_pos
    if args.phantom_pos is not None:
        PHANTOM_POS = tuple(args.phantom_pos)

    if args.cameras is not None:
        CAMERA_LIST = args.cameras
        print(f"Cameras: {CAMERA_LIST}")
    elif args.no_side_camera:
        CAMERA_LIST = [c for c in CAMERA_LIST if c != "side_camera"]
        print(f"Cameras: {CAMERA_LIST}")

    # Fixed perturbation mode
    if args.perturb is not None:
        x, y, z = args.perturb
        GRID_CELLS = [[x, x, y, y, z, z]]
        print(f"Fixed perturbation mode: X={x}mm Y={y}mm Z={z}mm")

    # Grid mode
    if args.grid_cells_file:
        with open(args.grid_cells_file, 'r') as f:
            GRID_CELLS = json.load(f)
        print(f"Grid mode: {len(GRID_CELLS)} cells loaded from {args.grid_cells_file}")

    if GRID_CELLS is not None:
        MAX_EPISODES = len(GRID_CELLS)

    main()
