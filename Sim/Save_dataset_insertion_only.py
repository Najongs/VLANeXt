"""
Insertion-only dataset collection.

1. 트로카 위치 고정/랜덤화
2. 기존 정렬 알고리즘으로 needle tip을 trocar entry까지 이동 (녹화 X)
3. (선택) 작은 perturbation 적용 — 불완전 정렬 상태에서 삽입 시작
4. 삽입 과정만 녹화 (정렬 X)
5. 목표 깊이 도달 + hold 시 에피소드 종료

핵심: 삽입 중 보정(correct-while-inserting) 전략 학습용 데이터.
정렬이 완벽하지 않아도 삽입하면서 각도를 보정하는 expert trajectory 생성.

Usage:
    # 기본 (완벽 정렬 후 삽입)
    python Sim/Save_dataset_insertion_only.py

    # 불완전 정렬에서 시작 (perturbation 적용)
    python Sim/Save_dataset_insertion_only.py --perturb


python run_parallel.py --script insertion --workers 10 --episodes 25 \
    --base-dir /data/public/NAS/VLANeXt/dataset/fine_align/insertion_data --phantom-pos 0.0 0.0

    # 랜덤 팬텀 + perturbation + 500개
    python Sim/Save_dataset_insertion_only.py \
        --save-dir dataset/insertion_data \
        --num-episodes 500 \
        --perturb --randomize-phantom-pos
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
# === Configuration ===
# ============================================================

MODEL_PATH = "meca_add.xml"
SAVE_DIR = "collected_data_insertion"
MAX_EPISODES = 1
IMG_WIDTH = 640
IMG_HEIGHT = 480

# --- 속도 ---
ALIGN_SPEED = 0.1           # 초기 정렬 속도 (m/s) — 녹화 전 이동용
APPROACH_SPEED = 0.005       # 최종 접근 속도 (m/s) — 녹화 중, entry까지 전진
INSERTION_SPEED = 0.0025     # 삽입 속도 (m/s) — 녹화 중

# --- 삽입 설정 ---
APPROACH_OFFSET_MM = 5.0          # align 모델 종료 위치 시뮬레이션: entry에서 trocar axis 방향으로 이만큼 뒤 (mm)
APPROACH_XY_OFFSET_MM = 2.0       # align 모델 xy 오차 시뮬레이션: trocar axis 수직 방향 랜덤 오프셋 (±mm)
TARGET_INSERTION_DEPTH = 0.0275   # 목표 삽입 깊이 (m) = 27.5mm
HOLD_DURATION_SEC = 1.0           # 삽입 완료 후 유지 시간 (초)

# --- Perturbation 설정 (불완전 정렬 시뮬레이션) ---
# align 모델의 실제 오차 범위를 모사: 위치는 작고 각도가 주 오차원
PERTURB_ENABLED = False
PERTURB_POS_XY_MM = 3.0     # XY 평면 perturbation 범위 (±mm) — align보다 훨씬 작음
PERTURB_POS_Z_MM = 2.0      # Z축 perturbation 범위 (±mm)
PERTURB_ANGLE_DEG = 5.0     # 각도 perturbation 범위 (±deg) — 주된 오차원

# --- 성공 조건 ---
ALIGN_THRESHOLD_M = 0.002   # pre-alignment 완료 판정 거리 (m)
ALIGN_HOLD_STEPS = 20       # threshold 이내 연속 유지 횟수

# --- Task Instruction ---
TASK_INSTRUCTION = "Insert the needle through the trocar opening while maintaining alignment"

# --- 기타 ---
ACTION_CLIP_MM = 1.0        # IK spike 방지용 delta position 클리핑 (mm)
TIMEOUT_SEC = 30.0          # 에피소드 전체 타임아웃 (초)

# --- Runtime globals (set via CLI) ---
RANDOM_SEED = None
RANDOMIZE_PHANTOM = False
PHANTOM_POS = None


# ============================================================
# === SimRecorder (align_only와 동일) ===
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
            keypoints_visibility=None, instruction=""):
        if not self.recording: return
        self.buffer.append({
            "ts": timestamp,
            "imgs": frames,
            "q": qpos,
            "p": ee_pose,
            "act": action,
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

                    # Insertion phase → gripper closed
                    action_gripper = np.full((len(data), 1), 1.0, dtype=np.float32)
                    act_data = np.concatenate([act_data, action_gripper], axis=-1)  # (N, 7)

                    proprio_gripper = np.full((len(data), 1), 1.0, dtype=np.float32)
                    p_data = np.concatenate([p_data, proprio_gripper], axis=-1)  # (N, 7)

                    obs.create_dataset("qpos", data=q_data, compression="gzip")
                    obs.create_dataset("ee_pose", data=p_data, compression="gzip")
                    obs.create_dataset("sensor_dist", data=sensor_data, compression="gzip")
                    f.create_dataset("action", data=act_data, compression="gzip")
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


def randomize_phantom_pos(model, data, phantom_id, rot_id):
    offset_x = np.random.uniform(-0.1, 0.1)
    offset_y = np.random.uniform(-0.4, 0.0)
    offset_z = 0.0

    model.body_pos[phantom_id] = np.array([offset_x, offset_y, offset_z])

    if offset_y >= -0.25:
        random_angle_deg = np.random.uniform(-15, 15)
    else:
        random_angle_deg = np.random.uniform(-15 - 90, 15 - 90)

    new_quat = np.zeros(4)
    mujoco.mju_euler2Quat(new_quat, [0, 0, np.deg2rad(random_angle_deg)], "xyz")
    model.body_quat[rot_id] = new_quat
    print(f">>> Randomize: Pos=({offset_x:.2f}, {offset_y:.2f}), Angle={random_angle_deg:.1f} deg")
    mujoco.mj_forward(model, data)
    return np.array([offset_x, offset_y, offset_z], dtype=np.float32), new_quat.astype(np.float32), np.float32(random_angle_deg)


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

    phantom_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "phantom_assembly")
    rotating_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rotating_assembly")

    recorder = SimRecorder(SAVE_DIR)
    ik_speed = 0.5

    def get_ee_pose_6d_scaled():
        if link6_id >= 0:
            pos = data.xpos[link6_id].copy() * 1000
            mat = data.xmat[link6_id].reshape(3, 3)
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
        """IK solver 1스텝 실행"""
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

    def get_sensor_dist():
        """needle tip에서 needle direction으로 ray cast → 거리(mm)"""
        curr_tip = data.site_xpos[tip_id].copy()
        curr_back = data.site_xpos[back_id].copy()
        needle_dir = (curr_tip - curr_back) / (np.linalg.norm(curr_tip - curr_back) + 1e-10)
        dist = mujoco.mj_ray(model, data, curr_tip, needle_dir, None, 1, link6_id, np.zeros(1, dtype=np.int32))
        return dist * 1000.0 if dist >= 0 else -1.0

    def record_step(last_ee_pose, phase):
        """현재 상태를 녹화하고 last_ee_pose 업데이트"""
        current_qpos_deg = np.rad2deg(data.qpos[:n_motors].copy())
        current_ee_pose_mm = get_ee_pose_6d_scaled()
        delta_ee_action = current_ee_pose_mm - last_ee_pose

        pos_mag = np.linalg.norm(delta_ee_action[:3])
        if pos_mag > ACTION_CLIP_MM:
            delta_ee_action[:3] *= ACTION_CLIP_MM / pos_mag

        frames = {}
        for cam_name in ["side_camera", "tool_camera", "top_camera"]:
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

        current_sensor_dist = get_sensor_dist()

        recorder.add(
            frames, current_qpos_deg, current_ee_pose_mm, delta_ee_action,
            data.time, phase, current_sensor_dist,
            needle_tip_mm=needle_tip_mm, trocar_entry_mm=trocar_entry_mm,
            keypoints_wrist=keypoints_wrist, keypoints_visibility=keypoints_visibility,
            instruction=TASK_INSTRUCTION,
        )
        return current_ee_pose_mm.copy()

    print(f"Starting Insertion Data Collection ({MAX_EPISODES} episodes)...")
    print(f"  Approach offset: {APPROACH_OFFSET_MM}mm (start behind entry)")
    print(f"  Perturbation: {'ON' if PERTURB_ENABLED else 'OFF'}")
    if PERTURB_ENABLED:
        print(f"  Perturb range: XY=±{PERTURB_POS_XY_MM}mm, Z=±{PERTURB_POS_Z_MM}mm, Angle=±{PERTURB_ANGLE_DEG}deg")
    pbar = tqdm(total=MAX_EPISODES, desc="Collecting", unit="ep")

    # ============================================================
    # Phase 0: 최초 1회 pre-alignment
    # ============================================================
    print("Running initial pre-alignment (one-time)...")
    mujoco.mj_resetData(model, data)
    home_pose = np.array([
        np.random.uniform(-0.45, 0.55),
        np.random.uniform(-0.4, -0.3),
        np.random.uniform(0.3, 0.4),
        0.0,
        np.random.uniform(0.45, 0.55),
        np.random.uniform(0.95, 1.05),
    ])
    data.qpos[:6] = home_pose
    mujoco.mj_forward(model, data)

    p_entry = data.site_xpos[target_entry_id].copy()
    p_depth = data.site_xpos[target_depth_id].copy()
    curr_tip = data.site_xpos[tip_id].copy()
    curr_back = data.site_xpos[back_id].copy()
    needle_len = np.linalg.norm(curr_tip - curr_back)

    axis_dir = (p_depth - p_entry) / (np.linalg.norm(p_depth - p_entry) + 1e-10)
    # Pre-align 목표: entry에서 APPROACH_OFFSET만큼 뒤 (align 모델 종료 위치 시뮬레이션)
    approach_offset_m = APPROACH_OFFSET_MM / 1000.0
    goal_tip = p_entry - (axis_dir * approach_offset_m)
    goal_back = goal_tip - (axis_dir * needle_len)

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

    aligned_qpos = data.qpos[:n_motors].copy()
    aligned_qvel = data.qvel[:n_motors].copy()
    print("Pre-alignment done.")

    # ============================================================
    # 에피소드 루프
    # ============================================================
    episode_count = 0
    while episode_count < MAX_EPISODES:
        # 정렬된 상태로 리셋
        mujoco.mj_resetData(model, data)
        data.qpos[:n_motors] = aligned_qpos
        data.qvel[:n_motors] = aligned_qvel
        mujoco.mj_forward(model, data)

        # --- 팬텀 랜덤화 ---
        phantom_offset = np.zeros(3, dtype=np.float32)
        phantom_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        phantom_angle_deg = np.float32(0.0)
        if PHANTOM_POS is not None and phantom_body_id >= 0:
            px, py = PHANTOM_POS
            model.body_pos[phantom_body_id] = np.array([px, py, 0.0])
            if py >= -0.25:
                rand_angle = np.random.uniform(-15, 15)
            else:
                rand_angle = np.random.uniform(-15 - 90, 15 - 90)
            new_quat = np.zeros(4)
            mujoco.mju_euler2Quat(new_quat, [0, 0, np.deg2rad(rand_angle)], "xyz")
            model.body_quat[rotating_id] = new_quat
            mujoco.mj_forward(model, data)
            phantom_offset = np.array([px, py, 0.0], dtype=np.float32)
            phantom_quat = new_quat.astype(np.float32)
            phantom_angle_deg = np.float32(rand_angle)
        elif RANDOMIZE_PHANTOM and phantom_body_id >= 0:
            phantom_offset, phantom_quat, phantom_angle_deg = randomize_phantom_pos(
                model, data, phantom_body_id, rotating_id)

        # 팬텀 이동 시 재정렬
        need_realign = (PHANTOM_POS is not None or RANDOMIZE_PHANTOM) and phantom_body_id >= 0
        if need_realign:
            p_entry = data.site_xpos[target_entry_id].copy()
            p_depth = data.site_xpos[target_depth_id].copy()
            curr_tip = data.site_xpos[tip_id].copy()
            curr_back = data.site_xpos[back_id].copy()
            needle_len_local = np.linalg.norm(curr_tip - curr_back)
            axis_dir_local = (p_depth - p_entry) / (np.linalg.norm(p_depth - p_entry) + 1e-10)
            re_goal_tip = p_entry - (axis_dir_local * approach_offset_m)
            re_goal_back = re_goal_tip - (axis_dir_local * needle_len_local)

            start_tip = data.site_xpos[tip_id].copy()
            start_back = data.site_xpos[back_id].copy()
            re_dist = np.linalg.norm(re_goal_tip - start_tip)
            re_duration = max(re_dist / ALIGN_SPEED, 0.1)
            re_start_time = data.time
            re_timer = 0

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
                        break

                if data.time - re_start_time > 50.0:
                    print(f"Re-alignment failed for phantom offset={phantom_offset}, skipping...")
                    break

            aligned_qpos = data.qpos[:n_motors].copy()
            aligned_qvel = data.qvel[:n_motors].copy()
            p_entry = data.site_xpos[target_entry_id].copy()
            p_depth = data.site_xpos[target_depth_id].copy()
            axis_dir = (p_depth - p_entry) / (np.linalg.norm(p_depth - p_entry) + 1e-10)
            goal_tip = p_entry - (axis_dir * approach_offset_m)
            goal_back = goal_tip - (axis_dir * needle_len)

        # ============================================================
        # XY offset 적용: align 모델의 xy 오차 시뮬레이션 (항상 적용, 녹화 X)
        # ============================================================
        # 삽입 방향: trocar axis (entry → depth)
        p_entry = data.site_xpos[target_entry_id].copy()
        p_depth = data.site_xpos[target_depth_id].copy()
        insert_axis = (p_depth - p_entry) / (np.linalg.norm(p_depth - p_entry) + 1e-10)

        # trocar axis에 수직인 두 방향 벡터 계산
        up_vec = np.array([0, 0, 1])
        lateral1 = np.cross(insert_axis, up_vec)
        lateral1 = lateral1 / (np.linalg.norm(lateral1) + 1e-10)
        lateral2 = np.cross(insert_axis, lateral1)
        lateral2 = lateral2 / (np.linalg.norm(lateral2) + 1e-10)

        xy_offset_1 = np.random.uniform(-APPROACH_XY_OFFSET_MM, APPROACH_XY_OFFSET_MM) / 1000.0
        xy_offset_2 = np.random.uniform(-APPROACH_XY_OFFSET_MM, APPROACH_XY_OFFSET_MM) / 1000.0
        xy_offset_vec = lateral1 * xy_offset_1 + lateral2 * xy_offset_2

        offset_tip = goal_tip + xy_offset_vec
        offset_back = goal_back + xy_offset_vec  # 방향 유지, 위치만 이동

        # IK로 xy offset 위치까지 이동 (녹화 X)
        move_dist = np.linalg.norm(xy_offset_vec)
        if move_dist > 1e-5:
            move_duration = max(move_dist / 0.05, 0.05)
            move_start_time = data.time
            for ps in range(3000):
                t = (data.time - move_start_time) / move_duration
                alpha = smooth_step(min(t, 1.0))
                interp_tip = (1 - alpha) * goal_tip + alpha * offset_tip
                interp_back = (1 - alpha) * goal_back + alpha * offset_back
                run_ik_step(interp_tip, interp_back)
                mujoco.mj_step(model, data)
                if t >= 1.0:
                    if np.linalg.norm(data.site_xpos[tip_id] - offset_tip) < 0.001:
                        for _ in range(100):
                            run_ik_step(offset_tip, offset_back)
                            mujoco.mj_step(model, data)
                        break
                    if ps > 2500:
                        break

        xy_dist_mm = np.linalg.norm(xy_offset_vec) * 1000
        print(f"  Episode {episode_count}: xy_offset={xy_dist_mm:.1f}mm "
              f"({xy_offset_1*1000:.1f}, {xy_offset_2*1000:.1f})")

        # ============================================================
        # Phase 1 (optional): Perturbation — 각도 오차 추가
        # ============================================================
        perturb_xyz = np.zeros(3)
        perturb_angle_rad = 0.0

        # 현재 시작점: xy offset 적용된 위치
        start_tip_for_record = offset_tip
        start_back_for_record = offset_back

        if PERTURB_ENABLED:
            perturb_xyz = np.array([
                np.random.uniform(-PERTURB_POS_XY_MM, PERTURB_POS_XY_MM) / 1000.0,
                np.random.uniform(-PERTURB_POS_XY_MM, PERTURB_POS_XY_MM) / 1000.0,
                np.random.uniform(-PERTURB_POS_Z_MM, PERTURB_POS_Z_MM) / 1000.0,
            ])
            perturb_angle_rad = np.deg2rad(np.random.uniform(-PERTURB_ANGLE_DEG, PERTURB_ANGLE_DEG))
            random_axis = np.random.randn(3)
            random_axis = random_axis / (np.linalg.norm(random_axis) + 1e-10)

            perturbed_tip = offset_tip + perturb_xyz
            rot_mat_perturb = np.eye(3)
            if abs(perturb_angle_rad) > 1e-6:
                K = np.array([
                    [0, -random_axis[2], random_axis[1]],
                    [random_axis[2], 0, -random_axis[0]],
                    [-random_axis[1], random_axis[0], 0],
                ])
                rot_mat_perturb = np.eye(3) + np.sin(perturb_angle_rad) * K + (1 - np.cos(perturb_angle_rad)) * (K @ K)
            perturbed_back_dir = rot_mat_perturb @ (offset_back - offset_tip)
            perturbed_back = perturbed_tip + perturbed_back_dir

            # IK로 perturbed 위치까지 이동 (녹화 X)
            move_dist_p = np.linalg.norm(perturbed_tip - offset_tip)
            move_duration = max(move_dist_p / 0.05, 0.1)
            move_start_time = data.time

            perturb_reached = False
            for ps in range(5000):
                t = (data.time - move_start_time) / move_duration
                alpha = smooth_step(min(t, 1.0))
                interp_tip = (1 - alpha) * offset_tip + alpha * perturbed_tip
                interp_back = (1 - alpha) * offset_back + alpha * perturbed_back

                run_ik_step(interp_tip, interp_back)
                mujoco.mj_step(model, data)

                if t >= 1.0:
                    if np.linalg.norm(data.site_xpos[tip_id] - perturbed_tip) < 0.001:
                        for _ in range(200):
                            run_ik_step(perturbed_tip, perturbed_back)
                            mujoco.mj_step(model, data)
                        perturb_reached = True
                        break
                    if ps > 4500:
                        break

            perturb_dist_mm = np.linalg.norm(perturb_xyz) * 1000
            print(f"    + perturbation: pos={perturb_dist_mm:.1f}mm, "
                  f"angle={np.rad2deg(perturb_angle_rad):.1f}deg "
                  f"[{'OK' if perturb_reached else 'IK_FAIL'}]")

            start_tip_for_record = perturbed_tip
            start_back_for_record = perturbed_back

        # ============================================================
        # Phase 2: 접근 + 삽입 녹화
        # ============================================================
        last_ee_pose = get_ee_pose_6d_scaled()

        episode_meta = {
            "aligned_qpos": np.rad2deg(aligned_qpos).astype(np.float32),
            "target_entry_world": p_entry.astype(np.float32),
            "target_depth_world": p_depth.astype(np.float32),
            "insertion_depth_m": np.float32(TARGET_INSERTION_DEPTH),
            "approach_xy_offset_mm": (xy_offset_vec * 1000).astype(np.float32),
        }
        if PERTURB_ENABLED:
            episode_meta["perturb_xyz_mm"] = (perturb_xyz * 1000).astype(np.float32)
            episode_meta["perturb_angle_deg"] = np.array(np.rad2deg(perturb_angle_rad), dtype=np.float32)
        if RANDOMIZE_PHANTOM or PHANTOM_POS is not None:
            episode_meta["phantom_offset"] = phantom_offset
            episode_meta["phantom_quat"] = phantom_quat
            episode_meta["phantom_angle_deg"] = phantom_angle_deg
        recorder.start(episode_meta)

        record_start_time = data.time
        step_count = 0
        accumulated_approach = 0.0   # 접근 진행량 (m)
        accumulated_depth = 0.0      # 삽입 진행량 (m)
        hold_start_time = None
        success = False
        current_phase = "approach"    # approach → insert → hold

        # 시작점: 현재 tip (offset 위치, perturbation 적용 가능)
        approach_base_tip = data.site_xpos[tip_id].copy()

        # 접근 목표: trocar entry 바로 앞 (0.1mm)
        approach_goal_tip = p_entry - (insert_axis * 0.0001)

        while True:
            if current_phase == "approach":
                # Phase A: 현재 위치(xy offset + perturb)에서 entry로 수렴하며 전진
                accumulated_approach += APPROACH_SPEED * model.opt.timestep
                approach_total = approach_offset_m - 0.0001  # offset에서 entry 0.1mm 앞까지
                approach_progress = min(accumulated_approach / max(approach_total, 1e-6), 1.0)

                # 시작점에서 axis 방향으로 전진한 naive 위치
                naive_tip = approach_base_tip + (insert_axis * accumulated_approach)
                # ideal 위치: trocar axis 위의 정확한 위치
                ideal_tip = goal_tip + (insert_axis * accumulated_approach)
                # 접근하면서 점진적으로 ideal(trocar axis)로 수렴
                correction_alpha = smooth_step(approach_progress)
                target_tip_pos = (1 - correction_alpha) * naive_tip + correction_alpha * ideal_tip

                target_back_pos = target_tip_pos - (insert_axis * needle_len)

                if approach_progress >= 1.0:
                    current_phase = "insert"

            elif current_phase == "insert":
                # Phase B: 삽입 진행
                accumulated_depth += INSERTION_SPEED * model.opt.timestep

                target_tip_pos = approach_goal_tip + (insert_axis * accumulated_depth)
                target_back_pos = target_tip_pos - (insert_axis * needle_len)

                if accumulated_depth >= TARGET_INSERTION_DEPTH:
                    current_phase = "hold"
                    hold_start_time = data.time

            else:  # hold
                # Phase C: 삽입 완료 후 위치 유지
                if hold_start_time is None:
                    hold_start_time = data.time
                target_tip_pos = approach_goal_tip + (insert_axis * TARGET_INSERTION_DEPTH)
                target_back_pos = target_tip_pos - (insert_axis * needle_len)

                if data.time - hold_start_time >= HOLD_DURATION_SEC:
                    success = True
                    break

            run_ik_step(target_tip_pos, target_back_pos)
            mujoco.mj_step(model, data)
            step_count += 1

            # 녹화 (67 스텝마다) — phase 번호: approach=1, insert=2, hold=2
            if step_count % 67 == 0:
                phase_num = 1 if current_phase == "approach" else 2
                last_ee_pose = record_step(last_ee_pose, phase=phase_num)

            # 타임아웃
            if data.time - record_start_time > TIMEOUT_SEC:
                break

        if success and len(recorder.buffer) > 0:
            depth_mm = accumulated_depth * 1000
            sensor_val = get_sensor_dist()
            print(f"  Episode {episode_count}: SUCCESS (depth={depth_mm:.1f}mm, sensor={sensor_val:.1f}mm)")
            recorder.save_async()
            episode_count += 1
            pbar.update(1)
        else:
            reason = "Timeout" if data.time - record_start_time > TIMEOUT_SEC else "Unknown"
            print(f"  Episode {episode_count} discarded. Reason: {reason}")
            recorder.discard()

    pbar.close()
    recorder.wait_for_all()
    print(f"\nAll collections finished! ({episode_count} episodes saved to {SAVE_DIR})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Insertion-only dataset collection")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help="Output directory for h5 files")
    parser.add_argument("--num-episodes", type=int, default=MAX_EPISODES,
                        help="Number of episodes to collect")
    parser.add_argument("--perturb", action="store_true", default=False,
                        help="Enable perturbation (imperfect alignment start)")
    parser.add_argument("--perturb-pos-xy", type=float, default=PERTURB_POS_XY_MM,
                        help="XY perturbation range in mm (default: 3.0)")
    parser.add_argument("--perturb-pos-z", type=float, default=PERTURB_POS_Z_MM,
                        help="Z perturbation range in mm (default: 2.0)")
    parser.add_argument("--perturb-angle", type=float, default=PERTURB_ANGLE_DEG,
                        help="Angle perturbation range in deg (default: 5.0)")
    parser.add_argument("--approach-offset", type=float, default=APPROACH_OFFSET_MM,
                        help="Start position behind entry in mm (default: 5.0)")
    parser.add_argument("--approach-xy-offset", type=float, default=APPROACH_XY_OFFSET_MM,
                        help="Random XY offset perpendicular to trocar axis in mm (default: 2.0)")
    parser.add_argument("--insertion-depth", type=float, default=TARGET_INSERTION_DEPTH * 1000,
                        help="Target insertion depth in mm (default: 27.5)")
    parser.add_argument("--insertion-speed", type=float, default=INSERTION_SPEED * 1000,
                        help="Insertion speed in mm/s (default: 2.5)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--randomize-phantom-pos", dest="randomize_phantom_pos",
                        action="store_true", default=False,
                        help="Enable phantom position randomization per episode")
    parser.add_argument("--no-randomize-phantom-pos", dest="randomize_phantom_pos",
                        action="store_false")
    parser.add_argument("--phantom-pos", type=float, nargs=2, default=None,
                        metavar=("X", "Y"),
                        help="Fixed phantom position (x, y)")
    args = parser.parse_args()

    # Override globals
    SAVE_DIR = args.save_dir
    MAX_EPISODES = args.num_episodes
    APPROACH_OFFSET_MM = args.approach_offset
    APPROACH_XY_OFFSET_MM = args.approach_xy_offset
    PERTURB_ENABLED = args.perturb
    PERTURB_POS_XY_MM = args.perturb_pos_xy
    PERTURB_POS_Z_MM = args.perturb_pos_z
    PERTURB_ANGLE_DEG = args.perturb_angle
    TARGET_INSERTION_DEPTH = args.insertion_depth / 1000.0
    INSERTION_SPEED = args.insertion_speed / 1000.0
    RANDOM_SEED = args.seed
    RANDOMIZE_PHANTOM = args.randomize_phantom_pos
    if args.phantom_pos is not None:
        PHANTOM_POS = tuple(args.phantom_pos)

    main()
