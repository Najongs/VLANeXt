"""
Fine-alignment only dataset collection.

1. 트로카 위치 고정 (랜덤화 없음)
2. 기존 정렬 알고리즘으로 needle tip을 trocar entry까지 이동 (녹화 X)
3. 랜덤 perturbation 적용
4. 미세 정렬만 녹화 (삽입 X)
5. 정렬 완료 시 에피소드 종료

Usage:
    python Save_dataset_align_only.py
"""

import os
os.environ['MUJOCO_GL'] = 'egl'

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

# --- 정렬 속도 ---
ALIGN_SPEED = 0.1          # 초기 정렬 속도 (m/s) — 녹화 전 이동용
FINE_ALIGN_SPEED = 0.005    # 미세 정렬 속도 (m/s) — 녹화 중

# --- Perturbation 설정 (미세 정렬 시작 전 흐트러뜨리는 범위) ---
PERTURB_POS_XY_MM = 15.0    # XY 평면 perturbation 범위 (±mm)
PERTURB_POS_Z_MM = 10.0     # Z축 perturbation 범위 (±mm)
PERTURB_ANGLE_DEG = 10.0    # 각도 perturbation 범위 (±deg)

# --- 성공 조건 ---
ALIGN_THRESHOLD_M = 0.002   # needle tip - trocar entry 거리 (m)
ALIGN_HOLD_STEPS = 20       # threshold 이내 연속 유지 횟수

# --- Task Instruction ---
TASK_INSTRUCTION = "Align the needle to the trocar opening"

# --- Holding (정렬 완료 후 자세 유지 녹화) ---
HOLD_RECORD_STEPS = 5           # 정렬 완료 후 녹화 control steps

# --- 기타 ---
ACTION_CLIP_MM = 1.0        # IK spike 방지용 delta position 클리핑 (mm)
TIMEOUT_SEC = 10.0          # 에피소드 전체 타임아웃 (초)

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

                    # Fine alignment only → gripper always open
                    action_gripper = np.full((len(data), 1), -1.0, dtype=np.float32)
                    act_data = np.concatenate([act_data, action_gripper], axis=-1)  # (N, 7)

                    proprio_gripper = np.full((len(data), 1), 0.0, dtype=np.float32)
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


def main():
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

    recorder = SimRecorder(SAVE_DIR)

    # 초기 home pose (정렬 시작점)
    home_pose = np.array([
        np.random.uniform(-0.45, 0.55),
        np.random.uniform(-0.4, -0.3),
        np.random.uniform(0.3, 0.4),
        0.0,
        np.random.uniform(0.45, 0.55),
        np.random.uniform(0.95, 1.05),
    ])
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
    goal_tip = p_entry - (axis_dir * 0.0001)
    goal_back = p_entry - (axis_dir * (0.0001 + needle_len))

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
    while episode_count < MAX_EPISODES:
        # 정렬된 상태로 즉시 리셋
        mujoco.mj_resetData(model, data)
        data.qpos[:n_motors] = aligned_qpos
        data.qvel[:n_motors] = aligned_qvel
        mujoco.mj_forward(model, data)

        # ============================================================
        # Phase 1: Perturbation 적용 (녹화 X)
        # ============================================================

        perturb_xyz = np.array([
            np.random.uniform(-PERTURB_POS_XY_MM, PERTURB_POS_XY_MM) / 1000.0,
            np.random.uniform(-PERTURB_POS_XY_MM, PERTURB_POS_XY_MM) / 1000.0,
            np.random.uniform(-PERTURB_POS_Z_MM, PERTURB_POS_Z_MM) / 1000.0,
        ])
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

        # IK로 perturbed 위치까지 이동
        for ps in range(3000):
            run_ik_step(perturbed_tip, perturbed_back)
            mujoco.mj_step(model, data)
            if np.linalg.norm(data.site_xpos[tip_id] - perturbed_tip) < 0.001:
                for _ in range(200):
                    run_ik_step(perturbed_tip, perturbed_back)
                    mujoco.mj_step(model, data)
                break

        perturb_dist_mm = np.linalg.norm(perturb_xyz) * 1000
        print(f"  Episode {episode_count}: perturbation applied "
              f"(pos={perturb_dist_mm:.1f}mm, angle={np.rad2deg(perturb_angle_rad):.1f}deg)")

        # ============================================================
        # Phase 2: 미세 정렬 녹화
        # ============================================================
        last_ee_pose = get_ee_pose_6d_scaled()
        recorder.start({
            "aligned_qpos": np.rad2deg(aligned_qpos).astype(np.float32),
            "perturb_xyz_mm": (perturb_xyz * 1000).astype(np.float32),
            "perturb_angle_deg": np.array(np.rad2deg(perturb_angle_rad), dtype=np.float32),
            "target_entry_world": p_entry.astype(np.float32),
            "target_depth_world": p_depth.astype(np.float32),
        })

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
            progress = smooth_step((data.time - fine_traj_start) / fine_duration) if fine_duration > 0 else 1.0

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
            pbar.update(1)
        else:
            reason = "Timeout" if not success else "Empty buffer"
            print(f"  Episode {episode_count} discarded. Reason: {reason}")
            recorder.discard()

    pbar.close()
    recorder.wait_for_all()
    print(f"\nAll collections finished! ({episode_count} episodes saved to {SAVE_DIR})")


if __name__ == "__main__":
    main()
