"""
# python Save_dataset.py --no-randomize-phantom-pos

# python run_parallel.py \
#     --script full --workers 5 --episodes 5 \
#     --base-dir /data/public/NAS/VLANeXt/dataset/fine_align/approach_test \
#     --phantom-pos 0.0 0.0 --no-insertion

# python run_parallel.py --script approach --workers 5 --episodes 5 \
#     --base-dir /data/public/NAS/VLANeXt/dataset/fine_align/approach_data \
#     --phantom-pos 0.0 0.0 --no-insertion

python run_parallel.py --script approach --workers 20 --episodes 500 \
    --base-dir /data/public/NAS/VLANeXt/dataset/fine_align/approach_data \
    --no-insertion
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

    # World → Camera frame
    p_cam = cam_mat.T @ (point_3d - cam_pos)

    # Perspective projection (MuJoCo convention: camera looks along -Z)
    f = img_h / (2.0 * np.tan(np.deg2rad(fovy) / 2.0))
    u = -f * (p_cam[0] / p_cam[2]) + (img_w - 1) / 2.0
    v =  f * (p_cam[1] / p_cam[2]) + (img_h - 1) / 2.0

    return np.array([u / img_w, v / img_h], dtype=np.float32)


# === Configuration ===
MODEL_PATH = "meca_add.xml"
SAVE_DIR = "collected_data_sim_clean"
MAX_EPISODES = 1
IMG_WIDTH = 640
IMG_HEIGHT = 480
CAMERA_LIST = ["side_camera", "tool_camera", "top_camera"]
TARGET_INSERTION_DEPTH = 0.0275
ALIGN_SPEED = 0.005      # 정렬 단계 속도: 0.02 m/s (~200 steps)
# Velocity profile shape during approach trajectory.
# 0.5 = pure cubic smoothstep (slow tails dominate), 0.2 = 20% accel + 60% cruise + 20% decel,
# 0.1 = sharper trapezoid. Lower = more time at constant velocity, less "stop-near-goal" bias.
APPROACH_ACCEL_FRAC = 0.1
INSERTION_SPEED = 0.0025  # 삽입 단계 속도: 0.003 m/s (초당 3mm)
TASK_INSTRUCTION = "Approach the needle tip to the small grey circular trocar port on the eye model, next to the larger lens opening"
ACTION_CLIP_MM = 1.0  # phase 전환 시 IK spike 방지: delta position 클리핑 (mm)
MAX_CTRL_STEPS = 500        # 녹화 control step 상한 (초과 시 에피소드 폐기)
HOLD_STEPS = 10             # 도달 후 hold 프레임 수 (control steps, action≈0 기록)
RETREAT_MM = 1.0           # goal_tip을 trocar entry에서 뒤로 빼는 거리 (mm) — align과 동일
WARMUP_STEPS = 500          # 녹화 전 J6 settling 대기 (sim steps, 67 control step ≈ 7 control frames)

# === Recorder Class (수정됨: sensor_dist 저장 로직 추가) ===
class SimRecorder:
    def __init__(self, output_dir):
        self.out = pathlib.Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.buffer = []
        self.episode_metadata = {}
        self.recording = False
        self.is_saving = False
        self.save_threads = []  # 저장 스레드 추적

    def start(self, episode_metadata=None):
        # 이전 저장이 진행 중이어도 새 에피소드 시작 가능
        # (각 에피소드는 독립적인 버퍼 사용)
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
                    act_data = np.array([x['act'] for x in data], dtype=np.float32)  # (N, 6)
                    ts_data = np.array([x['ts'] for x in data], dtype=np.float32)
                    phase_data = np.array([x['phase'] for x in data], dtype=np.int32)
                    sensor_data = np.array([x['sd'] for x in data], dtype=np.float32)

                    # Gripper: phase 1(정렬)→open, phase 2(삽입)→closed
                    # Action gripper: -1=open, 1=closed (DROID convention)
                    action_gripper = np.where(phase_data >= 2, 1.0, -1.0).astype(np.float32).reshape(-1, 1)
                    act_data = np.concatenate([act_data, action_gripper], axis=-1)  # (N, 7)

                    # Proprio gripper: 0=open, 1=closed (DROID convention)
                    proprio_gripper = np.where(phase_data >= 2, 1.0, 0.0).astype(np.float32).reshape(-1, 1)
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
                        sim_grip = np.where(phase_data >= 2, 1.0, -1.0).astype(np.float32).reshape(-1, 1)
                        sim_act_arr = np.concatenate([sim_act_arr, sim_grip], axis=-1)
                        f.create_dataset("action_sim", data=sim_act_arr, compression="gzip")

                    f.create_dataset("timestamp", data=ts_data, compression="gzip")
                    f.create_dataset("phase", data=phase_data, compression="gzip")

                    # Spatial auxiliary data (needle_tip, trocar, 2D keypoints from wrist camera)
                    if data[0].get("needle_tip_mm") is not None:
                        needle_tip_data = np.array([x['needle_tip_mm'] for x in data], dtype=np.float32)
                        trocar_entry_data = np.array([x['trocar_entry_mm'] for x in data], dtype=np.float32)
                        kp_wrist_data = np.array([x['keypoints_wrist'] for x in data], dtype=np.float32)
                        kp_vis_data = np.array([x['keypoints_visibility'] for x in data], dtype=np.float32)
                        obs.create_dataset("needle_tip_pos", data=needle_tip_data, compression="gzip")
                        obs.create_dataset("trocar_entry_pos", data=trocar_entry_data, compression="gzip")
                        obs.create_dataset("keypoints_wrist", data=kp_wrist_data, compression="gzip")
                        obs.create_dataset("keypoints_visibility", data=kp_vis_data, compression="gzip")

                    # Language instruction 저장
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
                print(f"❌ Save Failed: {e}")
            finally:
                self.is_saving = False

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = self.out / f"episode_{timestamp}.h5"
        t = threading.Thread(target=worker, args=(data_snapshot, metadata_snapshot, fname))
        t.start()
        self.save_threads.append(t)  # 스레드 추적

    def discard(self):
        self.buffer = []
        self.recording = False

    def wait_for_all(self):
        """모든 저장 스레드가 완료될 때까지 대기"""
        if self.save_threads:
            print(f"\n⏳ Waiting for {len(self.save_threads)} files to finish saving...")
            for t in self.save_threads:
                t.join()
            print("✅ All files saved successfully!")
            self.save_threads = []

# === Helper Functions ===
def smooth_step(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def trapezoid_step(t, accel_frac=0.2):
    """Trapezoidal velocity profile with cubic-smoothed accel/decel.
    accel_frac in (0, 0.5): fraction of total duration spent on accel (= decel).
    Cruise covers 1 - 2*accel_frac. Total area = 1 (position(0)=0, position(1)=1).
    Less time at slow tails near goal → less "stop-near-goal" bias in learned policy.
    """
    t = np.clip(t, 0.0, 1.0)
    ta = float(accel_frac)
    if ta <= 0 or ta >= 0.5:
        return smooth_step(t)
    v_max = 1.0 / (1.0 - ta)
    if t < ta:
        u = t / ta
        return v_max * ta * (u**3 - 0.5 * u**4)
    if t < 1 - ta:
        return v_max * ta * 0.5 + v_max * (t - ta)
    u = (1 - t) / ta
    return 1.0 - v_max * ta * (u**3 - 0.5 * u**4)

def randomize_phantom_pos(model, data, phantom_id, rot_id, base_pos=np.zeros(3),
                           assembly_id=-1, assembly_base_pos=np.zeros(3)):
    # 1. 위치 이동 (Translation) — plate 영역 내 제약
    offset_x = np.random.uniform(-0.025, 0.025)
    offset_y = np.random.uniform(-0.025, 0.075)
    offset_z = np.random.uniform(0.0, 0.05)  # optical_plate + trocar 통째로 상승

    model.body_pos[phantom_id] = base_pos + np.array([offset_x, offset_y, 0.0])
    if assembly_id >= 0:
        model.body_pos[assembly_id] = assembly_base_pos + np.array([0.0, 0.0, offset_z])

    random_angle_deg = float(np.random.uniform(-25, 25))

    new_quat = np.zeros(4)
    mujoco.mju_euler2Quat(new_quat, [0, 0, np.deg2rad(random_angle_deg)], "xyz")
    model.body_quat[rot_id] = new_quat
    print(f">>> Randomize: Pos=({offset_x:.3f}, {offset_y:.3f}, Z+{offset_z:.3f}), Angle={random_angle_deg:.1f} deg")
    mujoco.mj_forward(model, data)
    return np.array([offset_x, offset_y, offset_z], dtype=np.float32), new_quat.astype(np.float32), np.float32(random_angle_deg)

# === Args ===
NO_INSERTION = False
RANDOMIZE_PHANTOM = False
PHANTOM_POS = None  # (x, y) 고정 위치

def _parse_args():
    parser = argparse.ArgumentParser(description="Record simulation dataset.")
    parser.add_argument(
        "--randomize-phantom-pos",
        dest="randomize_phantom_pos",
        action="store_true",
        default=False,
        help="Enable phantom position randomization.",
    )
    parser.add_argument(
        "--no-randomize-phantom-pos",
        dest="randomize_phantom_pos",
        action="store_false",
        help="Disable phantom position randomization.",
    )
    parser.add_argument(
        "--no-insertion",
        action="store_true",
        default=False,
        help="Stop after alignment (no insertion phase).",
    )
    parser.add_argument(
        "--phantom-pos", type=float, nargs=2, default=None,
        metavar=("X", "Y"),
        help="Fixed phantom position (x, y). e.g. --phantom-pos 0.0 -0.2",
    )
    parser.add_argument(
        "--no-side-camera", action="store_true",
        help="Skip side_camera rendering/saving (saves storage)",
    )
    parser.add_argument(
        "--hold-steps", type=int, default=None,
        help="Number of hold frames to record after reaching target (default: 50)",
    )
    parser.add_argument(
        "--retreat-mm", type=float, default=RETREAT_MM,
        help="Retreat goal_tip from trocar entry along -axis_dir (mm, default: 10)",
    )
    parser.add_argument(
        "--cameras", type=str, nargs="+", default=None,
        help="Explicit camera list (overrides --no-side-camera)",
    )
    return parser.parse_args()

# === Main Script ===
def main():
    global NO_INSERTION, PHANTOM_POS, RANDOMIZE_PHANTOM, CAMERA_LIST, HOLD_STEPS, RETREAT_MM
    args = _parse_args()
    if args.hold_steps is not None:
        HOLD_STEPS = args.hold_steps
    RETREAT_MM = args.retreat_mm
    if args.no_insertion:
        NO_INSERTION = True
    if args.phantom_pos is not None:
        PHANTOM_POS = tuple(args.phantom_pos)
    if args.randomize_phantom_pos:
        RANDOMIZE_PHANTOM = True
    if args.cameras is not None:
        CAMERA_LIST = args.cameras
        print(f"Cameras: {CAMERA_LIST}")
    elif args.no_side_camera:
        CAMERA_LIST = [c for c in CAMERA_LIST if c != "side_camera"]
        print(f"Cameras: {CAMERA_LIST}")
    print(f"Loading Model: {MODEL_PATH}")
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=IMG_HEIGHT, width=IMG_WIDTH)
    
    try:
        tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "needle_tip")
        back_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "needle_back")
        target_entry_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "trocar_target")
        target_depth_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "trocar_depth")
        phantom_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trocar_assembly")
        rotating_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rotating_assembly")
        phantom_assembly_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "phantom_assembly")
        phantom_base_pos = model.body_pos[phantom_body_id].copy() if phantom_body_id >= 0 else np.zeros(3)
        phantom_assembly_base_pos = model.body_pos[phantom_assembly_id].copy() if phantom_assembly_id >= 0 else np.zeros(3)
        link6_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "6_Link") 
        n_motors = model.nu
        dof = model.nv
    except Exception as e:
        print(f"⚠️ Warning: Some IDs not found: {e}")
        phantom_body_id = -1

    recorder = SimRecorder(SAVE_DIR)
    current_speed = 0.5 # np.random.uniform(0.3, 0.6)

    def get_ee_pose_6d_scaled():
        """Get needle-tip world pose (x, y, z, rx, ry, rz). Tip = flange + 177.5mm Z."""
        # TCP shifted to needle_tip site. See src/utils/tip_frame.py.
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

    print(f"🚀 Starting Headless Collection...")
    pbar = tqdm(total=MAX_EPISODES, desc="Collecting", unit="ep")

    episode_count = 0
    while episode_count < MAX_EPISODES:
        mujoco.mj_resetData(model, data)
        # 초기 home pose (정렬 시작점) — 고정
        home_pose = np.array([0.75, -0.5, 0.5, 0, 0.6, 1.0])
        data.qpos[:6] = home_pose
        phantom_offset = np.zeros(3, dtype=np.float32)
        phantom_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        phantom_angle_deg = np.float32(0.0)
        if PHANTOM_POS is not None and phantom_body_id >= 0:
            px, py = PHANTOM_POS
            model.body_pos[phantom_body_id] = phantom_base_pos + np.array([px, py, 0.0])
            rand_angle = float(np.random.uniform(-25, 25))
            new_quat = np.zeros(4)
            mujoco.mju_euler2Quat(new_quat, [0, 0, np.deg2rad(rand_angle)], "xyz")
            model.body_quat[rotating_id] = new_quat
            mujoco.mj_forward(model, data)
            phantom_offset = np.array([px, py, 0.0], dtype=np.float32)
            phantom_quat = new_quat.astype(np.float32)
            phantom_angle_deg = np.float32(rand_angle)
        elif RANDOMIZE_PHANTOM:
            phantom_offset, phantom_quat, phantom_angle_deg = randomize_phantom_pos(
                model, data, phantom_body_id, rotating_id, phantom_base_pos,
                phantom_assembly_id, phantom_assembly_base_pos)
        mujoco.mj_forward(model, data)
        
        last_ee_pose = get_ee_pose_6d_scaled()
        task_state, traj_start_time, insertion_started, accumulated_depth, align_timer, traj_initialized, hold_start_time = 1, data.time, False, 0.0, 0, False, None
        hold_frame_count = 0  # hold 프레임 카운터 (NO_INSERTION 모드용)
        
        p_entry, p_depth = data.site_xpos[target_entry_id].copy(), data.site_xpos[target_depth_id].copy()
        start_tip, start_back = data.site_xpos[tip_id].copy(), data.site_xpos[back_id].copy()
        needle_len = np.linalg.norm(start_tip - start_back)
        
        recorder.start({
            "initial_qpos": np.rad2deg(data.qpos[:n_motors].copy()).astype(np.float32),
            "phantom_offset": phantom_offset,
            "phantom_quat": phantom_quat,
            "phantom_angle_deg": np.array(phantom_angle_deg, dtype=np.float32),
            "target_entry_world": p_entry.astype(np.float32),
            "target_depth_world": p_depth.astype(np.float32),
        })
        step_count, success = 0, False

        # --- Warm-up: J6 settling (IK 돌리되 녹화 안 함) ---
        for _ in range(WARMUP_STEPS):
            curr_tip_w = data.site_xpos[tip_id].copy()
            curr_back_w = data.site_xpos[back_id].copy()
            # IK: 현재 위치 유지 (target = current)
            target_tip_pos_w = curr_tip_w.copy()
            target_back_pos_w = curr_back_w.copy()

            err_tip_w = target_tip_pos_w - curr_tip_w
            err_back_w = target_back_pos_w - curr_back_w
            tip_rot_mat_w = data.site_xmat[tip_id].reshape(3, 3)
            offset_angle_w = np.deg2rad(180 + 30)
            offset_local_vec_w = np.array([np.cos(offset_angle_w), np.sin(offset_angle_w), 0])
            current_side_vec_w = tip_rot_mat_w @ offset_local_vec_w
            needle_axis_w = (curr_tip_w - curr_back_w) / (np.linalg.norm(curr_tip_w - curr_back_w) + 1e-10)
            target_side_vec_w = np.cross(needle_axis_w, np.array([0, 0, 1]))
            target_side_vec_w = target_side_vec_w / np.linalg.norm(target_side_vec_w) if np.linalg.norm(target_side_vec_w) > 1e-3 else np.array([1, 0, 0])
            err_roll_w = np.cross(current_side_vec_w, target_side_vec_w)

            jac_tip_w = np.zeros((6, dof))
            jac_back_w = np.zeros((3, dof))
            mujoco.mj_jacSite(model, data, jac_tip_w[:3], jac_tip_w[3:], tip_id)
            mujoco.mj_jacSite(model, data, jac_back_w, None, back_id)

            J1 = jac_tip_w[:3, :n_motors]
            e1 = err_tip_w * 50.0
            if np.linalg.norm(e1) > 1.0:
                e1 = e1 / np.linalg.norm(e1) * 1.0
            J1_pinv = np.linalg.pinv(J1, rcond=1e-4)
            dq1 = J1_pinv @ e1
            P1 = np.eye(n_motors) - (J1_pinv @ J1)
            J2_proj = jac_back_w[:, :n_motors] @ P1
            dq2 = np.linalg.pinv(J2_proj, rcond=1e-4) @ ((err_back_w * 50.0) - jac_back_w[:, :n_motors] @ dq1)
            P2 = P1 - (np.linalg.pinv(J2_proj, rcond=1e-4) @ J2_proj)
            J3_proj = jac_tip_w[3:, :n_motors] @ P2
            dq3 = np.linalg.pinv(J3_proj, rcond=1e-4) @ ((err_roll_w * 10.0) - jac_tip_w[3:, :n_motors] @ (dq1 + dq2))

            data.ctrl[:n_motors] = data.qpos[:n_motors] + (dq1 + dq2 + dq3) * current_speed
            mujoco.mj_step(model, data)

        # Warm-up 후 상태 재초기화
        last_ee_pose = get_ee_pose_6d_scaled()
        traj_start_time = data.time

        while True:
            t_curr = data.time
            curr_tip, curr_back = data.site_xpos[tip_id].copy(), data.site_xpos[back_id].copy()
            
            # --- 1. Expert Trajectory Logic (2-State System) ---
            if task_state == 1:  # State 1: Align (정렬)
                if not traj_initialized:
                    traj_start_time, start_tip_pos, start_back_pos, traj_initialized = t_curr, curr_tip.copy(), curr_back.copy(), True
                    # 거리 기반 동적 duration 계산: duration = distance / speed
                    axis_dir_init = (p_depth - p_entry) / (np.linalg.norm(p_depth - p_entry) + 1e-10)
                    retreat_m = RETREAT_MM / 1000.0
                    goal_tip_init = p_entry - (axis_dir_init * retreat_m)
                    align_distance = np.linalg.norm(goal_tip_init - start_tip_pos)
                    dynamic_duration = align_distance / ALIGN_SPEED
                progress = trapezoid_step((t_curr - traj_start_time) / dynamic_duration, APPROACH_ACCEL_FRAC) if dynamic_duration > 0 else 1.0
                axis_dir = (p_depth - p_entry) / (np.linalg.norm(p_depth - p_entry) + 1e-10)
                retreat_m = RETREAT_MM / 1000.0
                goal_tip, goal_back = p_entry - (axis_dir * retreat_m), p_entry - (axis_dir * (retreat_m + needle_len))
                target_tip_pos, target_back_pos = (1 - progress) * start_tip_pos + progress * goal_tip, (1 - progress) * start_back_pos + progress * goal_back
                if progress >= 1.0:
                    if np.linalg.norm(curr_tip - goal_tip) < 0.002: align_timer += 1
                    else: align_timer = 0
                    if align_timer > 50:
                        if NO_INSERTION:
                            task_state = 3  # Hold 상태로 전환
                        else:
                            task_state, insertion_started = 2, False
            elif task_state == 3:  # State 3: Hold (NO_INSERTION 모드 — 도달 후 위치 유지)
                # goal 위치 고정, IK가 현재 위치 유지 → action ≈ 0 프레임 기록
                axis_dir = (p_depth - p_entry) / (np.linalg.norm(p_depth - p_entry) + 1e-10)
                retreat_m = RETREAT_MM / 1000.0
                target_tip_pos = p_entry - (axis_dir * retreat_m)
                target_back_pos = target_tip_pos - (axis_dir * needle_len)
            elif task_state == 2:  # State 2: Insert + Hold (삽입 + 대기 통합)
                if not insertion_started:
                    phase3_base_tip, insertion_started, accumulated_depth, hold_start_time = curr_tip.copy(), True, 0.0, None

                axis_dir = (p_depth - p_entry) / (np.linalg.norm(p_depth - p_entry) + 1e-10)

                # 삽입이 완료되지 않은 경우: 계속 삽입
                if accumulated_depth < TARGET_INSERTION_DEPTH:
                    accumulated_depth += INSERTION_SPEED * model.opt.timestep
                    target_tip_pos = phase3_base_tip + (axis_dir * accumulated_depth)
                    target_back_pos = target_tip_pos - (axis_dir * needle_len)
                    # 목표 깊이 도달 시 대기 타이머 시작
                    if accumulated_depth >= TARGET_INSERTION_DEPTH:
                        hold_start_time = data.time
                # 삽입이 완료된 경우: 위치 고정 및 대기
                else:
                    if hold_start_time is None:
                        hold_start_time = data.time
                    # 목표를 최종 삽입 깊이로 고정
                    target_tip_pos = phase3_base_tip + (axis_dir * TARGET_INSERTION_DEPTH)
                    target_back_pos = target_tip_pos - (axis_dir * needle_len)
                    # 1초 대기 후 성공
                    if data.time - hold_start_time >= 1.0: success = True; break

            # --- 2. IK Solver ---
            err_tip, err_back = target_tip_pos - curr_tip, target_back_pos - curr_back
            tip_rot_mat = data.site_xmat[tip_id].reshape(3, 3)
            
            # 210도 오프셋
            offset_angle = np.deg2rad(180+30)
            offset_local_vec = np.array([np.cos(offset_angle), np.sin(offset_angle), 0])
            current_side_vec = tip_rot_mat @ offset_local_vec
            
            # current_side_vec = tip_rot_mat @ np.array([1, 0, 0])
            
            needle_axis_curr = (curr_tip - curr_back) / (np.linalg.norm(curr_tip - curr_back) + 1e-10)
            target_side_vec = np.cross(needle_axis_curr, np.array([0, 0, 1]))
            target_side_vec = target_side_vec / np.linalg.norm(target_side_vec) if np.linalg.norm(target_side_vec) > 1e-3 else np.array([1, 0, 0])
            err_roll = np.cross(current_side_vec, target_side_vec)
            
            jac_tip_full, jac_back = np.zeros((6, dof)), np.zeros((3, dof))
            mujoco.mj_jacSite(model, data, jac_tip_full[:3], jac_tip_full[3:], tip_id)
            mujoco.mj_jacSite(model, data, jac_back, None, back_id)
            
            J_p1, e_p1 = jac_tip_full[:3, :n_motors], (err_tip * 50.0)
            if np.linalg.norm(e_p1) > 1.0: e_p1 = e_p1 / np.linalg.norm(e_p1) * 1.0
            J_p1_pinv = np.linalg.pinv(J_p1, rcond=1e-4)
            dq_p1 = J_p1_pinv @ e_p1
            P_null_1 = np.eye(n_motors) - (J_p1_pinv @ J_p1)
            J_p2_proj = jac_back[:, :n_motors] @ P_null_1
            dq_p2 = np.linalg.pinv(J_p2_proj, rcond=1e-4) @ ((err_back * 50.0) - jac_back[:, :n_motors] @ dq_p1)
            P_null_2 = P_null_1 - (np.linalg.pinv(J_p2_proj, rcond=1e-4) @ J_p2_proj)
            J_p3_proj = jac_tip_full[3:, :n_motors] @ P_null_2
            dq_p3 = np.linalg.pinv(J_p3_proj, rcond=1e-4) @ ((err_roll * 10.0) - jac_tip_full[3:, :n_motors] @ (dq_p1 + dq_p2))
            
            data.ctrl[:n_motors] = data.qpos[:n_motors] + (dq_p1 + dq_p2 + dq_p3) * current_speed
            
            # --- 3. Sensor & Step ---
            p_sensor = data.site_xpos[tip_id].copy()
            needle_dir = (curr_tip - curr_back) / (np.linalg.norm(curr_tip - curr_back) + 1e-10)
            dist_to_surface = mujoco.mj_ray(model, data, p_sensor, needle_dir, None, 1, link6_id, np.zeros(1, dtype=np.int32))
            current_sensor_dist = dist_to_surface * 1000.0 if dist_to_surface >= 0 else -1.0

            mujoco.mj_step(model, data)
            step_count += 1
            
            # --- 4. Save ---
            if step_count % 67 == 0:
                current_qpos_deg = np.rad2deg(data.qpos[:n_motors].copy())
                current_ee_pose_mm = get_ee_pose_6d_scaled()
                delta_ee_action = current_ee_pose_mm - last_ee_pose
                delta_ee_action[3:6] = (delta_ee_action[3:6] + np.pi) % (2 * np.pi) - np.pi

                # Phase 전환 시 IK spike 방지: position delta 클리핑
                pos_mag = np.linalg.norm(delta_ee_action[:3])
                if pos_mag > ACTION_CLIP_MM:
                    delta_ee_action[:3] *= ACTION_CLIP_MM / pos_mag

                frames = {}
                for cam_name in CAMERA_LIST:
                    renderer.update_scene(data, camera=cam_name)
                    frames[cam_name] = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)

                # Spatial auxiliary data: 3D positions (mm) and 2D keypoints (wrist camera only)
                needle_tip_mm = data.site_xpos[tip_id].copy() * 1000
                trocar_entry_mm = data.site_xpos[target_entry_id].copy() * 1000

                tip_uv_wrist = project_to_2d(data.site_xpos[tip_id], model, data, "tool_camera", IMG_WIDTH, IMG_HEIGHT)
                trocar_uv_wrist = project_to_2d(data.site_xpos[target_entry_id], model, data, "tool_camera", IMG_WIDTH, IMG_HEIGHT)
                keypoints_wrist = np.concatenate([tip_uv_wrist, trocar_uv_wrist]).astype(np.float32)  # (4,)

                # Visibility: 1 if keypoint is within [0,1] image bounds, 0 otherwise
                tip_visible = float(0.0 <= tip_uv_wrist[0] <= 1.0 and 0.0 <= tip_uv_wrist[1] <= 1.0)
                trocar_visible = float(0.0 <= trocar_uv_wrist[0] <= 1.0 and 0.0 <= trocar_uv_wrist[1] <= 1.0)
                keypoints_visibility = np.array([tip_visible, trocar_visible], dtype=np.float32)  # (2,)

                recorder.add(frames, current_qpos_deg, current_ee_pose_mm, delta_ee_action, data.time, task_state, current_sensor_dist,
                             needle_tip_mm=needle_tip_mm, trocar_entry_mm=trocar_entry_mm,
                             keypoints_wrist=keypoints_wrist, keypoints_visibility=keypoints_visibility,
                             instruction=TASK_INSTRUCTION)
                last_ee_pose = current_ee_pose_mm.copy()

                # Hold 프레임 카운터: task_state==3이면 hold 프레임 기록 중
                if task_state == 3:
                    hold_frame_count += 1
                    if hold_frame_count >= HOLD_STEPS:
                        success = True; break

            if data.time - traj_start_time > 50.0: break
            if len(recorder.buffer) >= MAX_CTRL_STEPS: break

        if success:
            recorder.save_async()
            episode_count += 1
            pbar.update(1)
        else:
            # 왜 실패했는지 출력
            if len(recorder.buffer) >= MAX_CTRL_STEPS:
                reason = f"MaxSteps({MAX_CTRL_STEPS})"
            elif data.time - traj_start_time > 50.0:
                reason = "Timeout"
            elif task_state == 1:
                reason = "Failed to Align"
            elif task_state == 2:
                reason = "Failed to Insert/Hold"
            else:
                reason = "Unknown"
            print(f"  ⚠️ Episode {episode_count} discarded. Reason: {reason}")
            recorder.discard()

    pbar.close()
    recorder.wait_for_all()  # 모든 저장 완료 대기
    print("\n✅ All Collections Finished!")

if __name__ == "__main__":
    main()
