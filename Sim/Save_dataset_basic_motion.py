"""
Basic Motion Pre-training Dataset Collection.

1방향 1에피소드: 랜덤 시작 자세 → EE-relative 방향으로 N스텝 이동 → 저장.
팬텀/트로카 없이 순수 공간 이동만 녹화.

Usage:
    # 한 방향만
    python Sim/Save_dataset_basic_motion.py --direction left --num-episodes 100 \
        --save-dir dataset/basic_motion --no-side-camera

    # 전체 10방향
    python Sim/Save_dataset_basic_motion.py --direction all --num-episodes 100 \
        --save-dir dataset/basic_motion --no-side-camera
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
from math import sqrt

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, total=None: x


# ============================================================
# === Configuration ===
# ============================================================

MODEL_PATH = "meca_add.xml"
SAVE_DIR = "dataset/basic_motion"
MAX_EPISODES = 100
IMG_WIDTH = 640
IMG_HEIGHT = 480
CAMERA_LIST = ["side_camera", "tool_camera", "top_camera"]

FINE_ALIGN_SPEED = 0.005    # m/s — basic motion speed
STEPS_PER_EPISODE = 50      # recorded control steps per episode
ACTION_CLIP_MM = 1.0        # delta position clipping (mm)

# --- 10 directions in WORLD frame (fixed vectors) ---
# World: Z=up, gravity=(0,0,-9.81). X/Y assignment TBD after visual check.
DIRECTIONS = {
    "left":       np.array([ 0.0,  1.0,  0.0]),
    "right":      np.array([ 0.0, -1.0,  0.0]),
    "up":         np.array([ 0.0,  0.0,  1.0]),
    "down":       np.array([ 0.0,  0.0, -1.0]),
    "left_up":    np.array([ 0.0,  1.0,  1.0]) / sqrt(2),
    "left_down":  np.array([ 0.0,  1.0, -1.0]) / sqrt(2),
    "right_up":   np.array([ 0.0, -1.0,  1.0]) / sqrt(2),
    "right_down": np.array([ 0.0, -1.0, -1.0]) / sqrt(2),
    "forward":    np.array([ 1.0,  0.0,  0.0]),
    "backward":   np.array([-1.0,  0.0,  0.0]),
}

# Direction name → language instruction
DIRECTION_INSTRUCTIONS = {
    "left": "Left",
    "right": "Right",
    "up": "Up",
    "down": "Down",
    "left_up": "Left Up",
    "left_down": "Left Down",
    "right_up": "Right Up",
    "right_down": "Right Down",
    "forward": "Forward",
    "backward": "Backward",
}

# --- Random start joint ranges (safe, within joint limits) ---
JOINT_RANGES = [
    (-0.5, 0.5),    # J1 (base rotation)
    (-0.3, 0.3),    # J2 (shoulder pitch)
    (-0.5, 0.2),    # J3 (elbow pitch)
    (-0.3, 0.3),    # J4 (roll)
    (0.4, 1.0),     # J5 (wrist pitch, biased more downward)
    (-1.0, 1.0),    # J6
]

# Overridable by run_parallel.py
RANDOM_SEED = None
DIRECTION = None  # set by CLI or run_parallel

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
            instruction=""):
        if not self.recording:
            return
        self.buffer.append({
            "ts": timestamp,
            "imgs": frames,
            "q": qpos,
            "p": ee_pose,
            "act": action,
            "phase": phase,
            "sd": sensor_dist,
            "instruction": instruction,
        })

    def save_async(self):
        if not self.buffer:
            return
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

                    # Gripper always open
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

                    # Language instruction (direction name)
                    instruction = data[0].get("instruction", "")
                    f.create_dataset("language_instruction", data=instruction.encode("utf-8"))

                    # Metadata
                    for key, value in metadata.items():
                        if isinstance(value, str):
                            meta_grp.create_dataset(key, data=value.encode("utf-8"))
                        else:
                            value_arr = np.asarray(value)
                            if value_arr.shape == ():
                                meta_grp.create_dataset(key, data=value_arr)
                            else:
                                meta_grp.create_dataset(key, data=value_arr, compression="gzip")

                    # Images (JPEG encoded)
                    first_imgs = data[0]["imgs"]
                    for cam_name in first_imgs.keys():
                        jpeg_list = []
                        for step in data:
                            img = step["imgs"][cam_name]
                            success, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                            if success:
                                jpeg_list.append(buf.flatten())
                            else:
                                jpeg_list.append(np.zeros(1, dtype=np.uint8))

                        dt = h5py.special_dtype(vlen=np.dtype('uint8'))
                        dset = img_grp.create_dataset(cam_name, (len(jpeg_list),), dtype=dt)
                        for i, code in enumerate(jpeg_list):
                            dset[i] = code

            except Exception as e:
                print(f"Save Failed: {e}")
            finally:
                self.is_saving = False

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
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


def get_ee_direction_vectors(data, link6_id):
    """Get EE local frame axes in world coordinates."""
    mat = data.xmat[link6_id].reshape(3, 3)
    return mat[:, 0].copy(), mat[:, 1].copy(), mat[:, 2].copy()


def main():
    if RANDOM_SEED is not None:
        np.random.seed(RANDOM_SEED)
        print(f"Random seed: {RANDOM_SEED}")

    direction_name = DIRECTION
    if direction_name is None:
        print("Error: --direction is required")
        return

    if direction_name not in DIRECTIONS:
        print(f"Error: Unknown direction '{direction_name}'. Choose from: {list(DIRECTIONS.keys())}")
        return

    direction_vec = DIRECTIONS[direction_name]
    instruction = DIRECTION_INSTRUCTIONS[direction_name]

    print(f"Loading Model: {MODEL_PATH}")
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=IMG_HEIGHT, width=IMG_WIDTH)

    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "needle_tip")
    back_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "needle_back")
    link6_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "6_Link")
    n_motors = model.nu
    dof = model.nv
    ik_speed = 0.5

    # Save to direction-specific subdirectory
    save_path = pathlib.Path(SAVE_DIR) / direction_name
    recorder = SimRecorder(str(save_path))

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
        """IK solver 1 step."""
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

    print(f"Starting Basic Motion Collection: direction={direction_name}, episodes={MAX_EPISODES}")
    pbar = tqdm(total=MAX_EPISODES, desc=f"Collecting [{direction_name}]", unit="ep")

    episode_count = 0
    discard_count = 0

    while episode_count < MAX_EPISODES:
        # --- Reset with random joint angles ---
        mujoco.mj_resetData(model, data)
        # 초기 home pose (정렬 시작점)
        home_pose = np.array([
            np.random.uniform(-0.5, 0.5),    # J1 (base rotation)
            np.random.uniform(-0.6, -0.4),    # J2 (shoulder pitch)
            np.random.uniform(0.75, 0.25),    # J3 (elbow pitch)
            np.random.uniform(-0.3, 0.3),    # J4 (roll)
            np.random.uniform(0.4, 0.6),     # J5 (wrist pitch)
            np.random.uniform(0.9, 1.1),    # J6
        ])
        data.qpos[:6] = home_pose
        mujoco.mj_forward(model, data)

        # Stabilize for a few sim steps
        for _ in range(200):
            data.ctrl[:n_motors] = data.qpos[:n_motors]
            mujoco.mj_step(model, data)

        # Warm-up: run direction movement without recording to skip initial transient
        warmup_tip = data.site_xpos[tip_id].copy()
        warmup_back = data.site_xpos[back_id].copy()
        for ws in range(500):
            disp = direction_vec * FINE_ALIGN_SPEED * model.opt.timestep * (ws + 1)
            run_ik_step(warmup_tip + disp, warmup_back + disp)
            mujoco.mj_step(model, data)

        # Get initial EE state
        last_ee_pose = get_ee_pose_6d_scaled()
        curr_tip = data.site_xpos[tip_id].copy()
        curr_back = data.site_xpos[back_id].copy()
        needle_len = np.linalg.norm(curr_tip - curr_back)

        episode_meta = {
            "initial_qpos": np.rad2deg(home_pose).astype(np.float32),
            "direction": direction_name,
        }
        recorder.start(episode_meta)

        step_count = 0
        ctrl_step_count = 0
        episode_ok = True
        fail_reason = ""

        # Check initial state validity
        if np.any(np.isnan(data.qpos[:n_motors])) or np.any(np.isnan(data.site_xpos[tip_id])):
            episode_ok = False
            fail_reason = "nan_init"

        prev_tip = data.site_xpos[tip_id].copy() if episode_ok else None

        # Accumulate target from initial position (prevents gravity drift)
        init_tip = data.site_xpos[tip_id].copy()
        init_back = data.site_xpos[back_id].copy()
        elapsed_sim_steps = 0

        for ctrl_idx in range(STEPS_PER_EPISODE):
            if not episode_ok:
                break

            # Fixed world-frame direction (no EE-relative recomputation needed)
            direction_world = direction_vec

            # Run 67 sim steps (one control step = one recorded frame)
            for sim_step in range(67):
                elapsed_sim_steps += 1

                # NaN check inside sim loop — break early on IK divergence
                curr_tip_now = data.site_xpos[tip_id].copy()
                curr_back_now = data.site_xpos[back_id].copy()
                if np.any(np.isnan(curr_tip_now)) or np.any(np.isnan(data.qpos[:n_motors])):
                    episode_ok = False
                    fail_reason = "nan"
                    break

                # Target = initial position + accumulated displacement
                displacement = direction_world * FINE_ALIGN_SPEED * model.opt.timestep * elapsed_sim_steps
                target_tip = init_tip + displacement
                target_back = init_back + displacement

                run_ik_step(target_tip, target_back)

                # Sensor (ray cast from needle tip)
                needle_dir = (curr_tip_now - curr_back_now) / (np.linalg.norm(curr_tip_now - curr_back_now) + 1e-10)
                dist_to_surface = mujoco.mj_ray(model, data, curr_tip_now, needle_dir, None, 1, link6_id, np.zeros(1, dtype=np.int32))
                current_sensor_dist = dist_to_surface * 1000.0 if dist_to_surface >= 0 else -1.0

                mujoco.mj_step(model, data)
                step_count += 1

            if not episode_ok:
                break

            # Check joint limits (any joint near limit → discard)
            qpos_now = data.qpos[:n_motors].copy()
            for j in range(min(n_motors, model.njnt)):
                jnt_range = model.jnt_range[j]
                if jnt_range[1] - jnt_range[0] < 1e-6:
                    continue
                margin = (jnt_range[1] - jnt_range[0]) * 0.02
                if qpos_now[j] <= jnt_range[0] + margin or qpos_now[j] >= jnt_range[1] - margin:
                    episode_ok = False
                    fail_reason = f"joint_limit_j{j}"
                    break
            if not episode_ok:
                break

            # Check for large velocity (IK divergence)
            qvel_mag = np.linalg.norm(data.qvel[:n_motors])
            if qvel_mag > 50.0:
                episode_ok = False
                fail_reason = "ik_diverge_vel"
                break

            # Check tip didn't jump too far (IK divergence)
            curr_tip_check = data.site_xpos[tip_id].copy()
            tip_jump = np.linalg.norm(curr_tip_check - prev_tip) * 1000  # mm
            if tip_jump > 10.0:
                episode_ok = False
                fail_reason = "ik_diverge_jump"
                break
            prev_tip = curr_tip_check

            # Record frame
            current_qpos_deg = np.rad2deg(data.qpos[:n_motors].copy())
            current_ee_pose_mm = get_ee_pose_6d_scaled()
            delta_ee_action = current_ee_pose_mm - last_ee_pose
            delta_ee_action[3:6] = (delta_ee_action[3:6] + np.pi) % (2 * np.pi) - np.pi

            # Clip position delta
            pos_mag = np.linalg.norm(delta_ee_action[:3])
            if pos_mag > ACTION_CLIP_MM:
                delta_ee_action[:3] *= ACTION_CLIP_MM / pos_mag

            frames = {}
            for cam_name in CAMERA_LIST:
                renderer.update_scene(data, camera=cam_name)
                frames[cam_name] = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)

            recorder.add(
                frames, current_qpos_deg, current_ee_pose_mm, delta_ee_action,
                data.time, 0,  # phase=0 (basic motion)
                current_sensor_dist,
                instruction=instruction,
            )
            last_ee_pose = current_ee_pose_mm.copy()
            ctrl_step_count += 1

        # Save or discard
        if episode_ok and ctrl_step_count >= STEPS_PER_EPISODE:
            recorder.save_async()
            episode_count += 1
            pbar.update(1)
        else:
            if not fail_reason:
                fail_reason = f"ShortEpisode({ctrl_step_count})"
            recorder.discard()
            discard_count += 1
            if discard_count % 10 == 0:
                print(f"  Discarded {discard_count} episodes so far (last: {fail_reason})")

    pbar.close()
    recorder.wait_for_all()
    print(f"\nCollection finished! {episode_count} episodes saved to {save_path}")
    if discard_count > 0:
        print(f"  ({discard_count} episodes discarded)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basic motion pre-training dataset collection")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help="Base output directory")
    parser.add_argument("--num-episodes", type=int, default=MAX_EPISODES,
                        help="Number of episodes per direction")
    parser.add_argument("--direction", type=str, required=True,
                        choices=list(DIRECTIONS.keys()) + ["all"],
                        help="Direction to collect (or 'all' for all 10)")
    parser.add_argument("--steps-per-episode", type=int, default=STEPS_PER_EPISODE,
                        help="Recorded control steps per episode (default: 50)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    parser.add_argument("--no-side-camera", action="store_true",
                        help="Skip side_camera rendering/saving")
    args = parser.parse_args()

    SAVE_DIR = args.save_dir
    MAX_EPISODES = args.num_episodes
    STEPS_PER_EPISODE = args.steps_per_episode

    if args.seed is not None:
        RANDOM_SEED = args.seed

    if args.no_side_camera:
        CAMERA_LIST = [c for c in CAMERA_LIST if c != "side_camera"]
        print(f"Cameras: {CAMERA_LIST}")

    if args.direction == "all":
        # Run all 10 directions sequentially
        for dir_name in DIRECTIONS.keys():
            print(f"\n{'='*60}")
            print(f"  Direction: {dir_name}")
            print(f"{'='*60}")
            DIRECTION = dir_name
            if args.seed is not None:
                # Different seed per direction for variety
                RANDOM_SEED = args.seed + list(DIRECTIONS.keys()).index(dir_name) * 1000
            main()
    else:
        DIRECTION = args.direction
        main()
