#!/usr/bin/env python
"""
Digital Twin: Real-to-Sim Synchronization
Streams Real Mecademic Robot State → MuJoCo Simulation

Architecture:
- Real robot state sampler in separate thread
- Simulation mirror runs in main thread (MuJoCo viewer)
- State synchronization via thread-safe queue
- Gamepad control of real robot → visualization in simulation

python /home/irom/NAS/VLA/Insertion_VLA_Sim/digital_twin/real_to_sim.py
"""

import time
import threading
import logging
import numpy as np
from termcolor import colored
import cv2
import contextlib
import pathlib
from collections import deque

import mujoco
import mujoco.viewer
import mecademicpy.robot as mdr
import depthai as dai
import pygame
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
try:
    from scipy.spatial.transform import Rotation as R
except Exception:
    R = None

# === Configuration ===
ROBOT_ADDRESS = "192.168.0.100"
MODEL_PATH = "../Sim/meca_add.xml"
HOME_JOINTS = (0, 0, 0, 0, 0, 0) # (30, -20, 20, 0, 30, 60)
CONTROL_FREQUENCY = 15  # Hz
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent / "recordings" / "real_to_sim"
PLOT_HISTORY = 300
PLOT_UPDATE_HZ = 10  # Hz

# Gamepad settings (same as Robot_action.py)
SCALE_POS = 0.8
SCALE_HAT = 0.3
SCALE_Z = 0.5
SCALE_ROT = 0.3
DEADZONE = 0.2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# Real Robot State Sampler (from Robot_action.py)
# ============================================================
class RtSampler(threading.Thread):
    def __init__(self, robot, rate_hz=100):
        super().__init__(daemon=True)
        self.robot = robot
        self.dt = 1.0 / float(rate_hz)
        self.stop_evt = threading.Event()
        self.lock = threading.Lock()
        self.latest_q = np.zeros(6)
        self.latest_p = np.zeros(6)

    def stop(self):
        self.stop_evt.set()

    def get_latest_data(self):
        with self.lock:
            return self.latest_q.copy(), self.latest_p.copy()

    def run(self):
        logger.info("🤖 Starting robot state sampler...")
        
        # Set configuration for MovePose (Shoulder, Elbow, Wrist)
        self.robot.SetConf(1, 1, 1)
        
        # Initial read
        initial_success = False
        for _ in range(10):
            try:
                q = list(self.robot.GetJoints())
                p = list(self.robot.GetPose())
                if q and len(q) >= 6 and p and len(p) >= 6:
                    with self.lock:
                        self.latest_q = np.array(q[:6])
                        self.latest_p = np.array(p[:6])
                    logger.info(f"✅ Initial robot state acquired")
                    initial_success = True
                    break
            except Exception as e:
                logger.debug(f"Initial read failed: {e}")
            time.sleep(0.1)

        if not initial_success:
            logger.warning("⚠️ Failed to get initial robot state!")

        next_t = time.time()
        while not self.stop_evt.is_set():
            try:
                q = list(self.robot.GetJoints())
                p = list(self.robot.GetPose())

                if q and len(q) >= 6 and p and len(p) >= 6:
                    with self.lock:
                        self.latest_q = np.array(q[:6])
                        self.latest_p = np.array(p[:6])
            except Exception as e:
                pass

            next_t += self.dt
            sleep_time = next_t - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)

# ============================================================
# Gamepad Controller (from Robot_action.py)
# ============================================================
class GamepadController:
    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        self.joystick = None
        self.control_mode = 1
        self.mode_switch_cooldown = 0
        self.smoothing_enabled = False
        self.smoothing_cooldown = 0
        self.current_action = np.zeros(6)
        self.acceleration_rate = 0.2
        self.deceleration_rate = 0.7

        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            logger.info(f"🎮 Gamepad connected: {self.joystick.get_name()}")
        else:
            logger.warning("⚠️ No gamepad found (robot will stay in home position)")

    def get_action(self):
        if not self.joystick:
            return np.zeros(6), False, False, False, False

        pygame.event.pump()

        # Mode switching
        btn_mode_switch = self.joystick.get_button(6) if self.joystick.get_numbuttons() > 6 else False
        if btn_mode_switch and self.mode_switch_cooldown == 0:
            self.control_mode = (self.control_mode % 3) + 1
            logger.info(colored(f"🔧 Control Mode: {self.control_mode}", "yellow"))
            self.mode_switch_cooldown = 10
        if self.mode_switch_cooldown > 0:
            self.mode_switch_cooldown -= 1

        # Smoothing toggle
        btn_smoothing = self.joystick.get_button(2) if self.joystick.get_numbuttons() > 2 else False
        if btn_smoothing and self.smoothing_cooldown == 0:
            self.smoothing_enabled = not self.smoothing_enabled
            status = "ON" if self.smoothing_enabled else "OFF"
            logger.info(colored(f"🌊 Smoothing: {status}", "cyan"))
            self.smoothing_cooldown = 10
        if self.smoothing_cooldown > 0:
            self.smoothing_cooldown -= 1

        # Analog sticks
        y_stick_raw = self.joystick.get_axis(1)
        x_stick_raw = -self.joystick.get_axis(0)
        rs_x_raw = self.joystick.get_axis(3)
        rs_y_raw = -self.joystick.get_axis(4)

        # Deadzone
        y_stick = y_stick_raw if abs(y_stick_raw) > DEADZONE else 0.0
        x_stick = x_stick_raw if abs(x_stick_raw) > DEADZONE else 0.0
        rs_x = rs_x_raw if abs(rs_x_raw) > DEADZONE else 0.0
        rs_y = rs_y_raw if abs(rs_y_raw) > DEADZONE else 0.0

        y_stick *= SCALE_POS
        x_stick *= SCALE_POS

        # Triggers and bumpers
        lt = (self.joystick.get_axis(2) + 1) / 2
        rt = (self.joystick.get_axis(5) + 1) / 2
        lb = self.joystick.get_button(4)
        rb = self.joystick.get_button(5)

        # D-Pad
        hat_x, hat_y = self.joystick.get_hat(0)
        y_hat = -hat_y * SCALE_HAT
        x_hat = -hat_x * SCALE_HAT

        # Combine movement
        y = y_stick + y_hat
        x = x_stick + x_hat

        # Tool frame compensation (60° rotation)
        angle = np.radians(60)
        x_rotated = x * np.cos(angle) - y * np.sin(angle)
        y_rotated = x * np.sin(angle) + y * np.cos(angle)
        x, y = x_rotated, y_rotated

        # Rotation mapping based on control mode
        if self.control_mode == 1:
            rx = rs_y * SCALE_ROT
            ry = rs_x * SCALE_ROT
            rz = (rb - lb) * SCALE_ROT * 1.5
            z = (rt - lt) * SCALE_Z
        elif self.control_mode == 2:
            rx = rs_y * SCALE_ROT
            rz = rs_x * SCALE_ROT * 1.5
            ry = (rb - lb) * SCALE_ROT
            z = (rt - lt) * SCALE_Z
        else:  # Mode 3
            rx = rs_y * SCALE_ROT
            ry = (rb - lb) * SCALE_ROT
            rz = (rt - lt) * SCALE_ROT * 2.0
            z = 0

        # Rotation compensation
        angle = np.radians(60)
        rx_rotated = rx * np.cos(angle) - ry * np.sin(angle)
        ry_rotated = rx * np.sin(angle) + ry * np.cos(angle)
        rx, ry = rx_rotated, ry_rotated

        target_action = np.array([y, x, z, rx, ry, rz])

        # Smoothing
        if self.smoothing_enabled:
            if np.linalg.norm(target_action) < 0.01:
                self.current_action += (target_action - self.current_action) * self.deceleration_rate
            else:
                self.current_action += (target_action - self.current_action) * self.acceleration_rate
            action = np.where(np.abs(self.current_action) < 0.001, 0, self.current_action)
        else:
            action = target_action
            self.current_action = target_action

        # Buttons
        btn_rec = self.joystick.get_button(1)  # B
        btn_disc = self.joystick.get_button(0) # A
        btn_home = self.joystick.get_button(3)  # Y
        btn_exit = self.joystick.get_button(7) if self.joystick.get_numbuttons() > 7 else False  # START

        return action, btn_rec, btn_disc, btn_home, btn_exit

# ============================================================
# Camera Manager (from Robot_action.py)
# ============================================================
class OAKCameraManager:
    def __init__(self, width=640, height=480):
        self.width, self.height = width, height
        self.stack = contextlib.ExitStack()
        self.queues = []
        self.last_frames = {}
        self.camera_ids = []
        self.num_cameras = 0

    def initialize_cameras(self):
        infos = dai.Device.getAllAvailableDevices()
        if not infos:
            logger.warning("⚠️ No OAK cameras found")
            self.num_cameras = 0
            return 0

        self.camera_ids = []
        for info in infos:
            p = dai.Pipeline()
            c = p.create(dai.node.ColorCamera)
            c.setBoardSocket(dai.CameraBoardSocket.CAM_A)
            c.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            c.setPreviewSize(self.width, self.height)
            c.setInterleaved(False)

            # Manual focus for camera ID starting with "19"
            camera_id = info.getMxId()
            self.camera_ids.append(camera_id)
            if camera_id.startswith("19"):
                c.initialControl.setManualFocus(101)
                logger.info(f"📷 Camera {camera_id}: Manual focus set to 101")
            else:
                logger.info(f"📷 Camera {camera_id}: Auto focus")

            out = p.create(dai.node.XLinkOut)
            out.setStreamName("rgb")
            c.preview.link(out.input)
            d = self.stack.enter_context(dai.Device(p, info, dai.UsbSpeed.SUPER))
            self.queues.append(d.getOutputQueue("rgb", 4, False))

        self.num_cameras = len(self.queues)
        return self.num_cameras

    def get_frames(self):
        frames = {}
        for i, q in enumerate(self.queues):
            f = q.tryGet()
            if f:
                self.last_frames[i] = f.getCvFrame()
            if i in self.last_frames:
                frames[f"camera{i+1}"] = self.last_frames[i]
        return frames

    def close(self):
        self.stack.close()

# ============================================================
# Live EE Plotter
# ============================================================
class LiveEEPlotter:
    def __init__(self, history_len=PLOT_HISTORY):
        self.enabled = plt is not None
        if not self.enabled:
            logger.warning("⚠️ matplotlib not available: EE plot disabled")
            return

        self.history_len = history_len
        self.t = deque(maxlen=history_len)
        self.real = [deque(maxlen=history_len) for _ in range(6)]
        self.sim = [deque(maxlen=history_len) for _ in range(6)]

        plt.ion()
        self.fig, self.axes = plt.subplots(6, 1, figsize=(6, 10), sharex=True)
        self.fig.canvas.manager.set_window_title("EE Position (Real vs Sim)")

        labels = ["x (mm)", "y (mm)", "z (mm)", "rx (rad)", "ry (rad)", "rz (rad)"]
        self.lines_real = []
        self.lines_sim = []
        for i, ax in enumerate(self.axes):
            lr, = ax.plot([], [], color="tab:blue", label="real")
            ls, = ax.plot([], [], color="tab:orange", label="sim")
            ax.set_ylabel(labels[i])
            ax.grid(True, alpha=0.3)
            self.lines_real.append(lr)
            self.lines_sim.append(ls)
        self.axes[-1].set_xlabel("samples")
        self.axes[0].legend(loc="upper right")

    def update(self, real_ee, sim_ee, step_idx):
        if not self.enabled:
            return
        self.t.append(step_idx)
        for i in range(6):
            self.real[i].append(float(real_ee[i]))
            self.sim[i].append(float(sim_ee[i]))

        for i in range(6):
            self.lines_real[i].set_data(self.t, self.real[i])
            self.lines_sim[i].set_data(self.t, self.sim[i])
            self.axes[i].relim()
            self.axes[i].autoscale_view()

        self.fig.canvas.draw_idle()
        plt.pause(0.001)

def _body_pose_mm_rad_world(data, body_id):
    if body_id < 0:
        return np.zeros(6, dtype=np.float32)
    p = data.xpos[body_id].copy()
    r = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
    pos_mm = p * 1000.0
    if R is None:
        sy = np.sqrt(r[0,0]**2 + r[1,0]**2)
        if sy > 1e-6:
            rx = np.arctan2(r[2,1], r[2,2])
            ry = np.arctan2(-r[2,0], sy)
            rz = np.arctan2(r[1,0], r[0,0])
        else:
            rx = np.arctan2(-r[1,2], r[1,1])
            ry = np.arctan2(-r[2,0], sy)
            rz = 0
        return np.array([pos_mm[0], pos_mm[1], pos_mm[2], rx, ry, rz], dtype=np.float32)

    euler = R.from_matrix(r).as_euler("xyz", degrees=True)
    rx, ry, rz = np.deg2rad(euler)
    return np.array([pos_mm[0], pos_mm[1], pos_mm[2], rx, ry, rz], dtype=np.float32)

# ============================================================
# Data Recorder (Real/Sim)
# ============================================================
class DataRecorder:
    def __init__(self, output_dir):
        self.out = pathlib.Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.recording = False
        self.real_buf = []
        self.sim_buf = []
        self.step = 0

    def start(self):
        self.real_buf = []
        self.sim_buf = []
        self.recording = True
        self.step = 0
        logger.info("🔴 Recording started")

    def discard(self):
        self.real_buf = []
        self.sim_buf = []
        self.recording = False
        logger.warning("🗑️ Recording discarded")

    def add(self, real_ee, real_q, real_act, sim_ee, sim_q, sim_act, ts):
        if not self.recording:
            return
        self.real_buf.append({"ee": real_ee, "q": real_q, "act": real_act, "ts": ts})
        self.sim_buf.append({"ee": sim_ee, "q": sim_q, "act": sim_act, "ts": ts})
        self.step += 1

    def save(self):
        if not self.recording or not self.real_buf:
            self.recording = False
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        real_path = self.out / f"real_episode_{timestamp}.npz"
        sim_path = self.out / f"sim_episode_{timestamp}.npz"

        def pack(buf):
            ee = np.stack([x["ee"] for x in buf]).astype(np.float32)
            q = np.stack([x["q"] for x in buf]).astype(np.float32)
            act = np.stack([x["act"] for x in buf]).astype(np.float32)
            ts = np.stack([x["ts"] for x in buf]).astype(np.float32)
            return ee, q, act, ts

        real_ee, real_q, real_act, real_ts = pack(self.real_buf)
        sim_ee, sim_q, sim_act, sim_ts = pack(self.sim_buf)

        np.savez_compressed(real_path, ee_pose=real_ee, qpos=real_q, action=real_act, timestamp=real_ts)
        np.savez_compressed(sim_path, ee_pose=sim_ee, qpos=sim_q, action=sim_act, timestamp=sim_ts)

        logger.info(f"✅ Saved real data: {real_path}")
        logger.info(f"✅ Saved sim data: {sim_path}")

        self.recording = False
        self.real_buf = []
        self.sim_buf = []

def _pose_deg_to_rad(pose_deg):
    pose = np.array(pose_deg, dtype=np.float32)
    pose[3:6] = np.deg2rad(pose[3:6])
    return pose

def _quat_wxyz_to_euler_xyz(quat):
    w, x, y, z = quat
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    return np.array([roll, pitch, yaw], dtype=np.float64)

def _action_deg_to_rad(action):
    act = np.array(action, dtype=np.float32)
    act[3:6] = np.deg2rad(act[3:6])
    return act

# ============================================================
# Main
# ============================================================
def main():
    logger.info(colored("=== Digital Twin: Real-to-Sim ===", "cyan"))

    # Load MuJoCo model
    logger.info(f"📦 Loading model: {MODEL_PATH}")
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    # Link6 body for EE position (simulation)
    try:
        link6_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "6_Link")
    except:
        logger.warning("⚠️ Link6 body not found")
        link6_id = -1

    # Initialize gamepad and cameras
    gamepad = GamepadController()
    camera_mgr = OAKCameraManager()
    recorder = DataRecorder(OUTPUT_DIR)
    plotter = LiveEEPlotter()

    try:
        # Connect to robot
        logger.info(f"🔌 Connecting to robot at {ROBOT_ADDRESS}...")
        robot = mdr.Robot()
        robot.Connect(address=ROBOT_ADDRESS)

        if not robot.IsConnected():
            logger.error("❌ Failed to connect to robot")
            return

        logger.info("✅ Connected! Activating and homing...")
        robot.ActivateAndHome()
        robot.SetRealTimeMonitoring(1)

        logger.info("🏠 Moving to home position...")
        robot.MoveJoints(*HOME_JOINTS)
        robot.WaitIdle()
        logger.info("✅ Robot ready!")

        # Start state sampler
        sampler = RtSampler(robot, rate_hz=100)
        sampler.start()

        # Home pose comparison (real vs sim using DH FK)
        try:
            time.sleep(0.2)
            robot_q, robot_p = sampler.get_latest_data()
            data.qpos[:6] = np.deg2rad(robot_q)
            mujoco.mj_forward(model, data)
            sim_ee = _body_pose_mm_rad_world(data, link6_id)
            real_ee = _pose_deg_to_rad(robot_p)
            logger.info(f"[HOME] real EE (mm,rad): {real_ee}")
            logger.info(f"[HOME] sim  EE (mm,rad): {sim_ee}")
            pos_diff = np.linalg.norm(sim_ee[:3] - real_ee[:3])
            logger.info(f"[HOME] Position difference: {pos_diff:.2f} mm")
        except Exception as e:
            logger.warning(f"[HOME] compare failed: {e}")

        # Initialize cameras
        logger.info("📷 Initializing cameras...")
        num_cameras = camera_mgr.initialize_cameras()
        if num_cameras > 0:
            logger.info(f"✅ {num_cameras} camera(s) initialized")
        else:
            logger.warning("⚠️ No cameras available")

        logger.info(colored("\n=== CONTROLS ===", "cyan"))
        logger.info(" [Gamepad]    Control real robot (see Robot_action.py for details)")
        logger.info(" [B/A]        Rec(Start/Save) / Discard")
        logger.info(" [Y Button]   Go to home position")
        logger.info(" [START]      Exit program")
        logger.info(" Real robot movements will be mirrored in simulation!\n")

        # Set initial simulation pose
        data.qpos[:6] = np.deg2rad(HOME_JOINTS)
        mujoco.mj_forward(model, data)
        # Launch viewer
        with mujoco.viewer.launch_passive(model, data) as viewer:
            step_count = 0
            last_control_time = time.time()
            control_dt = 1.0 / CONTROL_FREQUENCY
            last_action = np.zeros(6, dtype=np.float32)
            last_plot_time = 0.0

            while viewer.is_running():
                step_start = time.time()

                # 0. Safety Check (auto-recover on robot error)
                try:
                    if robot.GetStatusRobot().error_status:
                        logger.warning("⚠️ Robot error detected. Auto-resetting...")
                        robot.ResetError()
                        time.sleep(0.1)
                        robot.ResumeMotion()
                        time.sleep(0.5)
                        continue
                except Exception as e:
                    logger.debug(f"Safety check failed: {e}")

                # Get real robot state
                robot_q, robot_p = sampler.get_latest_data()
                real_ee = _pose_deg_to_rad(robot_p)

                # Update simulation to match robot
                # Convert degrees to radians for MuJoCo
                sim_qpos = np.deg2rad(robot_q)
                data.qpos[:6] = sim_qpos
                mujoco.mj_forward(model, data)
                sim_q_deg = np.rad2deg(data.qpos[:6].copy()).astype(np.float32)
                sim_ee = _body_pose_mm_rad_world(data, link6_id)

                # Get camera frames
                frames = camera_mgr.get_frames()

                # Control robot with gamepad
                current_time = time.time()
                if current_time - last_control_time >= control_dt:
                    action, btn_rec, btn_disc, btn_home, btn_exit = gamepad.get_action()
                    last_action = action.copy()

                    if btn_exit:
                        logger.info(colored("🛑 Exit button pressed", "red"))
                        break

                    if btn_home:
                        logger.info(colored("🏠 Going home...", "yellow"))
                        try:
                            robot.MoveJoints(*HOME_JOINTS)
                            robot.WaitIdle()
                        except Exception as e:
                            logger.error(f"Home failed: {e}")

                    if btn_rec:
                        if not recorder.recording:
                            recorder.start()
                            time.sleep(0.3)
                        else:
                            recorder.save()
                            time.sleep(0.3)

                    if btn_disc and recorder.recording:
                        recorder.discard()
                        time.sleep(0.3)

                    # Send movement command to robot
                    if np.any(np.abs(action) > 0.001):
                        try:
                            robot.MoveLinRelTrf(*[float(x) for x in action])
                        except Exception as e:
                            logger.debug(f"Move command failed: {e}")

                    last_control_time = current_time

                # Live plot (real vs sim EE)
                if plotter.enabled:
                    if current_time - last_plot_time >= 1.0 / float(PLOT_UPDATE_HZ):
                        plotter.update(real_ee, sim_ee, step_count)
                        last_plot_time = current_time

                # Record data (real/sim)
                if recorder.recording:
                    real_q = np.array(robot_q, dtype=np.float32)
                    real_act = _action_deg_to_rad(last_action)

                    sim_q = sim_q_deg
                    sim_act = real_act.copy()

                    recorder.add(real_ee, real_q, real_act, sim_ee, sim_q, sim_act, time.time())

                # Display camera feeds
                if num_cameras > 0 and step_count % 2 == 0:
                    img_list = []
                    for i in range(num_cameras):
                        key = f"camera{i+1}"
                        if key in frames:
                            img = frames[key].copy()
                        else:
                            img = np.zeros((camera_mgr.height, camera_mgr.width, 3), dtype=np.uint8)
                        cv2.putText(img, key, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                   0.7, (255, 255, 0), 2)
                        img_list.append(img)

                    if img_list:
                        try:
                            # Combine horizontally
                            if len(img_list) > 1:
                                heights = [img.shape[0] for img in img_list]
                                if len(set(heights)) > 1:
                                    target_h = min(heights)
                                    resized = []
                                    for img in img_list:
                                        if img.shape[0] != target_h:
                                            aspect = img.shape[1] / img.shape[0]
                                            target_w = int(target_h * aspect)
                                            resized.append(cv2.resize(img, (target_w, target_h)))
                                        else:
                                            resized.append(img)
                                    combined = np.hstack(resized)
                                else:
                                    combined = np.hstack(img_list)
                            else:
                                combined = img_list[0]

                            # Add status text
                            cv2.putText(combined, "Real-to-Sim Mirror",
                                       (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                       (0, 255, 0), 2)

                            # Display joint positions
                            joint_text = f"Joints: [{', '.join([f'{x:.1f}' for x in robot_q])}]"
                            cv2.putText(combined, joint_text,
                                       (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                       (255, 255, 255), 1)

                            cv2.imshow("Real Robot Cameras", combined)

                            if cv2.waitKey(1) == ord('q'):
                                break

                        except Exception as e:
                            logger.debug(f"Display error: {e}")

                # Sync viewer
                viewer.sync()
                step_count += 1

                # Rate limiting
                elapsed = time.time() - step_start
                if elapsed < model.opt.timestep:
                    time.sleep(model.opt.timestep - elapsed)

    except KeyboardInterrupt:
        logger.info("\n⏹️ Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
    finally:
        # Cleanup
        if 'sampler' in locals():
            sampler.stop()
            sampler.join(timeout=2.0)

        if 'robot' in locals() and robot.IsConnected():
            robot.DeactivateRobot()
            robot.Disconnect()

        if 'camera_mgr' in locals():
            camera_mgr.close()

        cv2.destroyAllWindows()
        logger.info("✅ Cleanup complete")

if __name__ == "__main__":
    main()
