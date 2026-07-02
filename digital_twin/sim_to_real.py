#!/usr/bin/env python
"""
Digital Twin: Sim-to-Real Synchronization
Streams MuJoCo simulation state → Real Mecademic Robot

Architecture:
- Simulation runs in main thread (MuJoCo viewer)
- Real robot controller in separate thread
- State synchronization via thread-safe queue
"""

import os
import sys
from pathlib import Path
import time
import threading
import logging
import numpy as np
import queue
from termcolor import colored
import cv2
import contextlib

import mujoco
import mujoco.viewer
import mecademicpy.robot as mdr
import depthai as dai

# === Configuration ===
ROBOT_ADDRESS = "192.168.0.100"
REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = str(REPO_ROOT / "Sim" / "meca_add.xml")
SYNC_FREQUENCY = 15  # Hz (match robot control frequency)
HOME_JOINTS = (30, -20, 20, 0, 30, 60)

# Position control gains
POSITION_GAIN = 1.0  # Scale factor for position matching
ROTATION_GAIN = 1.0  # Scale factor for rotation matching

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# Real Robot Controller Thread
# ============================================================
class RealRobotController(threading.Thread):
    def __init__(self, robot_address, command_queue, state_queue):
        super().__init__(daemon=True)
        self.robot_address = robot_address
        self.command_queue = command_queue  # Receives target poses from sim
        self.state_queue = state_queue      # Sends current robot state back
        self.robot = None
        self.running = True
        self.last_target_pose = None

    def run(self):
        logger.info(f"🔌 Connecting to robot at {self.robot_address}...")

        try:
            self.robot = mdr.Robot()
            self.robot.Connect(address=self.robot_address)

            if not self.robot.IsConnected():
                logger.error(f"❌ Failed to connect to robot")
                return

            logger.info("✅ Robot connected! Activating and homing...")
            self.robot.ActivateAndHome()
            self.robot.SetRealTimeMonitoring(1)

            logger.info("🏠 Moving to home position...")
            self.robot.MoveJoints(*HOME_JOINTS)
            self.robot.WaitIdle()
            logger.info("✅ Robot ready for sync!")

            # Main control loop
            control_dt = 1.0 / SYNC_FREQUENCY
            next_time = time.time()

            while self.running:
                try:
                    # Get target pose from simulation
                    target_pose = self.command_queue.get(timeout=0.1)
                    self.last_target_pose = target_pose

                    # Send current state back
                    try:
                        current_joints = np.array(list(self.robot.GetJoints()[:6]))
                        current_pose = np.array(list(self.robot.GetPose()[:6]))
                        self.state_queue.put({
                            'qpos': current_joints,
                            'ee_pose': current_pose,
                            'timestamp': time.time()
                        })
                    except:
                        pass

                    # Execute motion to match simulation
                    if target_pose is not None:
                        # Convert from simulation coordinates to robot coordinates
                        # Sim uses: [x, y, z, rx, ry, rz] in meters/radians
                        # Robot uses: [x, y, z, rx, ry, rz] in mm/degrees

                        # Option 1: Joint-space control (more accurate)
                        if 'qpos' in target_pose:
                            target_joints = target_pose['qpos']
                            try:
                                self.robot.MoveJoints(*target_joints.tolist())
                            except Exception as e:
                                logger.debug(f"Joint move failed: {e}")

                        # Option 2: Cartesian-space control
                        elif 'ee_pose' in target_pose:
                            target_ee = target_pose['ee_pose']
                            # Convert meters to mm, radians to degrees
                            target_mm = target_ee[:3] * 1000.0
                            target_deg = np.rad2deg(target_ee[3:])
                            target_cart = np.concatenate([target_mm, target_deg])

                            try:
                                self.robot.MovePose(*target_cart.tolist())
                            except Exception as e:
                                logger.debug(f"Cartesian move failed: {e}")

                    # Rate limiting
                    next_time += control_dt
                    sleep_time = next_time - time.time()
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Control loop error: {e}")

        except Exception as e:
            logger.error(f"❌ Robot controller error: {e}")
        finally:
            if self.robot and self.robot.IsConnected():
                self.robot.DeactivateRobot()
                self.robot.Disconnect()
                logger.info("🔌 Robot disconnected")

    def stop(self):
        self.running = False

# ============================================================
# Simulation State Publisher
# ============================================================
class SimulationPublisher:
    def __init__(self, model, data):
        self.model = model
        self.data = data

        # Get site IDs
        try:
            self.tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "needle_tip")
            self.back_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "needle_back")
        except:
            logger.warning("⚠️ Needle sites not found in model")
            self.tip_id = -1
            self.back_id = -1

    def get_current_state(self):
        """Extract current simulation state for robot synchronization"""
        state = {
            'timestamp': self.data.time,
            'qpos': np.rad2deg(self.data.qpos[:6].copy()),  # Convert to degrees
        }

        # Get end-effector pose if sites are available
        if self.tip_id >= 0:
            tip_pos = self.data.site_xpos[self.tip_id].copy()  # In meters
            tip_mat = self.data.site_xmat[self.tip_id].reshape(3, 3)

            # Convert rotation matrix to Euler angles (XYZ convention)
            sy = np.sqrt(tip_mat[0,0]**2 + tip_mat[1,0]**2)
            if sy > 1e-6:
                rx = np.arctan2(tip_mat[2,1], tip_mat[2,2])
                ry = np.arctan2(-tip_mat[2,0], sy)
                rz = np.arctan2(tip_mat[1,0], tip_mat[0,0])
            else:
                rx = np.arctan2(-tip_mat[1,2], tip_mat[1,1])
                ry = np.arctan2(-tip_mat[2,0], sy)
                rz = 0

            state['ee_pose'] = np.concatenate([tip_pos, [rx, ry, rz]])

        return state

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
# Main
# ============================================================
def main():
    logger.info(colored("=== Digital Twin: Sim-to-Real ===", "cyan"))

    # Load MuJoCo model
    logger.info(f"📦 Loading model: {MODEL_PATH}")
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    # Initialize simulation publisher
    sim_publisher = SimulationPublisher(model, data)

    # Initialize cameras
    camera_mgr = OAKCameraManager()
    logger.info("📷 Initializing cameras...")
    num_cameras = camera_mgr.initialize_cameras()
    if num_cameras > 0:
        logger.info(f"✅ {num_cameras} camera(s) initialized")
    else:
        logger.warning("⚠️ No cameras available")

    # Create communication queues
    command_queue = queue.Queue(maxsize=1)  # Sim → Robot
    state_queue = queue.Queue(maxsize=10)   # Robot → Display

    # Start robot controller
    robot_controller = RealRobotController(ROBOT_ADDRESS, command_queue, state_queue)
    robot_controller.start()

    # Wait for robot to be ready
    time.sleep(3.0)

    # Set initial pose
    home_pose = np.array([0, 0, 0, 0, 0, 0])
    data.qpos[:6] = home_pose
    mujoco.mj_forward(model, data)

    logger.info(colored("\n=== CONTROLS ===", "cyan"))
    logger.info(" [Mouse]      Rotate/Pan view")
    logger.info(" [Arrow Keys] Move simulation (will sync to robot)")
    logger.info(" [Q]          Quit")
    logger.info(" [R]          Reset to home\n")

    # Launch viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        last_sync_time = time.time()
        sync_dt = 1.0 / SYNC_FREQUENCY
        step_count = 0

        # Variables for robot state display
        robot_qpos = None
        robot_ee_pose = None
        sync_error = 0.0

        while viewer.is_running():
            step_start = time.time()

            # Simple keyboard control for testing
            # (In production, this would be your autonomous controller)
            viewer.sync()

            # Step simulation
            mujoco.mj_step(model, data)
            step_count += 1

            # Publish state to robot at sync frequency
            current_time = time.time()
            if current_time - last_sync_time >= sync_dt:
                sim_state = sim_publisher.get_current_state()

                # Send to robot (non-blocking)
                try:
                    command_queue.put_nowait(sim_state)
                except queue.Full:
                    command_queue.get()  # Remove old command
                    command_queue.put_nowait(sim_state)

                last_sync_time = current_time

                # Get robot feedback
                try:
                    robot_state = state_queue.get_nowait()
                    robot_qpos = robot_state['qpos']
                    robot_ee_pose = robot_state['ee_pose']

                    # Calculate sync error (position difference)
                    if 'ee_pose' in sim_state:
                        sim_pos = sim_state['ee_pose'][:3] * 1000.0  # Convert to mm
                        robot_pos = robot_ee_pose[:3]
                        sync_error = np.linalg.norm(sim_pos - robot_pos)

                except queue.Empty:
                    pass

            # Display camera feeds
            if num_cameras > 0 and step_count % 2 == 0:
                frames = camera_mgr.get_frames()
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

                        cv2.putText(combined, "Sim-to-Real Cameras",
                                   (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                   (0, 255, 0), 2)

                        cv2.imshow("Sim-to-Real Cameras", combined)

                        if cv2.waitKey(1) == ord('q'):
                            break

                    except Exception as e:
                        logger.debug(f"Display error: {e}")

            # Display status every 30 steps
            if step_count % 30 == 0:
                sim_qpos = np.rad2deg(data.qpos[:6])

                print(f"\r🔄 Sync Status | "
                      f"Sim Joints: [{', '.join([f'{x:.1f}' for x in sim_qpos])}] | "
                      f"Error: {sync_error:.2f}mm    ", end='')

            # Rate limiting
            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)

    # Cleanup
    if "camera_mgr" in locals():
        camera_mgr.close()
    cv2.destroyAllWindows()

    robot_controller.stop()
    robot_controller.join(timeout=5.0)
    logger.info("\n✅ Shutdown complete")

if __name__ == "__main__":
    main()
